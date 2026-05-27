"""OddBall (N2/P3) EEG Processing Pipeline.

Processes BrainVision EEG files from the auditory oddball paradigm through
ICA artifact removal, epoching, ERP averaging, and N2/P3 difference wave
computation. Generates a self-contained HTML report per recording using
MNE-Python.

Stimuli: MAU (speech syllable) or TONE (pure tone).
Design:  Intervention (treat) vs Control, Pre (v1) vs Post (v2).
Markers: S 1 = standard, S 2 = deviant, S 3 = button-press response.

Usage:
    python process_oddball.py --input <vhdr_or_folder> --output <output_dir>
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import io
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Russian/Unicode text
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                  errors='replace')

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA

# ── Constants ────────────────────────────────────────────────────────────────

FILTER_L_FREQ = 0.1
FILTER_H_FREQ = 30.0
RESAMPLE_FREQ = 256
ICA_N_COMPONENTS = 35          # spec: components 1-35 (0_Алгоритм препроцессинга)
ICA_METHOD = 'infomax'         # spec: Infomax / runica.m
ICA_MAX_ITER = 1000
EPOCH_TMIN = -0.3
EPOCH_TMAX = 1.2
BASELINE = (-0.3, 0)
EPOCH_REJECT = dict(eeg=150e-6)  # 150 uV peak-to-peak amplitude threshold
MONTAGE_NAME = 'standard_1005'   # supports extended 10-10/10-5 channel names

# N2 measurement window (seconds) — fronto-central, 150-200 ms
N2_TMIN = 0.15
N2_TMAX = 0.20

# P3 measurement window (seconds) — centro-parietal, 300-400 ms
P3_TMIN = 0.30
P3_TMAX = 0.40

# Electrodes of interest for N2 (fronto-central)
N2_CHANNELS = ['Fz', 'FCz', 'Cz', 'F3', 'F4', 'FC1', 'FC2']

# Electrodes of interest for P3 (centro-parietal)
P3_CHANNELS = ['Cz', 'CPz', 'Pz', 'CP1', 'CP2', 'P3', 'P4']

# Full electrode set for per-electrode tables (Tech Work Table 6.1)
TARGET_ELECTRODES = [
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FCz', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CPz', 'CP6',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO3', 'POz', 'PO4', 'Oz',
]

# Non-EEG channel names to drop
NON_EEG_PATTERNS = ['ECG', 'EOG']

# QC thresholds
QC_MAX_EPOCH_REJECT_PCT = 25    # >25% rejected epochs → QC fail
QC_MAX_MISSING_TARGET_CHS = 2   # ≥2 target channels missing → QC fail
QC_MAX_RAW_ARTIFACT_PCT = 15    # >15% raw segments with artifacts → QC fail

# Normative thresholds by age group, with literature references.
NORMS = {
    'child': {
        'n2_amp_normal':  (-15.0, -2.0),    # uV, negative at Fz/FCz
        'n2_amp_warn':    (-20.0, -0.5),
        'n2_lat_normal':  (150, 350),        # ms
        'n2_lat_warn':    (100, 400),
        'p3_amp_normal':  (3.0, 25.0),       # uV, positive at Pz/Cz
        'p3_amp_warn':    (1.0, 35.0),
        'p3_lat_normal':  (250, 600),        # ms
        'p3_lat_warn':    (200, 700),
        'erp_amp_max':    30.0,
        'min_epochs_std': 100,
        'min_epochs_dev': 25,
        'baseline_max':   0.5,
        'reject_warn':    30,
        'reject_fail':    50,
    },
    'unknown': {
        'n2_amp_normal':  (-15.0, -2.0),
        'n2_amp_warn':    (-20.0, -0.5),
        'n2_lat_normal':  (150, 350),
        'n2_lat_warn':    (100, 400),
        'p3_amp_normal':  (3.0, 25.0),
        'p3_amp_warn':    (1.0, 35.0),
        'p3_lat_normal':  (250, 600),
        'p3_lat_warn':    (200, 700),
        'erp_amp_max':    30.0,
        'min_epochs_std': 100,
        'min_epochs_dev': 25,
        'baseline_max':   0.5,
        'reject_warn':    30,
        'reject_fail':    50,
    },
}

# ── Longitudinal subject mapping (Структура файлов.xlsx) ───────────────────
# M_y3 subjects re-recorded as L_y1 at v2. Key = L_y1 stem, value = M_y3 subject_id.
LONGITUDINAL_MAP = {
    'L_y1_001': 'M_y3_NewLab_038',
    'L_y1_002': 'M_y3_NewLab_29',
    'L_y1_003': 'M_y3_NewLab_031',
    'L_y1_005': 'M_y3_NewLab_032',
    'L_y1_006': 'M_y3_NewLab_030',
    'L_y1_007': 'M_y3_NewLab_036',
    'L_y1_009': 'M_y3_NewLab_043',
    'L_y1_010': 'M_y3_NewLab_040',
    'L_y1_011': 'M_y3_NewLab_042',
    'L_y1_013': 'M_y3_NewLab_044',
}

# ── Authoritative group assignment (Структура файлов.xlsx) ─────────────────
# Keyed by canonical subject_id (after longitudinal remapping).
GROUP_ASSIGNMENT = {
    # Co_y6 — treatment
    'Co_y6_020': 'treat', 'Co_y6_021': 'treat', 'Co_y6_022': 'treat',
    'Co_y6_024': 'treat', 'Co_y6_025': 'treat', 'Co_y6_026': 'treat',
    'Co_y6_027': 'treat', 'Co_y6_029': 'treat', 'Co_y6_037': 'treat',
    'Co_y6_038': 'treat', 'Co_y6_039': 'treat', 'Co_y6_043': 'treat',
    'Co_y6_044': 'treat', 'Co_y6_045': 'treat', 'Co_y6_050': 'treat',
    'Co_y6_053': 'treat', 'Co_y6_056': 'treat', 'Co_y6_058': 'treat',
    # Co_y6 — control
    'Co_y6_041': 'control', 'Co_y6_042': 'control', 'Co_y6_046': 'control',
    'Co_y6_049': 'control', 'Co_y6_054': 'control', 'Co_y6_060': 'control',
    'Co_y6_061': 'control', 'Co_y6_062': 'control', 'Co_y6_063': 'control',
    'Co_y6_064': 'control', 'Co_y6_066': 'control', 'Co_y6_067': 'control',
    'Co_y6_068': 'control', 'Co_y6_069': 'control', 'Co_y6_070': 'control',
    'Co_y6_071': 'control',
    # M_y3 — treatment
    'M_y3_NewLab_002': 'treat', 'M_y3_NewLab_003': 'treat',
    'M_y3_NewLab_004': 'treat', 'M_y3_NewLab_005': 'treat',
    'M_y3_NewLab_009': 'treat', 'M_y3_NewLab_012': 'treat',
    'M_y3_NewLab_015': 'treat', 'M_y3_NewLab_016': 'treat',
    # M_y3 — control
    'M_y3_NewLab_006': 'control', 'M_y3_NewLab_007': 'control',
    'M_y3_NewLab_017': 'control',
    # Longitudinal M_y3 — treatment (L_y1 mapped back to M_y3 IDs above)
    'M_y3_NewLab_29': 'treat', 'M_y3_NewLab_030': 'treat',
    'M_y3_NewLab_031': 'treat', 'M_y3_NewLab_032': 'treat',
    'M_y3_NewLab_036': 'treat', 'M_y3_NewLab_038': 'treat',
    'M_y3_NewLab_042': 'treat', 'M_y3_NewLab_043': 'treat',
    'M_y3_NewLab_044': 'treat',
    # Longitudinal M_y3 — control
    'M_y3_NewLab_040': 'control',
}

# ── Subject processing notes (from manual EEGLAB sessions) ─────────────────
SUBJECT_NOTES = {
    'M_y3_NewLab_29':  'MAU: 22 ICA components removed in manual processing (very noisy)',
    'M_y3_NewLab_044': 'TONE: manual processing used FIR 2-40 Hz (different filter)',
    'M_y3_NewLab_031': 'OddBall noted as 2:40 in manual log',
    'M_y3_NewLab_036': 'Electrode placement issue noted: "глаза на затылке"',
}

REFERENCES = """
REFERENCES — normative values and thresholds used in this validation:

[1] Polich J (2007).
    "Updating P300: An integrative theory of P3a and P3b."
    Clinical Neurophysiology, 118(10):2128-2148.
    doi:10.1016/j.clinph.2007.04.019
    >> P3b (target/deviant oddball) maximal at Pz, amplitude 5-20 uV
       in adults, latency 250-500 ms. Children show larger amplitudes
       and longer latencies.
    >> Used for: P3 amplitude and latency normal ranges.

[2] Johnstone SJ, Barry RJ, Anderson JW, Coyle SF (1996).
    "Age-related changes in child and adolescent event-related potential
    component morphology, amplitude and latency to standard and target
    stimuli in an auditory oddball task."
    International Journal of Psychophysiology, 24(3):223-238.
    doi:10.1016/S0167-8760(96)00065-7
    >> Children 6-8 years: P3 latency 350-600 ms, N2 latency 200-350 ms.
    >> P3 amplitude in children can reach 20-25 uV.
    >> Used for: child N2/P3 latency and amplitude ranges.

[3] Duncan CC, Barry RJ, Connolly JF, Fischer C, Michie PT, et al. (2009).
    "Event-related potentials in clinical research: Guidelines for eliciting,
    recording, and quantifying mismatch negativity, P300, and N400."
    Clinical Neurophysiology, 120(11):1883-1908.
    doi:10.1016/j.clinph.2009.07.045
    >> Minimum epochs: 20-30 deviant, 100+ standard for stable P3.
    >> Amplitude rejection: +/-75 to +/-150 uV typical.
    >> Used for: minimum epoch counts, rejection thresholds.

[4] Patel SH, Bhatt R, Engel AK (2005).
    "N200 and P300 as indices of cognitive and emotional processing."
    International Journal of Psychophysiology, 57(1):1-9.
    >> N200 reflects early deviance detection and conflict monitoring.
    >> Maximal at fronto-central sites (Fz, FCz, Cz), negative polarity.
    >> Used for: N2 channel selection and polarity checks.

[5] Picton TW, Bentin S, Berg P, Donchin E, Hillyard SA, et al. (2000).
    "Guidelines for using human event-related potentials to study
    cognition: Recording standards and publication criteria."
    Psychophysiology, 37(2):127-152.
    doi:10.1111/1469-8986.3720127
    >> Baseline correction: pre-stimulus mean ~ 0 uV.
    >> ERP amplitudes exceeding 20-25 uV suggest residual artifact.
    >> Used for: baseline and ERP amplitude sanity checks.

[6] Courchesne E (1978).
    "Neurophysiological correlates of cognitive development: changes
    in long-latency event-related potentials from childhood to adulthood."
    Electroencephalography and Clinical Neurophysiology, 45(4):468-482.
    >> N2 in children: larger amplitude (-5 to -15 uV), longer latency
       (200-350 ms) compared to adults.
    >> Used for: child N2 amplitude and latency warning ranges.
"""


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir, recording_id):
    """Configure logging to console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{recording_id}_{timestamp}.log'

    logger = logging.getLogger(f'oddball_{recording_id}')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%H:%M:%S')

    fh = logging.FileHandler(str(log_file), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f'Log file: {log_file}')
    return logger


# ── File Discovery ───────────────────────────────────────────────────────────

def discover_vhdr_files(input_path):
    """Find .vhdr files in the input path."""
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == '.vhdr':
        return [p]
    if p.is_dir():
        files = sorted(p.rglob('*.vhdr'))
        if not files:
            raise FileNotFoundError(f'No .vhdr files found in {p}')
        return files
    raise FileNotFoundError(f'Input path does not exist: {p}')


def extract_file_info(vhdr_path):
    """Extract metadata from filename and path structure.

    Handles three cohort naming schemes:
      - Co_y6_NNN          — 6-year-olds, v2 indicated by _v2 in filename
      - M_y3_NewLab_NNN    — 3-year-olds, v2 indicated by _v2 suffix
      - L_y1_NNN           — longitudinal v2 of an M_y3 subject (LONGITUDINAL_MAP)

    Returns dict with keys:
        subject_id   — canonical ID (L_y1 remapped to M_y3 counterpart)
        original_id  — raw ID as extracted from filename (before remapping)
        cohort       — 'Co_y6' | 'M_y3' | 'L_y1'
        visit        — 'v1' | 'v2'
        stimulus     — 'MAU' | 'TONE'
        group        — 'control' | 'treat'
        recording_id — unique key for output naming
    """
    fname = vhdr_path.stem  # e.g. Co_y6_041_Mau or Co_y6_041_v2_mau
    parent = vhdr_path.parent.name.lower()  # e.g. 'mau control'

    # ── Group from parent folder (fallback) ──
    folder_group = 'unknown'
    if 'control' in parent:
        folder_group = 'control'
    elif 'treat' in parent:
        folder_group = 'treat'

    # ── Stimulus type from parent folder or filename ──
    stimulus = 'unknown'
    if 'mau' in parent or 'mau' in fname.lower():
        stimulus = 'MAU'
    elif 'tone' in parent or 'tone' in fname.lower():
        stimulus = 'TONE'
    elif 'hunt' in parent or 'hunt' in fname.lower():
        stimulus = 'HUNT'

    # ── Visit from filename ──
    # Match _v2 followed by separator, dot, or end-of-string
    visit = 'v1'
    if re.search(r'_v2(?:[_.]|$)', fname, re.IGNORECASE):
        visit = 'v2'

    # ── Subject ID ──
    # Patterns: Co_y6_041, M_y3_NewLab_006, L_y1_010
    match = re.match(
        r'((?:Co_[Yy]6|M_y3_NewLab|L_y1)_\d+)',
        fname, re.IGNORECASE
    )
    if match:
        subject_id = match.group(1)
    else:
        # Fallback: take everything before stimulus/visit suffix
        subject_id = re.sub(r'[_]?v?2?[_]?(?:mau|tone|hunt).*$', '',
                            fname, flags=re.IGNORECASE)

    # Normalize case for Co_Y6 → Co_y6
    if subject_id.upper().startswith('CO_Y6'):
        subject_id = 'Co_y6' + subject_id[5:]

    original_id = subject_id

    # ── Longitudinal remapping: L_y1 → M_y3 (always v2) ──
    if subject_id in LONGITUDINAL_MAP:
        subject_id = LONGITUDINAL_MAP[subject_id]
        visit = 'v2'  # L_y1 files are always the second measurement

    # ── Cohort from canonical subject_id ──
    parts = subject_id.split('_')
    cohort = f'{parts[0]}_{parts[1]}' if len(parts) >= 2 else 'unknown'

    # ── Authoritative group assignment ──
    group = GROUP_ASSIGNMENT.get(subject_id, folder_group)

    # ── Recording ID (unique output key) ──
    recording_id = f'{subject_id}_{visit}_{stimulus}'.lower()

    return {
        'subject_id': subject_id,
        'original_id': original_id,
        'cohort': cohort,
        'visit': visit,
        'stimulus': stimulus,
        'group': group,
        'recording_id': recording_id,
    }


# ── Channel Preparation ─────────────────────────────────────────────────────

def prepare_channels(raw, logger):
    """Set channel types, drop non-EEG channels, set montage."""
    ch_names = raw.ch_names
    logger.info(f'Channels ({len(ch_names)}): {ch_names[:10]}... '
                f'(showing first 10)')
    logger.info(f'Sampling rate: {raw.info["sfreq"]} Hz')
    logger.info(f'Duration: {raw.times[-1]:.1f} s')

    # ── Identify and set non-EEG channel types ──
    # Keep EOG/ECG for ICA correlation — they are dropped after ICA step
    eog_ch = None
    for ch in ch_names:
        ch_upper = ch.upper()
        if ch_upper == 'EOG':
            raw.set_channel_types({ch: 'eog'})
            eog_ch = ch
            logger.info(f'Set {ch} channel type to eog')
        elif ch_upper == 'ECG':
            raw.set_channel_types({ch: 'ecg'})
            logger.info(f'Set {ch} channel type to ecg')

    if eog_ch is None:
        logger.warning('No EOG channel found — will use Fp1 as proxy')

    # ── Drop BIP / non-standard channels ──
    bip_chs = [ch for ch in raw.ch_names if ch.startswith('BIP')]
    if bip_chs:
        raw.drop_channels(bip_chs)
        logger.info(f'Dropped {len(bip_chs)} BIP channel(s): {bip_chs}')

    nonstandard = [ch for ch in raw.ch_names if '_' in ch]
    if nonstandard:
        raw.drop_channels(nonstandard)
        logger.info(f'Dropped non-standard channel(s): {nonstandard}')

    # ── Set montage (ignore missing for extended channel names) ──
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    raw.set_montage(montage, on_missing='ignore')
    n_with_pos = sum(1 for ch in raw.info['chs']
                     if np.any(ch['loc'][:3] != 0))
    logger.info(f'Set {MONTAGE_NAME} montage '
                f'({n_with_pos}/{len(raw.ch_names)} channels with positions)')

    return raw, eog_ch


# ── Bad Channel Detection ────────────────────────────────────────────────────

def detect_bad_channels(raw, logger):
    """Detect bad EEG channels using variance and per-segment amplitude.

    Two criteria:
      1. Robust z-score on per-channel variance (z > 5 = noisy, var < 1% median = flat)
      2. Channels exceeding +/-150 uV in >50% of 1-second segments
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    all_eeg_names = [raw.ch_names[i] for i in eeg_picks]
    all_data = raw.get_data(picks=eeg_picks)

    # Only check channels with montage positions (needed for interpolation)
    valid_mask = [np.any(raw.info['chs'][p]['loc'][:3] != 0) for p in eeg_picks]
    eeg_ch_names = [ch for ch, v in zip(all_eeg_names, valid_mask) if v]
    data = all_data[valid_mask]

    if not eeg_ch_names:
        logger.info('No channels with montage positions to check')
        return []

    bad_chs = set()

    # --- Method 1: Variance-based ---
    ch_var = np.var(data, axis=1)
    median_var = np.median(ch_var)
    mad_var = np.median(np.abs(ch_var - median_var))

    if mad_var > 0:
        z_scores = 0.6745 * (ch_var - median_var) / mad_var
        for i, ch in enumerate(eeg_ch_names):
            if z_scores[i] > 5.0:
                bad_chs.add(ch)
                logger.info(f'  Bad channel (high variance): {ch} '
                            f'(z={z_scores[i]:.1f})')
            elif ch_var[i] < median_var * 0.01:
                bad_chs.add(ch)
                logger.info(f'  Bad channel (flat): {ch}')

    # --- Method 2: Segment-based ---
    sfreq = raw.info['sfreq']
    segment_len = int(sfreq)
    n_segments = data.shape[1] // segment_len
    threshold = 150e-6

    if n_segments > 0:
        seg_data = data[:, :n_segments * segment_len].reshape(
            len(eeg_ch_names), n_segments, segment_len)
        exceeds = np.any(np.abs(seg_data) > threshold, axis=2)
        bad_pct = exceeds.mean(axis=1) * 100

        for i, ch in enumerate(eeg_ch_names):
            if ch in bad_chs:
                continue
            if bad_pct[i] > 50:
                bad_chs.add(ch)
                logger.info(f'  Bad channel (>50% segments artifacted): {ch} '
                            f'({bad_pct[i]:.0f}%)')

    bad_list = sorted(bad_chs)
    if bad_list:
        raw.info['bads'] = list(set(raw.info.get('bads', []) + bad_list))
        logger.warning(f'Bad channels detected ({len(bad_list)}): {bad_list}')
    else:
        logger.info('No bad channels detected')

    return bad_list


# ── Event Mapping ────────────────────────────────────────────────────────────

def build_event_id(annotations_event_id, logger):
    """Map BrainVision annotations to condition names.

    Matches S 1 / S  1 / S   1 → standard
              S 2 / S  2 / S   2 → deviant
              S 3 / S  3 / S   3 → response
    Handles variable whitespace in marker descriptions.
    """
    event_id = {}
    patterns = {
        'standard': re.compile(r'S\s+1$'),
        'deviant':  re.compile(r'S\s+2$'),
        'response': re.compile(r'S\s+3$'),
    }

    for annot_key, annot_val in annotations_event_id.items():
        for condition, pattern in patterns.items():
            if pattern.search(annot_key.strip()):
                event_id[condition] = annot_val
                logger.debug(f'  Mapped "{annot_key}" -> {condition} '
                             f'(event_id={annot_val})')
                break

    return event_id


# ── Behavioral Data Extraction ───────────────────────────────────────────────

def extract_behavioral_data(events, event_id, sfreq, logger):
    """Extract response times and accuracy from S2→S3 event pairs.

    Returns dict with: hit_count, miss_count, false_alarm_count,
    hit_rate, mean_rt, std_rt, all_rts.
    """
    behavioral = {
        'hit_count': 0, 'miss_count': 0, 'false_alarm_count': 0,
        'hit_rate': 0.0, 'mean_rt_ms': 0.0, 'std_rt_ms': 0.0,
        'all_rts_ms': [],
    }

    if 'deviant' not in event_id or 'response' not in event_id:
        logger.info('  Behavioral: deviant or response events missing')
        return behavioral

    dev_code = event_id['deviant']
    resp_code = event_id['response']
    std_code = event_id.get('standard', None)

    dev_samples = events[events[:, 2] == dev_code, 0]
    resp_samples = events[events[:, 2] == resp_code, 0]

    rt_min_samples = int(0.1 * sfreq)   # 100 ms minimum RT
    rt_max_samples = int(1.5 * sfreq)   # 1500 ms maximum RT

    matched_responses = set()

    for dev_s in dev_samples:
        # Find first response within [100, 1500] ms after deviant
        candidates = resp_samples[
            (resp_samples > dev_s + rt_min_samples) &
            (resp_samples < dev_s + rt_max_samples)
        ]
        if len(candidates) > 0:
            resp_s = candidates[0]
            rt_ms = (resp_s - dev_s) / sfreq * 1000
            behavioral['all_rts_ms'].append(round(rt_ms, 1))
            behavioral['hit_count'] += 1
            matched_responses.add(resp_s)
        else:
            behavioral['miss_count'] += 1

    # False alarms: responses not matched to any deviant
    behavioral['false_alarm_count'] = len(
        [r for r in resp_samples if r not in matched_responses])

    n_deviants = len(dev_samples)
    if n_deviants > 0:
        behavioral['hit_rate'] = round(
            behavioral['hit_count'] / n_deviants * 100, 1)

    rts = behavioral['all_rts_ms']
    if rts:
        behavioral['mean_rt_ms'] = round(np.mean(rts), 1)
        behavioral['std_rt_ms'] = round(np.std(rts), 1)

    logger.info(f'  Behavioral: {behavioral["hit_count"]} hits, '
                f'{behavioral["miss_count"]} misses, '
                f'{behavioral["false_alarm_count"]} false alarms '
                f'(hit rate {behavioral["hit_rate"]}%)')
    if rts:
        logger.info(f'  RT: {behavioral["mean_rt_ms"]} +/- '
                    f'{behavioral["std_rt_ms"]} ms')

    return behavioral


# ── Per-Electrode ERP Metrics ────────────────────────────────────────────────

def measure_erp_component(evoked, tmin, tmax, polarity='negative'):
    """Measure ERP component metrics at each EEG electrode.

    Args:
        evoked: MNE Evoked object.
        tmin, tmax: Time window in seconds.
        polarity: 'negative' for N2 (find min), 'positive' for P3 (find max).

    Returns dict: {channel: {'mean_amp', 'peak_amp', 'peak_lat', 'area_lat_50'}}
    """
    t_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    if not t_mask.any():
        return {}
    times_ms = evoked.times[t_mask] * 1000
    results = {}

    for ch_idx, ch_name in enumerate(evoked.ch_names):
        if evoked.get_channel_types()[ch_idx] != 'eeg':
            continue
        data_uv = evoked.data[ch_idx, t_mask] * 1e6

        if polarity == 'negative':
            peak_amp = float(data_uv.min())
            peak_lat = float(times_ms[np.argmin(data_uv)])
        else:
            peak_amp = float(data_uv.max())
            peak_lat = float(times_ms[np.argmax(data_uv)])

        # 50% area latency
        cumulative = np.cumsum(np.abs(data_uv))
        half_area = cumulative[-1] / 2.0
        area_idx = np.searchsorted(cumulative, half_area)
        area_lat = float(times_ms[min(area_idx, len(times_ms) - 1)])

        results[ch_name] = {
            'mean_amp': float(data_uv.mean()),
            'peak_amp': peak_amp,
            'peak_lat': peak_lat,
            'area_lat_50': area_lat,
        }
    return results


# ── Table Plotting ───────────────────────────────────────────────────────────

def plot_electrode_table(metrics, title, highlight_chs=None):
    """Create a matplotlib figure with a per-electrode metrics table.

    Rows for highlight_chs are highlighted green.
    """
    if not metrics:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f'{title}: no data', ha='center', va='center')
        ax.axis('off')
        return fig

    col_labels = ['Electrode', 'Mean amp (uV)', 'Peak amp (uV)',
                  'Peak lat (ms)', 'Area lat 50% (ms)']
    cell_text = []
    cell_colors = []

    sorted_chs = sorted(metrics.keys())
    for ch in sorted_chs:
        m = metrics[ch]
        row = [ch, f'{m["mean_amp"]:.2f}', f'{m["peak_amp"]:.2f}',
               f'{m["peak_lat"]:.0f}', f'{m["area_lat_50"]:.0f}']
        cell_text.append(row)
        is_hl = highlight_chs and ch in highlight_chs
        bg = '#d4edda' if is_hl else 'white'
        cell_colors.append([bg] * len(row))

    n_rows = len(cell_text)
    fig_h = max(4.0, 0.3 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=['#d0d0d0'] * len(col_labels),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    if highlight_chs:
        ax.text(0.01, 0.01, 'Green = target electrodes',
                transform=ax.transAxes, fontsize=7, va='bottom')

    fig.tight_layout()
    return fig


def plot_erp_profile_table(n2_metrics, p3_metrics, title):
    """Create a combined N2/P3 profile table at key channels.

    Layout: N2 channels (Fz, FCz, Cz) | P3 channels (Cz, CPz, Pz)
    4 metrics each: mean amp, peak amp, peak lat, area lat 50%
    """
    metric_labels = ['Mean amp (uV)', 'Peak amp (uV)',
                     'Peak lat (ms)', 'Area lat 50% (ms)']
    metric_keys = ['mean_amp', 'peak_amp', 'peak_lat', 'area_lat_50']

    n2_display = ['Fz', 'FCz', 'Cz']
    p3_display = ['Cz', 'CPz', 'Pz']
    all_chs = n2_display + [''] + p3_display  # separator column

    col_labels = [''] + n2_display + [''] + p3_display
    cell_text = []
    cell_colors = []

    # Region header row
    header = ['']
    header_colors = ['#d0d0d0']
    for ch in n2_display:
        header.append('N2 (fronto-central)')
        header_colors.append('#cce5ff')
    header.append('')
    header_colors.append('#f0f0f0')
    for ch in p3_display:
        header.append('P3 (centro-parietal)')
        header_colors.append('#fff3cd')
    cell_text.append(header)
    cell_colors.append(header_colors)

    for label, key in zip(metric_labels, metric_keys):
        row = [label]
        row_colors = ['#d0d0d0']
        for ch in n2_display:
            if n2_metrics and ch in n2_metrics:
                val = n2_metrics[ch][key]
                row.append(f'{val:.2f}' if 'amp' in key else f'{val:.0f}')
            else:
                row.append('—')
            row_colors.append('#e8f4fd')
        row.append('')
        row_colors.append('#f0f0f0')
        for ch in p3_display:
            if p3_metrics and ch in p3_metrics:
                val = p3_metrics[ch][key]
                row.append(f'{val:.2f}' if 'amp' in key else f'{val:.0f}')
            else:
                row.append('—')
            row_colors.append('#fff8e1')
        cell_text.append(row)
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=['#d0d0d0'] * len(col_labels),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    fig.tight_layout()
    return fig


def plot_behavioral_table(behavioral, title):
    """Create a behavioral performance summary table."""
    rows = [
        ['Hits (correct responses)', str(behavioral['hit_count'])],
        ['Misses (no response to deviant)', str(behavioral['miss_count'])],
        ['False alarms', str(behavioral['false_alarm_count'])],
        ['Hit rate (%)', f'{behavioral["hit_rate"]:.1f}'],
        ['Mean RT (ms)', f'{behavioral["mean_rt_ms"]:.1f}'],
        ['SD RT (ms)', f'{behavioral["std_rt_ms"]:.1f}'],
    ]
    colors = [['white', 'white']] * len(rows)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    table = ax.table(cellText=rows,
                     colLabels=['Metric', 'Value'],
                     cellColours=colors,
                     colColours=['#d0d0d0', '#d0d0d0'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    fig.tight_layout()
    return fig


def plot_erp_overlay(evoked_std, evoked_dev, diff_wave, channels, title):
    """Plot Standard vs Deviant vs Difference overlay at selected channels."""
    available = [ch for ch in channels
                 if ch in evoked_std.ch_names and
                    ch in evoked_dev.ch_names and
                    ch in diff_wave.ch_names]
    if not available:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f'{title}: no available channels', ha='center')
        ax.axis('off')
        return fig

    n_ch = len(available)
    fig, axes = plt.subplots(1, n_ch, figsize=(5 * n_ch, 4), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight='bold')

    times_ms = evoked_std.times * 1000

    for i, ch in enumerate(available):
        ax = axes[0, i]
        idx_std = evoked_std.ch_names.index(ch)
        idx_dev = evoked_dev.ch_names.index(ch)
        idx_diff = diff_wave.ch_names.index(ch)

        ax.plot(times_ms, evoked_std.data[idx_std] * 1e6,
                'b-', label='Standard', linewidth=1.2)
        ax.plot(times_ms, evoked_dev.data[idx_dev] * 1e6,
                'r-', label='Deviant', linewidth=1.2)
        ax.plot(times_ms, diff_wave.data[idx_diff] * 1e6,
                'k--', label='Difference', linewidth=1.0, alpha=0.7)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.5)

        # Shade N2 and P3 windows
        ax.axvspan(N2_TMIN * 1000, N2_TMAX * 1000,
                   alpha=0.08, color='blue', label='N2 window')
        ax.axvspan(P3_TMIN * 1000, P3_TMAX * 1000,
                   alpha=0.08, color='red', label='P3 window')

        ax.set_title(ch, fontsize=11)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (uV)')
        ax.legend(fontsize=7, loc='upper right')
        ax.invert_yaxis()

    fig.tight_layout()
    return fig


# ── QC Checks ────────────────────────────────────────────────────────────────

def compute_raw_artifact_pct(raw, logger):
    """Estimate percentage of 1-second raw segments with artifacts.

    A segment is artifacted when >10% of non-bad EEG channels exceed
    +/-150 uV (minimum 2 channels).
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data = raw.get_data(picks=eeg_picks)
    n_channels = data.shape[0]
    sfreq = raw.info['sfreq']
    segment_len = int(sfreq)
    n_segments = data.shape[1] // segment_len

    if n_segments == 0 or n_channels == 0:
        return 0.0

    threshold = 150e-6
    min_bad_chs = max(2, int(np.ceil(n_channels * 0.10)))

    seg_data = data[:, :n_segments * segment_len].reshape(
        n_channels, n_segments, segment_len)
    exceeds = np.any(np.abs(seg_data) > threshold, axis=2)
    n_bad_chs_per_seg = exceeds.sum(axis=0)
    bad_segments = int((n_bad_chs_per_seg >= min_bad_chs).sum())

    pct = (bad_segments / n_segments) * 100
    logger.info(f'  Raw artifact estimate: {bad_segments}/{n_segments} '
                f'segments ({pct:.1f}%) '
                f'[>={min_bad_chs}/{n_channels} channels criterion]')
    return pct


def run_qc_checks(raw, epochs, n_dropped, recording_id, logger,
                   bad_channels=None):
    """Run QC checks per Tech Work criteria.

    Returns dict with qc_pass (bool) and individual check results.
    """
    qc = {
        'recording_id': recording_id,
        'qc_pass': True,
        'checks': {},
    }

    if bad_channels:
        qc['bad_channels_detected'] = bad_channels
        qc['n_bad_channels'] = len(bad_channels)

    # Check 1: Epoch rejection rate
    total = len(epochs) + n_dropped
    reject_pct = (n_dropped / total * 100) if total > 0 else 0
    ch1_pass = reject_pct <= QC_MAX_EPOCH_REJECT_PCT
    qc['checks']['epoch_reject_pct'] = {
        'value': round(reject_pct, 1),
        'threshold': QC_MAX_EPOCH_REJECT_PCT,
        'pass': ch1_pass,
    }
    if not ch1_pass:
        qc['qc_pass'] = False
        logger.warning(f'  QC FAIL: epoch rejection {reject_pct:.1f}% '
                       f'> {QC_MAX_EPOCH_REJECT_PCT}%')
    else:
        logger.info(f'  QC PASS: epoch rejection {reject_pct:.1f}%')

    # Check 2: Target channels present
    report_chs = list(set(N2_CHANNELS + P3_CHANNELS))
    missing = [ch for ch in report_chs if ch not in epochs.ch_names]
    ch2_pass = len(missing) < QC_MAX_MISSING_TARGET_CHS
    qc['checks']['missing_target_channels'] = {
        'missing': missing,
        'count': len(missing),
        'threshold': QC_MAX_MISSING_TARGET_CHS,
        'pass': ch2_pass,
    }
    if not ch2_pass:
        qc['qc_pass'] = False
        logger.warning(f'  QC FAIL: {len(missing)} target channels missing: '
                       f'{missing}')
    else:
        logger.info(f'  QC PASS: {len(missing)} target channels missing')

    # Check 3: Raw data artifact percentage
    artifact_pct = compute_raw_artifact_pct(raw, logger)
    ch3_pass = artifact_pct <= QC_MAX_RAW_ARTIFACT_PCT
    qc['checks']['raw_artifact_pct'] = {
        'value': round(artifact_pct, 1),
        'threshold': QC_MAX_RAW_ARTIFACT_PCT,
        'pass': ch3_pass,
    }
    if not ch3_pass:
        qc['qc_pass'] = False
        logger.warning(f'  QC FAIL: raw artifact {artifact_pct:.1f}% '
                       f'> {QC_MAX_RAW_ARTIFACT_PCT}%')
    else:
        logger.info(f'  QC PASS: raw artifact {artifact_pct:.1f}%')

    status = 'PASS' if qc['qc_pass'] else 'FAIL'
    logger.info(f'  QC RESULT: {status}')
    return qc


# ── Post-Processing Validation ───────────────────────────────────────────────

def validate_results(recording_id, file_info, epochs, evokeds, diff_wave,
                     n2_metrics, p3_metrics, ica, eog_indices, n_dropped,
                     behavioral, logger):
    """Run validation checks on processed data and log a structured report.

    Uses age-appropriate normative thresholds from published literature.
    Logs PASS/WARNING/FAIL for each check, with a full reference list.
    """
    # All subjects in this study are children
    age_group = 'child'
    norms = NORMS[age_group]

    logger.info('')
    logger.info('=' * 70)
    logger.info('  VALIDATION REPORT: %s', recording_id)
    logger.info('=' * 70)

    # Log file info
    logger.info('--- Recording Info ---')
    logger.info('  Subject:   %s', file_info['subject_id'])
    logger.info('  Cohort:    %s', file_info['cohort'])
    logger.info('  Visit:     %s', file_info['visit'])
    logger.info('  Stimulus:  %s', file_info['stimulus'])
    logger.info('  Group:     %s', file_info['group'])
    logger.info('  Norms:     CHILD')

    warnings_count = 0
    fails_count = 0

    def log_check(name, status, detail, refs=''):
        nonlocal warnings_count, fails_count
        ref_str = f'  [{refs}]' if refs else ''
        if status == 'FAIL':
            fails_count += 1
            logger.error('  [FAIL]    %-30s  %s%s', name, detail, ref_str)
        elif status == 'WARNING':
            warnings_count += 1
            logger.warning('  [WARNING] %-30s  %s%s', name, detail, ref_str)
        else:
            logger.info('  [PASS]    %-30s  %s%s', name, detail, ref_str)

    # ── 1. Epoch counts ──────────────────────────────────────────────────
    logger.info('--- Epoch Counts [ref 3] ---')
    event_id = epochs.event_id

    for cond, min_ep in [('standard', norms['min_epochs_std']),
                         ('deviant', norms['min_epochs_dev'])]:
        if cond not in event_id:
            log_check(f'{cond} epochs', 'FAIL',
                      'condition missing from data', 'ref 3')
            continue
        n = len(epochs[cond])
        if n == 0:
            log_check(f'{cond} epochs', 'FAIL',
                      f'{n} epochs (none survived)', 'ref 3')
        elif n < min_ep:
            log_check(f'{cond} epochs', 'WARNING',
                      f'{n} epochs (< {min_ep} recommended minimum)',
                      'ref 3')
        else:
            log_check(f'{cond} epochs', 'PASS', f'{n} epochs', 'ref 3')

    total_epochs = len(epochs) + n_dropped
    drop_pct = (n_dropped / total_epochs * 100) if total_epochs > 0 else 0
    if drop_pct > norms['reject_fail']:
        log_check('Epoch rejection rate', 'FAIL',
                  f'{n_dropped} dropped ({drop_pct:.1f}% > '
                  f'{norms["reject_fail"]}% limit)', 'ref 3')
    elif drop_pct > norms['reject_warn']:
        log_check('Epoch rejection rate', 'WARNING',
                  f'{n_dropped} dropped ({drop_pct:.1f}%)', 'ref 3')
    else:
        log_check('Epoch rejection rate', 'PASS',
                  f'{n_dropped} dropped ({drop_pct:.1f}%)', 'ref 3')

    # ── 2. ICA quality ───────────────────────────────────────────────────
    logger.info('--- ICA [ref 5] ---')
    n_excluded = len(eog_indices)
    if n_excluded == 0:
        log_check('ICA EOG exclusion', 'WARNING',
                  'no components excluded (eye artifacts may remain)', 'ref 5')
    elif n_excluded > 5:
        log_check('ICA EOG exclusion', 'WARNING',
                  f'{n_excluded} excluded (unusually many)', 'ref 5')
    else:
        log_check('ICA EOG exclusion', 'PASS',
                  f'{n_excluded} component(s) excluded: '
                  f'{list(eog_indices)}', 'ref 5')

    # ── 3. Baseline ──────────────────────────────────────────────────────
    logger.info('--- Baseline [ref 5] ---')
    if 'standard' in evokeds:
        std = evokeds['standard']
        bl_mask = std.times <= 0
        bl_data = std.data[:, bl_mask] * 1e6
        bl_mean_abs = np.abs(bl_data.mean())
        bl_mean_val = bl_data.mean()
        if bl_mean_abs > norms['baseline_max']:
            log_check('Baseline correction', 'WARNING',
                      f'mean = {bl_mean_val:.4f} uV '
                      f'(|mean| > {norms["baseline_max"]} uV)', 'ref 5')
        else:
            log_check('Baseline correction', 'PASS',
                      f'mean = {bl_mean_val:.4f} uV', 'ref 5')

    # ── 4. ERP amplitudes ────────────────────────────────────────────────
    erp_max = norms['erp_amp_max']
    logger.info('--- ERP Amplitudes [ref 5] ---')
    for name, evk in evokeds.items():
        amp_range = evk.data * 1e6
        amp_min, amp_max = amp_range.min(), amp_range.max()
        peak = max(abs(amp_min), abs(amp_max))
        if peak > erp_max:
            log_check(f'{name} amplitude', 'WARNING',
                      f'range [{amp_min:.1f}, {amp_max:.1f}] uV '
                      f'(peak {peak:.1f} > {erp_max} uV limit)', 'ref 5')
        else:
            log_check(f'{name} amplitude', 'PASS',
                      f'range [{amp_min:.1f}, {amp_max:.1f}] uV', 'ref 5')

    # ── 5. N2 checks (difference wave) ───────────────────────────────────
    logger.info('--- N2 Component [ref 4,6] ---')
    if diff_wave is not None:
        available_n2 = [ch for ch in N2_CHANNELS
                        if ch in diff_wave.ch_names]

        if not available_n2:
            log_check('N2 polarity', 'FAIL',
                      'no fronto-central channels found', 'ref 4')
        else:
            # Polarity check: N2 in difference wave should be negative
            t_mask = (diff_wave.times >= N2_TMIN) & \
                     (diff_wave.times <= N2_TMAX)
            neg_count = 0
            for ch in available_n2:
                ch_idx = diff_wave.ch_names.index(ch)
                mean_val = (diff_wave.data[ch_idx, t_mask] * 1e6).mean()
                if mean_val < 0:
                    neg_count += 1

            if neg_count == len(available_n2):
                log_check('N2 polarity', 'PASS',
                          f'negative at all {neg_count} fronto-central '
                          f'channels', 'ref 4')
            elif neg_count >= len(available_n2) * 0.5:
                log_check('N2 polarity', 'WARNING',
                          f'negative at {neg_count}/{len(available_n2)} '
                          f'channels', 'ref 4')
            else:
                log_check('N2 polarity', 'FAIL',
                          f'negative at only {neg_count}/'
                          f'{len(available_n2)} channels', 'ref 4')

            # N2 peak amplitude and latency at best channel
            amp_normal = norms['n2_amp_normal']
            amp_warn = norms['n2_amp_warn']
            lat_normal = norms['n2_lat_normal']
            lat_warn = norms['n2_lat_warn']

            best_ch, best_peak, best_lat = None, 0, 0
            for ch in available_n2:
                if n2_metrics and ch in n2_metrics:
                    peak_val = n2_metrics[ch]['peak_amp']
                    if peak_val < best_peak:
                        best_peak = peak_val
                        best_lat = n2_metrics[ch]['peak_lat']
                        best_ch = ch

            if best_ch:
                if amp_normal[0] <= best_peak <= amp_normal[1]:
                    log_check('N2 amplitude', 'PASS',
                              f'{best_peak:.2f} uV at {best_ch}', 'ref 6')
                elif amp_warn[0] <= best_peak <= amp_warn[1]:
                    log_check('N2 amplitude', 'WARNING',
                              f'{best_peak:.2f} uV at {best_ch} '
                              f'(outside normal range)', 'ref 6')
                else:
                    log_check('N2 amplitude', 'FAIL',
                              f'{best_peak:.2f} uV at {best_ch}', 'ref 6')

                if lat_normal[0] <= best_lat <= lat_normal[1]:
                    log_check('N2 latency', 'PASS',
                              f'{best_lat:.0f} ms at {best_ch}', 'ref 2')
                elif lat_warn[0] <= best_lat <= lat_warn[1]:
                    log_check('N2 latency', 'WARNING',
                              f'{best_lat:.0f} ms at {best_ch}', 'ref 2')
                else:
                    log_check('N2 latency', 'FAIL',
                              f'{best_lat:.0f} ms at {best_ch}', 'ref 2')

    # ── 6. P3 checks (difference wave) ───────────────────────────────────
    logger.info('--- P3 Component [ref 1,2] ---')
    if diff_wave is not None:
        available_p3 = [ch for ch in P3_CHANNELS
                        if ch in diff_wave.ch_names]

        if not available_p3:
            log_check('P3 polarity', 'FAIL',
                      'no centro-parietal channels found', 'ref 1')
        else:
            # Polarity check: P3 in difference wave should be positive
            t_mask = (diff_wave.times >= P3_TMIN) & \
                     (diff_wave.times <= P3_TMAX)
            pos_count = 0
            for ch in available_p3:
                ch_idx = diff_wave.ch_names.index(ch)
                mean_val = (diff_wave.data[ch_idx, t_mask] * 1e6).mean()
                if mean_val > 0:
                    pos_count += 1

            if pos_count == len(available_p3):
                log_check('P3 polarity', 'PASS',
                          f'positive at all {pos_count} centro-parietal '
                          f'channels', 'ref 1')
            elif pos_count >= len(available_p3) * 0.5:
                log_check('P3 polarity', 'WARNING',
                          f'positive at {pos_count}/{len(available_p3)} '
                          f'channels', 'ref 1')
            else:
                log_check('P3 polarity', 'FAIL',
                          f'positive at only {pos_count}/'
                          f'{len(available_p3)} channels', 'ref 1')

            # P3 peak amplitude and latency at best channel
            amp_normal = norms['p3_amp_normal']
            amp_warn = norms['p3_amp_warn']
            lat_normal = norms['p3_lat_normal']
            lat_warn = norms['p3_lat_warn']

            best_ch, best_peak, best_lat = None, 0, 0
            for ch in available_p3:
                if p3_metrics and ch in p3_metrics:
                    peak_val = p3_metrics[ch]['peak_amp']
                    if peak_val > best_peak:
                        best_peak = peak_val
                        best_lat = p3_metrics[ch]['peak_lat']
                        best_ch = ch

            if best_ch:
                if amp_normal[0] <= best_peak <= amp_normal[1]:
                    log_check('P3 amplitude', 'PASS',
                              f'{best_peak:.2f} uV at {best_ch}', 'ref 1')
                elif amp_warn[0] <= best_peak <= amp_warn[1]:
                    log_check('P3 amplitude', 'WARNING',
                              f'{best_peak:.2f} uV at {best_ch} '
                              f'(outside normal range)', 'ref 1')
                else:
                    log_check('P3 amplitude', 'FAIL',
                              f'{best_peak:.2f} uV at {best_ch}', 'ref 1')

                if lat_normal[0] <= best_lat <= lat_normal[1]:
                    log_check('P3 latency', 'PASS',
                              f'{best_lat:.0f} ms at {best_ch}', 'ref 2')
                elif lat_warn[0] <= best_lat <= lat_warn[1]:
                    log_check('P3 latency', 'WARNING',
                              f'{best_lat:.0f} ms at {best_ch}', 'ref 2')
                else:
                    log_check('P3 latency', 'FAIL',
                              f'{best_lat:.0f} ms at {best_ch}', 'ref 2')

    # ── 7. Behavioral checks ────────────────────────────────────────────
    logger.info('--- Behavioral ---')
    if behavioral['hit_rate'] > 0:
        if behavioral['hit_rate'] >= 50:
            log_check('Hit rate', 'PASS',
                      f'{behavioral["hit_rate"]:.1f}%')
        else:
            log_check('Hit rate', 'WARNING',
                      f'{behavioral["hit_rate"]:.1f}% (low)')

        if behavioral['mean_rt_ms'] > 0:
            if 200 <= behavioral['mean_rt_ms'] <= 1200:
                log_check('Mean RT', 'PASS',
                          f'{behavioral["mean_rt_ms"]:.0f} ms')
            else:
                log_check('Mean RT', 'WARNING',
                          f'{behavioral["mean_rt_ms"]:.0f} ms (unusual)')
    else:
        log_check('Behavioral', 'WARNING',
                  'no button-press responses detected (passive oddball?)')

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('--- Summary ---')
    if fails_count == 0 and warnings_count == 0:
        logger.info('  RESULT: ALL CHECKS PASSED')
    elif fails_count == 0:
        logger.warning('  RESULT: PASSED with %d warning(s)', warnings_count)
    else:
        logger.error('  RESULT: %d FAIL(s), %d warning(s)',
                     fails_count, warnings_count)

    # ── References ───────────────────────────────────────────────────────
    for line in REFERENCES.strip().splitlines():
        logger.info('  %s', line)
    logger.info('=' * 70)

    return fails_count, warnings_count


# ── Main Pipeline ────────────────────────────────────────────────────────────

def process_single_file(vhdr_path, output_dir, log_dir, logger):
    """Run the full OddBall processing pipeline on a single recording."""
    file_info = extract_file_info(vhdr_path)
    recording_id = file_info['recording_id']
    logger.info(f'Processing {recording_id}')
    logger.info(f'  Subject:  {file_info["subject_id"]}')
    if file_info['original_id'] != file_info['subject_id']:
        logger.info(f'  Original: {file_info["original_id"]} '
                    f'(remapped from longitudinal L_y1)')
    logger.info(f'  Cohort:   {file_info["cohort"]}')
    logger.info(f'  Visit:    {file_info["visit"]}')
    logger.info(f'  Stimulus: {file_info["stimulus"]}')
    logger.info(f'  Group:    {file_info["group"]}')
    logger.info(f'  Input:    {vhdr_path}')

    # ── Subject notes from manual processing ──
    sid = file_info['subject_id']
    if sid in SUBJECT_NOTES:
        logger.warning(f'  NOTE: {SUBJECT_NOTES[sid]}')

    # ── 1. Load raw data ─────────────────────────────────────────────────
    logger.info('── 1. Load raw data ──')
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True,
                                       verbose='WARNING')
    logger.info(f'Loaded: {len(raw.ch_names)} channels, '
                f'{raw.info["sfreq"]} Hz, {raw.times[-1]:.1f} s')

    # ── 2. Prepare channels & montage ────────────────────────────────────
    logger.info('── 2. Prepare channels & montage ──')
    raw, eog_ch = prepare_channels(raw, logger)

    # ── 3. Bandpass filter ───────────────────────────────────────────────
    logger.info(f'── 3. Bandpass filter: {FILTER_L_FREQ}–{FILTER_H_FREQ} Hz ──')
    raw.filter(FILTER_L_FREQ, FILTER_H_FREQ, verbose='WARNING')

    # ── 4. Resample ──────────────────────────────────────────────────────
    orig_sfreq = raw.info['sfreq']
    logger.info(f'── 4. Resample: {orig_sfreq} Hz -> {RESAMPLE_FREQ} Hz ──')
    raw.resample(RESAMPLE_FREQ, verbose='WARNING')

    # ── 5. Detect bad channels ───────────────────────────────────────────
    logger.info('── 5. Detect bad channels ──')
    bad_channels = detect_bad_channels(raw, logger)

    # ── 6. ICA artifact removal ──────────────────────────────────────────
    logger.info('── 6. ICA artifact removal ──')
    raw_before_ica = raw.copy()

    ica = ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD,
              max_iter=ICA_MAX_ITER, random_state=42)
    ica.fit(raw, verbose='WARNING')
    logger.info(f'ICA fitted: {ica.n_components_} components')

    # Auto-detect EOG artifacts
    eog_proxy = eog_ch
    if eog_proxy is None and 'Fp1' in raw.ch_names:
        eog_proxy = 'Fp1'
        logger.info('Using Fp1 as EOG proxy')

    eog_indices = []
    eog_scores = None
    if eog_proxy:
        eog_indices, eog_scores = ica.find_bads_eog(
            raw, ch_name=eog_proxy, threshold=2.5, verbose='WARNING')
        if not eog_indices:
            logger.info('No EOG components at threshold=2.5, retrying at 2.0')
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, ch_name=eog_proxy, threshold=2.0, verbose='WARNING')
        ica.exclude = eog_indices
        if eog_indices:
            logger.info(f'EOG components excluded: {eog_indices}')
        else:
            logger.warning('No EOG components found — '
                           'ICA applied without exclusions')
    else:
        logger.warning('No EOG proxy available — skipping ICA exclusion')

    ica.apply(raw, verbose='WARNING')
    logger.info('ICA applied')

    # Drop EOG/ECG now that ICA is done
    drop_after_ica = [ch for ch in raw.ch_names
                      if ch.upper() in NON_EEG_PATTERNS]
    if drop_after_ica:
        raw.drop_channels(drop_after_ica)
        logger.info(f'Dropped non-EEG channel(s) after ICA: {drop_after_ica}')

    # ── 7. Interpolate bad channels ──────────────────────────────────────
    if raw.info['bads']:
        n_bads = len(raw.info['bads'])
        logger.info(f'── 7. Interpolating {n_bads} bad channel(s): '
                    f'{raw.info["bads"]} ──')
        raw.interpolate_bads(reset_bads=True, verbose='WARNING')
        logger.info('Bad channels interpolated')
    else:
        logger.info('── 7. No bad channels to interpolate ──')

    # ── 8. Re-reference to average ───────────────────────────────────────
    logger.info('── 8. Re-reference to average ──')
    raw.set_eeg_reference('average', verbose='WARNING')

    # ── 9. Extract events & create epochs ────────────────────────────────
    logger.info('── 9. Extract events & create epochs ──')
    events, all_event_id = mne.events_from_annotations(raw, verbose='WARNING')
    event_id = build_event_id(all_event_id, logger)
    logger.info(f'Event mapping: {event_id}')

    if 'standard' not in event_id or 'deviant' not in event_id:
        raise ValueError(f'Missing standard/deviant events in data. '
                         f'Found annotations: {list(all_event_id.keys())}')

    # Count events before epoching
    for cond, code in event_id.items():
        n_events = np.sum(events[:, 2] == code)
        logger.info(f'  {cond}: {n_events} events')

    # Extract behavioral data before epoching (uses response events)
    behavioral = extract_behavioral_data(
        events, event_id, raw.info['sfreq'], logger)

    # Create epochs (only standard + deviant, not response)
    epoch_event_id = {k: v for k, v in event_id.items()
                      if k in ('standard', 'deviant')}

    epochs = mne.Epochs(raw, events, epoch_event_id,
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                        baseline=BASELINE, preload=True,
                        reject=EPOCH_REJECT,
                        verbose='WARNING')

    n_dropped = len(epochs.drop_log) - len(epochs)
    if n_dropped:
        logger.info(f'  Dropped {n_dropped} epochs by amplitude rejection '
                    f'(threshold: {EPOCH_REJECT["eeg"] * 1e6:.0f} uV)')

    for cond in epoch_event_id:
        n = len(epochs[cond])
        logger.info(f'  {cond}: {n} epochs')
    logger.info(f'  Total: {len(epochs)} epochs')

    # ── 10. Compute ERPs ─────────────────────────────────────────────────
    logger.info('── 10. Compute evoked responses ──')
    evokeds = {}

    evokeds['standard'] = epochs['standard'].average()
    evokeds['standard'].comment = 'standard'

    evokeds['deviant'] = epochs['deviant'].average()
    evokeds['deviant'].comment = 'deviant'

    # ── 11. Compute difference wave (Deviant - Standard) ─────────────────
    logger.info('── 11. Compute difference wave ──')
    diff_wave = mne.combine_evoked(
        [evokeds['deviant'], evokeds['standard']], weights=[1, -1])
    diff_wave.comment = 'difference (deviant - standard)'

    # ── 12. Measure N2 and P3 components ─────────────────────────────────
    logger.info('── 12. Measure N2 and P3 components ──')

    # N2: negative peak in difference wave at fronto-central sites
    n2_metrics = measure_erp_component(diff_wave, N2_TMIN, N2_TMAX,
                                       polarity='negative')
    logger.info(f'  N2 measured at {len(n2_metrics)} channels')

    # P3: positive peak in difference wave at centro-parietal sites
    p3_metrics = measure_erp_component(diff_wave, P3_TMIN, P3_TMAX,
                                       polarity='positive')
    logger.info(f'  P3 measured at {len(p3_metrics)} channels')

    # ── 13. Generate report ──────────────────────────────────────────────
    logger.info('── 13. Generate report ──')
    report_title = (f'OddBall report: {file_info["subject_id"]} '
                    f'({file_info["stimulus"]}, {file_info["visit"]}, '
                    f'{file_info["group"]})')
    report = mne.Report(title=report_title, verbose='WARNING')

    # Section 1: ICA
    report.add_ica(
        ica=ica,
        title='ICA and automatic artifact components',
        inst=raw_before_ica,
        eog_evoked=None,
        eog_scores=eog_scores,
        tags=('ica', 'qc'),
        n_jobs=1,
    )

    # Section 2: Epochs
    report.add_epochs(
        epochs=epochs,
        title='Epochs after rejection',
        tags=('epochs', 'qc'),
        psd=True,
    )

    # Section 3-4: Evoked responses
    for key in ('standard', 'deviant'):
        report.add_evokeds(
            evokeds=evokeds[key],
            titles=key,
            tags=('evoked', 'erp'),
            n_time_points=None,
        )

    # Section 5: Difference wave
    report.add_evokeds(
        evokeds=diff_wave,
        titles='Difference (Deviant - Standard)',
        tags=('evoked', 'difference'),
    )

    # Section 6: ERP overlay at N2 channels
    fig = plot_erp_overlay(
        evokeds['standard'], evokeds['deviant'], diff_wave,
        ['Fz', 'FCz', 'Cz'],
        'Standard vs Deviant — fronto-central (N2 channels)')
    report.add_figure(fig, title='ERP overlay — N2 channels',
                      tags=('erp', 'overlay'))
    plt.close(fig)

    # Section 7: ERP overlay at P3 channels
    fig = plot_erp_overlay(
        evokeds['standard'], evokeds['deviant'], diff_wave,
        ['Cz', 'CPz', 'Pz'],
        'Standard vs Deviant — centro-parietal (P3 channels)')
    report.add_figure(fig, title='ERP overlay — P3 channels',
                      tags=('erp', 'overlay'))
    plt.close(fig)

    # Section 8: N2 per-electrode table
    if n2_metrics:
        # Full table with target highlighting
        fig = plot_electrode_table(
            {ch: n2_metrics[ch] for ch in n2_metrics
             if ch in TARGET_ELECTRODES},
            f'N2 component ({N2_TMIN*1000:.0f}-{N2_TMAX*1000:.0f} ms) '
            f'— target electrodes',
            highlight_chs=N2_CHANNELS)
        report.add_figure(fig, title='N2 — electrode table',
                          tags=('n2', 'electrodes'))
        plt.close(fig)

    # Section 9: P3 per-electrode table
    if p3_metrics:
        fig = plot_electrode_table(
            {ch: p3_metrics[ch] for ch in p3_metrics
             if ch in TARGET_ELECTRODES},
            f'P3 component ({P3_TMIN*1000:.0f}-{P3_TMAX*1000:.0f} ms) '
            f'— target electrodes',
            highlight_chs=P3_CHANNELS)
        report.add_figure(fig, title='P3 — electrode table',
                          tags=('p3', 'electrodes'))
        plt.close(fig)

    # Section 10: Combined N2/P3 profile table
    fig = plot_erp_profile_table(
        n2_metrics, p3_metrics,
        'N2/P3 profile — difference wave (Deviant - Standard)')
    report.add_figure(fig, title='N2/P3 profile table',
                      tags=('n2', 'p3', 'profile'))
    plt.close(fig)

    # Section 11: Behavioral table
    fig = plot_behavioral_table(
        behavioral,
        f'Behavioral performance — {file_info["stimulus"]}')
    report.add_figure(fig, title='Behavioral performance',
                      tags=('behavioral',))
    plt.close(fig)

    # ── 14. QC checks ────────────────────────────────────────────────────
    logger.info('── 14. QC checks ──')
    qc_result = run_qc_checks(raw, epochs, n_dropped, recording_id, logger,
                               bad_channels=bad_channels)

    # ── 15. Save outputs ─────────────────────────────────────────────────
    logger.info('── 15. Save outputs ──')
    output_dir = Path(output_dir)
    rec_dir = output_dir / recording_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Save HTML report
    report_path = rec_dir / f'{recording_id}_report.html'
    report.save(str(report_path), overwrite=True, open_browser=False,
                verbose='WARNING')
    logger.info(f'Report saved: {report_path}')

    # Save evoked + difference as .fif for group analysis
    all_evokeds = list(evokeds.values()) + [diff_wave]
    fif_path = rec_dir / f'{recording_id}-ave.fif'
    mne.write_evokeds(str(fif_path), all_evokeds, overwrite=True,
                      verbose='WARNING')
    logger.info(f'Evokeds saved: {fif_path}')

    # Save QC result as JSON
    qc_path = rec_dir / f'{recording_id}_qc.json'
    with open(str(qc_path), 'w', encoding='utf-8') as f:
        json.dump(qc_result, f, indent=2, ensure_ascii=False)
    logger.info(f'QC saved: {qc_path} '
                f'({"PASS" if qc_result["qc_pass"] else "FAIL"})')

    # Save metrics as JSON (N2, P3, behavioral, file info)
    metrics = {
        'file_info': file_info,
        'n2': {ch: n2_metrics[ch] for ch in N2_CHANNELS
               if ch in n2_metrics} if n2_metrics else {},
        'p3': {ch: p3_metrics[ch] for ch in P3_CHANNELS
               if ch in p3_metrics} if p3_metrics else {},
        'n2_all_electrodes': n2_metrics,
        'p3_all_electrodes': p3_metrics,
        'behavioral': {k: v for k, v in behavioral.items()
                       if k != 'all_rts_ms'},
        'epochs': {
            'standard': len(epochs['standard']),
            'deviant': len(epochs['deviant']),
            'dropped': n_dropped,
            'total_before_reject': len(epochs) + n_dropped,
        },
        'bad_channels': bad_channels,
        'ica_excluded': list(eog_indices),
        'qc_pass': qc_result['qc_pass'],
    }
    metrics_path = rec_dir / f'{recording_id}_metrics.json'

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(str(metrics_path), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    logger.info(f'Metrics saved: {metrics_path}')

    # ── 16. Validate results ─────────────────────────────────────────────
    validate_results(recording_id, file_info, epochs, evokeds, diff_wave,
                     n2_metrics, p3_metrics, ica, eog_indices, n_dropped,
                     behavioral, logger)

    return report_path


# ── Parallel Worker ─────────────────────────────────────────────────────────

def _process_worker(vhdr_path, output_dir, log_dir, verbose=False):
    """Worker function for parallel file processing (runs in subprocess)."""
    if not verbose:
        mne.set_log_level('WARNING')
    file_info = extract_file_info(vhdr_path)
    recording_id = file_info['recording_id']
    logger = setup_logging(log_dir, recording_id)
    try:
        report_path = process_single_file(vhdr_path, output_dir,
                                          log_dir, logger)
        return ('ok', recording_id, str(report_path))
    except Exception:
        logger.exception(f'Failed to process {vhdr_path}')
        return ('fail', recording_id, str(vhdr_path))


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='OddBall (N2/P3) EEG Processing Pipeline')
    parser.add_argument('--input', required=True,
                        help='Path to .vhdr file or folder with recordings')
    parser.add_argument('--output', required=True,
                        help='Output directory for reports')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Number of parallel workers (default: 1, '
                             'use -j 4 or higher on server)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose MNE output')
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.verbose:
        mne.set_log_level('WARNING')

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Discover files
    vhdr_files = discover_vhdr_files(input_path)
    print(f'Found {len(vhdr_files)} .vhdr file(s)')

    # Resolve log directory relative to script location
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'logs'

    successes = []
    failures = []

    if args.jobs == 1:
        # ── Sequential processing ──
        for vhdr_path in vhdr_files:
            file_info = extract_file_info(vhdr_path)
            recording_id = file_info['recording_id']
            logger = setup_logging(log_dir, recording_id)

            try:
                report_path = process_single_file(vhdr_path, output_dir,
                                                  log_dir, logger)
                successes.append((recording_id, report_path))
            except Exception:
                logger.exception(f'Failed to process {vhdr_path}')
                failures.append((recording_id, vhdr_path))
    else:
        # ── Parallel processing ──
        print(f'Using {args.jobs} parallel workers')
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {}
            for vhdr_path in vhdr_files:
                future = executor.submit(_process_worker, vhdr_path,
                                         output_dir, log_dir, args.verbose)
                futures[future] = vhdr_path

            for future in as_completed(futures):
                try:
                    status, rid, path = future.result()
                except Exception as exc:
                    vhdr = futures[future]
                    rid = extract_file_info(vhdr)['recording_id']
                    print(f'  FAIL {rid}: worker error: {exc}')
                    failures.append((rid, str(vhdr)))
                    continue

                if status == 'ok':
                    successes.append((rid, path))
                    print(f'  OK   {rid}')
                else:
                    failures.append((rid, path))
                    print(f'  FAIL {rid}')

    # Summary
    print(f'\nDone: {len(successes)} succeeded, {len(failures)} failed')
    for rid, path in successes:
        print(f'  OK   {rid}: {path}')
    for rid, path in failures:
        print(f'  FAIL {rid}: {path}')


if __name__ == '__main__':
    main()
