"""OddBall Group Analysis — Grand Averages, Statistics, and Comparison Reports.

Loads individual *_metrics.json and *-ave.fif files from processed recordings,
computes grand averages per condition (Group x Stimulus x Visit), runs
statistical comparisons, generates topographic maps and per-electrode tables.

Outputs:
  - HTML report with grand average ERPs, topomaps, tables, statistics
  - oddball_group_table.csv   (grand average per-electrode metrics)
  - oddball_raw_arrays.csv    (per-subject per-electrode values)
  - oddball_statistics.csv    (N2/P3 statistical test results)

Usage:
    python group_analysis_oddball.py --input <outputs_dir> --output <group_dir>
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import csv
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == 'win32':
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                      errors='replace')

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

# ── Constants ────────────────────────────────────────────────────────────────

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
GROUP_ASSIGNMENT = {
    'Co_y6_020': 'treat', 'Co_y6_021': 'treat', 'Co_y6_022': 'treat',
    'Co_y6_024': 'treat', 'Co_y6_025': 'treat', 'Co_y6_026': 'treat',
    'Co_y6_027': 'treat', 'Co_y6_029': 'treat', 'Co_y6_037': 'treat',
    'Co_y6_038': 'treat', 'Co_y6_039': 'treat', 'Co_y6_043': 'treat',
    'Co_y6_044': 'treat', 'Co_y6_045': 'treat', 'Co_y6_050': 'treat',
    'Co_y6_053': 'treat', 'Co_y6_056': 'treat', 'Co_y6_058': 'treat',
    'Co_y6_041': 'control', 'Co_y6_042': 'control', 'Co_y6_046': 'control',
    'Co_y6_049': 'control', 'Co_y6_054': 'control', 'Co_y6_060': 'control',
    'Co_y6_061': 'control', 'Co_y6_062': 'control', 'Co_y6_063': 'control',
    'Co_y6_064': 'control', 'Co_y6_066': 'control', 'Co_y6_067': 'control',
    'Co_y6_068': 'control', 'Co_y6_069': 'control', 'Co_y6_070': 'control',
    'Co_y6_071': 'control',
    'M_y3_NewLab_002': 'treat', 'M_y3_NewLab_003': 'treat',
    'M_y3_NewLab_004': 'treat', 'M_y3_NewLab_005': 'treat',
    'M_y3_NewLab_009': 'treat', 'M_y3_NewLab_012': 'treat',
    'M_y3_NewLab_015': 'treat', 'M_y3_NewLab_016': 'treat',
    'M_y3_NewLab_006': 'control', 'M_y3_NewLab_007': 'control',
    'M_y3_NewLab_017': 'control',
    'M_y3_NewLab_29': 'treat', 'M_y3_NewLab_030': 'treat',
    'M_y3_NewLab_031': 'treat', 'M_y3_NewLab_032': 'treat',
    'M_y3_NewLab_036': 'treat', 'M_y3_NewLab_038': 'treat',
    'M_y3_NewLab_042': 'treat', 'M_y3_NewLab_043': 'treat',
    'M_y3_NewLab_044': 'treat',
    'M_y3_NewLab_040': 'control',
}

# Conditions stored in -ave.fif (from process_oddball.py)
ALL_CONDITIONS = ('standard', 'deviant', 'difference (deviant - standard)')

# Condition labels for reports
DIFF_COMMENT = 'difference (deviant - standard)'


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'group_analysis_oddball_{timestamp}.log'

    logger = logging.getLogger('group_oddball')
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


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data(input_dir, logger):
    """Load all metrics JSON and evoked FIF files from output directories.

    Returns:
        recordings: list of dicts with keys:
            'file_info', 'metrics', 'evokeds' (dict of condition->Evoked),
            'qc_pass', 'recording_id'
    """
    input_dir = Path(input_dir)
    recordings = []

    # Find all recording subdirectories (each has *_metrics.json)
    metric_files = sorted(input_dir.rglob('*_metrics.json'))
    if not metric_files:
        raise FileNotFoundError(f'No *_metrics.json files found in {input_dir}')

    logger.info(f'Found {len(metric_files)} recording(s)')

    for mf in metric_files:
        rec_dir = mf.parent
        recording_id = mf.stem.replace('_metrics', '')

        # Load metrics
        with open(str(mf), 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        file_info = metrics.get('file_info', {})
        qc_pass = metrics.get('qc_pass', True)

        # Apply longitudinal remapping (handles outputs from pre-fix runs)
        sid = file_info.get('subject_id', '')
        if sid in LONGITUDINAL_MAP:
            file_info['original_id'] = sid
            file_info['subject_id'] = LONGITUDINAL_MAP[sid]
            file_info['visit'] = 'v2'
            sid = file_info['subject_id']
            logger.debug(f'  Remapped {file_info["original_id"]} -> {sid} (v2)')

        # Apply authoritative group assignment
        if sid in GROUP_ASSIGNMENT:
            file_info['group'] = GROUP_ASSIGNMENT[sid]

        # Load evokeds
        fif_path = rec_dir / f'{recording_id}-ave.fif'
        if not fif_path.exists():
            logger.warning(f'  {recording_id}: no .fif file, skipping')
            continue

        try:
            evoked_list = mne.read_evokeds(str(fif_path), verbose='WARNING')
        except Exception as e:
            logger.error(f'  {recording_id}: failed to load .fif: {e}')
            continue

        evokeds = {}
        for evk in evoked_list:
            if evk.comment:
                evokeds[evk.comment] = evk

        recordings.append({
            'recording_id': recording_id,
            'file_info': file_info,
            'metrics': metrics,
            'evokeds': evokeds,
            'qc_pass': qc_pass,
        })

        logger.info(f'  {recording_id}: {file_info.get("group","?")} / '
                     f'{file_info.get("stimulus","?")} / '
                     f'{file_info.get("visit","?")} '
                     f'(QC={"PASS" if qc_pass else "FAIL"})')

    return recordings


def group_recordings(recordings):
    """Group recordings by stimulus x group x visit.

    Returns dict: {(stimulus, group, visit): [recording, ...]}
    """
    groups = {}
    for rec in recordings:
        fi = rec['file_info']
        key = (fi.get('stimulus', '?'),
               fi.get('group', '?'),
               fi.get('visit', '?'))
        groups.setdefault(key, []).append(rec)
    return groups


# ── ERP Measurement ──────────────────────────────────────────────────────────

def measure_component(evoked, tmin, tmax, polarity='negative', electrodes=None):
    """Measure ERP component at each electrode in the given time window.

    Args:
        evoked: MNE Evoked object.
        tmin, tmax: Time window in seconds.
        polarity: 'negative' (find min, for N2) or 'positive' (find max, for P3).
        electrodes: list of channel names to measure, or None for all EEG.

    Returns: {channel: {'mean_amp', 'peak_amp', 'peak_lat', 'area_lat_50'}}
    """
    t_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    if not t_mask.any():
        return {}
    times_ms = evoked.times[t_mask] * 1000
    results = {}

    for ch_idx, ch_name in enumerate(evoked.ch_names):
        if evoked.get_channel_types()[ch_idx] != 'eeg':
            continue
        if electrodes is not None and ch_name not in electrodes:
            continue
        data_uv = evoked.data[ch_idx, t_mask] * 1e6

        if polarity == 'negative':
            peak_amp = float(data_uv.min())
            peak_lat = float(times_ms[np.argmin(data_uv)])
        else:
            peak_amp = float(data_uv.max())
            peak_lat = float(times_ms[np.argmax(data_uv)])

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


# ── Grand Averaging ──────────────────────────────────────────────────────────

def compute_grand_averages(grouped, condition, logger):
    """Compute grand average for a condition within each group key.

    Args:
        grouped: {(stim, group, visit): [recordings]}
        condition: string, e.g. 'standard', 'deviant', DIFF_COMMENT

    Returns: {(stim, group, visit): Evoked}
    """
    result = {}
    for key, recs in grouped.items():
        evokeds_list = []
        for rec in recs:
            if condition in rec['evokeds']:
                evokeds_list.append(rec['evokeds'][condition])

        if not evokeds_list:
            continue

        if len(evokeds_list) == 1:
            grand = evokeds_list[0].copy()
        else:
            grand = mne.grand_average(evokeds_list, interpolate_bads=True,
                                       drop_bads=False)
        grand.comment = f'{condition} [{key[0]} {key[1]} {key[2]}]'
        result[key] = grand
        logger.debug(f'  Grand avg {condition} {key}: '
                     f'{len(evokeds_list)} recordings')

    return result


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_grand_erp_overlay(grand_avgs_std, grand_avgs_dev, grand_avgs_diff,
                           channels, title):
    """Plot grand average ERP overlay (Standard vs Deviant vs Difference)
    for multiple group keys side by side.
    """
    keys = sorted(set(grand_avgs_std.keys()) & set(grand_avgs_dev.keys()))
    if not keys:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f'{title}: no data', ha='center', va='center')
        ax.axis('off')
        return fig

    n_keys = len(keys)
    n_ch = len(channels)
    fig, axes = plt.subplots(n_keys, n_ch,
                             figsize=(5 * n_ch, 3.5 * n_keys),
                             squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for row, key in enumerate(keys):
        evk_std = grand_avgs_std[key]
        evk_dev = grand_avgs_dev[key]
        evk_diff = grand_avgs_diff.get(key, None)
        times_ms = evk_std.times * 1000

        for col, ch in enumerate(channels):
            ax = axes[row, col]

            if ch not in evk_std.ch_names or ch not in evk_dev.ch_names:
                ax.text(0.5, 0.5, f'{ch} N/A', ha='center', va='center')
                ax.axis('off')
                continue

            idx_s = evk_std.ch_names.index(ch)
            idx_d = evk_dev.ch_names.index(ch)

            ax.plot(times_ms, evk_std.data[idx_s] * 1e6,
                    'b-', label='Standard', linewidth=1.2)
            ax.plot(times_ms, evk_dev.data[idx_d] * 1e6,
                    'r-', label='Deviant', linewidth=1.2)
            if evk_diff and ch in evk_diff.ch_names:
                idx_df = evk_diff.ch_names.index(ch)
                ax.plot(times_ms, evk_diff.data[idx_df] * 1e6,
                        'k--', label='Difference', linewidth=1.0, alpha=0.7)

            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
            ax.axvline(0, color='gray', linestyle=':', linewidth=0.5)
            ax.axvspan(N2_TMIN * 1000, N2_TMAX * 1000,
                       alpha=0.06, color='blue')
            ax.axvspan(P3_TMIN * 1000, P3_TMAX * 1000,
                       alpha=0.06, color='red')

            label = f'{key[0]} {key[1]} {key[2]}'
            ax.set_title(f'{ch} — {label}', fontsize=9)
            ax.set_xlabel('Time (ms)', fontsize=8)
            ax.set_ylabel('uV', fontsize=8)
            ax.invert_yaxis()
            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc='upper right')

    fig.tight_layout()
    return fig


def plot_group_comparison_overlay(grand_avgs_diff, channels, stim, title):
    """Plot Treat vs Control difference waves at selected channels.

    Shows v1 and v2 for both groups on the same axes.
    """
    available = [ch for ch in channels]
    n_ch = len(available)
    fig, axes = plt.subplots(1, n_ch, figsize=(5 * n_ch, 4), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight='bold')

    line_styles = {
        (stim, 'treat', 'v1'):   ('red', '-',  'Treat v1 (pre)'),
        (stim, 'treat', 'v2'):   ('red', '--', 'Treat v2 (post)'),
        (stim, 'control', 'v1'): ('blue', '-',  'Control v1 (pre)'),
        (stim, 'control', 'v2'): ('blue', '--', 'Control v2 (post)'),
    }

    for col, ch in enumerate(available):
        ax = axes[0, col]
        plotted = False
        for key, (color, ls, label) in line_styles.items():
            if key not in grand_avgs_diff:
                continue
            evk = grand_avgs_diff[key]
            if ch not in evk.ch_names:
                continue
            idx = evk.ch_names.index(ch)
            times_ms = evk.times * 1000
            ax.plot(times_ms, evk.data[idx] * 1e6,
                    color=color, linestyle=ls, label=label, linewidth=1.3)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, f'{ch}: no data', ha='center', va='center')
            ax.axis('off')
            continue

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.5)
        ax.axvspan(N2_TMIN * 1000, N2_TMAX * 1000,
                   alpha=0.06, color='blue')
        ax.axvspan(P3_TMIN * 1000, P3_TMAX * 1000,
                   alpha=0.06, color='red')
        ax.set_title(ch, fontsize=11)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('uV')
        ax.legend(fontsize=7, loc='upper right')
        ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_three_panel_topomap(evokeds_dict, keys, tmin, tmax, param,
                             title, logger):
    """Plot topomaps for up to 4 group-keys side by side.

    param: 'mean_amp' or 'peak_amp' or 'peak_lat'
    """
    present_keys = [k for k in keys if k in evokeds_dict]
    n = len(present_keys)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f'{title}: no data', ha='center', va='center')
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight='bold')

    for ax, key in zip(axes, present_keys):
        evk = evokeds_dict[key]
        t_mask = (evk.times >= tmin) & (evk.times <= tmax)

        values = np.zeros(len(evk.ch_names))
        for ch_idx in range(len(evk.ch_names)):
            if evk.get_channel_types()[ch_idx] != 'eeg':
                continue
            data_uv = evk.data[ch_idx, t_mask] * 1e6
            if param == 'mean_amp':
                values[ch_idx] = data_uv.mean()
            elif param == 'peak_amp':
                values[ch_idx] = data_uv.min() if 'N2' in title else data_uv.max()
            elif param == 'peak_lat':
                if 'N2' in title:
                    values[ch_idx] = evk.times[t_mask][np.argmin(data_uv)] * 1000
                else:
                    values[ch_idx] = evk.times[t_mask][np.argmax(data_uv)] * 1000

        try:
            mne.viz.plot_topomap(values, evk.info, axes=ax, show=False)
        except Exception as e:
            ax.text(0.5, 0.5, f'Topo error:\n{e}',
                    ha='center', va='center', fontsize=7,
                    transform=ax.transAxes)

        label = f'{key[1]} {key[2]}' if len(key) >= 3 else str(key)
        ax.set_title(label, fontsize=9)

    fig.tight_layout()
    return fig


def plot_comparison_table(grouped, component, tmin, tmax, polarity,
                          electrodes, title, logger):
    """Per-electrode table comparing conditions (group x visit).

    Columns: electrode, then one column per condition key.
    """
    keys = sorted(grouped.keys())
    present_keys = [k for k in keys if k in grouped]
    if not present_keys:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f'{title}: no data', ha='center', va='center')
        ax.axis('off')
        return fig

    # Measure for each condition
    all_metrics = {}
    for key in present_keys:
        evk = grouped[key]
        all_metrics[key] = measure_component(evk, tmin, tmax, polarity,
                                              electrodes)

    col_labels = ['Electrode']
    for k in present_keys:
        col_labels.append(f'{k[1]} {k[2]}')

    cell_text = []
    for ch in electrodes:
        row = [ch]
        for k in present_keys:
            m = all_metrics[k]
            if ch in m:
                row.append(f'{m[ch]["peak_amp"]:.2f} ({m[ch]["peak_lat"]:.0f}ms)')
            else:
                row.append('—')
        cell_text.append(row)

    n_rows = len(cell_text)
    fig_h = max(4.0, 0.35 * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(3.5 * len(col_labels), fig_h))
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     colColours=['#d0d0d0'] * len(col_labels),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    fig.tight_layout()
    return fig


# ── Statistical Tests ────────────────────────────────────────────────────────

def run_statistics(recordings, logger):
    """Run group comparison statistics on N2/P3 metrics.

    Tests:
      1. Between-group (Treat vs Control) at each visit: Mann-Whitney U
      2. Within-group pre vs post (v1 vs v2): Wilcoxon signed-rank

    Returns list of result dicts for CSV export.
    """
    results = []

    for stim in ('MAU', 'TONE'):
        stim_recs = [r for r in recordings
                     if r['file_info'].get('stimulus') == stim]
        if not stim_recs:
            continue

        # Organise by group and visit
        data = {}  # {(group, visit): {subject_id: metrics}}
        for rec in stim_recs:
            fi = rec['file_info']
            key = (fi.get('group', '?'), fi.get('visit', '?'))
            sid = fi.get('subject_id', '?')
            data.setdefault(key, {})[sid] = rec['metrics']

        # ── Between-group: Treat vs Control at each visit ──
        for visit in ('v1', 'v2'):
            treat_data = data.get(('treat', visit), {})
            control_data = data.get(('control', visit), {})

            if len(treat_data) < 2 or len(control_data) < 2:
                logger.info(f'  {stim} {visit}: insufficient data for '
                            f'between-group test '
                            f'(treat={len(treat_data)}, '
                            f'control={len(control_data)})')
                continue

            for component, comp_key, channels in [
                ('N2', 'n2', N2_CHANNELS),
                ('P3', 'p3', P3_CHANNELS),
            ]:
                for ch in channels:
                    vals_treat = []
                    vals_control = []
                    for m in treat_data.values():
                        comp_data = m.get(comp_key, {})
                        if ch in comp_data:
                            vals_treat.append(comp_data[ch]['peak_amp'])
                    for m in control_data.values():
                        comp_data = m.get(comp_key, {})
                        if ch in comp_data:
                            vals_control.append(comp_data[ch]['peak_amp'])

                    if len(vals_treat) >= 2 and len(vals_control) >= 2:
                        u_stat, p_val = stats.mannwhitneyu(
                            vals_treat, vals_control,
                            alternative='two-sided')
                        results.append({
                            'stimulus': stim,
                            'test': 'between_group',
                            'comparison': f'treat_vs_control_{visit}',
                            'component': component,
                            'metric': 'peak_amp',
                            'channel': ch,
                            'n_treat': len(vals_treat),
                            'n_control': len(vals_control),
                            'mean_treat': np.mean(vals_treat),
                            'mean_control': np.mean(vals_control),
                            'statistic': u_stat,
                            'p_value': p_val,
                        })

        # ── Within-group: v1 vs v2 (paired by subject) ──
        for group in ('treat', 'control'):
            v1_data = data.get((group, 'v1'), {})
            v2_data = data.get((group, 'v2'), {})
            paired_sids = sorted(set(v1_data.keys()) & set(v2_data.keys()))

            if len(paired_sids) < 3:
                logger.info(f'  {stim} {group}: insufficient paired data '
                            f'for within-group test '
                            f'({len(paired_sids)} paired)')
                continue

            for component, comp_key, channels in [
                ('N2', 'n2', N2_CHANNELS),
                ('P3', 'p3', P3_CHANNELS),
            ]:
                for ch in channels:
                    vals_v1 = []
                    vals_v2 = []
                    for sid in paired_sids:
                        c1 = v1_data[sid].get(comp_key, {})
                        c2 = v2_data[sid].get(comp_key, {})
                        if ch in c1 and ch in c2:
                            vals_v1.append(c1[ch]['peak_amp'])
                            vals_v2.append(c2[ch]['peak_amp'])

                    if len(vals_v1) >= 3:
                        try:
                            w_stat, p_val = stats.wilcoxon(vals_v1, vals_v2)
                        except ValueError:
                            continue
                        results.append({
                            'stimulus': stim,
                            'test': 'within_group',
                            'comparison': f'{group}_v1_vs_v2',
                            'component': component,
                            'metric': 'peak_amp',
                            'channel': ch,
                            'n_paired': len(vals_v1),
                            'mean_v1': np.mean(vals_v1),
                            'mean_v2': np.mean(vals_v2),
                            'statistic': w_stat,
                            'p_value': p_val,
                        })

    logger.info(f'  Statistical tests: {len(results)} comparisons')
    return results


def plot_statistics_table(stat_results, title):
    """Plot statistics results as a table figure."""
    if not stat_results:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f'{title}: no results (need more data)',
                ha='center', va='center')
        ax.axis('off')
        return fig

    col_labels = ['Stimulus', 'Test', 'Comparison', 'Component',
                  'Channel', 'N', 'Statistic', 'p-value', 'Sig.']
    cell_text = []
    cell_colors = []

    for r in stat_results:
        n_str = (f'{r.get("n_treat","")}/{r.get("n_control","")}'
                 if r['test'] == 'between_group'
                 else str(r.get('n_paired', '')))
        p = r['p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

        row = [
            r['stimulus'], r['test'], r['comparison'], r['component'],
            r['channel'], n_str, f'{r["statistic"]:.1f}',
            f'{p:.4f}', sig,
        ]
        cell_text.append(row)

        if p < 0.05:
            cell_colors.append(['#d4edda'] * len(row))
        else:
            cell_colors.append(['white'] * len(row))

    n_rows = len(cell_text)
    fig_h = max(4.0, 0.35 * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=['#d0d0d0'] * len(col_labels),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)

    fig.tight_layout()
    return fig


# ── CSV Export ───────────────────────────────────────────────────────────────

def export_raw_arrays(recordings, output_path, logger):
    """Export per-subject per-electrode N2/P3 metrics for downstream analysis."""
    with open(str(output_path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Recording_ID', 'Subject_ID', 'Cohort', 'Group', 'Stimulus',
            'Visit', 'Component', 'Electrode',
            'Mean_Amplitude_uV', 'Peak_Amplitude_uV',
            'Peak_Latency_ms', 'Area_Latency_50pct_ms',
            'QC_Pass',
        ])

        for rec in recordings:
            fi = rec['file_info']
            rid = rec['recording_id']
            qc = rec['qc_pass']

            diff = rec['evokeds'].get(DIFF_COMMENT)
            if diff is None:
                continue

            # N2
            n2 = measure_component(diff, N2_TMIN, N2_TMAX, 'negative')
            for ch, m in sorted(n2.items()):
                writer.writerow([
                    rid, fi.get('subject_id', ''), fi.get('cohort', ''),
                    fi.get('group', ''), fi.get('stimulus', ''),
                    fi.get('visit', ''), 'N2', ch,
                    f'{m["mean_amp"]:.4f}', f'{m["peak_amp"]:.4f}',
                    f'{m["peak_lat"]:.1f}', f'{m["area_lat_50"]:.1f}',
                    qc,
                ])

            # P3
            p3 = measure_component(diff, P3_TMIN, P3_TMAX, 'positive')
            for ch, m in sorted(p3.items()):
                writer.writerow([
                    rid, fi.get('subject_id', ''), fi.get('cohort', ''),
                    fi.get('group', ''), fi.get('stimulus', ''),
                    fi.get('visit', ''), 'P3', ch,
                    f'{m["mean_amp"]:.4f}', f'{m["peak_amp"]:.4f}',
                    f'{m["peak_lat"]:.1f}', f'{m["area_lat_50"]:.1f}',
                    qc,
                ])

    logger.info(f'Raw arrays CSV saved: {output_path}')


def export_statistics(stat_results, output_path, logger):
    """Export statistical test results to CSV."""
    with open(str(output_path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Stimulus', 'Test', 'Comparison', 'Component', 'Metric',
            'Channel', 'N', 'Statistic', 'P_Value', 'Significant',
        ])

        for r in stat_results:
            n_str = (f'{r.get("n_treat","")}/{r.get("n_control","")}'
                     if r['test'] == 'between_group'
                     else str(r.get('n_paired', '')))
            sig = 'yes' if r['p_value'] < 0.05 else 'no'
            writer.writerow([
                r['stimulus'], r['test'], r['comparison'], r['component'],
                r['metric'], r['channel'], n_str,
                f'{r["statistic"]:.4f}', f'{r["p_value"]:.6f}', sig,
            ])

    logger.info(f'Statistics CSV saved: {output_path}')


def export_group_summary(grouped, output_path, logger):
    """Export grand average per-electrode metrics to CSV."""
    with open(str(output_path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Stimulus', 'Group', 'Visit', 'Component', 'Electrode',
            'Mean_Amplitude_uV', 'Peak_Amplitude_uV',
            'Peak_Latency_ms', 'Area_Latency_50pct_ms',
            'N_Recordings',
        ])

        for key, recs in sorted(grouped.items()):
            stim, group, visit = key
            # Collect difference waves
            diff_evokeds = [r['evokeds'][DIFF_COMMENT] for r in recs
                            if DIFF_COMMENT in r['evokeds']]
            if not diff_evokeds:
                continue

            if len(diff_evokeds) == 1:
                grand = diff_evokeds[0]
            else:
                grand = mne.grand_average(diff_evokeds,
                                           interpolate_bads=True,
                                           drop_bads=False)

            for comp, tmin, tmax, pol in [('N2', N2_TMIN, N2_TMAX, 'negative'),
                                          ('P3', P3_TMIN, P3_TMAX, 'positive')]:
                metrics = measure_component(grand, tmin, tmax, pol,
                                             TARGET_ELECTRODES)
                for ch in TARGET_ELECTRODES:
                    if ch not in metrics:
                        continue
                    m = metrics[ch]
                    writer.writerow([
                        stim, group, visit, comp, ch,
                        f'{m["mean_amp"]:.4f}', f'{m["peak_amp"]:.4f}',
                        f'{m["peak_lat"]:.1f}', f'{m["area_lat_50"]:.1f}',
                        len(diff_evokeds),
                    ])

    logger.info(f'Group summary CSV saved: {output_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='OddBall Group Analysis — Grand Averages, Statistics, '
                    'and Comparison Reports')
    parser.add_argument('--input', required=True,
                        help='Directory containing individual outputs '
                             '(with *_metrics.json and *-ave.fif)')
    parser.add_argument('--output', required=True,
                        help='Output directory for group report and CSVs')
    parser.add_argument('--include-qc-fail', action='store_true',
                        help='Include recordings that failed QC')
    args = parser.parse_args()

    mne.set_log_level('WARNING')

    script_dir = Path(__file__).resolve().parent
    logger = setup_logging(script_dir / 'logs')
    logger.info('=' * 60)
    logger.info('OddBall Group Analysis')
    logger.info('=' * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────
    logger.info('── 1. Loading individual data ──')
    recordings = load_all_data(args.input, logger)

    if not args.include_qc_fail:
        before = len(recordings)
        recordings = [r for r in recordings if r['qc_pass']]
        excluded = before - len(recordings)
        if excluded:
            logger.info(f'  Excluded {excluded} QC-failed recordings '
                        f'(use --include-qc-fail to include)')
    logger.info(f'  Total recordings for analysis: {len(recordings)}')

    if len(recordings) < 1:
        logger.error('No recordings available for analysis')
        sys.exit(1)

    # ── 2. Group recordings ──────────────────────────────────────────────
    logger.info('── 2. Grouping recordings ──')
    grouped = group_recordings(recordings)
    for key, recs in sorted(grouped.items()):
        logger.info(f'  {key}: {len(recs)} recordings')

    # ── 3. Compute grand averages ────────────────────────────────────────
    logger.info('── 3. Computing grand averages ──')
    ga_standard = compute_grand_averages(grouped, 'standard', logger)
    ga_deviant = compute_grand_averages(grouped, 'deviant', logger)
    ga_diff = compute_grand_averages(grouped, DIFF_COMMENT, logger)

    # ── 4. Run statistics ────────────────────────────────────────────────
    logger.info('── 4. Running statistical tests ──')
    stat_results = run_statistics(recordings, logger)

    # ── 5. Export CSVs ───────────────────────────────────────────────────
    logger.info('── 5. Exporting CSVs ──')
    export_raw_arrays(recordings, output_dir / 'oddball_raw_arrays.csv',
                      logger)
    export_group_summary(grouped, output_dir / 'oddball_group_table.csv',
                         logger)
    if stat_results:
        export_statistics(stat_results,
                          output_dir / 'oddball_statistics.csv', logger)

    # ── 6. Generate report ───────────────────────────────────────────────
    logger.info('── 6. Generating group report ──')
    report = mne.Report(
        title='OddBall Group Analysis: Treat vs Control',
        verbose='WARNING')

    # Section A: Grand average ERP overlays per stimulus
    for stim in ('MAU', 'TONE'):
        stim_keys_std = {k: v for k, v in ga_standard.items()
                         if k[0] == stim}
        stim_keys_dev = {k: v for k, v in ga_deviant.items()
                         if k[0] == stim}
        stim_keys_diff = {k: v for k, v in ga_diff.items()
                          if k[0] == stim}

        if stim_keys_std:
            fig = plot_grand_erp_overlay(
                stim_keys_std, stim_keys_dev, stim_keys_diff,
                ['Fz', 'Cz', 'Pz'],
                f'{stim} — Grand average ERPs (Fz, Cz, Pz)')
            report.add_figure(fig,
                              title=f'{stim} — Grand average ERPs',
                              tags=('erp', stim.lower()))
            plt.close(fig)

    # Section B: Group comparison overlays (Treat vs Control diff waves)
    for stim in ('MAU', 'TONE'):
        stim_diff = {k: v for k, v in ga_diff.items() if k[0] == stim}
        if stim_diff:
            # N2 channels
            fig = plot_group_comparison_overlay(
                stim_diff, ['Fz', 'FCz', 'Cz'], stim,
                f'{stim} — Difference wave: Treat vs Control (N2 channels)')
            report.add_figure(fig,
                              title=f'{stim} — Group comparison N2',
                              tags=('comparison', 'n2', stim.lower()))
            plt.close(fig)

            # P3 channels
            fig = plot_group_comparison_overlay(
                stim_diff, ['Cz', 'CPz', 'Pz'], stim,
                f'{stim} — Difference wave: Treat vs Control (P3 channels)')
            report.add_figure(fig,
                              title=f'{stim} — Group comparison P3',
                              tags=('comparison', 'p3', stim.lower()))
            plt.close(fig)

    # Section C: Topographic maps (difference waves)
    for stim in ('MAU', 'TONE'):
        stim_diff = {k: v for k, v in ga_diff.items() if k[0] == stim}
        if not stim_diff:
            continue

        keys_order = [
            (stim, 'treat', 'v1'), (stim, 'treat', 'v2'),
            (stim, 'control', 'v1'), (stim, 'control', 'v2'),
        ]

        # N2 topomaps
        fig = plot_three_panel_topomap(
            stim_diff, keys_order, N2_TMIN, N2_TMAX, 'mean_amp',
            f'{stim} — N2 mean amplitude topomap (difference wave)',
            logger)
        report.add_figure(fig,
                          title=f'{stim} — N2 topomap',
                          tags=('topomap', 'n2', stim.lower()))
        plt.close(fig)

        # P3 topomaps
        fig = plot_three_panel_topomap(
            stim_diff, keys_order, P3_TMIN, P3_TMAX, 'mean_amp',
            f'{stim} — P3 mean amplitude topomap (difference wave)',
            logger)
        report.add_figure(fig,
                          title=f'{stim} — P3 topomap',
                          tags=('topomap', 'p3', stim.lower()))
        plt.close(fig)

    # Section D: Per-electrode comparison tables
    for stim in ('MAU', 'TONE'):
        stim_diff = {k: v for k, v in ga_diff.items() if k[0] == stim}
        if not stim_diff:
            continue

        # N2 table
        fig = plot_comparison_table(
            stim_diff, 'N2', N2_TMIN, N2_TMAX, 'negative',
            ['Fz', 'FCz', 'Cz', 'F3', 'F4'],
            f'{stim} — N2 peak amplitude & latency (difference wave)',
            logger)
        report.add_figure(fig,
                          title=f'{stim} — N2 comparison table',
                          tags=('table', 'n2', stim.lower()))
        plt.close(fig)

        # P3 table
        fig = plot_comparison_table(
            stim_diff, 'P3', P3_TMIN, P3_TMAX, 'positive',
            ['Cz', 'CPz', 'Pz', 'P3', 'P4'],
            f'{stim} — P3 peak amplitude & latency (difference wave)',
            logger)
        report.add_figure(fig,
                          title=f'{stim} — P3 comparison table',
                          tags=('table', 'p3', stim.lower()))
        plt.close(fig)

    # Section E: Statistics table
    if stat_results:
        fig = plot_statistics_table(
            stat_results,
            'Statistical comparisons — N2/P3 peak amplitude')
        report.add_figure(fig,
                          title='Statistical comparisons',
                          tags=('statistics',))
        plt.close(fig)
    else:
        logger.info('  No statistical tests to plot '
                    '(need more recordings for group comparisons)')

    # Save report
    report_path = output_dir / 'oddball_group_report.html'
    report.save(str(report_path), overwrite=True, open_browser=False,
                verbose='WARNING')
    logger.info(f'Report saved: {report_path}')
    logger.info('Done.')


if __name__ == '__main__':
    main()
