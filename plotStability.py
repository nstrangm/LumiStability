import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import uproot

TRIGGER_BASES = ['FT0', 'FDD', 'ZDC']

# Cross-section map: dataset_name -> trigger_base -> cross_section (in mb)
# trigger_base is the prefix before the first underscore (e.g. FT0, FDD, ZDC)
CROSS_SECTIONS = {
    "LHC25ad (pO)": {
        'FT0': 372,  # Dummy value in mb
        'FDD': 345,      # Dummy value in mb
        'ZDC': 387,      # Dummy value in mb
    },
    "LHC25ae (OO)": {
        'FT0': 1151,  # Dummy value in mb
        'FDD': 1100,      # Dummy value in mb
        'ZDC': 1216       # Dummy value in mb
    },
    "LHC24apq (pp-Ref)": {
        'FT0': 46.0,  # Dummy value in mb
        'FDD': 40.7,      # Dummy value in mb
    },
    "LHC23 (pp)": {
        'FT0': 52.7,  # Dummy value in mb
        'FDD': 49.62,      # Dummy value in mb
    },
    "LHC24 (pp)": {
        'FT0': 53.49,  # Dummy value in mb
        'FDD': 50.86,      # Dummy value in mb
    },
    "LHC25 (pp)": {
        'FT0': 52.71,  # https://indico.cern.ch/event/1633849/#preview:5700356 Slide 7, Scan 10824, average v0
        'FDD': 48.36,  # https://indico.cern.ch/event/1633849/#preview:5700356 Slide 8, Scan 10824, average v0
    },
    "LHC26 (pp)": {
        'FT0': 53.39,  # https://indico.cern.ch/event/1732937/#preview:6014966 Last slide, FBCT average
        'FDD': 50.42,  # https://indico.cern.ch/event/1732937/#preview:6014966 Last slide, FBCT average
    }
}

# Display labels for trigger suffixes in legends; unlisted suffixes are shown as-is
SUFFIX_LABELS = {
    'B': 'Any BC',
    'SLD5': 'SL - 5',
    'SLD50': 'SL - 50',
    'SLD100': 'SL - 100',
    'SLD300': 'SL - 300',
    'L100': 'L - 100',
}

# Colors/markers to cycle through
TRIGGER_STYLES = [
    ('#0072B2', 'o'),   # blue circle
    ('#D55E00', 's'),   # vermilion square
    ('#009E73', '^'),   # green triangle
    ('#CC79A7', 'D'),   # rose diamond
    ('#E69F00', 'v'),   # orange triangle
    ('#56B4E9', 'P'),   # sky blue plus-filled
    ('#332288', 'X'),   # indigo x-filled
    ('#88CCEE', 'h'),   # light blue hexagon
    ('#44AA99', '*'),   # teal star
    ('#117733', '+'),   # dark green plus
    ('#999933', 'x'),   # olive x
    ('#AA4499', '<'),   # purple left triangle
    ('#000000', '>'),   # black right triangle
    ('#EE7733', '1'),   # orange tri-down
    ('#33BBEE', '2'),   # cyan tri-up
    ('#EE3377', '3'),   # magenta tri-left
    ('#BBBBBB', '4'),   # gray tri-right
    ('#228833', 'p'),   # green pentagon
    ('#CCBB44', 'H'),   # yellow-green hexagon
    ('#4477AA', 'd'),   # blue thin diamond
]


def get_trigger_base(column_name):
    """Extract the trigger base name (e.g. 'FT0' from 'FT0_B')."""
    for base in TRIGGER_BASES:
        if column_name.startswith(base + '_'):
            return base
    return None


def get_suffix_label(suffix):
    """Return the legend label for a trigger suffix, falling back to the suffix itself."""
    return format_legend_label(SUFFIX_LABELS.get(suffix, suffix))


def format_legend_label(label):
    """Replace FT0 with TVX in a legend label."""
    return label.replace('FT0', 'TVX')


def get_trigger_suffix(column_name):
    """Extract the shared suffix after the trigger base (e.g. 'SL_FDD100')."""
    base = get_trigger_base(column_name)
    if not base:
        return None
    return column_name[len(base) + 1:]


def get_cross_section(dataset_name, trigger_column):
    """
    Get the cross section for a given dataset and trigger column name.
    
    Args:
        dataset_name (str): The dataset name (e.g. 'LHC24apq (pp Ref)')
        trigger_column (str): Column name (e.g. 'FT0_B')
    
    Returns:
        float: Cross section in mb, or None if not found
    """
    base = get_trigger_base(trigger_column)
    if base and dataset_name in CROSS_SECTIONS:
        return CROSS_SECTIONS[dataset_name].get(base)
    return None


def resolve_trigger_columns(requested_items, available_columns):
    """
    Resolve CLI trigger inputs to concrete dataframe columns.

    Each requested item can be either:
    - a full column name like 'FT0_SL_FDD100'
    - a shared suffix like 'SL_FDD100', which expands to
      'FT0_SL_FDD100' and 'FDD_SL_FDD100' when present
    """
    resolved = []
    missing = []
    available_set = set(available_columns)

    for item in requested_items:
        if item in available_set:
            resolved.append(item)
            continue

        expanded = []
        for base in ['FT0', 'FDD', "ZDC"]:
            candidate = f"{base}_{item}"
            if candidate in available_set:
                expanded.append(candidate)

        if expanded:
            resolved.extend(expanded)
        else:
            missing.append(item)

    deduplicated = list(dict.fromkeys(resolved))
    return deduplicated, missing


def build_matched_luminosity_ratios(active_triggers):
    """
    Build FDD / FT0 and ZDC / FT0 luminosity ratios by matching trigger suffixes.

    Args:
        active_triggers (list): tuples shaped like
            (lumi_column, original_column, color, marker, ...)

    Returns:
        list: (ratio_name, numerator_lumi, denominator_lumi, color, marker, suffix)
    """
    triggers_by_suffix = {}
    suffix_order = []

    for trigger in active_triggers:
        lumi_column, original_column, color, marker = trigger[:4]
        base = get_trigger_base(original_column)
        suffix = get_trigger_suffix(original_column)

        if base not in {'FT0', 'FDD', 'ZDC'} or suffix is None:
            continue

        if suffix not in triggers_by_suffix:
            triggers_by_suffix[suffix] = {}
            suffix_order.append(suffix)

        triggers_by_suffix[suffix][base] = (lumi_column, original_column, color, marker)

    ratio_definitions = []
    for suffix in suffix_order:
        matched = triggers_by_suffix[suffix]
        if 'FT0' not in matched:
            print(f"Warning: Could not build FDD/FT0 or ZDC/FT0 luminosity ratio for '{suffix}'")
            continue

        denominator_lumi, denominator_name, _, _ = matched['FT0']
        matched_numerator = False
        for numerator_base in ('FDD', 'ZDC'):
            if numerator_base not in matched:
                continue

            numerator_lumi, numerator_name, color, marker = matched[numerator_base]
            ratio_name = f"{numerator_name} / {denominator_name}"
            ratio_definitions.append((ratio_name, numerator_lumi, denominator_lumi, color, marker, suffix))
            matched_numerator = True

        if not matched_numerator:
            print(f"Warning: Could not build FDD/FT0 or ZDC/FT0 luminosity ratio for '{suffix}'")

    return ratio_definitions


def get_direct_luminosity_ratio_series(data, ratio_definitions):
    """Compute direct FDD/FT0 and ZDC/FT0 luminosity ratio series for plotting."""
    ratio_series = []

    for ratio_name, numerator, denominator, color, marker, suffix in ratio_definitions:
        direct_ratio = data[numerator] / data[denominator]
        ratio_series.append((ratio_name, direct_ratio, color, marker, suffix))

    return ratio_series


def normalize_ratio_by_median(raw_ratio):
    """Normalize a ratio by its median when that scale is defined and nonzero."""
    median_ratio = raw_ratio.median()
    if not np.isfinite(median_ratio) or median_ratio == 0:
        return raw_ratio, median_ratio, None
    return raw_ratio / median_ratio, median_ratio, 1.0 / median_ratio


def configure_time_axis(ax, start_time, end_time):
    """Choose adaptive time ticks, extending short spans by one tick at each end."""
    time_span = end_time - start_time
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8, interval_multiples=True)
    if time_span <= pd.Timedelta(days=14):
        ticks = locator.tick_values(start_time, end_time)
        first_interval = ticks[1] - ticks[0]
        last_interval = ticks[-1] - ticks[-2]
        ticks = np.concatenate(([ticks[0] - first_interval], ticks,
                                [ticks[-1] + last_interval]))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        formatter = mdates.DateFormatter('%d.%m.\n%H:%M')
        label_padding = pd.to_timedelta(max(0.0,
            mdates.date2num(start_time) - ticks[0],
            ticks[-1] - mdates.date2num(end_time)
        ), unit='D')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=15)
    else:
        ax.xaxis.set_major_locator(locator)
        formatter = mdates.ConciseDateFormatter(locator)
        label_padding = pd.Timedelta(0)
    ax.xaxis.set_major_formatter(formatter)
    if time_span > pd.Timedelta(days=14):
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=15)
    return time_span, label_padding


def combine_legends(ax_left, ax_right, loc='center left', fontsize=15, bbox_to_anchor=None):
    """Combine legend entries from both axes into one legend on the left axis."""
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()

    if handles_left or handles_right:
        ax_left.legend(handles_left + handles_right, labels_left + labels_right,
                       loc=loc, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)


def get_run_tick_style(n_runs):
    """Return adaptive (fontsize, rotation) for run-number x-axis labels."""
    if n_runs <= 10:
        return 15, 45
    if n_runs <= 20:
        return 12, 45
    if n_runs <= 35:
        return 10, 45
    if n_runs <= 60:
        return 9, 60
    if n_runs <= 100:
        return 8, 75
    if n_runs <= 150:
        return 7, 90
    if n_runs <= 250:
        return 6, 90
    if n_runs <= 350:
        return 5, 90
    if n_runs <= 500:
        return 4, 90
    return 2, 90


def get_run_figure_width(n_runs):
    """Return adaptive figure width (inches) for dense run-number plots."""
    return min(45, max(16, n_runs * 0.09))

def load_vdm_run_numbers():
    """Load VdM run numbers from the list beside this script."""
    vdm_runs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VdMRuns.txt')
    with open(vdm_runs_path) as vdm_runs_file:
        return {
            int(run.strip())
            for run in vdm_runs_file.read().replace('\n', ',').split(',')
            if run.strip()
        }


def get_vdm_run_intervals(df, vdm_run_numbers):
    """Return VdM runs present in the data and their observed time intervals."""
    numeric_runs = pd.to_numeric(df['runNumber'], errors='coerce')
    vdm_mask = numeric_runs.isin(vdm_run_numbers)
    present_runs = set(numeric_runs[vdm_mask].astype(int))

    run_times = df.loc[vdm_mask].assign(
        _vdm_run=numeric_runs[vdm_mask].astype(int)
    ).groupby('_vdm_run')['datetime'].agg(['min', 'max'])
    intervals = list(run_times[['min', 'max']].itertuples(index=False, name=None))
    return present_runs, intervals


def highlight_vdm_time_intervals(ax, intervals):
    """Shade the observed time interval for each VdM run."""
    for start_time, end_time in intervals:
        if start_time == end_time:
            start_time -= pd.Timedelta(seconds=30)
            end_time += pd.Timedelta(seconds=30)
        ax.axvspan(start_time, end_time, color='gold', alpha=0.2, zorder=0)


def highlight_vdm_run_positions(ax, run_numbers, vdm_run_numbers):
    """Shade the discrete x-axis position of each VdM run."""
    for position, run_number in enumerate(run_numbers):
        if int(run_number) in vdm_run_numbers:
            ax.axvspan(position - 0.5, position + 0.5, color='gold', alpha=0.2, zorder=0)


def style_run_tick_labels(ax, run_numbers, fontsize, rotation):
    """Apply adaptive style to run tick labels and highlight selected run IDs in red."""
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=rotation, ha='right', fontsize=fontsize)


def plot_luminosity_ratio_with_FT0_B(df, ratio_series, plot_title, output_dir, vdm_intervals):
    """Plot matched luminosity ratios with FT0_B interaction rate on a right axis."""
    if not ratio_series:
        print("Warning: No matched luminosity ratios available for dual-axis time plot")
        return

    if 'FT0_B' not in df.columns:
        print("Warning: FT0_B not found, skipping dual-axis time plot")
        return

    ft0_rate_khz = df['FT0_B'] / 1000

    fig, ax_left = plt.subplots(1, 1, figsize=(16, 8))
    ax_right = ax_left.twinx()
    highlight_vdm_time_intervals(ax_left, vdm_intervals)

    fig.suptitle(f'Luminosity Ratios and FT0_B Rate vs Time - {plot_title}',
                 fontsize=20, fontweight='bold')

    all_ratios = []
    for ratio_name, direct_ratio, color, marker, suffix in ratio_series:
        valid_ratios = direct_ratio.replace([np.inf, -np.inf], np.nan)
        all_ratios.extend(valid_ratios.dropna().tolist())
        ax_left.plot(df['datetime'], valid_ratios,
                     marker,
                     markersize=4,
                     alpha=0.7,
                     color=color,
                     label=get_suffix_label(suffix),
                     markeredgewidth=0.5)

    ax_right.plot(df['datetime'], ft0_rate_khz,
                  color='black',
                  linestyle='-',
                  linewidth=2,
                  alpha=0.85,
                  label='FT0_B Rate')

    ax_left.set_xlabel('Time (UTC)', fontsize=20, fontweight='bold')
    ax_left.set_ylabel('Luminosity Ratio (FDD or ZDC / FT0)', fontsize=20, fontweight='bold')
    ax_right.set_ylabel('FT0_B Rate (kHz)', fontsize=20, fontweight='bold')
    ax_left.grid(True, alpha=0.3)
    ax_left.tick_params(axis='both', which='major', labelsize=15)
    ax_right.tick_params(axis='y', which='major', labelsize=15)

    if all_ratios:
        data_min = min(all_ratios)
        data_max = max(all_ratios)
        margin = (data_max - data_min) * 0.05
        if margin == 0:
            margin = max(abs(data_max) * 0.05, 0.05)
        ax_left.set_ylim(data_min - margin, data_max + margin)
    else:
        ax_left.set_ylim(0, 2)

    rate_min = ft0_rate_khz.min()
    rate_max = ft0_rate_khz.max()
    rate_margin = (rate_max - rate_min) * 0.05
    if rate_margin == 0:
        rate_margin = max(abs(rate_max) * 0.05, 0.05)
    ax_right.set_ylim(max(0, rate_min - rate_margin), rate_max + rate_margin)

    time_span, label_padding = configure_time_axis(ax_left, df['datetime'].min(), df['datetime'].max())

    time_padding = max(time_span * 0.02, label_padding)
    ax_left.set_xlim(df['datetime'].min() - time_padding, df['datetime'].max() + time_padding)

    combine_legends(ax_left, ax_right)
    plt.tight_layout(pad=2.0)

    output_file = os.path.join(output_dir, 'luminosity_ratio_with_FT0_B.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Dual-axis luminosity ratio time plot saved as: {output_file}")


def plot_luminosity_ratio_per_run_with_FT0_B(run_positions, run_numbers, run_averages,
                                                ratio_series, plot_title, output_dir,
                                                vdm_run_numbers):
    """Plot per-run matched luminosity ratios with FT0_B average rate on a right axis."""
    if not ratio_series:
        print("Warning: No matched luminosity ratios available for dual-axis per-run plot")
        return

    if 'FT0_B' not in run_averages.columns:
        print("Warning: FT0_B not found, skipping dual-axis per-run plot")
        return

    run_fig_width = get_run_figure_width(len(run_numbers))
    fig, ax_left = plt.subplots(1, 1, figsize=(run_fig_width, 8))
    ax_right = ax_left.twinx()
    highlight_vdm_run_positions(ax_left, run_numbers, vdm_run_numbers)

    # Put a custom title if the plot title contains (pO)
    if "(pp-Ref)" in plot_title:
        fig.suptitle(r'$\mathbf{ALICE\ this\ thesis}$ - pp, ${\sqrt{s}}$ = 5.36 TeV', fontsize=20)
    elif "23" in plot_title:
        fig.suptitle(r'$\mathbf{ALICE\ this\ thesis}$ - pp, ${\sqrt{s}}$ = 13.6 TeV - 2023', fontsize=20)
    elif "24" in plot_title:
        fig.suptitle(r'$\mathbf{ALICE\ this\ thesis}$ - pp, ${\sqrt{s}}$ = 13.6 TeV - 2024', fontsize=20)
    elif "25" in plot_title:
        fig.suptitle(r'$\mathbf{ALICE\ this\ thesis}$ - pp, ${\sqrt{s}}$ = 13.6 TeV - 2025', fontsize=20)
    elif "26" in plot_title:
        fig.suptitle(r'$\mathbf{ALICE\ this\ thesis}$ - pp, ${\sqrt{s}}$ = 13.6 TeV - 2026', fontsize=20)
    else:
        fig.suptitle(f'Per-Run Luminosity Ratios and TVX Rate - {plot_title}', fontsize=20, fontweight='bold')

    all_ratios = []
    marker_shapes = ['s', '^', 'v', 'D', 'P', 'X', 'o']
    marker_sizes = [9, 9, 9, 10, 10, 9, 7]
    for ratio_index, (ratio_name, direct_ratio, color, marker, suffix) in enumerate(ratio_series):
        if ratio_index == 4:
            color = "#00CC29"
        if ratio_index == 3:
            color = '#CC00CC'
        marker = marker_shapes[ratio_index] if ratio_index < len(marker_shapes) else marker
        markersize = marker_sizes[ratio_index] if ratio_index < len(marker_sizes) else 5
        valid_ratios = direct_ratio.replace([np.inf, -np.inf], np.nan)
        all_ratios.extend(valid_ratios.dropna().tolist())
        ax_left.plot(run_positions, valid_ratios,
                     marker,
                     markersize=markersize,
                     alpha=0.9,
                     color=color,
                     markerfacecolor=color,
                    #  markeredgecolor='black',
                     markeredgewidth=0.8,
                     label=get_suffix_label(suffix),
                     linestyle='-',
                     linewidth=1.5)

    ax_right.plot(run_positions, run_averages['FT0_B'],
                  color='black',
                  marker='o',
                  markersize=10,
                  linestyle='-',
                  linewidth=2,
                  alpha=0.85,
                  label='TVX Rate')

    ax_left.set_xlabel('Run Number', fontsize=20, fontweight='bold', labelpad=20)
    
    if "pp" in plot_title:
        ax_left.set_ylabel('Luminosity Ratio (FDD / TVX)', fontsize=20, fontweight='bold', labelpad=20)
    else:
        ax_left.set_ylabel('Luminosity Ratio (FDD or ZDC / FT0)', fontsize=20, fontweight='bold', labelpad=20)
    
    ax_right.set_ylabel('TVX Rate (kHz)', fontsize=20, fontweight='bold', labelpad=20)
    ax_left.grid(True, alpha=0.3)
    ax_left.tick_params(axis='both', which='major', labelsize=15)
    ax_right.tick_params(axis='y', which='major', labelsize=15)
    ax_left.set_xticks(run_positions)
    ax_left.set_xticklabels(run_numbers)
    run_tick_fontsize, run_tick_rotation = get_run_tick_style(len(run_numbers))
    if "LHC26" in plot_title:
        run_number_fontsize = run_tick_fontsize*1.2
    else:
        run_number_fontsize = run_tick_fontsize * 1.5
    style_run_tick_labels(ax_left, run_numbers, run_number_fontsize, run_tick_rotation)

    if all_ratios:
        data_min = min(all_ratios)
        data_max = max(all_ratios)
        margin = (data_max - data_min) * 0.05
        if margin == 0:
            margin = max(abs(data_max) * 0.05, 0.05)
        ax_left.set_ylim(data_min - margin, data_max + margin)
    else:
        ax_left.set_ylim(0, 2)

    # rate_min = run_averages['FT0_B'].min()
    rate_min = 0
    rate_max = run_averages['FT0_B'].max()
    rate_margin = (rate_max - rate_min) * 0.05
    if rate_margin == 0:
        rate_margin = max(abs(rate_max) * 0.05, 0.05)
    ax_right.set_ylim(max(0, rate_min - rate_margin), rate_max + rate_margin)

    legend_anchor = (0.7, 0.22) if "LHC26" in plot_title else (0.13, 0.4)
    combine_legends(ax_left, ax_right, loc='center left', bbox_to_anchor=legend_anchor)
    plt.tight_layout(pad=2.0)

    output_file = os.path.join(output_dir, 'luminosity_ratio_with_FT0_B_per_run.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Dual-axis luminosity ratio per-run plot saved as: {output_file}")
    
    # ========================
    # CREATE AND SAVE ROOT HISTOGRAMS: PER-RUN DATA
    # ========================
    
    print(f"\nCreating per-run histograms for ROOT export...")
    
    # Dictionary to store histograms
    root_histograms = {}
    
    # Create histograms for each luminosity ratio
    for ratio_name, direct_ratio, color, marker, suffix in ratio_series:
        valid_ratios = direct_ratio.replace([np.inf, -np.inf], np.nan)
        valid_mask = np.isfinite(valid_ratios)
        
        if valid_mask.sum() > 0:
            # Create bin edges: one bin per run, centered at run position
            # Bins go from -0.5 to n-0.5, so run 0 is in bin [−0.5, 0.5], etc.
            n_runs = len(run_numbers)
            bin_edges = np.arange(-0.5, n_runs, 1.0)
            
            # Histogram values are the ratio values at each run position
            hist_values = valid_ratios.to_numpy()
            
            hist_name = f"lumi_ratio_{suffix.replace('_', '')}"
            root_histograms[hist_name] = (bin_edges, hist_values)
            
            print(f"  {suffix}:")
            print(f"    Histogram name: {hist_name}")
            print(f"    Number of runs: {n_runs}")
            print(f"    Mean ratio: {valid_ratios[valid_mask].mean():.3f}")
    
    # Create histogram for FT0_B rate
    if 'FT0_B' in run_averages.columns:
        ft0_rate = run_averages['FT0_B'].to_numpy()
        valid_mask = np.isfinite(ft0_rate)
        
        if valid_mask.sum() > 0:
            n_runs = len(run_numbers)
            bin_edges = np.arange(-0.5, n_runs, 1.0)
            
            hist_name = "FT0_B_rate"
            root_histograms[hist_name] = (bin_edges, ft0_rate)
            
            print(f"  FT0_B Rate:")
            print(f"    Histogram name: {hist_name}")
            print(f"    Number of runs: {n_runs}")
            print(f"    Mean rate: {ft0_rate[valid_mask].mean():.3f} kHz")
    
    if root_histograms:
        root_file = os.path.join(output_dir, 'luminosity_ratio_FT0_B_per_run.root')
        
        with uproot.recreate(root_file) as file:
            for hist_name, (bin_edges, hist_values) in root_histograms.items():
                file[hist_name] = (hist_values, bin_edges)
        
        print(f"\nROOT histograms saved to: {root_file}")
        print(f"  Contains {len(root_histograms)} histogram(s): {list(root_histograms.keys())}")
        
        # Create a ROOT macro to set bin errors to 0 and add axis labels
        macro_file = os.path.join(output_dir, 'set_histogram_metadata.C')
        with open(macro_file, 'w') as f:
            f.write('void set_histogram_metadata() {\n')
            f.write(f'  TFile* f = TFile::Open("{root_file}", "UPDATE");\n')
            f.write('  if (!f || f->IsZombie()) {\n')
            f.write('    std::cout << "Error opening file!" << std::endl;\n')
            f.write('    return;\n')
            f.write('  }\n\n')
            
            # Add run number array
            f.write('  int run_numbers[] = {')
            f.write(', '.join(str(int(run)) for run in run_numbers))
            f.write('};\n')
            f.write(f'  int n_runs = {len(run_numbers)};\n\n')
            
            for hist_name, (bin_edges, hist_values) in root_histograms.items():
                safe_name = hist_name.replace('-', '_')
                f.write(f'  TH1* h_{safe_name} = (TH1*)f->Get("{hist_name}");\n')
                f.write(f'  if (h_{safe_name}) {{\n')
                f.write(f'    h_{safe_name}->GetXaxis()->SetTitle("Run number");\n')
                
                if "FT0_B_rate" in hist_name:
                    f.write(f'    h_{safe_name}->GetYaxis()->SetTitle("Average Rate (kHz)");\n')
                else:
                    f.write(f'    h_{safe_name}->GetYaxis()->SetTitle("Luminosity Ratio (FDD or ZDC / FT0)");\n')
                
                f.write(f'    for (int i = 0; i < n_runs; i++) {{\n')
                f.write(f'      h_{safe_name}->GetXaxis()->SetBinLabel(i+1, Form("%d", run_numbers[i]));\n')
                f.write(f'      h_{safe_name}->SetBinError(i+1, 0);\n')
                f.write(f'    }}\n')
                
                if len(run_numbers) > 10:
                    f.write(f'    h_{safe_name}->GetXaxis()->LabelsOption("v");\n')
                
                f.write(f'    h_{safe_name}->Write("", TObject::kOverwrite);\n')
                f.write(f'  }}\n\n')
            
            f.write('  f->Close();\n')
            f.write('  std::cout << "Histogram metadata updated successfully!" << std::endl;\n')
            f.write('}\n')
        
        print(f"\nROOT macro created: {macro_file}")
        print(f"  Run this macro to set uncertainties to 0 and add axis labels:")
        print(f"  root -l -b -q {macro_file}")
    else:
        print("No valid histograms to save")


def load_data(input_dirs):
    """Load and concatenate Output.csv from the specified directories."""
    all_dfs = []
    for dir_path in input_dirs:
        dir_path = os.path.join("Data/", dir_path)
        csv_path = os.path.join(dir_path, "Output.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                all_dfs.append(df)
                print(f"  Loaded {len(df)} rows from {csv_path}")
        else:
            print(f"  Warning: {csv_path} not found, skipping")
    if not all_dfs:
        print("Error: No Output.csv files found!")
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values('timestamp', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"Total: {len(combined)} rows")
    return combined


def plot_stability(df, triggers, plot_title, output_dir, vdm_intervals):
    """
    Plot trigger rates vs time.
    Keeps the exact same formatting as the original script.
    
    Args:
        df (DataFrame): The data (with 'datetime' column already set)
        triggers (list): List of (column_name, color, marker) tuples
        plot_title (str): Title for the plot
        output_dir (str): Directory to save the plot
    """
    
    # Convert all rates from Hz to kHz by dividing by 1000
    df = df.copy()
    for col, _, _ in triggers:
        df[col] = df[col] / 1000
    
    # Create figure with two subplots (one above the other)
    # Increased figure size to accommodate larger fonts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 13), 
                                   height_ratios=[3, 1],  # Top plot 3x taller than bottom
                                   sharex=True)  # Share x-axis between plots
    highlight_vdm_time_intervals(ax1, vdm_intervals)
    highlight_vdm_time_intervals(ax2, vdm_intervals)
    
    # Increase main title font size
    fig.suptitle(f'Trigger Rates and Ratios vs Time - {plot_title}', fontsize=20, fontweight='bold')
    
    print(f"Active triggers: {[col for col, _, _ in triggers]}")
    
    # ========================
    # TOP PLOT: Absolute rates
    # ========================
    
    # Plot each active trigger rate on the same axes
    for col, color, marker in triggers:
        ax1.plot(df['datetime'], df[col], 
                marker,                 
                markersize=5,           # Slightly larger markers
                alpha=0.7,              
                color=color,            
                label=format_legend_label(col),  # Use column name as label
                markeredgewidth=0.5)    
    
    # Set y-axis to start at 0
    ax1.set_ylim(bottom=0)
    
    # Find fill transitions and run transitions
    fill_changes = df.groupby('fill_number').agg({
        'datetime': ['min', 'max']
    }).reset_index()
    fill_changes.columns = ['fill_number', 'start_time', 'end_time']
    
    run_changes = df.groupby(['runNumber', 'fill_number']).agg({
        'datetime': ['min', 'max']
    }).reset_index()
    run_changes.columns = ['runNumber', 'fill_number', 'start_time', 'end_time']
    
    print(f"Found {len(fill_changes)} fills and {len(run_changes)} runs:")
    for _, fill_row in fill_changes.iterrows():
        print(f"  Fill {fill_row['fill_number']}: "
              f"{fill_row['start_time'].strftime('%I:%M %p')} - {fill_row['end_time'].strftime('%I:%M %p')}")
        runs_in_fill = run_changes[run_changes['fill_number'] == fill_row['fill_number']]
        for _, run_row in runs_in_fill.iterrows():
            print(f"    Run {run_row['runNumber']}: "
                  f"{run_row['start_time'].strftime('%I:%M %p')} - {run_row['end_time'].strftime('%I:%M %p')}")
    
    # Add fill and run annotations to top plot
    y_min, y_max = ax1.get_ylim()
    fill_bar_y_min = 0.02      
    fill_bar_y_max = 0.08     
    run_bar_y_min = 0.09     
    run_bar_y_max = 0.15      
    
    fill_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    run_colors = ['steelblue', 'darkgreen', 'darkred', 'gold', 'mediumvioletred']
    
    # Add fill annotations (bottom bar) to top plot
    for i, (_, fill_row) in enumerate(fill_changes.iterrows()):
        color = fill_colors[i % len(fill_colors)]
        ax1.axvspan(fill_row['start_time'], fill_row['end_time'], 
                   ymin=fill_bar_y_min, ymax=fill_bar_y_max,
                   alpha=0.6, color=color)
        
        middle_time = fill_row['start_time'] + (fill_row['end_time'] - fill_row['start_time']) / 2
        annotation_text = f"Fill {fill_row['fill_number']}"
        text_y = y_min + (y_max - y_min) * (fill_bar_y_min + fill_bar_y_max) / 2
        
        ax1.text(middle_time, text_y, annotation_text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=15,  # Increased font size
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add run annotations (top bar) to top plot
    for i, (_, run_row) in enumerate(run_changes.iterrows()):
        fill_index = list(fill_changes['fill_number']).index(run_row['fill_number'])
        color = run_colors[fill_index % len(run_colors)]
        
        ax1.axvspan(run_row['start_time'], run_row['end_time'], 
                   ymin=run_bar_y_min, ymax=run_bar_y_max,
                   alpha=0.6, color=color)
        
        middle_time = run_row['start_time'] + (run_row['end_time'] - run_row['start_time']) / 2
        annotation_text = f"{run_row['runNumber']}"
        text_y = y_min + (y_max - y_min) * (run_bar_y_min + run_bar_y_max) / 2
        
        ax1.text(middle_time, text_y, annotation_text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=15,   # Increased font size
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Customize top plot with larger, bold fonts
    ax1.set_ylabel('Rate (kHz)', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=15)
    
    # Increase tick label font sizes
    ax1.tick_params(axis='both', which='major', labelsize=15)
    
    # ========================
    # BOTTOM PLOT: Rate ratios
    # ========================
    
    # Use first trigger as reference
    reference_col = triggers[0][0]
    reference_name = triggers[0][0]
    
    # Create ratio definitions based on active triggers
    ratio_definitions = []
    for col, color, marker in triggers[1:]:
        ratio_name = f"{col} / {reference_name}"
        ratio_definitions.append((ratio_name, col, reference_col, color, marker))
    
    if not ratio_definitions:
        print("Warning: No ratios to plot (only one trigger active)")
        # Hide the ratio subplot if no ratios to show
        ax2.set_visible(False)
        fig.subplots_adjust(hspace=0.15)  # Increased spacing for larger fonts
    else:
        print(f"\nRatio statistics (reference: {reference_name}):")
        
        # Plot ratios
        for ratio_name, numerator, denominator, color, marker in ratio_definitions:
            # Calculate raw ratio, handling potential division by zero
            raw_ratio = df[numerator] / df[denominator]
            
            # Calculate the median of the raw ratio instead of mean
            # Median is more robust to outliers than mean
            normalized_ratio, median_ratio, scale_factor = normalize_ratio_by_median(raw_ratio)
            
            # Update statistics output to show both mean and median
            finite_ratio = raw_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            mean_ratio = finite_ratio.mean()
            std_ratio = finite_ratio.std()
            
            relative_std = (f"{100 * std_ratio / median_ratio:.1f}%"
                        if np.isfinite(median_ratio) and median_ratio != 0
                        else "undefined (zero or unavailable median)")
            print(f"  {ratio_name}: median = {median_ratio:.3f}, mean = {mean_ratio:.3f}, "
                f"std = {std_ratio:.3f} ({relative_std})")
            
            ax2.plot(df['datetime'], normalized_ratio,
                    marker,
                    markersize=4,  # Slightly larger markers
                    alpha=0.7,
                    color=color,
                          label=format_legend_label(
                              f"{ratio_name} (x{scale_factor:.3f})" if scale_factor is not None
                              else f"{ratio_name} (unnormalized)"),
                    markeredgewidth=0.5)
        
        # Customize bottom plot with larger, bold fonts
        ax2.set_xlabel('Time (UTC)', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Ratio (median-normalized where defined)', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=15)
        
        # Increase tick label font sizes
        ax2.tick_params(axis='both', which='major', labelsize=15)
        
        # Add horizontal line at ratio = 1 for reference
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=1.005, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=0.995, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Set y-limits based on the normalized data (using median to match plotting)
        all_normalized_ratios = []
        for ratio_name, numerator, denominator, color, marker in ratio_definitions:
            raw_ratio = df[numerator] / df[denominator]
            normalized_ratio, _, _ = normalize_ratio_by_median(raw_ratio)
            finite_normalized = normalized_ratio.replace([np.inf, -np.inf], np.nan)
            all_normalized_ratios.extend(finite_normalized.dropna().tolist())
        
        # Set limits to show variation around 1
        if all_normalized_ratios:
            data_min = min(all_normalized_ratios)
            data_max = max(all_normalized_ratios)
            margin = (data_max - data_min) * 0.05  # 5% margin of the data range
            ax2.set_ylim(data_min - margin, data_max + margin)
        else:
            ax2.set_ylim(0.95, 1.05)  # Default ±5% range
    
    # ============================
    # X-AXIS FORMATTING IMPROVEMENTS
    # ============================
    
    # Use the full data range to select an appropriate tick density.
    time_span, label_padding = configure_time_axis(ax2, df['datetime'].min(), df['datetime'].max())
    total_hours = time_span.total_seconds() / 3600
    
    ax2.tick_params(axis='x', which='major', length=8)
    
    # Add some padding around the data
    time_padding = max(time_span * 0.02, label_padding)
    ax1.set_xlim(df['datetime'].min() - time_padding, df['datetime'].max() + time_padding)
    
    print(f"Data time span: {total_hours:.1f} hours")
    print(f"Time range: {df['datetime'].min().strftime('%I:%M %p')} - {df['datetime'].max().strftime('%I:%M %p')}")
    
    # Adjust layout with more padding for larger fonts
    plt.tight_layout(pad=2.0)  # Increased padding
    
    # Save the plot
    output_file = os.path.join(output_dir, f'stability_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Plot saved as: {output_file}")


def plot_luminosity(df, triggers, plot_title, output_dir, dataset_name, vdm_intervals):
    """
    Plot luminosity (rate / cross_section) vs time.
    
    Args:
        df (DataFrame): The data (with 'datetime' column already set)
        triggers (list): List of (column_name, color, marker) tuples
        plot_title (str): Title for the plot
        output_dir (str): Directory to save the plot
        dataset_name (str): Dataset name for cross section lookup
    """
    
    df = df.copy()
    
    # Filter triggers that have valid cross sections and compute luminosity
    active_triggers = []
    for col, color, marker in triggers:
        cross_section = get_cross_section(dataset_name, col)
        if cross_section is not None and cross_section > 0:
            lumi_column = f"{col}_Lumi"
            df[lumi_column] = df[col] / cross_section / 1000  # Luminosity in Hz/mb -> kHz/mb
            active_triggers.append((lumi_column, col, color, marker, cross_section))
            print(f"Applied cross section {cross_section} mb to {col}")
        elif cross_section == 0:
            print(f"Warning: Cross section is 0 for {col}, skipping")
        else:
            print(f"Warning: No cross section found for {col} in dataset '{dataset_name}', skipping")
    
    if not active_triggers:
        print("Error: No triggers with valid cross sections available!")
        return
    
    print(f"Active triggers: {[orig for _, orig, _, _, _ in active_triggers]}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, 
                                   figsize=(16, 12), 
                                   height_ratios=[2, 1],
                                   sharex=True)
    highlight_vdm_time_intervals(ax1, vdm_intervals)
    highlight_vdm_time_intervals(ax2, vdm_intervals)

    # Put a custom title if the plot title contains (pO)
    if "(pO)" in plot_title:
        fig.suptitle(r'$\mathbf{ALICE\ this\ thesis}$ - pO, ${\sqrt{s_{\mathrm{NN}}}}$ = 9.62 TeV', fontsize=20)
    elif "(OO)" in plot_title:
        fig.suptitle(r'$\mathbf{ALICE\ this\ thesis}$ - OO, ${\sqrt{s_{\mathrm{NN}}}}$ = 5.36 TeV', fontsize=20)
    else:
        fig.suptitle(r'$\mathbf{ALICE\ this\ thesis}$ - ' + plot_title, fontsize=20)
    
    
    # ========================
    # TOP PLOT: Absolute luminosity
    # ========================
    
    for trigger_index, (lumi_column, orig_col, color, marker, _) in enumerate(active_triggers):
        ax1.plot(df['datetime'], df[lumi_column], 
                marker,                 
                markersize=10 if trigger_index == 0 else 3 if trigger_index == 1 else 1,
                alpha=0.9,              
                color=color,            
                label=format_legend_label(orig_col.replace('_B', '')),
                markerfacecolor=color,
                markeredgewidth=0.8)
    
    ax1.set_ylim(bottom=0)
    
    # Find fill and run transitions
    fill_changes = df.groupby('fill_number').agg({
        'datetime': ['min', 'max']
    }).reset_index()
    fill_changes.columns = ['fill_number', 'start_time', 'end_time']
    
    run_changes = df.groupby(['runNumber', 'fill_number']).agg({
        'datetime': ['min', 'max']
    }).reset_index()
    run_changes.columns = ['runNumber', 'fill_number', 'start_time', 'end_time']
    
    # Add fill and run annotations
    y_min, y_max = ax1.get_ylim()
    fill_bar_y_min = 0.02
    fill_bar_y_max = 0.08
    run_bar_y_min = 0.09
    run_bar_y_max = 0.15
    
    fill_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    run_colors = ['steelblue', 'darkgreen', 'darkred', 'gold', 'mediumvioletred']
    
    for i, (_, fill_row) in enumerate(fill_changes.iterrows()):
        color = fill_colors[i % len(fill_colors)]
        ax1.axvspan(fill_row['start_time'], fill_row['end_time'], 
                   ymin=fill_bar_y_min, ymax=fill_bar_y_max,
                   alpha=0.6, color=color)
        
        middle_time = fill_row['start_time'] + (fill_row['end_time'] - fill_row['start_time']) / 2
        annotation_text = f"Fill {fill_row['fill_number']}"
        text_y = y_min + (y_max - y_min) * (fill_bar_y_min + fill_bar_y_max) / 2
        
        ax1.text(middle_time, text_y, annotation_text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=15,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    for i, (_, run_row) in enumerate(run_changes.iterrows()):
        fill_index = list(fill_changes['fill_number']).index(run_row['fill_number'])
        color = run_colors[fill_index % len(run_colors)]
        
        ax1.axvspan(run_row['start_time'], run_row['end_time'], 
                   ymin=run_bar_y_min, ymax=run_bar_y_max,
                   alpha=0.6, color=color)
        
        middle_time = run_row['start_time'] + (run_row['end_time'] - run_row['start_time']) / 2
        annotation_text = f"{run_row['runNumber']}"
        text_y = y_min + (y_max - y_min) * (run_bar_y_min + run_bar_y_max) / 2
        
        ax1.text(middle_time, text_y, annotation_text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=15,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax1.set_ylabel('Luminosity (Hz/μb)', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    
    # ========================
    # BOTTOM PLOT: Luminosity ratios
    # ========================
    
    ratio_definitions = build_matched_luminosity_ratios(active_triggers)
    ratio_series = get_direct_luminosity_ratio_series(df, ratio_definitions)
    
    if not ratio_definitions:
        print("Warning: No matched FDD/FT0 or ZDC/FT0 luminosity ratios to plot")
        ax2.set_visible(False)
        fig.subplots_adjust(hspace=0.15)
    else:
        print(f"\nLuminosity ratio statistics (FDD or ZDC / FT0, matched by BC suffix):")
        
        for ratio_name, direct_ratio, color, marker, suffix in ratio_series:
            
            finite_ratio = direct_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if finite_ratio.empty:
                print(f"  {suffix}: no finite values, skipping statistics")
            else:
                mean_ratio = finite_ratio.mean()
                median_ratio = finite_ratio.median()
                std_ratio = finite_ratio.std()
                relative_std = (f"{100 * std_ratio / mean_ratio:.1f}%"
                                if np.isfinite(mean_ratio) and mean_ratio != 0
                                else "undefined (zero or unavailable mean)")
                print(f"  {suffix}: median = {median_ratio:.3f}, mean = {mean_ratio:.3f}, "
                      f"std = {std_ratio:.3f} ({relative_std})")
            
            ax2.plot(df['datetime'], direct_ratio,
                    marker,
                    markersize=3,
                    alpha=0.9,
                    color=color,
                    label=format_legend_label(ratio_name.replace('_B', '')),
                    markeredgewidth=0.5)
        
        ax2.set_xlabel('Time (UTC)', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Luminosity Ratio', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        
        # Set y-limits based on direct ratios
        all_ratios = []
        for ratio_name, direct_ratio, color, marker, suffix in ratio_series:
            valid_ratios = direct_ratio.replace([np.inf, -np.inf], np.nan).dropna().tolist()
            all_ratios.extend(valid_ratios)
        
        if all_ratios:
            data_min = min(all_ratios)
            data_max = max(all_ratios)
            margin = (data_max - data_min) * 0.05
            ax2.set_ylim(data_min - margin, data_max + margin)
        else:
            ax2.set_ylim(0, 2)
    
    # X-axis formatting
    time_span, label_padding = configure_time_axis(ax2, df['datetime'].min(), df['datetime'].max())
    total_hours = time_span.total_seconds() / 3600
    
    ax2.tick_params(axis='x', which='major', length=8)
    
    time_padding = max(time_span * 0.02, label_padding)
    ax1.set_xlim(df['datetime'].min() - time_padding, df['datetime'].max() + time_padding)
    
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    output_file = os.path.join(output_dir, f'luminosity_plot.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Luminosity plot saved as: {output_file}")

    plot_luminosity_ratio_with_FT0_B(df, ratio_series, plot_title, output_dir, vdm_intervals)
    
    # ========================
    # CREATE AND SAVE ROOT HISTOGRAMS: LUMINOSITY vs TIME
    # ========================
    
    print(f"\nCreating luminosity vs time histograms for ROOT export...")
    
    # Convert datetime to Unix timestamp (seconds since epoch) for binning
    df['timestamp_seconds'] = df['datetime'].astype('int64') / 1e9
    
    # Time binning: typically 1 minute (60 seconds) per bin
    time_bin_width = 60  # seconds per bin
    
    # Determine time range
    time_min = df['timestamp_seconds'].min()
    time_max = df['timestamp_seconds'].max()
    n_time_bins = int((time_max - time_min) / time_bin_width) + 1
    
    # Dictionary to store histograms
    root_histograms = {}
    
    for lumi_column, orig_col, color, marker, xs in active_triggers:
        lumi_values = df[lumi_column].to_numpy()
        time_values = df['timestamp_seconds'].to_numpy()
        
        valid_mask = np.isfinite(lumi_values) & np.isfinite(time_values)
        lumi_clean = lumi_values[valid_mask]
        time_clean = time_values[valid_mask]
        
        if len(lumi_clean) > 0:
            hist_values, bin_edges = np.histogram(time_clean, 
                                                  bins=n_time_bins,
                                                  range=(time_min, time_max),
                                                  weights=lumi_clean)
            
            counts, _ = np.histogram(time_clean, 
                                    bins=n_time_bins,
                                    range=(time_min, time_max))
            
            hist_values = np.divide(hist_values, counts, where=counts > 0, out=np.zeros_like(hist_values))
            
            hist_name = f"lumi_{orig_col}"
            root_histograms[hist_name] = (bin_edges, hist_values)
            
            print(f"  {orig_col}:")
            print(f"    Histogram name: {hist_name}")
            print(f"    Time range: {time_min:.0f} - {time_max:.0f} seconds (Unix timestamp)")
            print(f"    Number of time bins: {n_time_bins} ({time_bin_width}s per bin)")
            print(f"    Mean luminosity: {np.mean(lumi_clean):.3f} Hz/μb")
        else:
            print(f"  {orig_col}: No valid data for histogram")
    
    if root_histograms:
        root_file = os.path.join(output_dir, f'luminosity_histogram.root')
        
        with uproot.recreate(root_file) as file:
            for hist_name, (bin_edges, hist_values) in root_histograms.items():
                file[hist_name] = (hist_values, bin_edges)
        
        print(f"\nROOT histograms saved to: {root_file}")
        print(f"  Contains {len(root_histograms)} histogram(s): {list(root_histograms.keys())}")
        print(f"  Note: Histograms have default ROOT uncertainties. To set uncertainties to 0,")
        print(f"        modify them in ROOT after reading: h->SetBinError(i, 0) for all bins.")
    else:
        print("No valid histograms to save")


def plot_stability_per_run(df, triggers, plot_title, output_dir, vdm_run_numbers):
    """
    Plot average trigger rates per run number.
    
    Args:
        df (DataFrame): The data
        triggers (list): List of (column_name, color, marker) tuples
        plot_title (str): Title for the plot
        output_dir (str): Directory to save the plot
    """
    
    cols = [col for col, _, _ in triggers]
    
    # Calculate average rates per run
    run_averages = df.groupby('runNumber')[cols].mean().reset_index()
    
    # Convert rates from Hz to kHz
    for col in cols:
        run_averages[col] = run_averages[col] / 1000
    
    print(f"Active triggers: {cols}")
    
    # Create discrete x-axis positions for each run (evenly spaced)
    run_numbers = run_averages['runNumber'].tolist()
    x_positions = np.arange(len(run_numbers))  # 0, 1, 2, 3, ...
    run_tick_fontsize, run_tick_rotation = get_run_tick_style(len(run_numbers))
    run_fig_width = get_run_figure_width(len(run_numbers))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(run_fig_width, 13), 
                                   height_ratios=[3, 1],
                                   sharex=True)
    highlight_vdm_run_positions(ax1, run_numbers, vdm_run_numbers)
    highlight_vdm_run_positions(ax2, run_numbers, vdm_run_numbers)
    
    fig.suptitle(f'Average Trigger Rates and Ratios per Run - {plot_title}', fontsize=20, fontweight='bold')
    
    # ========================
    # TOP PLOT: Absolute rates per run
    # ========================
    
    for col, color, marker in triggers:
        ax1.plot(x_positions, run_averages[col], 
                marker,                 
                markersize=8,
                alpha=0.7,              
                color=color,            
                label=format_legend_label(col),
                markeredgewidth=0.5,
                linestyle='-',
                linewidth=1.5)
    
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('Average Rate (kHz)', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    
    # Set x-axis ticks to show run numbers at discrete positions
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(run_numbers)
    
    # ========================
    # BOTTOM PLOT: Rate ratios per run
    # ========================
    
    # Use first trigger as reference
    reference_col = triggers[0][0]
    
    ratio_definitions = []
    for col, color, marker in triggers[1:]:
        ratio_name = f"{col} / {reference_col}"
        ratio_definitions.append((ratio_name, col, reference_col, color, marker))
    
    if not ratio_definitions:
        print("Warning: No ratios to plot (only one trigger active)")
        ax2.set_visible(False)
        fig.subplots_adjust(hspace=0.15)
    else:
        print(f"\nPer-run ratio statistics (reference: {reference_col}):")
        
        for ratio_name, numerator, denominator, color, marker in ratio_definitions:
            raw_ratio = run_averages[numerator] / run_averages[denominator]
            normalized_ratio, median_ratio, scale_factor = normalize_ratio_by_median(raw_ratio)
            
            finite_ratio = raw_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            mean_ratio = finite_ratio.mean()
            std_ratio = finite_ratio.std()
            
            relative_std = (f"{100 * std_ratio / median_ratio:.1f}%"
                        if np.isfinite(median_ratio) and median_ratio != 0
                        else "undefined (zero or unavailable median)")
            print(f"  {ratio_name}: median = {median_ratio:.3f}, mean = {mean_ratio:.3f}, "
                f"std = {std_ratio:.3f} ({relative_std})")
            
            ax2.plot(x_positions, normalized_ratio,
                    marker,
                    markersize=7,
                    alpha=0.7,
                    color=color,
                          label=format_legend_label(
                              f"{ratio_name} (x{scale_factor:.3f})" if scale_factor is not None
                              else f"{ratio_name} (unnormalized)"),
                    markeredgewidth=0.5,
                    linestyle='-',
                    linewidth=1.5)
        
        ax2.set_xlabel('Run Number', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Ratio (median-normalized where defined)', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        
        # Set x-axis ticks to show run numbers at discrete positions
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(run_numbers)
        
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=1.005, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=0.995, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Set y-limits based on normalized data
        all_normalized_ratios = []
        for ratio_name, numerator, denominator, color, marker in ratio_definitions:
            raw_ratio = run_averages[numerator] / run_averages[denominator]
            normalized_ratio, _, _ = normalize_ratio_by_median(raw_ratio)
            finite_normalized = normalized_ratio.replace([np.inf, -np.inf], np.nan)
            all_normalized_ratios.extend(finite_normalized.dropna().tolist())
        
        if all_normalized_ratios:
            data_min = min(all_normalized_ratios)
            data_max = max(all_normalized_ratios)
            margin = (data_max - data_min) * 0.05
            ax2.set_ylim(data_min - margin, data_max + margin)
        else:
            ax2.set_ylim(0.95, 1.05)
    
    # Rotate x-axis labels for better readability
    style_run_tick_labels(ax2, run_numbers, run_tick_fontsize, run_tick_rotation)
    
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    output_file = os.path.join(output_dir, f'stability_plot_per_run.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Per-run plot saved as: {output_file}")


def plot_luminosity_per_run(df, triggers, plot_title, output_dir, dataset_name,
                            vdm_run_numbers):
    """
    Plot average luminosity (rate / cross_section) per run number.
    
    Args:
        df (DataFrame): The data
        triggers (list): List of (column_name, color, marker) tuples
        plot_title (str): Title for the plot
        output_dir (str): Directory to save the plot
        dataset_name (str): Dataset name for cross section lookup
    """
    
    df = df.copy()
    
    # Apply cross sections and calculate luminosity
    active_triggers = []
    for col, color, marker in triggers:
        cross_section = get_cross_section(dataset_name, col)
        if cross_section is not None and cross_section > 0:
            lumi_col = f"{col}_Lumi"
            df[lumi_col] = df[col] / cross_section  # Luminosity in Hz/mb
            active_triggers.append((lumi_col, col, color, marker))
            print(f"Applied cross section {cross_section} mb to {col}")
        elif cross_section == 0:
            print(f"Warning: Cross section is 0 for {col}, skipping")
        else:
            print(f"Warning: No cross section found for {col}, skipping")
    
    if not active_triggers:
        print("Error: No luminosity columns could be created (no valid cross sections)")
        return
    
    lumi_cols = [lc for lc, _, _, _ in active_triggers]
    rate_columns = [col for col in ['FT0_B'] if col in df.columns]
    
    # Calculate average luminosity per run
    aggregation_columns = lumi_cols + rate_columns
    run_averages = df.groupby('runNumber')[aggregation_columns].mean().reset_index()

    if 'FT0_B' in run_averages.columns:
        run_averages['FT0_B'] = run_averages['FT0_B'] / 1000
    
    print(f"Active triggers: {[orig for _, orig, _, _ in active_triggers]}")
    
    # Create discrete x-axis positions for each run
    run_numbers = run_averages['runNumber'].tolist()
    x_positions = np.arange(len(run_numbers))
    run_tick_fontsize, run_tick_rotation = get_run_tick_style(len(run_numbers))
    run_fig_width = get_run_figure_width(len(run_numbers))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(run_fig_width, 13), 
                                   height_ratios=[3, 1],
                                   sharex=True)
    highlight_vdm_run_positions(ax1, run_numbers, vdm_run_numbers)
    highlight_vdm_run_positions(ax2, run_numbers, vdm_run_numbers)
    
    fig.suptitle(f'Average Luminosity and Ratios per Run - {plot_title}', fontsize=20, fontweight='bold')
    
    # ========================
    # TOP PLOT: Absolute luminosity per run
    # ========================
    
    for lumi_col, orig_col, color, marker in active_triggers:
        ax1.plot(x_positions, run_averages[lumi_col], 
                marker,                 
                markersize=8,
                alpha=0.7,              
                color=color,            
                label=format_legend_label(orig_col),
                markeredgewidth=0.5,
                linestyle='-',
                linewidth=1.5)
    
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('Average Luminosity (μb⁻¹s⁻¹)', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    
    # Set x-axis ticks to show run numbers at discrete positions
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(run_numbers)
    
    # ========================
    # BOTTOM PLOT: Luminosity ratios per run
    # ========================
    
    ratio_definitions = build_matched_luminosity_ratios(active_triggers)
    ratio_series = get_direct_luminosity_ratio_series(run_averages, ratio_definitions)
    
    if not ratio_definitions:
        print("Warning: No matched FDD/FT0 or ZDC/FT0 luminosity ratios to plot")
        ax2.set_visible(False)
        fig.subplots_adjust(hspace=0.15)
    else:
        print(f"\nPer-run luminosity ratio statistics (FDD or ZDC / FT0, matched by BC suffix):")
        
        for ratio_name, direct_ratio, color, marker, suffix in ratio_series:
            
            finite_ratio = direct_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if finite_ratio.empty:
                print(f"  {suffix}: no finite values, skipping statistics")
            else:
                mean_ratio = finite_ratio.mean()
                median_ratio = finite_ratio.median()
                std_ratio = finite_ratio.std()
                relative_std = (f"{100 * std_ratio / mean_ratio:.1f}%"
                                if np.isfinite(mean_ratio) and mean_ratio != 0
                                else "undefined (zero or unavailable mean)")
                print(f"  {suffix}: median = {median_ratio:.3f}, mean = {mean_ratio:.3f}, "
                      f"std = {std_ratio:.3f} ({relative_std})")
            
            ax2.plot(x_positions, direct_ratio,
                    marker,
                    markersize=7,
                    alpha=0.7,
                    color=color,
                    label=format_legend_label(ratio_name),
                    markeredgewidth=0.5,
                    linestyle='-',
                    linewidth=1.5)
        
        ax2.set_xlabel('Run Number', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Luminosity Ratio', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        
        # Set x-axis ticks to show run numbers at discrete positions
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(run_numbers)
        
        # Set y-limits based on direct ratios
        all_ratios = []
        for ratio_name, direct_ratio, color, marker, suffix in ratio_series:
            valid_ratios = direct_ratio.replace([np.inf, -np.inf], np.nan).dropna().tolist()
            all_ratios.extend(valid_ratios)
        
        if all_ratios:
            data_min = min(all_ratios)
            data_max = max(all_ratios)
            margin = (data_max - data_min) * 0.05
            ax2.set_ylim(data_min - margin, data_max + margin)
        else:
            ax2.set_ylim(0, 2)
    
    # Rotate x-axis labels for better readability
    style_run_tick_labels(ax2, run_numbers, run_tick_fontsize, run_tick_rotation)
    
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    output_file= os.path.join(output_dir, f'luminosity_plot_per_run.png')
    # ticks = np.linspace(0, len(run_numbers)-1, min(len(run_numbers), 50), dtype=int)
    # plt.xticks(ticks, [run_numbers[i] for i in ticks], rotation=45, ha='right', fontsize=2)

    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Per-run luminosity plot saved as: {output_file}")

    plot_luminosity_ratio_per_run_with_FT0_B(
        x_positions,
        run_numbers,
        run_averages,
        ratio_series,
        plot_title,
        output_dir,
        vdm_run_numbers
    )


def plot_scatter_rate_vs_lumi_ratio(df, triggers, plot_title, output_dir, dataset_name,
                                    vdm_run_numbers):
    """
    Create a scatter plot with multiple luminosity ratios on y-axis
    and FT0_B rate on x-axis, one point per run for each ratio.
    Ratios are built by matching FDD/FT0 and ZDC/FT0 triggers with the same suffix.
    
    Args:
        df (DataFrame): The data
        triggers (list): List of (column_name, color, marker) tuples
        plot_title (str): Title for the plot
        output_dir (str): Directory to save the plot
        dataset_name (str): Dataset name for cross section lookup
    """
    
    if 'FT0_B' not in df.columns:
        print("Error: FT0_B not found in data, skipping scatter plot")
        return

    # Build luminosity columns for triggers with valid cross sections
    active_triggers = []
    for col, color, marker in triggers:
        xs = get_cross_section(dataset_name, col)
        if xs is not None and xs > 0:
            lumi_col = f"{col}_Lumi"
            active_triggers.append((lumi_col, col, color, marker))

    ratio_definitions = build_matched_luminosity_ratios(active_triggers)
    if not ratio_definitions:
        print("Error: No matched FDD/FT0 or ZDC/FT0 luminosity ratios found for scatter plot, skipping")
        return

    # Calculate luminosities in the event-level dataframe
    df = df.copy()
    for lumi_col, orig_col, _, _ in active_triggers:
        xs = get_cross_section(dataset_name, orig_col)
        df[lumi_col] = df[orig_col] / xs

    # Per-run averages for luminosities and FT0_B rate
    lumi_cols = [lumi_col for lumi_col, _, _, _ in active_triggers]
    aggregation_columns = lumi_cols + ['FT0_B']
    run_data = df.groupby('runNumber')[aggregation_columns].mean().reset_index()
    run_data['rate_kHz'] = run_data['FT0_B'] / 1000

    ratio_series = get_direct_luminosity_ratio_series(run_data, ratio_definitions)
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    fig.suptitle(f'FT0_B Rate vs Luminosity Ratios - {plot_title}', 
                 fontsize=20, fontweight='bold')

    vdm_legend_added = False
    for ratio_name, direct_ratio, color, marker, suffix in ratio_series:
        valid_ratios = direct_ratio.replace([np.inf, -np.inf], np.nan)
        valid_mask = np.isfinite(valid_ratios) & np.isfinite(run_data['rate_kHz'])

        ax.scatter(run_data.loc[valid_mask, 'rate_kHz'],
                   valid_ratios[valid_mask],
                   s=90,
                   alpha=0.75,
                   color=color,
                   edgecolors='black',
                   linewidths=0.7,
                   marker=marker,
                   label=get_suffix_label(suffix))

        vdm_mask = valid_mask & run_data['runNumber'].astype(int).isin(vdm_run_numbers)
        if vdm_mask.any():
            ax.scatter(run_data.loc[vdm_mask, 'rate_kHz'],
                       valid_ratios[vdm_mask],
                       s=190,
                       alpha=0.9,
                       color='gold',
                       edgecolors='black',
                       linewidths=1.0,
                       marker='*',
                       label='VdM run' if not vdm_legend_added else '_nolegend_')
            vdm_legend_added = True

    ax.set_xlabel('FT0_B Rate (kHz)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Luminosity Ratio (FDD or ZDC / FT0)', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(loc='best', fontsize=12)
    
    print(f"\nScatter plot statistics:")
    print(f"  Number of runs: {len(run_data)}")
    print(f"  FT0_B rate range: {run_data['rate_kHz'].min():.2f} - {run_data['rate_kHz'].max():.2f} kHz")
    for ratio_name, direct_ratio, _, _, suffix in ratio_series:
        valid_ratios = direct_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_ratios) > 0:
            print(f"  {suffix}: range = {valid_ratios.min():.3f} - {valid_ratios.max():.3f}, "
                  f"mean = {valid_ratios.mean():.3f}")
    
    plt.tight_layout(pad=2.0)
    
    output_file = os.path.join(output_dir, f'scatter_rate_vs_lumi_ratio.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Scatter plot saved as: {output_file}")
    
    # Save data to text file (one row per run, one column per ratio)
    txt_file = os.path.join(output_dir, f'scatter_data.txt')

    ratio_column_names = []
    for _, _, _, _, suffix in ratio_series:
        ratio_column_names.append(f"Ratio_{suffix}")

    ratio_frame = pd.DataFrame({'Run Number': run_data['runNumber'],
                                'FT0_B Rate (kHz)': run_data['rate_kHz']})
    for i, (_, direct_ratio, _, _, suffix) in enumerate(ratio_series):
        ratio_frame[ratio_column_names[i]] = direct_ratio.replace([np.inf, -np.inf], np.nan)

    with open(txt_file, 'w') as f:
        f.write(f"# Scatter plot data\n")
        f.write(f"# Plot title: {plot_title}\n")
        f.write(f"# X-axis: FT0_B Rate (kHz)\n")
        f.write(f"# Y-series: luminosity ratios (FDD or ZDC / FT0), matched by suffix\n")
        f.write("#\n")
        header_cols = ['Run Number', 'FT0_B Rate (kHz)'] + ratio_column_names
        f.write(' | '.join(header_cols) + "\n")
        f.write('-' * (len(' | '.join(header_cols))) + "\n")

        for _, row in ratio_frame.iterrows():
            values = [f"{int(row['Run Number'])}", f"{row['FT0_B Rate (kHz)']:.6f}"]
            for rc in ratio_column_names:
                v = row[rc]
                values.append("nan" if pd.isna(v) else f"{v:.6f}")
            f.write(' | '.join(values) + "\n")
    
    print(f"Scatter data saved as: {txt_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot stability / luminosity from Output.csv files.')
    parser.add_argument('--t', required=True,
                        help='Comma-separated list of BC suffixes or full trigger columns, e.g. "SL_FDD100,SL_FDD5" or "FT0_SL_FDD100,FDD_SL_FDD100"')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name for cross sections and plot titles, e.g. "LHC24apq (pp Ref)"')
    parser.add_argument('--outdir', required=True,
                        help='Output directory name for plots, e.g. "SL_Scan"')
    parser.add_argument('--input', required=True,
                        help='Comma-separated list of Output_* directories to read, e.g. "Output_651230" or "Output_651230,Output_535345"')
    args = parser.parse_args()

    requested_triggers = [t.strip() for t in args.t.split(',') if t.strip()]
    dataset_name = args.dataset
    output_dir = args.outdir
    input_dirs = [d.strip() for d in args.input.split(',')]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Requested: {requested_triggers}")
    print(f"Dataset  : {dataset_name}")
    print(f"Output   : {output_dir}")
    print(f"Input    : {input_dirs}")
    print()

    # Load data from specified directories
    df = load_data(input_dirs)
    if df is None:
        return

    trigger_names, missing = resolve_trigger_columns(requested_triggers, df.columns)

    # Validate requested columns exist
    if missing:
        print(f"\nError: requested trigger suffixes/columns not found: {missing}")
        print(f"Available trigger columns:")
        meta_cols = {'timestamp', 'time in fill', 'fill_number', 'runNumber'}
        for c in sorted(df.columns):
            if c not in meta_cols:
                print(f"  {c}")
        return

    print(f"Resolved : {trigger_names}")

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    listed_vdm_runs = load_vdm_run_numbers()
    vdm_run_numbers, vdm_intervals = get_vdm_run_intervals(df, listed_vdm_runs)
    missing_vdm_runs = sorted(listed_vdm_runs - vdm_run_numbers)
    print(f"VdM runs present in data ({len(vdm_run_numbers)}): {sorted(vdm_run_numbers)}")
    print(f"VdM runs absent from data ({len(missing_vdm_runs)}): {missing_vdm_runs}")

    # Build trigger list with colors/markers
    triggers = []
    for i, col in enumerate(trigger_names):
        color, marker = TRIGGER_STYLES[i % len(TRIGGER_STYLES)]
        triggers.append((col, color, marker))

    plot_title = dataset_name

    # ==========================================
    # RATE PLOTS (interaction rates in kHz)
    # ==========================================
    
    print("\n" + "-"*60)
    print(f"Generating time-based RATE plot...")
    print("-"*60)
    plot_stability(df, triggers, plot_title, output_dir, vdm_intervals)
    
    print("\n" + "-"*60)
    print(f"Generating per-run RATE plot...")
    print("-"*60)
    plot_stability_per_run(df, triggers, plot_title, output_dir, vdm_run_numbers)
    
    # ==========================================
    # LUMINOSITY PLOTS (rate / cross_section)
    # ==========================================
    
    print("\n" + "-"*60)
    print(f"Generating time-based LUMINOSITY plot...")
    print("-"*60)
    plot_luminosity(df, triggers, plot_title, output_dir, dataset_name, vdm_intervals)
    
    print("\n" + "-"*60)
    print(f"Generating per-run LUMINOSITY plot...")
    print("-"*60)
    plot_luminosity_per_run(df, triggers, plot_title, output_dir, dataset_name,
                            vdm_run_numbers)
    
    # ==========================================
    # SCATTER PLOT (rate vs luminosity ratio)
    # ==========================================
    
    print("\n" + "-"*60)
    print(f"Generating SCATTER plot (rate vs lumi ratio)...")
    print("-"*60)
    plot_scatter_rate_vs_lumi_ratio(df, triggers, plot_title, output_dir, dataset_name,
                                    vdm_run_numbers)
    
    print("\n" + "="*80)
    print("All plots generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
