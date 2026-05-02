"""
plot_continuous_scale.py — Continuous scale PSNR curve plotting and analysis.

Uses existing continuous-scale eval data to generate publication-quality plots.
Supports reading from JSON and annotating training boundaries and x1 anomalies.

Usage:
  python plot_continuous_scale.py --input continuous_psnr.json --output figs/
  python plot_continuous_scale.py --input continuous_psnr.json --annotate
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def smooth(y, window=3):
    """Simple moving average smoothing"""
    return uniform_filter1d(y, size=window)


def plot_continuous_psnr(data, output_dir, annotate=True, smooth_window=1,
                         exclude_x1=True):
    """
    data: dict {model_name: {scale_str: psnr_value}}
          scale_str format: "x1.0", "x2.0", ...
    """
    os.makedirs(output_dir, exist_ok=True)

    STYLE = {
        'LIIF':    {'color': '#4C72B0', 'ls': '--', 'lw': 1.5, 'marker': None},
        'LTE':     {'color': '#DD8452', 'ls': '-',  'lw': 2.0, 'marker': None},
        'LTE-NoC': {'color': '#55A868', 'ls': '-',  'lw': 2.0, 'marker': None},
        'SC-INR':  {'color': '#C44E52', 'ls': '-',  'lw': 2.5, 'marker': None},
    }
    DEFAULT_STYLE = {'color': '#8172B2', 'ls': '-.', 'lw': 1.5, 'marker': None}

    # Collect all scales
    all_scales = set()
    for model_data in data.values():
        for k in model_data:
            s = float(k.lstrip('x'))
            if exclude_x1 and s < 1.1:
                continue
            all_scales.add(s)
    scales = sorted(all_scales)

    # ── Figure 1: Full range curve ──
    fig, ax = plt.subplots(figsize=(12, 5))

    for mname, model_data in data.items():
        style = STYLE.get(mname, DEFAULT_STYLE)
        ys, xs_plot = [], []
        for s in scales:
            key = f'x{s}'
            if key in model_data and model_data[key] is not None:
                xs_plot.append(s)
                ys.append(model_data[key])
        if not xs_plot:
            continue
        ys_plot = smooth(ys, smooth_window) if smooth_window > 1 else ys
        ax.plot(xs_plot, ys_plot, label=mname,
                color=style['color'], linestyle=style['ls'],
                linewidth=style['lw'])

    if annotate:
        ax.axvline(x=4.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
        ax.text(4.1, ax.get_ylim()[0] + 0.5, 'Training\nboundary (x4)',
                fontsize=8, color='gray', va='bottom')
        ax.axvspan(scales[0], 4.0, alpha=0.05, color='blue')
        ax.axvspan(4.0, scales[-1], alpha=0.05, color='red')

    ax.set_xlabel('Super-Resolution Scale', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Continuous Scale PSNR — BSD100', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(scales[0], scales[-1])

    plt.tight_layout()
    path1 = os.path.join(output_dir, 'continuous_psnr_full.pdf')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.savefig(path1.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path1}")

    # ── Figure 2: OOD region zoom (x4-x30) + smoothing ──
    fig, ax = plt.subplots(figsize=(10, 4))

    ood_scales = [s for s in scales if s >= 4.0]

    for mname, model_data in data.items():
        style = STYLE.get(mname, DEFAULT_STYLE)
        xs_plot, ys = [], []
        for s in ood_scales:
            key = f'x{s}'
            if key in model_data and model_data[key] is not None:
                xs_plot.append(s)
                ys.append(model_data[key])
        if not xs_plot:
            continue
        ys_plot = smooth(ys, max(smooth_window, 3))
        ax.plot(xs_plot, ys_plot, label=mname,
                color=style['color'], linestyle=style['ls'],
                linewidth=style['lw'])

    if annotate:
        ax.axvline(x=4.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
        ax.text(4.2, ax.get_ylim()[0], 'Training boundary',
                fontsize=8, color='gray')

    ax.set_xlabel('Super-Resolution Scale', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('OOD Generalization: Continuous Scale PSNR — BSD100', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path2 = os.path.join(output_dir, 'continuous_psnr_ood.pdf')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.savefig(path2.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")

    # ── Figure 3: Delta curve (LTE-NoC − LTE) ──
    if 'LTE' in data and 'LTE-NoC' in data:
        fig, ax = plt.subplots(figsize=(10, 4))
        ref = data['LTE']

        for mname in ['LTE-NoC']:
            if mname not in data:
                continue
            model_data = data[mname]
            style = STYLE.get(mname, DEFAULT_STYLE)
            xs_plot, ys = [], []
            for s in ood_scales:
                key = f'x{s}'
                if (key in model_data and key in ref
                        and model_data[key] is not None
                        and ref[key] is not None):
                    xs_plot.append(s)
                    ys.append(model_data[key] - ref[key])
            if not xs_plot:
                continue
            ys_plot = smooth(ys, max(smooth_window, 3))
            ax.plot(xs_plot, ys_plot, label=f'{mname} - LTE',
                    color=style['color'], linestyle=style['ls'],
                    linewidth=style['lw'])

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.fill_between(ood_scales, 0, 0, alpha=0)
        ax.set_xlabel('Scale', fontsize=12)
        ax.set_ylabel('Delta PSNR vs LTE (dB)', fontsize=12)
        ax.set_title('PSNR Difference: LTE-NoC minus LTE — OOD region', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path3 = os.path.join(output_dir, 'delta_psnr_noc_vs_lte.pdf')
        plt.savefig(path3, dpi=150, bbox_inches='tight')
        plt.savefig(path3.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path3}")

    # ── Statistical analysis ──
    print("\n" + "="*60)
    print("Statistical Analysis: OOD range (x4.5-x30) mean PSNR")
    print("="*60)
    ood_range = [s for s in scales if s > 4.0]
    for mname, model_data in data.items():
        vals = [model_data.get(f'x{s}') for s in ood_range]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"  {mname:<18}: mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}")

    if 'LTE' in data and 'LTE-NoC' in data:
        lte = data['LTE']
        noc = data['LTE-NoC']
        wins = sum(1 for s in ood_range
                   if f'x{s}' in lte and f'x{s}' in noc
                   and noc[f'x{s}'] is not None and lte[f'x{s}'] is not None
                   and noc[f'x{s}'] > lte[f'x{s}'])
        total = sum(1 for s in ood_range
                    if f'x{s}' in lte and f'x{s}' in noc)
        print(f"\n  LTE-NoC > LTE in OOD: {wins}/{total} "
              f"({100*wins/total:.1f}%)")

    # x1 anomaly report
    for mname, model_data in data.items():
        for k in ['x1.0', '1.0']:
            if k in model_data:
                print(f"  {mname} x1.0: {model_data[k]:.2f} dB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Continuous scale PSNR JSON file')
    parser.add_argument('--output', type=str, default='figs',
                        help='Output figure directory')
    parser.add_argument('--smooth', type=int, default=3,
                        help='Smoothing window size (1=no smoothing)')
    parser.add_argument('--annotate', action='store_true', default=True,
                        help='Annotate training boundary')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    print(f"Models found: {list(data.keys())}")
    plot_continuous_psnr(data, args.output,
                         annotate=args.annotate,
                         smooth_window=args.smooth)


if __name__ == '__main__':
    main()
