#!/usr/bin/env python3
"""Paper figures from a footprint-stage summary.json (stage 01 or the stage-02 evaluation).

  python3 figures.py --summary ../01-inference/score/summary.json --out ../../../paper/icme2027/figures --pair 2x4

Produces fig_grid.pdf (R over (u_y, u_x) per model and cell policy), fig_policy.pdf (cell-policy contrasts with
intervals and seed ranges) and fig_beyond.pdf (downscaling and cross-scale consistency).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

LABEL = {'liif': 'LIIF', 'lte': 'LTE', 'sc-inr': 'SC-INR', 'sc-inr-exact': 'Footprint-exact', 'liif-asym': 'LIIF + asym. aug.',
         'lte-asym': 'LTE + asym. aug.', 'sc-inr-nosinc': 'SC-INR w/o sinc', 'lte-linear': 'LTE linear readout',
         'sc-inr-nosinc-test': 'SC-INR, sinc off at test'}
ORDER = ['liif', 'lte', 'liif-asym', 'lte-asym', 'lte-linear', 'sc-inr-nosinc', 'sc-inr-nosinc-test', 'sc-inr', 'sc-inr-exact']
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948', '#52514e']
INK, QUIET, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
BLUES = LinearSegmentedColormap.from_list('seqblue', ['#f3f7fc', '#c6dbf2', '#7fb0e3', '#2a78d6', '#153c6b'])

plt.rcParams.update({'font.size': 8, 'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'xtick.color': QUIET, 'ytick.color': QUIET,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def models_in(summary):
    return [m for m in ORDER if m in summary['models']] + [m for m in summary['models'] if m not in ORDER]


def fig_grid(summary, pair, out):
    d = summary['descriptive']['R_grid_uy_ux']
    edges = ['0', '0.5', '1', '1.5', '2', '∞']
    models = [m for m in models_in(summary) if pair in d['rows_uy_cols_ux'][m]]
    policies = ['native', 'swap', 'iso']
    fig, axes = plt.subplots(len(models), 3, figsize=(7.0, 1.95 * len(models) + 0.5), squeeze=False)
    for i, m in enumerate(models):
        for j, pol in enumerate(policies):
            ax = axes[i, j]
            R = np.array(d['rows_uy_cols_ux'][m][pair][pol], dtype=float)
            im = ax.imshow(R, cmap=BLUES, vmin=0, vmax=1, origin='upper')
            for r in range(R.shape[0]):
                for c in range(R.shape[1]):
                    v = R[r, c]
                    ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=6.5, color='white' if v > 0.55 else INK)
            ax.set_xticks(np.arange(-0.5, 5), edges)
            ax.set_yticks(np.arange(-0.5, 5), edges)
            ax.tick_params(length=0)
            if i == 0:
                ax.set_title({'native': 'true per-axis cell', 'swap': 'axes swapped', 'iso': 'symmetric substitute'}[pol], fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(f'{LABEL.get(m, m)}\n$u_y$', color=INK)
            if i == len(models) - 1:
                ax.set_xlabel('$u_x$')
    sy, sx = pair.split('x')
    fig.suptitle(f'Recovery $R(u_y,u_x)$ at $(s_y,s_x)=({sy},{sx})$: only the analytic response uses the finely sampled axis', fontsize=8.5, color=INK, y=0.995)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cb.set_label('recovered share of band energy', color=INK)
    fig.savefig(Path(out) / 'fig_grid.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_policy(summary, out):
    d = summary['descriptive']
    models = [m for m in models_in(summary) if m in d['policy_contrasts']]
    keys = [('native-swap:in_support', 'true − swapped'), ('native-iso:in_support', 'true − symmetric substitute')]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.42 * len(models) + 1.6), gridspec_kw={'width_ratios': [1.3, 1]})
    ax = axes[0]
    y = np.arange(len(models))
    hgt = 0.36
    for k, (key, lab) in enumerate(keys):
        pts = [d['policy_contrasts'][m][key]['point_db'] for m in models]
        lo = [pts[i] - d['policy_contrasts'][m][key]['ci95'][0] for i, m in enumerate(models)]
        hi = [d['policy_contrasts'][m][key]['ci95'][1] - pts[i] for i, m in enumerate(models)]
        ax.barh(y + (k - 0.5) * hgt, pts, height=hgt * 0.9, color=SERIES[k], label=lab, xerr=[lo, hi], error_kw={'ecolor': INK, 'lw': 0.8, 'capsize': 2})
        for yi, p, l_, h_ in zip(y + (k - 0.5) * hgt, pts, lo, hi):
            xpos = p + h_ + 0.02 if p >= 0 else p - l_ - 0.02
            ax.text(xpos, yi, f'{p:+.3f}', va='center', ha='left' if p >= 0 else 'right', fontsize=6.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    xl = ax.get_xlim()
    ax.set_xlim(xl[0] - 0.12 * (xl[1] - xl[0]), xl[1] + 0.1 * (xl[1] - xl[0]))
    ax.set_yticks(y, [LABEL.get(m, m) for m in models])
    ax.invert_yaxis()
    ax.set_xlabel('PSNR-Y change (dB), eight in-range asymmetric pairs, 95% paired-image interval')
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax = axes[1]
    pairs = [k for k in d['seed_spread'][models[0]] if k.startswith('asym:')]
    for i, m in enumerate(models):
        sym = [d['seed_spread'][m][k]['range_db'] for k in d['seed_spread'][m] if k.startswith('sym:')]
        asym = [d['seed_spread'][m][k]['range_db'] for k in pairs]
        ax.scatter([i] * len(sym), sym, marker='_', s=120, color=QUIET, lw=1.2, label='symmetric scales' if i == 0 else None)
        ax.scatter(np.full(len(asym), i) + np.linspace(-0.18, 0.18, len(asym)), asym, s=14, color=SERIES[2], label='asymmetric pairs' if i == 0 else None, zorder=3)
    ax.set_yscale('log')
    ax.set_xticks(range(len(models)), [LABEL.get(m, m) for m in models], rotation=20, ha='right')
    ax.set_ylabel('seed range of PSNR-Y (dB, log)')
    ax.set_title('Spread across trained seeds', fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(Path(out) / 'fig_policy.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_beyond(summary, out):
    d = summary['descriptive']
    models = [m for m in models_in(summary) if m in d['downscale']]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ax = axes[0]
    names = ['nearest', 'bicubic (antialiased)'] + [LABEL.get(m, m) for m in models]
    vals = [d['downscale'][models[0]]['nearest_vs_box'], d['downscale'][models[0]]['bic_vs_box']] + [d['downscale'][m]['psnr_vs_box'] for m in models]
    cols = [QUIET, QUIET] + [SERIES[2] if m.startswith('sc-inr') else SERIES[0] for m in models]
    bars = ax.barh(range(len(names)), vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f'{v:.1f}', va='center', fontsize=7, color=INK)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel('PSNR against the 2×2 box average (dB), 0.5× downscaling')
    ax.set_title('Downscaling with models trained only for upscaling', fontsize=8, color=INK)
    ax.set_xlim(0, max(vals) + 6)
    ax.set_xlim(left=0)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax = axes[1]
    pairs = sorted(next(iter(d['consistency'].values())).keys())
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = [d['consistency'][m][p]['psnr_hibox_vs_lo'] for p in pairs]
        ax.bar(np.arange(len(pairs)) + (i - (len(models) - 1) / 2) * w, v, width=w * 0.9, color=SERIES[2] if m.startswith('sc-inr') else SERIES[0 if m == 'liif' else 1], label=LABEL.get(m, m))
        for x, y in zip(np.arange(len(pairs)) + (i - (len(models) - 1) / 2) * w, v):
            ax.text(x, y + 0.15, f'{y:.1f}', ha='center', fontsize=6, color=INK)
    ax.set_xticks(range(len(pairs)), [p.replace('cons:', '×').replace('-', ' vs ×') for p in pairs])
    ax.set_ylabel('consistency PSNR (dB)')
    ax.set_ylim(40, 46.5)
    ax.set_title('Box-averaged coarse output vs. direct coarse output', fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(Path(out) / 'fig_beyond.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pair', default='2x4')
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig_grid(s, a.pair, a.out)
    fig_policy(s, a.out)
    fig_beyond(s, a.out)
    print('wrote', sorted(p.name for p in Path(a.out).glob('fig_*.pdf')))
