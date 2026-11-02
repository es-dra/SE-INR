#!/usr/bin/env python3
"""LaTeX/markdown table rows from footprint summaries and benchmark jsonl files.

  python3 tables.py --summary ../01-inference/score/summary.json [--bench bench-1000ep.jsonl ...] [--tex out.tex]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

LABEL = {'liif': 'LIIF', 'lte': 'LTE', 'sc-inr': 'SC-INR', 'sc-inr-exact': 'Footprint-exact', 'liif-asym': 'LIIF + asym.\\ aug.',
         'lte-asym': 'LTE + asym.\\ aug.', 'sc-inr-nosinc': 'SC-INR w/o sinc', 'lte-linear': 'LTE linear readout',
         'sc-inr-nosinc-test': 'SC-INR, sinc off at test'}
ORDER = ['liif', 'lte', 'liif-asym', 'lte-asym', 'lte-linear', 'sc-inr-nosinc', 'sc-inr-nosinc-test', 'sc-inr', 'sc-inr-exact']


def fmt(x, s=True):
    return ('%+.3f' if s else '%.2f') % x


def asym_table(summary):
    d = summary['descriptive']['policy_contrasts']
    models = [m for m in ORDER if m in d] + [m for m in d if m not in ORDER]
    lines = ['% model & native-swap & native-iso & seed range (asym) & seed range (sym)']
    for m in models:
        ns, ni = d[m]['native-swap:in_support'], d[m]['native-iso:in_support']
        ss = summary['descriptive']['seed_spread'][m]
        asym_r = max(v['range_db'] for k, v in ss.items() if k.startswith('asym:'))
        sym_r = max(v['range_db'] for k, v in ss.items() if k.startswith('sym:'))
        lines.append(f"{LABEL.get(m, m)} & {fmt(ns['point_db'])} [{ns['ci95'][0]:+.3f}, {ns['ci95'][1]:+.3f}] & "
                     f"{fmt(ni['point_db'])} [{ni['ci95'][0]:+.3f}, {ni['ci95'][1]:+.3f}] & {asym_r:.2f} & {sym_r:.3f}\\\\")
    return '\n'.join(lines)


def beyond_table(summary):
    d = summary['descriptive']
    models = [m for m in ORDER if m in d['downscale']] + [m for m in d['downscale'] if m not in ORDER]
    any_m = models[0]
    lines = ['% model & consistency x4->x2 & consistency x3->x1.5 & downscale vs box & downscale vs bicubic-aa',
             f"nearest / antialiased bicubic & -- & -- & {d['downscale'][any_m]['nearest_vs_box']:.2f} / {d['downscale'][any_m]['bic_vs_box']:.2f} & "
             f"{d['downscale'][any_m]['nearest_vs_bic']:.2f} / --\\\\"]
    for m in models:
        c = d['consistency'][m]
        lines.append(f"{LABEL.get(m, m)} & {c['cons:4-2']['psnr_hibox_vs_lo']:.2f} & {c['cons:3-1.5']['psnr_hibox_vs_lo']:.2f} & "
                     f"{d['downscale'][m]['psnr_vs_box']:.2f} & {d['downscale'][m]['psnr_vs_bic']:.2f}\\\\")
    return '\n'.join(lines)


def standard_table(bench_files, datasets=('set5', 'set14', 'bsd100', 'urban100'), scales=(2, 3, 4, 6, 8, 12)):
    acc = collections.defaultdict(list)
    for f in bench_files:
        for line in open(f):
            r = json.loads(line)
            base = r['name'].rsplit('-s', 1)[0] if '-s' in r['name'] else r['name']
            acc[(base, r['dataset'], r['scale'])].append(r['psnr'])
    names = sorted({k[0] for k in acc})
    head = '% name & ' + ' & '.join(f'{d}x{s}' for d in datasets for s in scales)
    lines = [head]
    for n in names:
        cells = []
        for d in datasets:
            for s in scales:
                v = acc.get((n, d, s))
                cells.append(f'{np.mean(v):.2f}' if v else '--')
        seeds = max(len(acc.get((n, d, s), [])) for d in datasets for s in scales)
        lines.append(f'{n} ({seeds} seeds) & ' + ' & '.join(cells) + '\\\\')
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary')
    ap.add_argument('--bench', nargs='*', default=[])
    ap.add_argument('--tex')
    a = ap.parse_args()
    out = []
    if a.summary:
        s = json.loads(Path(a.summary).read_text())
        out += ['%% ---- asymmetric cell policies ----', asym_table(s), '', '%% ---- beyond coverage ----', beyond_table(s), '']
    if a.bench:
        out += ['%% ---- standard benchmarks ----', standard_table(a.bench)]
    text = '\n'.join(out)
    print(text)
    if a.tex:
        Path(a.tex).write_text(text + '\n')
