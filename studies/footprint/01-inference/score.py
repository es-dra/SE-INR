#!/usr/bin/env python3
"""Score a completed footprint run: five frozen gates plus descriptive tables. Reads only records.jsonl.gz."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GATES = 5
B = 10000
RNG_SEED = 20261101


def fnum(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def load(run_root):
    recs = []
    for shard in sorted((run_root / 'outputs').glob('*-seed*')):
        assert (shard / 'COMPLETE.json').exists(), f'{shard.name} incomplete'
        with gzip.open(shard / 'records.jsonl.gz', 'rt') as stream:
            for line in stream:
                recs.append(json.loads(line))
    return recs


def key_of(r):
    k = r['kind']
    if k == 'sym':
        return f"sym:{fnum(r['scale'])}"
    if k == 'asym':
        return f"asym:{fnum(r['sy'])}x{fnum(r['sx'])}:{r['policy']}"
    if k == 'cons':
        return f"cons:{fnum(r['hi'])}-{fnum(r['lo'])}"
    if k == 'down':
        return 'down'
    raise ValueError(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', required=True)
    args = ap.parse_args()
    root = Path(args.run_root).resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    cond = protocol['conditions']
    recs = load(root)
    out_dir = root / 'score'
    out_dir.mkdir(exist_ok=True)

    # ---------- index scalar metrics: table[model][seed][dataset][image][condition] -> value ----------
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    spectra = defaultdict(lambda: defaultdict(lambda: {'V': None, 'E': None, 'n': 0}))  # (model, cond, dataset) pooled
    grids = defaultdict(lambda: {'V': None, 'E': None, 'n': 0})                            # (model, pair, policy, dataset)
    for r in recs:
        k = key_of(r)
        slot = table[r['model']][str(r['seed'])][r['dataset']][r['image']]
        if r['kind'] in ('sym', 'asym'):
            slot[k] = r['psnr_y']
            acc = spectra[(r['model'], k)][r['dataset']]
            for name in ('V', 'E'):
                v = np.asarray(r['u'][name])
                acc[name] = v if acc[name] is None else acc[name] + v
            acc['n'] += 1
            if r['kind'] == 'asym':
                g = grids[(r['model'], f"{fnum(r['sy'])}x{fnum(r['sx'])}", r['policy'], r['dataset'])]
                for name in ('V', 'E'):
                    v = np.asarray(r['grid'][name])
                    g[name] = v if g[name] is None else g[name] + v
                g['n'] += 1
        elif r['kind'] == 'cons':
            slot[k] = r['psnr_hibox_vs_lo']
            for f in ('psnr_lo_vs_tbox', 'psnr_lo_vs_tbic', 'psnr_hibox_vs_tbox', 'psnr_hibox_vs_tbic', 'psnr_tbic_vs_tbox'):
                slot[f'{k}:{f}'] = r[f]
        elif r['kind'] == 'down':
            for f in ('psnr_vs_box', 'psnr_vs_bic', 'nearest_vs_box', 'nearest_vs_bic', 'bic_vs_box'):
                slot[f'down:{f}'] = r[f]

    models = sorted(table)
    seeds = sorted(set.intersection(*[set(table[m]) for m in models]))
    datasets = sorted(set.intersection(*[set(table[m][s]) for m in models for s in seeds]))
    images = {d: sorted(set.intersection(*[set(table[m][s][d]) for m in models for s in seeds])) for d in datasets}
    conditions = sorted(set.union(*[set(table[m][s][d][i]) for m in models for s in seeds for d in datasets for i in images[d]]))

    def arr(model, condition, dataset, seed=None):
        """Per-image vector, seeds averaged unless one seed is named."""
        use = [seed] if seed else seeds
        return np.array([[table[model][s][dataset][i][condition] for i in images[dataset]] for s in use]).mean(axis=0)

    # ---------- bootstrap machinery: one index draw per dataset shared by every statistic ----------
    rng = np.random.default_rng(RNG_SEED)
    draws = {d: rng.integers(0, len(images[d]), size=(B, len(images[d]))) for d in datasets}

    def balanced(per_dataset_vectors):
        """Point estimate and bootstrap distribution of the dataset-balanced mean of per-image vectors."""
        point = float(np.mean([v.mean() for v in per_dataset_vectors.values()]))
        boot = np.mean([per_dataset_vectors[d][draws[d]].mean(axis=1) for d in datasets], axis=0)
        return point, boot

    def interval(boot, two_sided=0.95):
        lo, hi = np.quantile(boot, [(1 - two_sided) / 2, 1 - (1 - two_sided) / 2])
        return [float(lo), float(hi)]

    def corrected_lower(boot):
        return float(np.quantile(boot, 0.05 / GATES))

    pairs_in = [f"{fnum(a)}x{fnum(b)}" for a, b in cond['asymmetric_pairs_sy_sx'] if max(a, b) <= 4]
    pairs_out = [f"{fnum(a)}x{fnum(b)}" for a, b in cond['asymmetric_pairs_sy_sx'] if max(a, b) > 4]
    cons_keys = [f"cons:{fnum(h)}-{fnum(l)}" for h, l in cond['consistency_pairs_hi_lo']]

    def contrast(model, pairs, a, b, seed=None):
        """Per-dataset per-image vector of mean over pairs of PSNR(policy a) - PSNR(policy b)."""
        return {d: np.mean([arr(model, f'asym:{p}:{a}', d, seed) - arr(model, f'asym:{p}:{b}', d, seed) for p in pairs], axis=0)
                for d in datasets}

    summary = {'version': protocol['version'], 'models': models, 'seeds': seeds, 'datasets': datasets,
               'images_per_dataset': {d: len(images[d]) for d in datasets}, 'records': len(recs),
               'bootstrap': {'resamples': B, 'rng_seed': RNG_SEED, 'gate_alpha_one_sided': 0.05 / GATES}}

    # ---------- gates ----------
    gates = {}
    sc = 'sc-inr'
    assert sc in models, models
    others = [m for m in models if m != sc]

    v = contrast(sc, pairs_in, 'native', 'swap')
    point, boot = balanced(v)
    per_seed = {s: balanced(contrast(sc, pairs_in, 'native', 'swap', s))[0] for s in seeds}
    gates['G1_scinr_axis_identified'] = {
        'statistic': 'SC-INR native - swap PSNR-Y, eight in-support pairs', 'point_db': point,
        'corrected_lower_bound': corrected_lower(boot), 'ci95': interval(boot), 'per_seed_balanced': per_seed,
        'pass': corrected_lower(boot) > 0 and all(x > 0 for x in per_seed.values())}

    for m in others:
        vm = contrast(m, pairs_in, 'native', 'swap')
        diff = {d: v[d] - vm[d] for d in datasets}
        point, boot = balanced(diff)
        gates[f'G2_scinr_exceeds_{m}'] = {
            'statistic': f'(native - swap)[SC-INR] - (native - swap)[{m}]', 'point_db': point,
            'corrected_lower_bound': corrected_lower(boot), 'ci95': interval(boot), 'pass': corrected_lower(boot) > 0}

    v3 = contrast(sc, pairs_in, 'native', 'iso')
    point, boot = balanced(v3)
    gates['G3_scinr_native_beats_iso'] = {
        'statistic': 'SC-INR native - iso PSNR-Y, eight in-support pairs', 'point_db': point,
        'corrected_lower_bound': corrected_lower(boot), 'ci95': interval(boot),
        'per_seed_balanced': {s: balanced(contrast(sc, pairs_in, 'native', 'iso', s))[0] for s in seeds},
        'pass': corrected_lower(boot) > 0}

    def cons_vec(model, seed=None):
        return {d: np.mean([arr(model, k, d, seed) for k in cons_keys], axis=0) for d in datasets}

    g4 = {'statistic': 'SC-INR consistency PSNR (box(hi) vs lo) minus each other model, two pairs', 'contrasts': {}}
    vs = cons_vec(sc)
    for m in others:
        vm = cons_vec(m)
        point, boot = balanced({d: vs[d] - vm[d] for d in datasets})
        g4['contrasts'][m] = {'point_db': point, 'corrected_lower_bound': corrected_lower(boot), 'ci95': interval(boot)}
    g4['pass'] = all(c['corrected_lower_bound'] > 0 for c in g4['contrasts'].values())
    gates['G4_consistency'] = g4
    summary['gates'] = gates
    summary['gates_passed'] = sum(int(g['pass']) for g in gates.values())

    # ---------- descriptive tables ----------
    desc = {}
    bal = {}
    for m in models:
        bal[m] = {}
        for c in conditions:
            if c.startswith(('sym:', 'asym:', 'down:')) or (c.startswith('cons:') and c.count(':') == 1):
                bal[m][c] = {'seed_mean': balanced({d: arr(m, c, d) for d in datasets})[0],
                             'per_seed': {s: balanced({d: arr(m, c, d, s) for d in datasets})[0] for s in seeds}}
    desc['balanced_psnr'] = bal

    pol = {}
    for m in models:
        pol[m] = {}
        for a, b in (('native', 'swap'), ('native', 'iso'), ('iso', 'swap')):
            for label, pairs in (('in_support', pairs_in), ('with_6', pairs_out)):
                if not pairs:
                    continue
                point, boot = balanced(contrast(m, pairs, a, b))
                pol[m][f'{a}-{b}:{label}'] = {
                    'point_db': point, 'ci95': interval(boot),
                    'per_seed': {s: balanced(contrast(m, pairs, a, b, s))[0] for s in seeds},
                    'per_pair': {p: balanced(contrast(m, [p], a, b))[0] for p in pairs}}
    desc['policy_contrasts'] = pol

    degr = {}
    for m in models:
        degr[m] = {}
        for sy, sx in cond['asymmetric_pairs_sy_sx']:
            p = f'{fnum(sy)}x{fnum(sx)}'
            ref = {d: 0.5 * (arr(m, f'sym:{fnum(sy)}', d) + arr(m, f'sym:{fnum(sx)}', d)) for d in datasets}
            point, boot = balanced({d: arr(m, f'asym:{p}:native', d) - ref[d] for d in datasets})
            degr[m][p] = {'native_minus_symmetric_reference_db': point, 'ci95': interval(boot)}
    desc['asymmetric_degradation'] = degr

    seed_spread = {}
    for m in models:
        seed_spread[m] = {}
        for c in ['sym:2', 'sym:4'] + [f'asym:{p}:native' for p in pairs_in]:
            per = [balanced({d: arr(m, c, d, s) for d in datasets})[0] for s in seeds]
            seed_spread[m][c] = {'per_seed': per, 'range_db': float(max(per) - min(per))}
    desc['seed_spread'] = seed_spread

    cons = {}
    for m in models:
        cons[m] = {}
        for k in cons_keys:
            cons[m][k] = {f: balanced({d: arr(m, f'{k}:{f}', d) for d in datasets})[0]
                          for f in ('psnr_lo_vs_tbox', 'psnr_lo_vs_tbic', 'psnr_hibox_vs_tbox', 'psnr_hibox_vs_tbic', 'psnr_tbic_vs_tbox')}
            cons[m][k]['psnr_hibox_vs_lo'] = balanced({d: arr(m, k, d) for d in datasets})[0]
    desc['consistency'] = cons

    down = {m: {f: balanced({d: arr(m, f'down:{f}', d) for d in datasets})[0]
                for f in ('psnr_vs_box', 'psnr_vs_bic', 'nearest_vs_box', 'nearest_vs_bic', 'bic_vs_box')} for m in models}
    desc['downscale'] = down

    def pooled_R(acc):
        V = sum(acc[d]['V'] for d in datasets)
        E = sum(acc[d]['E'] for d in datasets)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(E > 0, 1 - V / E, np.nan).tolist()
    prof = {}
    for m in models:
        prof[m] = {}
        for c in conditions:
            if (c.startswith('sym:') or c.startswith('asym:')) and (m, c) in spectra:
                prof[m][c] = pooled_R(spectra[(m, c)])
    desc['R_u_profiles'] = {'u_edges': [i * 0.25 for i in range(17)] + ['inf'], 'pooled_over_images_and_datasets': prof}

    gr = {}
    for m in models:
        gr[m] = {}
        for pair in pairs_in + pairs_out:
            gr[m][pair] = {}
            for policy in cond['cell_policies']:
                acc = {d: grids[(m, pair, policy, d)] for d in datasets}
                V = sum(np.asarray(acc[d]['V']) for d in datasets)
                E = sum(np.asarray(acc[d]['E']) for d in datasets)
                with np.errstate(divide='ignore', invalid='ignore'):
                    gr[m][pair][policy] = np.where(E > 0, 1 - V / E, np.nan).tolist()
    desc['R_grid_uy_ux'] = {'edges': [0, 0.5, 1, 1.5, 2, 'inf'], 'rows_uy_cols_ux': gr}
    summary['descriptive'] = desc

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    (out_dir / 'COMPLETE.json').write_text(json.dumps({'gates_passed': summary['gates_passed'], 'records': len(recs)}) + '\n')
    print(json.dumps({'gates': {k: {'point_db': v.get('point_db'), 'pass': v['pass']} for k, v in gates.items()},
                      'seeds': seeds, 'records': len(recs)}, indent=1, default=float))


if __name__ == '__main__':
    main()
