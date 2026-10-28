import json, collections, numpy as np
R = '/workspace/SE-INR/runs/claude-20261029-sp-pilot-v1'
res = collections.defaultdict(dict)
for line in open(f'{R}/results.txt'):
    a, t, *v = line.split()
    if v: res[a][t] = float(v[0])
cells = sorted(res['base'])
bal = {a: float(np.mean([res[a][t] for t in cells])) for a in res}
d = lambda a, b: bal[a] - bal[b]
prof = {}
for a in res:
    p = json.load(open(f'{R}/probe_{a}.json'))['rows']; E = collections.Counter(); T = collections.Counter()
    for r in p: E[(r['s'], r['lo'])] += r['err']['1.0']; T[(r['s'], r['lo'])] += r['tgt']
    prof[a] = {k: 1 - E[k] / T[k] for k in T}
dR = {k: prof['full'][k] - prof['base'][k] for k in prof['base']}
pos = {k: max(v, 0) for k, v in dR.items()}
share = sum(v for k, v in pos.items() if k[1] >= 0.75) / max(sum(pos.values()), 1e-12)
out = dict(n_cells=len(cells), balanced_psnr=bal,
           P1_full_minus_base=d('full', 'base'), P1_pass=d('full', 'base') >= 0.05,
           P2_full_minus_wide=d('full', 'wide'), P2_pass=d('full', 'wide') >= 0.03,
           P3_film_minus_base=d('film', 'base'), P3_uloss_minus_base=d('uloss', 'base'), wide_minus_base=d('wide', 'base'),
           P4_positive_gain_share_u_ge_0p75=share,
           per_scale={a: {s: float(np.mean([res[a][t] for t in cells if t.endswith(f'-{s}')])) for s in (2, 3, 4, 6)} for a in res},
           dR_full_minus_base={f'x{k[0]}/{k[1]}': round(v, 5) for k, v in sorted(dR.items())})
json.dump(out, open(f'{R}/score.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ('dR_full_minus_base',)}, indent=1, default=str))
