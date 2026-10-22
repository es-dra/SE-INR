import collections, json, numpy as np
R = {}
for line in open('results.txt'):
    a, t, *v = line.split()
    if v: R[(a, t)] = float(v[0])
dsets = ['set5', 'set14', 'bsd100', 'urban100']; scales = [2, 3, 4, 6]
def mean_seed(arm, t):
    vals = [R[(f'{arm}_s{s}', t)] for s in (1, 2) if (f'{arm}_s{s}', t) in R]
    return np.mean(vals) if vals else None
out = {'per_dataset': {}, 'avg_delta': {}, 'edsr_abs': {}}
for arm in ['tinydec', 'nocell', 'rdn', 'swinir']:
    for s in scales:
        ds = []
        for d in dsets:
            t = f'{d}-{s}'
            if arm == 'swinir': b, x = R.get(('edsr_s1', t)), R.get(('swinir_s1', t))
            else: b, x = mean_seed('edsr', t), mean_seed(arm, t)
            if b is None or x is None: continue
            out['per_dataset'][f'{arm}/{t}'] = round(x - b, 3); ds.append(x - b)
        out['avg_delta'][f'{arm}/x{s}'] = round(float(np.mean(ds)), 3)
for d in dsets:
    for s in scales: out['edsr_abs'][f'{d}-{s}'] = round(float(mean_seed('edsr', f'{d}-{s}')), 2)
json.dump(out, open('benchmark_table.json', 'w'), indent=1)
for k, v in out['avg_delta'].items(): print(k, v)
print(out['edsr_abs'])
for arm in ['tinydec', 'nocell', 'rdn', 'swinir']:
    print(arm, ' '.join(f"{d}:{out['per_dataset'].get(f'{arm}/{d}-4')}" for d in dsets), '(x4)')
