import json, glob, collections, numpy as np
R = {}
for f in glob.glob('out/*.json'):
    d = json.load(open(f)); R[d['model']] = {k: 1 - v[0]/v[1] for k, v in d['acc'].items()}
dec = {'liif': [m for m in R if m.startswith('liif_s')], 'lte': [m for m in R if m.startswith('lte_s')], 'sc-inr': [m for m in R if m.startswith('sc-inr_s')], 'gaussiansr': ['gaussiansr']}
mean = {d: {k: np.mean([R[m][k] for m in ms]) for k in R[ms[0]]} for d, ms in dec.items()}
bins = [round(b, 2) for b in np.arange(0.5, 2.0, .25)]
f1 = []; f2 = []; f3 = []
for ds in ['Set14', 'BSD100', 'Urban100']:
    for s in (2, 3, 4):
        for b in bins:
            if b >= s: continue
            k = f'{ds}|{s}|{b}'; v = [mean[d][k] for d in dec]; f1.append(max(v)-min(v))
        k = f'{ds}|{s}|1.0'; f3.append(np.mean([mean[d][k] for d in ['liif', 'lte', 'sc-inr']]) - R['bicubic'][k])
    for d in dec:
        for b in bins:
            v = [mean[d][f'{ds}|{s}|{b}'] for s in (2, 3, 4) if b < s]
            if len(v) > 1: f2.append(max(v)-min(v))
res = dict(F1=dict(ok=f'{sum(x <= .03 for x in f1)}/{len(f1)}', max=float(max(f1)), PASS=sum(x <= .03 for x in f1)/len(f1) >= .9),
           F2=dict(ok=f'{sum(x <= .05 for x in f2)}/{len(f2)}', max=float(max(f2)), PASS=sum(x <= .05 for x in f2)/len(f2) >= .8),
           F3=dict(gaps=[round(float(x), 3) for x in f3], PASS=all(x >= .2 for x in f3)),
           R_mean={d: {k: round(float(v), 4) for k, v in mm.items()} for d, mm in mean.items()}, R_bicubic={k: round(v, 4) for k, v in R['bicubic'].items()})
json.dump(res, open('score_v9.json', 'w'), indent=1, default=str); print(json.dumps({k: res[k] for k in ['F1', 'F2', 'F3']}, indent=1, default=str))
