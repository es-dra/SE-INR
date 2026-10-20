"""Two-seed mean PSNR deltas vs EDSR-LIIF with image-bootstrap 95% CI (per-image MSE pooled within seed, delta averaged over seeds)."""
import json, collections, numpy as np
rng = np.random.default_rng(20261020); B = 2000
V5 = '/workspace/SE-INR/claude-20261019-encoder-train-v5'; V8 = '/workspace/SE-INR/claude-20261020-seed2-v8'
def per_img(p):
    D = collections.defaultdict(collections.Counter)
    for r in json.load(open(p)): D[r['img']][r['s']] += r['err']['1.0']
    return D
runs = {(a, 1): per_img(f'{V5}/eval_{a}/rows.json') for a in ['edsr', 'rdn', 'tinydec', 'nocell']}
runs.update({(a, 2): per_img(f'{V8}/eval_{a}/rows.json') for a in ['edsr', 'rdn', 'tinydec', 'nocell']})
ids = sorted(runs[('edsr', 1)])
def d(sel, a, s):
    return np.mean([10*np.log10(sum(runs[('edsr', sd)][i][s] for i in sel)/sum(runs[(a, sd)][i][s] for i in sel)) for sd in (1, 2)])
out = {}
for a in ['tinydec', 'nocell', 'rdn']:
    for s in (2, 3, 4, 6):
        bs = [d(rng.choice(ids, len(ids)), a, s) for _ in range(B)]
        out[f'{a}_x{s}'] = dict(mean=round(d(ids, a, s), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                 seed1=round(10*np.log10(sum(runs[('edsr', 1)][i][s] for i in ids)/sum(runs[(a, 1)][i][s] for i in ids)), 3),
                                 seed2=round(10*np.log10(sum(runs[('edsr', 2)][i][s] for i in ids)/sum(runs[(a, 2)][i][s] for i in ids)), 3))
json.dump(out, open('table1_two_seed.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
