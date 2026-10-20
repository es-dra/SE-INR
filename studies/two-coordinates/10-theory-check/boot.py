"""Image-level bootstrap CIs for headline numbers (32 DIV2K 0869-0900 images; 2000 resamples)."""
import json, collections, numpy as np
rng = np.random.default_rng(20261020); B = 2000
def per_image(p, key):
    D = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.Counter()))
    for r in json.load(open(p)):
        k = key(r); D[r['img']][k].update({'tgt': r['tgt'], **{m: v for m, v in r['err'].items()}})
    return D
imgs = None; out = {}
def boot(stat, D):
    ids = sorted(D); arr = []
    for _ in range(B):
        sel = rng.choice(ids, len(ids)); arr.append(stat(D, sel))
    return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]
def agg(D, sel, k, field):
    return sum(D[i][k][field] for i in sel)
V2 = '/workspace/SE-INR/runs/claude-20261019-two-coordinates-v2/formal/rows.json'
D2 = per_image(V2, lambda r: (r['model'], r['s'], r['lo']))
ids = sorted(D2)
def spread(D, sel, s):
    best = 0
    for b in np.arange(0.5, min(s, 2.0), .25):
        v = [1 - agg(D, sel, (m, s, float(b)), '1.0')/agg(D, sel, (m, s, float(b)), 'tgt') for m in ['liif', 'lte', 'sc-inr']]
        best = max(best, max(v)-min(v))
    return best
for s in (2, 3, 4): out[f'decoder_max_spread_x{s}'] = dict(point=round(spread(D2, ids, s), 4), ci=boot(lambda D, sel: spread(D, sel, s), D2))
def cost(D, sel, s, m):
    n = sum(agg(D, sel, (mo, s, float(b)), '1.0') for mo in ['liif', 'lte', 'sc-inr'] for b in np.arange(0, s, .25))
    c = sum(agg(D, sel, (mo, s, float(b)), m) for mo in ['liif', 'lte', 'sc-inr'] for b in np.arange(0, s, .25))
    return 10*np.log10(c/n)
for s in (2, 3, 4, 6):
    for m in ['2.0', '0.5']: out[f'cell_m{m}_cost_db_x{s}'] = dict(point=round(cost(D2, ids, s, m), 4), ci=boot(lambda D, sel: cost(D, sel, s, m), D2))
V5 = '/workspace/SE-INR/claude-20261019-encoder-train-v5'
arms = {a: per_image(f'{V5}/eval_{a}/rows.json', lambda r: (r['s'], r['lo'])) for a in ['edsr', 'rdn', 'tinydec', 'nocell']}
arms['edsr50'] = per_image(f'{V5}/interim50_edsr/rows.json', lambda r: (r['s'], r['lo']))
arms['swinir50'] = per_image('/workspace/SE-INR/claude-20261020-swinir-v7/eval50_swinir/rows.json', lambda r: (r['s'], r['lo']))
def delta(sel, a, base, s):
    ea = sum(arms[a][i][(s, float(b))]['1.0'] for i in sel for b in np.arange(0, s, .25)); eb = sum(arms[base][i][(s, float(b))]['1.0'] for i in sel for b in np.arange(0, s, .25))
    return 10*np.log10(eb/ea)
for a, base in [('rdn', 'edsr'), ('tinydec', 'edsr'), ('nocell', 'edsr'), ('swinir50', 'edsr50')]:
    for s in (2, 3, 4, 6):
        out[f'{a}_vs_{base}_db_x{s}'] = dict(point=round(delta(ids, a, base, s), 4), ci=boot(lambda D, sel: delta(sel, a, base, s), D2))
json.dump(out, open('bootstrap_ci.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
