"""Score frozen gates of claude-20261019-cell-mismatch-v1. Read rows.json only."""
import json, sys, collections, numpy as np
from scipy.stats import spearmanr
rows = json.load(open(sys.argv[1])); MS = ['0.5','0.625','0.75','0.8','0.9091','1.0','1.1','1.25','1.3333','1.6','2.0']
PAIRS = [('1.1','0.9091'),('1.25','0.8'),('1.3333','0.75'),('1.6','0.625'),('2.0','0.5')]
ins = lambda s, m: 1 <= s/float(m) <= 4 + 1e-9
G = collections.defaultdict(list)
for r in rows: G[(r['model'], r['seed'], r['ds'], r['s'])].append(r)
out = {'H1': [], 'H2': [], 'H3': [], 'argmin': {}, 'cost_db': {}}
for g, rr in sorted(G.items()):
    model, seed, ds, s = g
    bins = collections.defaultdict(lambda: collections.Counter()); tgt = collections.Counter()
    for r in rr:
        for m, v in r['err'].items(): bins[r['lo']][m] += v
        tgt[r['lo']] += r['tgt']
    tot = {m: sum(bins[b][m] for b in bins) for m in MS}; C = {m: tot[m] - tot['1.0'] for m in MS}
    out['cost_db'][str(g)] = {m: 10*np.log10(tot[m]/tot['1.0']) for m in MS}
    insup = [m for m in MS if ins(s, m)]
    out['argmin'][str(g)] = min(insup, key=lambda m: tot[m])
    if s in (2, 3, 4):
        for m in insup:
            if m == '1.0' or abs(C[m]) <= 1e-7: continue
            d = {b: bins[b][m] - bins[b]['1.0'] for b in bins}
            share = sum(abs(v) for b, v in d.items() if b >= 1) / sum(abs(v) for v in d.values())
            out['H1'].append(dict(g=g, m=m, share=share, ok=share >= .8))
    if s in (2, 3):
        for big, small in PAIRS:
            if not (ins(s, big) and ins(s, small)): continue
            out['H2'].append(dict(g=g, pair=big, C_small=C[small], C_big=C[big], ok=C[small] > C[big], ratio=C[small]/max(C[big], 1e-9)))
            lr, R = [], []
            for b in sorted(bins):
                if tgt[b] <= 0: continue
                ds_ = bins[b][small]-bins[b]['1.0']; db_ = bins[b][big]-bins[b]['1.0']; eps = 1e-12
                lr.append(np.log((max(ds_, 0)+eps)/(max(db_, 0)+eps))); R.append(1-bins[b]['1.0']/tgt[b])
            rho = spearmanr(R, lr).correlation if len(R) > 2 else None
            out['H3'].append(dict(g=g, pair=big, rho=rho))
h1 = out['H1']; h2 = out['H2']
summ = dict(
  H1=dict(n=len(h1), ok=sum(x['ok'] for x in h1), frac=sum(x['ok'] for x in h1)/len(h1), median_share=float(np.median([x['share'] for x in h1])), PASS=sum(x['ok'] for x in h1)/len(h1) >= .9),
  H2=dict(n=len(h2), ok=sum(x['ok'] for x in h2), frac=sum(x['ok'] for x in h2)/len(h2), median_ratio=float(np.median([x['ratio'] for x in h2])), PASS=sum(x['ok'] for x in h2)/len(h2) >= .8 and float(np.median([x['ratio'] for x in h2])) >= 2,
          per_model={mo: f"{sum(x['ok'] for x in h2 if x['g'][0]==mo)}/{sum(1 for x in h2 if x['g'][0]==mo)}" for mo in ['liif','lte','sc-inr']}),
  H3=dict(n=len(out['H3']), negative=sum(1 for x in out['H3'] if x['rho'] is not None and x['rho'] < 0), median_rho=float(np.nanmedian([x['rho'] for x in out['H3'] if x['rho'] is not None]))))
json.dump(dict(summary=summ, detail=out), open(sys.argv[2], 'w'), indent=1, default=str)
print(json.dumps(summ, indent=1))
