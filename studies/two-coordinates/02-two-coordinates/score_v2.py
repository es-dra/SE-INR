import json, sys, collections, numpy as np
from scipy.stats import spearmanr
rows = json.load(open(sys.argv[1])); MS = ['0.5','0.625','0.75','0.8','0.9091','1.0','1.1','1.25','1.3333','1.6','2.0']
A = collections.defaultdict(collections.Counter); T = collections.Counter()
for r in rows:
    k = (r['model'], r['seed'], r['s'], r['lo']); A[k].update(r['err']); T[k] += r['tgt']
Rn = lambda k: 1 - A[k]['1.0']/T[k]
models = ['liif','lte','sc-inr']; seeds = sorted({k[1] for k in A}); res = {}
bins = [b for b in np.arange(0.5, 2.0, 0.25)]
# P1
p1 = []
for mo in models:
    for sd in seeds:
        for b in bins:
            v = [Rn((mo,sd,s,b)) for s in (2,3,4) if (mo,sd,s,b) in A]
            if len(v) >= 2: p1.append(max(v)-min(v))
res['P1'] = dict(n=len(p1), ok=int(sum(x <= .05 for x in p1)), max=float(max(p1)), PASS=sum(x <= .05 for x in p1)/len(p1) >= .9)
p2 = []
for s in (2,3,4):
    for b in bins:
        if b >= s: continue
        v = [np.mean([Rn((mo,sd,s,b)) for sd in seeds]) for mo in models]; p2.append(max(v)-min(v))
res['P2'] = dict(n=len(p2), ok=int(sum(x <= .03 for x in p2)), max=float(max(p2)), PASS=sum(x <= .03 for x in p2)/len(p2) >= .9)
p3 = []; peaks = []
for mo in models:
    for sd in seeds:
        for s in (2,3,4,6):
            H = collections.Counter()
            for b in np.arange(0, s, .25):
                k = (mo,sd,s,float(b))
                if k in A: H[int((b/(2*s))/0.0625 + 1e-9)] += A[k]['2.0'] - A[k]['1.0']
            pk = max(H, key=H.get)*0.0625; peaks.append(pk); p3.append(0.1875 <= pk < 0.375)
res['P3'] = dict(n=len(p3), ok=int(sum(p3)), peaks=collections.Counter(peaks), PASS=sum(p3)/len(p3) >= .9)
p4 = {}
for mo in models:
    xs = []; ys = []
    for s in (2,3,4,6):
        for b in np.arange(0, s, .25):
            if b/(2*s) < 0.1875: continue
            ks = [(mo,sd,s,float(b)) for sd in seeds if (mo,sd,s,float(b)) in A]
            xs.append(sum(T[k]-A[k]['1.0'] for k in ks)); ys.append(sum(A[k]['2.0']-A[k]['1.0'] for k in ks))
    p4[mo] = float(spearmanr(xs, ys).correlation)
res['P4'] = dict(rho=p4, PASS=all(v >= .7 for v in p4.values()))
p5 = []; am = {}; p6 = []; dbs = {}
for mo in models:
    for sd in seeds:
        c2 = []
        for s in (2,3,4,6):
            tot = {m: sum(A[k][m] for k in A if k[:3] == (mo,sd,s)) for m in MS}
            a = min(MS, key=tot.get); am[f'{mo}/{sd}/x{s}'] = a
            if s in (2,3,4): p5.append(a in ('0.9091','1.0','1.1'))
            c2.append(10*np.log10(tot['2.0']/tot['1.0'])); dbs[f'{mo}/{sd}/x{s}'] = {m: round(10*np.log10(tot[m]/tot['1.0']), 4) for m in ('0.5','2.0')}
        p6.append(all(c2[i] > c2[i+1] for i in range(3)))
res['P5'] = dict(n=len(p5), ok=int(sum(p5)), argmin=am, PASS=sum(p5)/len(p5) >= .8)
res['P6'] = dict(n=len(p6), ok=int(sum(p6)), PASS=sum(p6)/len(p6) >= .8)
res['cost_db_m0.5_m2'] = dbs
json.dump(res, open(sys.argv[2], 'w'), indent=1, default=str)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('argmin',)} if isinstance(v, dict) and k != 'cost_db_m0.5_m2' else '...' for k, v in res.items()}, indent=1, default=str))
