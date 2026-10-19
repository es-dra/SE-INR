import json, collections, numpy as np
def load(p):
    A = collections.defaultdict(collections.Counter); T = collections.Counter()
    for r in json.load(open(p)):
        k = (r['model'], r['seed'], r['s'], r['lo']); A[k].update(r['err']); T[k] += r['tgt']
    return A, T
A2, T2 = load('../claude-20261019-two-coordinates-v2/formal/rows.json')
base = lambda s, b: np.mean([1 - A2[(m, sd, s, b)]['1.0']/T2[(m, sd, s, b)] for m in ['liif', 'lte', 'sc-inr'] for sd in [1, 2, 3]])
res = {}
for name, d in [('gaussiansr', 'formal_gs'), ('rdn-btc', 'formal_btc')]:
    A, T = load(f'{d}/rows.json'); R = lambda s, b: 1 - A[(name, 0, s, b)]['1.0']/T[(name, 0, s, b)]
    prof = {f'x{s}': {f'{b:.2f}': [round(R(s, b), 4), round(base(s, b), 4)] for b in np.arange(0.5, min(s, 2.0), .25)} for s in (2, 3, 4)}
    d1 = [abs(R(s, b) - base(s, b)) for s in (2, 3, 4) for b in np.arange(0.5, min(s, 2.0), .25)]
    d2 = [R(s, b) > base(s, b) for s in (2, 3, 4) for b in np.arange(1.0, min(s, 2.0), .25)]
    peaks = {}
    for s in (2, 3, 4, 6):
        H = collections.Counter()
        for b in np.arange(0, s, .25):
            k = (name, 0, s, float(b)); H[int((b/(2*s))/0.0625 + 1e-9)] += A[k]['2.0'] - A[k]['1.0']
        peaks[s] = max(H, key=H.get)*0.0625
    tot = {s: {m: sum(A[k][m] for k in A if k[2] == s) for m in A[(name, 0, s, 0.0)]} for s in (2, 3, 4, 6)}
    res[name] = dict(profile=prof, Q1_within_003=f'{sum(x <= .03 for x in d1)}/{len(d1)}', Q1_PASS=sum(x <= .03 for x in d1)/len(d1) >= .9 if name == 'gaussiansr' else None,
                     Q2_above=f'{sum(d2)}/{len(d2)}', Q2_PASS=sum(d2)/len(d2) >= .8 if name == 'rdn-btc' else None,
                     Q3_peaks=peaks, Q3_PASS=all(.1875 <= v < .375 for v in peaks.values()),
                     argmin={s: min(t, key=t.get) for s, t in tot.items()}, cost_m2_db={s: round(10*np.log10(t['2.0']/t['1.0']), 3) for s, t in tot.items()},
                     native_psnr_nonDC={s: round(-10*np.log10(t['1.0']/32/3), 2) for s, t in tot.items()})
json.dump(res, open('score_v3.json', 'w'), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
