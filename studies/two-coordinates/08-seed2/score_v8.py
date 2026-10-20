import json, collections, numpy as np
def load(p):
    A = collections.defaultdict(collections.Counter); T = collections.Counter()
    for r in json.load(open(p)):
        k = (r['s'], r['lo']); A[k].update(r['err']); T[k] += r['tgt']
    return A, T
P = json.load(open('protocol_v8.json')); ref = P['seed1_reference_delta_db']
arms = {a: load(f'eval_{a}/rows.json') for a in ['edsr', 'rdn', 'tinydec', 'nocell']}
psnr = {a: {s: -10*np.log10(sum(A[k]['1.0'] for k in A if k[0] == s)/32/3) for s in (2, 3, 4, 6)} for a, (A, T) in arms.items()}
k1 = []; deltas = {}
for a in ['rdn', 'tinydec', 'nocell']:
    deltas[a] = []
    for i, s in enumerate((2, 3, 4, 6)):
        d = psnr[a][s] - psnr['edsr'][s]; r1 = ref[a][i]; deltas[a].append(round(d, 4))
        ok = abs(d) <= .02 if abs(r1) < .01 else (np.sign(d) == np.sign(r1) and .5*abs(r1) <= abs(d) <= 2*abs(r1))
        k1.append(bool(ok))
R = lambda A, T, k: 1 - A[k]['1.0']/T[k]
def loc(a, sign):
    (A, T), (E, ET) = arms[a], arms['edsr']; hi = []; lo = []
    for s in (2, 3, 4):
        for b in np.arange(0, min(s, 2.0), .25):
            d = R(A, T, (s, float(b))) - R(E, ET, (s, float(b)))
            if b >= 1: hi.append(sign*d > 0)
            elif b < .75: lo.append(abs(d) <= .01)
    return dict(hi=f'{sum(hi)}/{len(hi)}', lo=f'{sum(lo)}/{len(lo)}', PASS=sum(hi)/len(hi) >= .8 and sum(lo)/len(lo) >= .8)
res = dict(delta_seed2=deltas, psnr_seed2={a: {s: round(v, 4) for s, v in p.items()} for a, p in psnr.items()},
           K1=dict(ok=f'{sum(k1)}/12', PASS=sum(k1) >= 10), K2=loc('rdn', 1), K3=loc('tinydec', -1))
json.dump(res, open('score_v8.json', 'w'), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
