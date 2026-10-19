import json, collections, numpy as np
def load(p):
    A = collections.defaultdict(collections.Counter); T = collections.Counter()
    for r in json.load(open(p)):
        k = (r['s'], r['lo']); A[k].update(r['err']); T[k] += r['tgt']
    return A, T
E_, ET = load('eval_edsr/rows.json'); R_, RT = load('eval_rdn/rows.json')
R = lambda A, T, k: 1 - A[k]['1.0']/T[k]
hi = []; lo = []; prof = {}
for s in (2, 3, 4):
    for b in np.arange(0, min(s, 2.0), .25):
        k = (s, float(b)); d = R(R_, RT, k) - R(E_, ET, k); prof[f'x{s}/{b:.2f}'] = round(d, 4)
        if b >= 1: hi.append(d)
        elif b < .75: lo.append(d)
cost = lambda A: [10*np.log10(sum(A[k]['2.0'] for k in A if k[0] == s)/sum(A[k]['1.0'] for k in A if k[0] == s)) for s in (2, 3, 4, 6)]
psnr = lambda A: {s: round(-10*np.log10(sum(A[k]['1.0'] for k in A if k[0] == s)/32/3), 3) for s in (2, 3, 4, 6)}
ce, cr = cost(E_), cost(R_)
res = dict(T1=dict(ok=f'{sum(x > 0 for x in hi)}/{len(hi)}', PASS=sum(x > 0 for x in hi)/len(hi) >= .8),
           T2=dict(ok=f'{sum(abs(x) <= .01 for x in lo)}/{len(lo)}', PASS=sum(abs(x) <= .01 for x in lo)/len(lo) >= .8),
           T3=dict(edsr=[round(x, 3) for x in ce], rdn=[round(x, 3) for x in cr], PASS=all(c[i] > c[i+1] for c in (ce, cr) for i in range(3))),
           diff_rdn_minus_edsr=prof, psnr_nonDC=dict(edsr=psnr(E_), rdn=psnr(R_)))
json.dump(res, open('score_v5.json', 'w'), indent=1); print(json.dumps(res, indent=1))
