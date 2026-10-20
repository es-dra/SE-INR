import json, collections, numpy as np
def load(p):
    A = collections.defaultdict(collections.Counter); T = collections.Counter()
    for r in json.load(open(p)):
        k = (r['s'], r['lo']); A[k].update(r['err']); T[k] += r['tgt']
    return A, T
arms = {a: load(f'eval_{a}/rows.json') for a in ['edsr', 'rdn', 'nocell']}
Bc = json.load(open('bicubic_R.json'))
R = lambda A, T, k: 1 - A[k]['1.0']/T[k]
psnr = {a: {s: -10*np.log10(sum(A[k]['1.0'] for k in A if k[0] == s)/32/3) for s in (2, 3, 4, 6)} for a, (A, T) in arms.items()}
agree = []; below = {}; dR = {}
for s in (2, 3, 4):
    below[s] = False
    for b in np.arange(0, min(s, 2.0), .25):
        k = (s, float(b)); d = R(*arms['rdn'], k) - R(*arms['edsr'], k); dR[f'x{s}/{b:.2f}'] = round(d, 4)
        agree.append((d > .005) == (Bc[f'{s}|{b}'] < .9))
        if b < .75 and d > .005: below[s] = True
g = {s: psnr['rdn'][s] - psnr['edsr'][s] for s in (2, 3, 4, 6)}; nc = {s: psnr['edsr'][s] - psnr['nocell'][s] for s in (2, 3, 4, 6)}
res = dict(G1=dict(agree=f'{sum(agree)}/{len(agree)}', PASS=sum(agree)/len(agree) >= .8), G2=dict(per_scale=below, PASS=all(below.values())),
           G3=dict(gain={s: round(v, 4) for s, v in g.items()}, PASS=all(v > 0 for v in g.values())),
           G4=dict(loss={s: round(v, 4) for s, v in nc.items()}, PASS=nc[2] > nc[4] and nc[2] > nc[6] and nc[4] <= .05 and nc[6] <= .05),
           dR_rdn_minus_edsr=dR, R_bicubic=Bc, psnr=psnr)
json.dump(res, open('score_v12.json', 'w'), indent=1, default=str); print(json.dumps({k: res[k] for k in ['G1', 'G2', 'G3', 'G4']}, indent=1, default=str))
