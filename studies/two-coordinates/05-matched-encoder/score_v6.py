import json, collections, numpy as np
def load(p):
    A = collections.defaultdict(collections.Counter); T = collections.Counter()
    for r in json.load(open(p)):
        k = (r['s'], r['lo']); A[k].update(r['err']); T[k] += r['tgt']
    return A, T
arms = {a: load(f'eval_{a}/rows.json') for a in ['edsr', 'tinydec', 'nocell', 'rdn']}
psnr = {a: {s: -10*np.log10(sum(A[k]['1.0'] for k in A if k[0] == s)/32/3) for s in (2, 3, 4, 6)} for a, (A, T) in arms.items()}
d = lambda a, s: psnr[a][s] - psnr['edsr'][s]
(Ae, Te), (At, Tt) = arms['edsr'], arms['tinydec']
rb = [abs((1 - At[(s, b)]['1.0']/Tt[(s, b)]) - (1 - Ae[(s, b)]['1.0']/Te[(s, b)])) for s in (3, 4) for b in np.arange(0, 2, .25)]
res = dict(psnr_nonDC={a: {s: round(v, 4) for s, v in p.items()} for a, p in psnr.items()},
           delta_vs_edsr={a: {s: round(d(a, s), 4) for s in (2, 3, 4, 6)} for a in ['tinydec', 'nocell', 'rdn']},
           D1=dict(loss_x4=round(-d('tinydec', 4), 4), loss_x6=round(-d('tinydec', 6), 4), Rbins_within_001=f'{sum(x <= .01 for x in rb)}/{len(rb)}',
                   PASS=-d('tinydec', 4) <= .05 and -d('tinydec', 6) <= .05 and sum(x <= .01 for x in rb)/len(rb) >= .8),
           D2=dict(losses=[round(-d('nocell', s), 4) for s in (2, 3, 4, 6)], PASS=all(-d('nocell', a) > -d('nocell', b) for a, b in [(2, 3), (3, 4), (4, 6)]) and -d('nocell', 4) <= .03 and -d('nocell', 6) <= .03),
           D3=dict(rdn_gain_x4=round(d('rdn', 4), 4), tinydec_loss_x4=round(-d('tinydec', 4), 4), PASS=d('rdn', 4) > abs(d('tinydec', 4))))
json.dump(res, open('score_v6.json', 'w'), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
