import json, collections, numpy as np
def load(p, seed=None):
    A = collections.defaultdict(collections.Counter); T = collections.Counter()
    for r in json.load(open(p)):
        if seed is not None and r['seed'] != seed: continue
        k = (r['model'], r['s'], r['lo']); A[k].update(r['err']); T[k] += r['tgt']
    return A, T
A2, T2 = load('../claude-20261019-two-coordinates-v2/formal/rows.json', seed=1)
A2all, T2all = load('../claude-20261019-two-coordinates-v2/formal/rows.json')
R = lambda A, T, k: 1 - A[k]['1.0']/T[k]
res = {'R1': {}, 'R3': {}}
for mo in ['liif', 'lte', 'sc-inr']:
    Ae, Te = load(f'early_{mo}/rows.json')
    hi = []; lo = []; prof = {}
    for s in (2, 3, 4):
        for b in np.arange(0, min(s, 2.0), .25):
            k = (mo, s, float(b)); ke = k
            d = R(Ae, Te, ke) - R(A2, T2, k); prof[f'x{s}/{b:.2f}'] = round(d, 4)
            (hi if b >= 1 else lo if b < .75 else []).append(d)
    res['R1'][mo] = dict(hi_lower=f'{sum(x < 0 for x in hi)}/{len(hi)}', lo_within=f'{sum(abs(x) <= .01 for x in lo)}/{len(lo)}', PASS=sum(x < 0 for x in hi)/len(hi) >= .8 and sum(abs(x) <= .01 for x in lo)/len(lo) >= .8, diff_early_minus_final=prof)
    c = lambda A: 10*np.log10(sum(A[k]['2.0'] for k in A if k[1] == 2)/sum(A[k]['1.0'] for k in A if k[1] == 2))
    res['R3'][mo] = dict(early=round(c(Ae), 3), final=round(c(A2), 3))
res['R3']['PASS'] = sum(v['early'] < v['final'] for v in res['R3'].values() if isinstance(v, dict)) >= 2
Ae, Te = load('edsr_full/rows.json')
base = lambda b: np.mean([1 - A2all[(m, 4, b)]['1.0']/T2all[(m, 4, b)] for m in ['liif', 'lte', 'sc-inr']])
r2 = {f'{b:.2f}': [round(R(Ae, Te, ('edsr-full', 4, float(b))), 4), round(base(float(b)), 4)] for b in np.arange(0, 4, .25)}
ok = sum(R(Ae, Te, ('edsr-full', 4, float(b))) > base(float(b)) for b in (1.0, 1.25, 1.5, 1.75))
res['R2'] = dict(profile_edsrfull_vs_base=r2, above=f'{ok}/4', PASS=ok >= 3, psnr_nonDC_edsrfull=round(-10*np.log10(sum(Ae[k]['1.0'] for k in Ae)/32/3), 3), psnr_nonDC_base_lte_seed1=round(-10*np.log10(sum(A2[k]['1.0'] for k in A2 if k[0] == 'lte' and k[1] == 4)/32/3), 3))
json.dump(res, open('score_v4.json', 'w'), indent=1, default=str)
print(json.dumps({k: ({kk: (vv if kk != 'diff_early_minus_final' else '...') for kk, vv in v.items()} if k != 'R1' else {m: {kk: vv for kk, vv in d.items() if kk != 'diff_early_minus_final'} for m, d in v.items()}) for k, v in res.items()}, indent=1, default=str))
