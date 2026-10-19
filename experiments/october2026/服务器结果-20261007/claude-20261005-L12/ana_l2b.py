# Analysis for 预注册-L2b-20261005.md. Written and hashed before any delta was computed.
# R_full, L_act, L_pred recomputed from L2 outputs exactly as ana_l12.py; delta_b^2 from l2b outputs.
# Bands = ana_l12 / F1 bands on Chebyshev k (b1 [0,1/8), b2 [1/8,1/4), b3 [1/4,1/2]), DC excluded.
import json, sys, os
import numpy as np
from scipy.stats import spearmanr
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
HR = 240; NK = HR // 2 + 1; K = np.arange(NK)
SETS = ['Set14', 'BSD100', 'Urban100']; SIN = [2, 3, 4]
PAIRS = [('lte', 'lte-nocellphase'), ('sc-inr', 'sc-inr-nosinc')]
BANDS = [(0, 1 / 8, False), (1 / 8, 1 / 4, False), (1 / 4, 1 / 2, True)]
def bmask(i):
    lo, hi, inc = BANDS[i]; nu = K / HR
    return (nu >= lo) & ((nu <= hi + 1e-12) if inc else (nu < hi - 1e-12)) & (K > 0)
def ratio(num, den, m):
    imgs = sorted(den); assert sorted(num) == imgs
    return sum(np.asarray(num[i])[m].sum() for i in imgs) / sum(np.asarray(den[i])[m].sum() for i in imgs)
L2 = {ds: json.load(open(f'{D}/l2_{ds}.json')) for ds in SETS}
LB = {ds: json.load(open(f'{D}/l2b_{ds}.json')) for ds in SETS}
out = {}; H = {}
for full, abl in PAIRS:
    rows = []
    for ds in SETS:
        for s in SIN + [8]:
            for b in range(3):
                m = bmask(b); r = L2[ds]
                rf = 1 - ratio(r['E'][f'{full}|{s}|full'], r['S'], m)
                ra = 1 - ratio(r['E'][f'{full}|{s}|abl'], r['S'], m)
                rm = 1 - ratio(r['E'][f'{full}|{s}|marg'], r['S'], m)
                d2 = ratio(LB[ds]['D'][f'{full}|{s}'], LB[ds]['F'][f'{full}|{s}'], m)
                rows.append(dict(ds=ds, s=s, band=f'b{b + 1}', R=rf, L_act=rf - ra, L_pred=rf - rm, d2=d2, P1=d2 * rf, P2=d2 * rf * (1 - 2 * rf) ** 2))
    def rho(rs, k): return float(spearmanr([r[k] for r in rs], [r['L_act'] for r in rs]).correlation)
    conf = [r for r in rows if r['s'] in SIN]; x8 = [r for r in rows if r['s'] == 8]
    st = dict(rho_Lpred=rho(conf, 'L_pred'), rho_P1=rho(conf, 'P1'), rho_P2=rho(conf, 'P2'), rho_d2=rho(conf, 'd2'),
              x8=dict(rho_Lpred=rho(x8, 'L_pred'), rho_P1=rho(x8, 'P1'), rho_P2=rho(x8, 'P2'), rho_d2=rho(x8, 'd2')), rows=rows)
    out[full] = st
    print(f'\n== pair ({full}, {abl}) ==')
    print(f'{"set":9s} {"s":>2s} {"band":4s} {"R":>8s} {"L_act":>8s} {"L_pred":>8s} {"d2":>10s} {"P1":>10s} {"P2":>10s}')
    for r in rows:
        print(f'{r["ds"]:9s} {r["s"]:2d} {r["band"]:4s} {r["R"]:8.4f} {r["L_act"]:8.4f} {r["L_pred"]:8.4f} {r["d2"]:10.3e} {r["P1"]:10.3e} {r["P2"]:10.3e}')
    print({k: v for k, v in st.items() if k != 'rows'})
H['H-L2b-1'] = dict(value={k: (v['rho_P1'], v['rho_Lpred']) for k, v in out.items()}, rule='rho(P1,L_act) >= rho(L_pred,L_act) both pairs',
                    ok=all(v['rho_P1'] >= v['rho_Lpred'] for v in out.values()))
H['H-L2b-2'] = dict(value={k: (v['rho_P2'], v['rho_P1']) for k, v in out.items()}, rule='rho(P2,L_act) < rho(P1,L_act) both pairs',
                    ok=all(v['rho_P2'] < v['rho_P1'] for v in out.values()))
print('\n== 判据 ==')
for k, v in H.items(): print(f'{k}: {"成立" if v["ok"] else "不成立"}  value={v["value"]}  [{v["rule"]}]')
out['H'] = H
json.dump(out, open(os.path.join(os.path.dirname(D.rstrip('/')), 'l2b_result.json'), 'w'), indent=1, default=float)
