# Analysis for prereg 预注册-L1L2-20261005.md. Written and hashed BEFORE any L1/L2 output was opened.
# Decision rules exactly as prereg; Spearman = scipy.stats.spearmanr (average ranks for ties).
# Appendix (预注册-L1L2-附录-20261005.md): H2d vs G1 eps_b^2.
# Pre-declared handling (not in prereg text, fixed here before outputs exist):
#  - empty frequency bin -> NaN, medians use nanmedian;
#  - H2b with no cell having L_act>0.005 for a pair -> "不可判定" (reported, not counted as 成立).
import json, sys, os, glob
import numpy as np
from scipy.stats import spearmanr
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
HR = 240; NK = HR // 2 + 1; K = np.arange(NK)
SETS = ['Set14', 'BSD100', 'Urban100']
SIN, SOUT = [2, 3, 4], [6, 8, 12]
UE = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
NETS, PUB = ['liif', 'lte', 'sc-inr'], ['gsasr', 'contsr', 'thera']
out = {}

def mask_bin(x, i, edges):
    lo, hi = edges[i], edges[i + 1]
    m = (x >= lo - 1e-12) & ((x < hi - 1e-12) if i < len(edges) - 2 else (x <= hi + 1e-12))
    return m & (K > 0)

def R_of(E_imgs, S_imgs, m):
    imgs = sorted(S_imgs)
    assert sorted(E_imgs) == imgs, 'image sets differ'
    e = sum(np.asarray(E_imgs[i])[m].sum() for i in imgs); s = sum(np.asarray(S_imgs[i])[m].sum() for i in imgs)
    return np.nan if (m.sum() == 0 or s == 0) else 1 - e / s

# ------------------------------- L1 -------------------------------
L1 = {}
for ds in SETS:
    S, E = None, {}
    for f in sorted(glob.glob(f'{D}/l1_*_{ds}.json')):
        r = json.load(open(f))
        if S is None: S = r['S']
        assert sorted(r['S']) == sorted(S)
        E.update(r['E'])
    L1[ds] = (S, E)

def Ru(ds, mn, s, i):
    S, E = L1[ds]; return R_of(E[mn][str(s)], S, mask_bin(2 * s * K / HR, i, UE))
def Rnu(ds, mn, s, i):
    S, E = L1[ds]; return R_of(E[mn][str(s)], S, mask_bin(K / HR, i, [u / 6 for u in UE]))
def Dstats(ds, mn):
    ru = np.array([[Ru(ds, mn, s, i) for i in range(7)] for s in SIN + SOUT])
    rin, rout = ru[:3], ru[3:]
    din = np.nanmax(rin, 0) - np.nanmin(rin, 0)
    dout = np.nanmax(np.abs(rout - np.nanmean(rin, 0)), 0)
    rn = np.array([[Rnu(ds, mn, s, i) for i in range(7)] for s in SIN])
    dnu = np.nanmax(rn, 0) - np.nanmin(rn, 0)
    return dict(Din_med=float(np.nanmedian(din)), Dout_med=float(np.nanmedian(dout)), Dnu_med=float(np.nanmedian(dnu)),
                Din=din.tolist(), Dout=dout.tolist(), Dnu=dnu.tolist(), R_u={str(s): ru[j].tolist() for j, s in enumerate(SIN + SOUT)})

have = {mn for ds in SETS for mn in L1[ds][1]}
st = {(ds, mn): Dstats(ds, mn) for ds in SETS for mn in sorted(have) if all(mn in L1[d][1] for d in SETS)}
out['L1_stats'] = {f'{d}|{m}': v for (d, m), v in st.items()}
print('== L1: median over 7 u-bins ==')
print(f'{"set":9s} {"model":15s} {"D_in":>7s} {"D_nu":>7s} {"D_out":>7s}')
for (d, m), v in sorted(st.items(), key=lambda x: (x[0][1], x[0][0])):
    print(f'{d:9s} {m:15s} {v["Din_med"]:7.4f} {v["Dnu_med"]:7.4f} {v["Dout_med"]:7.4f}')
H = {}
lm = [st[(d, 'lmmse_generic')]['Din_med'] for d in SETS]
H['H1a'] = dict(value=lm, rule='lmmse_generic D_in med <=0.03 on all 3 sets', ok=all(x <= 0.03 for x in lm))
g9 = [(d, m) for m in NETS for d in SETS]
n = sum(st[g]['Din_med'] <= 0.05 for g in g9); H['H1b'] = dict(value=f'{n}/9', rule='D_in med <=0.05, >=8/9', ok=n >= 8)
n = sum(st[g]['Dnu_med'] > st[g]['Din_med'] for g in g9); H['H1c'] = dict(value=f'{n}/9', rule='D_nu med > D_u med, >=8/9', ok=n >= 8)
n = sum(st[g]['Dout_med'] > st[g]['Din_med'] for g in g9); H['H1d'] = dict(value=f'{n}/9', rule='D_out med > D_in med, >=8/9', ok=n >= 8)
gp = [(d, m) for m in PUB for d in SETS if (d, m) in st]
n = sum(st[g]['Din_med'] <= 0.05 for g in gp); H['H1e'] = dict(value=f'{n}/{len(gp)}', rule='public D_in med <=0.05, >=7/9', ok=(len(gp) == 9 and n >= 7))
if not H['H1a']['ok']:
    for k in ('H1b', 'H1c', 'H1d', 'H1e'): H[k]['note'] = 'H1a 未过：只报告不判定'
# descriptive
desc = {}
for d in SETS:
    for m in NETS + PUB:
        if (d, m) not in st: continue
        desc[f'{d}|{m}'] = {str(s): [Ru(d, m, s, i) - Ru(d, 'lmmse_generic', s, i) for i in range(3, 7)] for s in SIN + SOUT}
out['L1_desc_net_minus_lmmse_u_gt_1'] = desc
seedr = {}
for m in NETS:
    vals = []
    for d in SETS:
        if not all(x in L1[d][1] for x in (m, f'{m}@s2', f'{m}@s3')): continue
        for s in SIN + SOUT:
            for i in range(7):
                r = [Ru(d, x, s, i) for x in (m, f'{m}@s2', f'{m}@s3')]
                vals.append(np.nanmax(r) - np.nanmin(r))
    seedr[m] = dict(median=float(np.nanmedian(vals)) if vals else None, p90=float(np.nanpercentile(vals, 90)) if vals else None)
out['L1_desc_seed_range'] = seedr
print('desc: seed range of R(u) (median/p90 over set×scale×bin):', seedr)
print('desc: net - lmmse_generic, u-bins >1 (mean over the 4 bins):')
for k, v in desc.items(): print(' ', k, {s: round(float(np.nanmean(x)), 4) for s, x in v.items()})

# ------------------------------- L2 -------------------------------
PAIRS = [('lte', 'lte-nocellphase'), ('sc-inr', 'sc-inr-nosinc')]
EPS = {'b1': 6e-05, 'b2': 0.00552, 'b3': 0.07235}   # G1 closed-form eps_b^2 (appendix 预注册-L1L2-附录-20261005.md)
BANDS = [(0, 1 / 8, False), (1 / 8, 1 / 4, False), (1 / 4, 1 / 2, True)]
def bmask(i):
    lo, hi, inc = BANDS[i]; nu = K / HR
    return (nu >= lo) & ((nu <= hi + 1e-12) if inc else (nu < hi - 1e-12)) & (K > 0)
L2 = {ds: json.load(open(f'{D}/l2_{ds}.json')) for ds in SETS if os.path.exists(f'{D}/l2_{ds}.json')}
if len(L2) == 3:
    def R2(ds, full, s, tag, m):
        r = L2[ds]; return R_of(r['E'][f'{full}|{s}|{tag}'], r['S'], m)
    tab = {}
    for full, abl in PAIRS:
        rows = []
        for ds in SETS:
            for s in SIN + [8]:
                for b in range(3):
                    m = bmask(b); rf, ra, rm, rl = (R2(ds, full, s, t, m) for t in ('full', 'abl', 'marg', 'mlog'))
                    rows.append(dict(ds=ds, s=s, band=f'b{b + 1}', R_full=rf, R_abl=ra, R_marg=rm, L_act=rf - ra, L_pred=rf - rm, L_pred_log=rf - rl))
        tab[full] = rows
        conf = [r for r in rows if r['s'] in SIN]
        la = np.array([r['L_act'] for r in conf]); lp = np.array([r['L_pred'] for r in conf]); ll = np.array([r['L_pred_log'] for r in conf])
        rho = float(spearmanr(lp, la).correlation); rho_log = float(spearmanr(ll, la).correlation)
        eps = np.array([EPS[r['band']] for r in conf]); rho_eps = float(spearmanr(eps, la).correlation)
        sel = la > 0.005
        ratio = float(np.median(lp[sel] / la[sel])) if sel.any() else None
        fr = []; fr_s = {}
        for ds in SETS:
            for s in SIN + [8]:
                for img, v in L2[ds]['mse'][f'{full}|{s}'].items():
                    w = v['abl_marg'] < v['abl_full']; fr_s.setdefault(s, []).append(w)
                    if s in SIN: fr.append(w)
        h2z = max(v for ds in SETS for s in SIN + [8] for v in L2[ds]['h2z'][f'{full}|{s}'].values())
        out.setdefault('L2', {})[full] = dict(rows=rows, rho=rho, rho_log=rho_log, rho_eps=rho_eps, n_sel=int(sel.sum()), ratio_med=ratio,
                                               frac_c=float(np.mean(fr)), frac_c_by_scale={str(s): float(np.mean(x)) for s, x in fr_s.items()}, h2z_max=h2z)
        print(f'\n== L2 pair ({full}, {abl}) ==')
        print(f'{"set":9s} {"s":>2s} {"band":4s} {"R_full":>8s} {"R_abl":>8s} {"R_marg":>8s} {"L_act":>8s} {"L_pred":>8s} {"L_plog":>8s}')
        for r in rows:
            print(f'{r["ds"]:9s} {r["s"]:2d} {r["band"]:4s} {r["R_full"]:8.4f} {r["R_abl"]:8.4f} {r["R_marg"]:8.4f} {r["L_act"]:8.4f} {r["L_pred"]:8.4f} {r["L_pred_log"]:8.4f}')
        print(f'rho={rho:.3f} rho_eps={rho_eps:.3f} rho_log={rho_log:.3f} n(L_act>0.005)={int(sel.sum())} ratio_med={ratio} frac_c={np.mean(fr):.3f} by_s={ {s: round(float(np.mean(x)), 3) for s, x in fr_s.items()} } h2z_max={h2z:.3g}')
        # u-binned curves (descriptive)
        cur = {}
        for ds in SETS:
            for s in SIN + [8]:
                rr = []
                for i in range(7):
                    m = mask_bin(2 * s * K / HR, i, UE); rf, ra, rm = (R2(ds, full, s, t, m) for t in ('full', 'abl', 'marg'))
                    rr.append(dict(L_act=rf - ra, L_pred=rf - rm))
                cur[f'{ds}|{s}'] = rr
        out['L2'][full]['u_curves'] = cur
    P = out['L2']
    H['H2a'] = dict(value={k: v['rho'] for k, v in P.items()}, rule='Spearman >=0.7 both pairs', ok=all(v['rho'] >= 0.7 for v in P.values()))
    H['H2d'] = dict(value={k: dict(rho=v['rho'], rho_eps=v['rho_eps'], diff=v['rho'] - v['rho_eps']) for k, v in P.items()},
                    rule='rho(L_pred,L_act) - rho(eps_b^2,L_act) >= +0.05 both pairs', ok=all(v['rho'] - v['rho_eps'] >= 0.05 for v in P.values()))
    if any(v['ratio_med'] is None for v in P.values()):
        H['H2b'] = dict(value={k: v['ratio_med'] for k, v in P.items()}, rule='median L_pred/L_act in [0.5,2] both pairs', ok=None, note='不可判定：某对无 L_act>0.005 的格')
    else:
        H['H2b'] = dict(value={k: v['ratio_med'] for k, v in P.items()}, rule='median L_pred/L_act in [0.5,2] both pairs', ok=all(0.5 <= v['ratio_med'] <= 2 for v in P.values()))
    H['H2c'] = dict(value={k: v['frac_c'] for k, v in P.items()}, rule='frac MSE(abl,marg)<MSE(abl,full) >=0.8 both pairs', ok=all(v['frac_c'] >= 0.8 for v in P.values()))
    H['H2z'] = dict(value={k: v['h2z_max'] for k, v in P.items()}, rule='ablate max|Δ| over cells <=1e-6', ok=all(v['h2z_max'] <= 1e-6 for v in P.values()))
    if not H['H2z']['ok']: print('!! H2z FAILED: STOP and investigate before interpreting L2')
out['H'] = H
print('\n== 判据 ==')
for k, v in H.items():
    ok = '不可判定' if v['ok'] is None else ('成立' if v['ok'] else '不成立')
    print(f'{k}: {ok}  value={v["value"]}  [{v["rule"]}]' + (f'  ({v["note"]})' if 'note' in v else ''))
json.dump(out, open(os.path.join(os.path.dirname(D.rstrip('/')), 'l12_result.json'), 'w'), indent=1, default=float)
