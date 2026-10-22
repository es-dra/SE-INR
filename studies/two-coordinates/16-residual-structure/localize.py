"""Retrospective CPU analysis of saved sufficient statistics; no model calls.

Primary estimator fits one real transfer coefficient per model/scale/cell/bin,
pooling images and both fixed training seeds. Image allocations retain the
nonzero individual <P, epsilon> term; bootstrap and LOO refit the coefficient.
"""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

MODELS = ['liif', 'lte', 'sc-inr']
SCALES = [2, 3, 4, 6]
MUS = [0.5, 1.3333, 2.0]
PAIRS = [(2, 3), (3, 4), (4, 6), (2, 4), (3, 6), (2, 6)]
GROUPS = [('low', 0, 6), ('middle', 6, 12), ('high', 12, 16), ('all', 0, 16)]
SUM_KEYS = ['Q', 'G', 'H', 'D', 'U', 'V', 'E', 'K', 'epsilon_energy',
            'error_cross', 'native_projection_cross', 'e_epsilon', 'Jtotal',
            'epsilon_per_h2', 'H_per_h2', 'Jtotal_per_h2']
CHECKS = {}


def check(name, actual, expected, atol=2e-12, rtol=2e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(actual - expected)))
    CHECKS[name] = max(CHECKS.get(name, 0), error)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_saved(path):
    raw = json.loads(path.read_text())
    assert raw['stat_names'] == ['C', 'U', 'E', 'V', 'K', 'D', 'Z', 'G']
    names = sorted({r['image'] for r in raw['records']})
    assert len(names) == 14
    x = np.full((14, 3, 2, 4, 3, 17, 8), np.nan)
    dc = np.full((14, 3, 2, 4, 3, 8), np.nan)
    for r in raw['records']:
        idx = (names.index(r['image']), MODELS.index(r['model']), r['seed'] - 1,
               SCALES.index(r['scale']), MUS.index(r['mu']))
        assert np.isnan(x[idx]).all(), 'Duplicate record'
        x[idx], dc[idx] = r['bins'], r['dc']
    assert np.isfinite(x).all() and np.isfinite(dc).all()
    return x, dc, names, raw['metadata']


def measure(x, w=None):
    C, U, E, V, K, D, Z, T = np.moveaxis(x, -1, 0)
    check('B_identity', E - V, U + 2*C)
    if w is None:
        w = K/U
    ep = Z - 2*w*K + w*w*U
    if ep.min() < -2e-12:
        raise ValueError('Negative residual energy')
    ee = T - K - w*C
    pe = K - w*U
    Q = (1-w)**2 * (E-V)
    G = 2*w*(1-w)*C
    pcross = -2*(1-w)*pe
    H = ep - 2*ee + pcross
    check('exact_cost_identity', D, Q + G + H)
    Jtotal = Z + U - 2*K
    h2 = (2*(np.asarray(MUS)[None, :]-1)/np.asarray(SCALES)[:, None])**2
    h2 = h2[..., None]
    return dict(Q=Q, G=G, H=H, D=D, U=U, V=V, E=E, K=K,
                epsilon_energy=ep, e_epsilon=ee, error_cross=-2*ee,
                native_projection_cross=pcross, w=np.broadcast_to(w, U.shape),
                kappa=ep/U, rho=ee/np.sqrt(np.maximum(V*ep, 1e-300)),
                Jtotal=Jtotal,epsilon_per_h2=ep/h2,H_per_h2=H/h2,
                Jtotal_per_h2=Jtotal/h2)


def grouped(m, a, b):
    v = {key: m[key][..., a:b].sum(axis=-1) for key in SUM_KEYS}
    v['kappa'] = v['epsilon_energy']/v['U']
    v['rho'] = v['e_epsilon']/np.sqrt(np.maximum(v['V']*v['epsilon_energy'], 1e-300))
    v['normalized_residual_to_error'] = v['epsilon_energy']/v['V']
    v['w_energy_weighted'] = v['K']/v['U']
    return v


def factor_shapley(start, end, func):
    """Exact three-factor endpoint accounting, not causal identification."""
    result = np.zeros_like(start)
    for order in itertools.permutations(range(start.shape[0])):
        state = start.copy()
        old = func(state)
        for j in order:
            state[j] = end[j]
            new = func(state)
            result[j] += (old-new)/6
            old = new
    check('shapley_identity', result.sum(axis=0), func(start)-func(end))
    return result


def pair_values(m, start, end, a=0, b=16):
    # Leading dimensions, if present, precede (model, scale, multiplier, bin).
    i, j = SCALES.index(start), SCALES.index(end)
    u0, u1 = m['U'][..., i, :, a:b], m['U'][..., j, :, a:b]
    k0, k1 = m['kappa'][..., i, :, a:b], m['kappa'][..., j, :, a:b]
    fu = ((u0-u1)*(k0+k1)/2).sum(axis=-1)
    fk = ((k0-k1)*(u0+u1)/2).sum(axis=-1)
    v = grouped(m, a, b)
    diffs = {key: v[key][..., i, :] - v[key][..., j, :] for key in SUM_KEYS}
    check('U_kappa_band_identity', fu+fk, diffs['epsilon_energy'])
    gu = (v['U'][..., i, :]-v['U'][..., j, :]) * (v['kappa'][..., i, :]+v['kappa'][..., j, :])/2
    gk = (v['kappa'][..., i, :]-v['kappa'][..., j, :]) * (v['U'][..., i, :]+v['U'][..., j, :])/2
    check('U_kappa_group_identity', gu+gk, diffs['epsilon_energy'])
    # The signed correlation term is -2*rho*sqrt(V*epsilon_energy).
    st = np.stack([v['epsilon_energy'][..., i, :], v['V'][..., i, :], v['rho'][..., i, :]])
    en = np.stack([v['epsilon_energy'][..., j, :], v['V'][..., j, :], v['rho'][..., j, :]])
    shap = factor_shapley(st, en, lambda z: -2*z[2]*np.sqrt(np.maximum(z[0]*z[1], 0)))
    hs = (2*(np.asarray(MUS)-1)/start)**2
    ht = (2*(np.asarray(MUS)-1)/end)**2
    eta_s, eta_t = v['epsilon_per_h2'][...,i,:], v['epsilon_per_h2'][...,j,:]
    hfactor = (hs-ht)*(eta_s+eta_t)/2
    etafactor = (eta_s-eta_t)*(hs+ht)/2
    check('h2_normalized_epsilon_identity',hfactor+etafactor,diffs['epsilon_energy'])
    return dict(**{f'delta_{k}': val for k, val in diffs.items()},
                epsilon_drop_U=fu, epsilon_drop_kappa=fk,
                epsilon_drop_group_U=gu, epsilon_drop_group_kappa=gk,
                cross_drop_epsilon=shap[0], cross_drop_V=shap[1], cross_drop_rho=shap[2],
                epsilon_drop_h2=hfactor,epsilon_drop_normalized_response=etafactor)


def scalar_rows(m, labels):
    rows = []
    for group, a, b in GROUPS:
        v = grouped(m, a, b)
        for model in range(len(labels)+1):
            label = labels[model] if model < len(labels) else 'mean_of_models'
            select = lambda z: z[model] if model < len(labels) else z.mean(axis=0)
            for si, scale in enumerate(SCALES):
                for mi, mu in enumerate(MUS):
                    row = dict(model=label, group=group, r_min=a/32, r_max=b/32, scale=scale, mu=mu)
                    row.update(h=2*(mu-1)/scale,h2=(2*(mu-1)/scale)**2)
                    row.update({key: float(select(value)[si, mi]) for key, value in v.items()})
                    rows.append(row)
    return rows


def pair_rows(m, labels):
    rows = []
    for group, a, b in GROUPS:
        for start, end in PAIRS:
            p = pair_values(m, start, end, a, b)
            for model in range(len(labels)+1):
                label = labels[model] if model < len(labels) else 'mean_of_models'
                select = lambda z: z[model] if model < len(labels) else z.mean(axis=0)
                for mi, mu in enumerate(MUS):
                    row = dict(model=label, group=group, start=start, end=end, mu=mu)
                    row.update({key: float(select(value)[mi]) for key, value in p.items()})
                    rows.append(row)
    return rows


def band_rows(m, labels):
    rows = []
    for model in range(len(labels)+1):
        label = labels[model] if model < len(labels) else 'mean_of_models'
        select = lambda z: z[model] if model < len(labels) else z.mean(axis=0)
        for si, scale in enumerate(SCALES):
            for mi, mu in enumerate(MUS):
                for bi in range(m['H'].shape[-1]):
                    row = dict(model=label, scale=scale, mu=mu, bin=bi, r_min=bi/32, r_max=(bi+1)/32)
                    row.update(h=2*(mu-1)/scale,h2=(2*(mu-1)/scale)**2)
                    row.update({key: float(select(value)[si, mi, bi]) for key, value in m.items()})
                    rows.append(row)
    return rows


def image_allocations(x, names, m):
    # x: images, models, seeds, scales, multiplier, band, stat.
    im = measure(x, m['w'][None, :, None])
    for key in SUM_KEYS:
        check(f'image_pool_{key}', im[key].mean(axis=(0, 2)), m[key])
    per_image = {key: val.mean(axis=2) for key, val in im.items()}
    rows, ranking, concentration = [], [], []
    for group, a, b in GROUPS:
        g = grouped(per_image, a, b)
        for image, name in enumerate(names):
            for model in range(4):
                label = MODELS[model] if model < 3 else 'mean_of_models'
                select = lambda val: val[image, model] if model < 3 else val[image].mean(axis=0)
                for si, scale in enumerate(SCALES):
                    for mi, mu in enumerate(MUS):
                        row = dict(image=name, model=label, group=group, scale=scale, mu=mu)
                        row.update(h=2*(mu-1)/scale,h2=(2*(mu-1)/scale)**2)
                        row.update({key: float(select(val)[si, mi]) for key, val in g.items()})
                        rows.append(row)
        for start, end in PAIRS:
            i, j = SCALES.index(start), SCALES.index(end)
            for mi, mu in enumerate(MUS):
                v = (g['H'][:, :, i, mi]-g['H'][:, :, j, mi]).mean(axis=1)
                d = (g['D'][:, :, i, mi]-g['D'][:, :, j, mi]).mean(axis=1)
                order = np.argsort(-v)
                for rank, image in enumerate(order, start=1):
                    row = dict(group=group, start=start, end=end, mu=mu, rank=rank,
                               image=names[image], delta_H=float(v[image]),
                               contribution_to_mean_delta_H=float(v[image]/len(names)),
                               signed_share_of_delta_H=float(v[image]/v.sum()),
                               delta_D=float(d[image]))
                    for key in ['epsilon_energy', 'error_cross', 'native_projection_cross']:
                        row[f'delta_{key}'] = float((g[key][image, :, i, mi]-g[key][image, :, j, mi]).mean())
                    ranking.append(row)
                positive_sum = v[v>0].sum()
                concentration.append(dict(group=group, start=start, end=end, mu=mu,
                    top1=names[order[0]], top3=[names[k] for k in order[:3]],
                    top1_signed_share=float(v[order[0]]/v.sum()),
                    top3_signed_share=float(v[order[:3]].sum()/v.sum()),
                    top1_share_of_positive=float(v[order[0]]/positive_sum),
                    top3_share_of_positive=float(v[order[:3]].sum()/positive_sum),
                    negative_images=[names[k] for k in range(len(names)) if v[k]<0],
                    negative_signed_share=float(v[v<0].sum()/v.sum())))
    return rows, ranking, concentration


def loo(x, names):
    # x has already averaged the two fixed seeds.
    rows = []
    for ii, name in enumerate(names):
        m = measure(np.delete(x, ii, axis=0).mean(axis=0))
        p = pair_values(m, 2, 6)
        mid = pair_values(m, 2, 6, 6, 12)
        for mi, mu in enumerate(MUS):
            v = {key: float(val[:,mi].mean()) for key,val in p.items()}
            rows.append(dict(omitted_image=name, mu=mu, **v,
                H_share_of_D=v['delta_H']/v['delta_D'],
                middle_share_of_H=float(mid['delta_H'][:,mi].mean()/v['delta_H']),
                U_share_of_epsilon_drop=v['epsilon_drop_U']/v['delta_epsilon_energy'],
                kappa_share_of_epsilon_drop=v['epsilon_drop_kappa']/v['delta_epsilon_energy'],
                h2_share_of_epsilon_drop=v['epsilon_drop_h2']/v['delta_epsilon_energy'],
                normalized_response_share_of_epsilon_drop=v['epsilon_drop_normalized_response']/v['delta_epsilon_energy']))
    return rows


def bootstrap(x, n):
    rng = np.random.default_rng(20261024)
    draws = {pair: [] for pair in PAIRS}
    descriptors = []
    for start in range(0, n, 25):
        count = min(n-start, 25)
        indices = rng.integers(0, len(x), size=(count,len(x)))
        weights = np.stack([np.bincount(row,minlength=len(x)) for row in indices])/len(x)
        pooled = np.tensordot(weights,x,axes=(1,0))
        m = measure(pooled)
        g = grouped(m,0,16)
        descriptors.append(np.stack([g[k].mean(axis=1) for k in ['H','epsilon_energy','error_cross','kappa','rho','epsilon_per_h2','H_per_h2','Jtotal_per_h2']],axis=-1))
        for pair in PAIRS:
            p = pair_values(m,*pair)
            middle = pair_values(m,*pair,6,12)
            # Average model-specific fitted quantities, then ratios of averages.
            p = {key: val.mean(axis=1) for key,val in p.items()}
            vals = dict(delta_H=p['delta_H'], delta_D=p['delta_D'],
                        H_share_of_D=p['delta_H']/p['delta_D'],
                        epsilon_drop=p['delta_epsilon_energy'],
                        error_cross_drop=p['delta_error_cross'],
                        middle_share_of_H=middle['delta_H'].mean(axis=1)/p['delta_H'],
                        U_share_of_epsilon_drop=p['epsilon_drop_U']/p['delta_epsilon_energy'],
                        kappa_share_of_epsilon_drop=p['epsilon_drop_kappa']/p['delta_epsilon_energy'],
                        h2_share_of_epsilon_drop=p['epsilon_drop_h2']/p['delta_epsilon_energy'],
                        normalized_response_share_of_epsilon_drop=p['epsilon_drop_normalized_response']/p['delta_epsilon_energy'])
            draws[pair].append(vals)
    rows = []
    for pair, batches in draws.items():
        for mi,mu in enumerate(MUS):
            row = dict(start=pair[0],end=pair[1],mu=mu)
            for key in batches[0]:
                values = np.concatenate([batch[key][:,mi] for batch in batches])
                row[f'{key}_ci95'] = np.quantile(values,[.025,.975]).tolist()
                row[f'{key}_nonpositive_fraction'] = float((values<=0).mean())
            rows.append(row)
    d = np.concatenate(descriptors)
    scale_rows = []
    for si,s in enumerate(SCALES):
        for mi,mu in enumerate(MUS):
            row = dict(scale=s,mu=mu)
            for ki,key in enumerate(['H','epsilon_energy','error_cross','kappa','rho','epsilon_per_h2','H_per_h2','Jtotal_per_h2']):
                row[f'{key}_ci95']=np.quantile(d[:,si,mi,ki],[.025,.975]).tolist()
            scale_rows.append(row)
    return dict(seed=20261024,n=n,unit='paired image cluster; seeds fixed; w refit every draw',pairs=rows,scales=scale_rows)


def load_div2k(path):
    raw = {ast.literal_eval(k): v for k,v in json.loads(path.read_text()).items()}
    x = np.zeros((3,4,3,16,6))
    for (model,scale,mu,b),value in raw.items():
        x[MODELS.index(model),SCALES.index(scale),MUS.index(float(mu)),b]=value
    x /= 32*3*232**4
    C,U,E,V,K,D=np.moveaxis(x,-1,0)
    w=K/U;Q=(1-w)**2*(E-V);G=2*w*(1-w)*C;H=D-Q-G
    rows=[]
    for model in range(4):
        label=MODELS[model] if model<3 else 'mean_of_models'
        select=lambda z:z[model] if model<3 else z.mean(axis=0)
        for si,s in enumerate(SCALES):
            for mi,mu in enumerate(MUS):
                for bi in range(16):
                    rows.append(dict(model=label,scale=s,mu=mu,bin=bi,r_min=bi/32,r_max=(bi+1)/32,
                        **{k:float(select(v)[si,mi,bi]) for k,v in dict(H=H,D=D,Q=Q,G=G,U=U,V=V,w=w).items()}))
    return dict(scope='32 DIV2K crops, seed1; only six statistics, so H can be localized but epsilon/cross cannot be split',bands=rows)


def save_csv(path, rows):
    if not rows:return
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def figures(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,3,figsize=(15,8.3),layout='constrained')
    rows=[r for r in report['scales'] if r['model']=='mean_of_models' and r['group']=='all' and r['mu']==2]
    for key,label,color in [('H','Net remainder H','#bb4940'),('epsilon_energy','Residual energy','#4578a0'),('error_cross','Error correlation term','#bd9c41')]:
        ax[0,0].plot(SCALES,[r[key]*1e6 for r in rows],'o-',label=label,color=color)
    ax[0,0].axhline(0,lw=.5,color='#555555');ax[0,0].set(xticks=SCALES,xlabel='Scale',ylabel='RGB MSE x 1e6',title='A. Energy shrinks; alignment partly offsets it');ax[0,0].legend(fontsize=8)
    r=[r for r in report['bands'] if r['model']=='mean_of_models' and r['mu']==2]
    y={s:np.array([v['H'] for v in r if v['scale']==s]) for s in SCALES}
    f=(np.arange(16)+.5)/32
    ax[0,1].bar(f,(y[2]-y[6])*1e6,width=.027,color='#4b8c7b')
    ax[0,1].axvspan(.1875,.375,color='#4b8c7b',alpha=.12);ax[0,1].set(xlabel='Output frequency r',ylabel='H decrease x 1e6',title='B. 70.9% of H decrease at r = 0.1875-0.375')
    ranks=[r for r in report['image_ranking'] if r['group']=='all' and r['start']==2 and r['end']==6 and r['mu']==2]
    ax[0,2].barh([r['image'] for r in ranks][::-1],[r['signed_share_of_delta_H']*100 for r in ranks][::-1],color=['#bd6e64' if r['delta_H']<0 else '#4c7995' for r in ranks][::-1]);ax[0,2].axvline(0,lw=.6,color='#333333');ax[0,2].set(xlabel='Signed share of H decrease (%)',title='C. Contributions depend strongly on image')
    pairs=[r for r in report['pairs'] if r['group']=='all' and r['model']=='mean_of_models' and r['mu']==2 and (r['start'],r['end']) in [(2,3),(3,4),(4,6),(2,6)]]
    for i,row in enumerate(pairs):
        for key,color,off in [('epsilon_drop_U','#4578a0',-.17),('epsilon_drop_kappa','#bd9c41',.17)]:
            ax[1,0].bar(i+off,row[key]*1e6,width=.32,color=color,label={'epsilon_drop_U':'Native-output energy U','epsilon_drop_kappa':'Residual fraction kappa'}[key] if i==0 else None)
    ax[1,0].axhline(0,lw=.5,color='#555555');ax[1,0].set(xticks=range(len(pairs)),xticklabels=[f"{r['start']} to {r['end']}" for r in pairs],ylabel='Residual-energy decrease x 1e6',title='D. Residual fraction and available energy');ax[1,0].legend(fontsize=8)
    for model in MODELS:
        rr=[r for r in report['scales'] if r['model']==model and r['group']=='all' and r['mu']==2]
        ax[1,1].plot(SCALES,[r['rho'] for r in rr],'o-',label=model)
    ax[1,1].set(xticks=SCALES,xlabel='Scale',ylabel='Error-residual alignment rho',title='E. Alignment is weak, positive, and non-monotone');ax[1,1].legend(fontsize=8)
    loo=[r for r in report['leave_one_image_out'] if r['mu']==2]
    ax[1,2].barh([r['omitted_image'] for r in loo],[r['H_share_of_D']*100 for r in loo],color='#8786a3');ax[1,2].set(xlabel='H / measured cost decrease (%)',title='F. Refit after leaving one image out')
    fig.suptitle('Set14 saved predictions: structure of the non-scalar remainder (cell doubled)',fontsize=14)
    fig.supxlabel('Retrospective accounting; 14 center crops, 3 models, 2 fixed seeds; no new training or inference. 0 < r < 0.5.',fontsize=9)
    fig.savefig(out/'residual-localization.png',dpi=180);fig.savefig(out/'residual-localization.pdf');plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(11.5,8),layout='constrained')
    colors=['#4578a0','#bd9c41','#bb4940']
    for mu,color in zip(MUS,colors):
        rr=[r for r in report['scales'] if r['model']=='mean_of_models' and r['group']=='all' and r['mu']==mu]
        ax[0,0].plot(SCALES,[r['epsilon_energy']*1e6 for r in rr],'o-',color=color,label=f'm={mu:g}')
        ax[0,1].plot(SCALES,[r['epsilon_per_h2']*1e6 for r in rr],'o-',color=color,label=f'm={mu:g}')
    ax[0,0].set(xlabel='Scale',ylabel='Residual energy x 1e6',xticks=SCALES,title='A. Fixed cell multiplier: raw residual declines');ax[0,0].legend()
    ax[0,1].set(xlabel='Scale',ylabel='Residual energy / h squared x 1e6',xticks=SCALES,title='B. Per-step response does not decline uniformly');ax[0,1].legend()
    for key,label,color in [('Jtotal_per_h2','Total output change / h squared','#4578a0'),('epsilon_per_h2','Non-scalar energy / h squared','#bb4940'),('H_per_h2','Net remainder H / h squared','#bd9c41')]:
        ax[1,0].plot(SCALES,[r[key]*1e6 for r in rows],'o-',label=label,color=color)
    ax[1,0].set(xticks=SCALES,xlabel='Scale',ylabel='Normalized response x 1e6',title='C. Cell doubled: response and net error differ');ax[1,0].legend(fontsize=9)
    pp=[r for r in report['pairs'] if r['group']=='all' and r['model']=='mean_of_models' and r['start']==2 and r['end']==6]
    for i,row in enumerate(pp):
        for key,color,off in [('epsilon_drop_h2','#4578a0',-.17),('epsilon_drop_normalized_response','#bb4940',.17)]:
            ax[1,1].bar(i+off,row[key]*1e6,width=.32,color=color,label={'epsilon_drop_h2':'Step squared factor','epsilon_drop_normalized_response':'Normalized response factor'}[key] if i==0 else None)
    ax[1,1].axhline(0,color='#555555',lw=.6);ax[1,1].set(xticks=range(3),xticklabels=[f"m={r['mu']:g}" for r in pp],ylabel='Residual-energy decrease x 1e6',title='D. Scale 2 to 6: symmetric endpoint accounting');ax[1,1].legend(fontsize=9)
    fig.suptitle('Fixed multiplier also changes the step in the actual cell input',fontsize=14)
    fig.supxlabel('h = 2(m-1)/s per axis. Finite-difference normalization is not a matched-step experiment or a Jacobian estimate.',fontsize=9)
    fig.savefig(out/'finite-difference-normalization.png',dpi=180);fig.savefig(out/'finite-difference-normalization.pdf');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();base=Path(__file__).resolve().parents[1]/'15-energy-filter-decomposition'/'inputs'
    ap.add_argument('--saved',type=Path,default=base/'set14'/'stats.json');ap.add_argument('--v11',type=Path,default=base/'transfer_v11.json')
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'localization-v2');ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args();args.out.mkdir(exist_ok=False);started=time.perf_counter()
    x,dc,names,metadata=load_saved(args.saved)
    core=x[...,:16,:];xm=core.mean(axis=2);m=measure(xm.mean(axis=0))
    rows,ranks,concentration=image_allocations(core,names,m)
    report=dict(metadata=dict(scope='CPU saved-statistics analysis only',training_runs=0,inference_runs=0,
        model_pooling='fit per model, pool 14 images and two fixed seeds; average derived model quantities',
        image_allocation='pooled w held fixed; include -2(1-w)<P,epsilon> per image; only pooled cross vanishes',
        cell_input_coordinate='rel_cell=cell*H_LR, native=2/s; fixed multiplier mu has h=2*(mu-1)/s per axis',
        h_normalization='finite difference along direction (1,1), h is per-axis coordinate increment, not Euclidean displacement; not a Jacobian or controlled-step experiment',
        decomposition_caveat='U-kappa shares depend on fixed frequency partition; h2-response split is a different algebraic accounting, not additive to U-kappa',
        primary_frequency='DC excluded, 0<r<0.5, r=max(abs(fx),abs(fy)); 16 bins of width 1/32',
        uncertainty='paired image cluster, not training-seed population uncertainty',
        interpretation='descriptive endpoint accounting; no causal or held-out prediction claim',
        extraction_metadata=metadata),
        scales=scalar_rows(m,MODELS),pairs=pair_rows(m,MODELS),bands=band_rows(m,MODELS),
        image_allocations=rows,image_ranking=ranks,image_concentration=concentration,
        leave_one_image_out=loo(xm,names),bootstrap=bootstrap(xm,args.bootstrap),DIV2K=load_div2k(args.v11))
    # Fit each fixed trained model/seed separately, keeping both as descriptive instances.
    ms=measure(core.mean(axis=0).reshape(6,4,3,16,8))
    labels=[f'{model}_seed{seed}' for model in MODELS for seed in [1,2]]
    report['model_seed_scales']=scalar_rows(ms,labels);report['model_seed_pairs']=pair_rows(ms,labels)
    # Alternate within-image fit, then average derived quantities, clearly separated.
    own=measure(xm)
    report['per_image_refit_scales']=scalar_rows({k:v.mean(axis=0) for k,v in own.items()},MODELS)
    # Frequency completeness sensitivity: append Nyquist and DC as separate scalar groups.
    allx=np.concatenate([x,dc[...,None,:]],axis=-2).mean(axis=(0,2))
    allm=measure(allx)
    report['frequency_completeness']=[]
    for label,a,b in [('nonDC_core',0,16),('nyquist',16,17),('DC',17,18),('full',0,18)]:
        g=grouped(allm,a,b)
        for si,s in enumerate(SCALES):
            for mi,mu in enumerate(MUS):
                report['frequency_completeness'].append(dict(group=label,scale=s,mu=mu,
                    **{k:float(g[k][:,si,mi].mean()) for k in SUM_KEYS}))
    report['checks']=CHECKS
    report['manifest']=dict(inputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [args.saved,args.v11]],
        script=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),
        python=sys.version,numpy=np.__version__,elapsed_seconds=time.perf_counter()-started)
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for key in ['scales','pairs','bands','image_allocations','image_ranking','leave_one_image_out','model_seed_scales','model_seed_pairs','frequency_completeness']:
        save_csv(args.out/f'{key}.csv',report[key])
    save_csv(args.out/'div2k-bands.csv',report['DIV2K']['bands'])
    figures(report,args.out)
    print(json.dumps(dict(output=str(args.out),checks=CHECKS,elapsed_seconds=time.perf_counter()-started),indent=2))


if __name__=='__main__':main()
