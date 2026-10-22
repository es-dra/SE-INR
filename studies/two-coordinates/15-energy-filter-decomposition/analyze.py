"""Offline, retrospective decomposition; never imports a model or reads weights."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np

MODELS = ['liif', 'lte', 'sc-inr']
SCALES = [2, 3, 4, 6]
MUS = [0.5, 1.3333, 2.0]
PAIRS = [(2, 3), (3, 4), (4, 6), (2, 4), (3, 6), (2, 6)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(x):
    """Input (..., bands, statistics), scalar results have shape (...)."""
    C, U, E, V, K, D = [x[..., i] for i in range(6)]
    assert np.all(U > 0)
    B = E - V
    w = K / U
    a = (1 - w) ** 2
    Q = np.sum(a * B, axis=-1)
    G = np.sum(2 * w * (1 - w) * C, axis=-1)
    actual = D.sum(axis=-1)
    H = actual - Q - G
    out = dict(Q=Q, G=G, H=H, D=actual, V=V.sum(axis=-1), B=B, w=w, a=a)
    out['cost_db'] = 10 * np.log10(1 + actual / out['V'])
    if x.shape[-1] >= 8:
        Z, TPm = x[..., 6], x[..., 7]
        eps_energy = Z - K * K / U
        e_eps = TPm - K - w * C
        out['epsilon_energy'] = eps_energy.sum(axis=-1)
        out['epsilon_error_cross'] = (-2 * e_eps).sum(axis=-1)
        np.testing.assert_allclose(H, out['epsilon_energy'] + out['epsilon_error_cross'], rtol=1e-7, atol=2e-13)
    return out


def pair_terms(m, s, t):
    """Scale is axis -3 for band arrays, -2 for summed arrays."""
    i, j = SCALES.index(s), SCALES.index(t)
    bs, bt = m['B'][..., i, :, :], m['B'][..., j, :, :]
    aa, ab = m['a'][..., i, :, :], m['a'][..., j, :, :]
    fr = np.sum((bs - bt) * (aa + ab) / 2, axis=-1)
    fw = np.sum((aa - ab) * (bs + bt) / 2, axis=-1)
    dg = m['G'][..., i, :] - m['G'][..., j, :]
    dh = m['H'][..., i, :] - m['H'][..., j, :]
    dd = m['D'][..., i, :] - m['D'][..., j, :]
    dq = m['Q'][..., i, :] - m['Q'][..., j, :]
    np.testing.assert_allclose(fr + fw, dq, rtol=1e-10, atol=2e-15)
    np.testing.assert_allclose(fr + fw + dg + dh, dd, rtol=1e-10, atol=2e-15)
    return np.stack([fr, fw, dg, dh, dd, dq], axis=-1)


def describe(x, name):
    # x: models, scales, multipliers, bands, stats; already per-image RGB MSE.
    assert np.isfinite(x).all()
    C, U, E, V = [x[..., i] for i in range(4)]
    b_error = float(np.max(np.abs(E - V - U - 2 * C)))
    e_error = float(np.max(np.abs(E - E[:, :1, :1, :])))
    assert b_error < 1e-12 and e_error < 1e-12
    m = metrics(x)
    result = dict(name=name, checks={'B_identity_max_abs': b_error, 'fixed_E_max_abs': e_error}, scales=[], pairs=[], db_pairs=[])
    for mo in range(len(MODELS) + 1):
        label = MODELS[mo] if mo < len(MODELS) else 'mean_of_models'
        def select(z):
            return z[mo] if mo < len(MODELS) else z.mean(axis=0)
        for si, s in enumerate(SCALES):
            for mi, mu in enumerate(MUS):
                row = dict(model=label, scale=s, mu=mu)
                for k in ['Q', 'G', 'H', 'D', 'V', 'cost_db', 'epsilon_energy', 'epsilon_error_cross']:
                    if k in m:
                        row[k] = float(select(m[k])[si, mi])
                row['Q_over_D'] = row['Q'] / row['D'] if row['D'] else None
                row['scalar_exact_over_D'] = (row['Q'] + row['G']) / row['D'] if row['D'] else None
                result['scales'].append(row)
        for s, t in PAIRS:
            vals = select(pair_terms(m, s, t))
            si, ti = SCALES.index(s), SCALES.index(t)
            ds, dt, vs, vt = m['D'][:,si,:], m['D'][:,ti,:], m['V'][:,si,:], m['V'][:,ti,:]
            f = lambda d, v: 10*np.log10(1+d/v)
            numerator = select((f(ds,vs)-f(dt,vs)+f(ds,vt)-f(dt,vt))/2)
            denominator = select((f(ds,vs)-f(ds,vt)+f(dt,vs)-f(dt,vt))/2)
            measured_db_drop = select(f(ds,vs)-f(dt,vt))
            np.testing.assert_allclose(numerator+denominator, measured_db_drop, atol=2e-14)
            for mi, mu in enumerate(MUS):
                fr, fw, dg, dh, dd, dq = map(float, vals[mi])
                result['pairs'].append(dict(model=label, start=s, end=t, mu=mu,
                    R=fr, w=fw, G=dg, H=dh, delta_D=dd, delta_Q=dq,
                    shares_of_actual=[v/dd for v in (fr, fw, dg, dh)] if abs(dd)>1e-15 else None,
                    shares_of_proxy=[fr/dq, fw/dq] if abs(dq)>1e-15 else None))
                result['db_pairs'].append(dict(model=label,start=s,end=t,mu=mu,
                    cell_cost_numerator=float(numerator[mi]),baseline_MSE_denominator=float(denominator[mi]),
                    measured_db_drop=float(measured_db_drop[mi])))
    result['band_diagnostics'] = dict(
        negative_B_count=int(np.sum(m['B'] < 0)), total_B_values=int(m['B'].size),
        w_above_one_count=int(np.sum(m['w'] > 1)), w_below_zero_count=int(np.sum(m['w'] < 0)),
        w_min=float(m['w'].min()), w_max=float(m['w'].max()),
        w_mean_models=m['w'].mean(axis=0).tolist(),
        recovery_mean_models=(m['B']/E).mean(axis=0).tolist(),
        negative_B_contribution_Q=float(np.sum(m['a'] * np.minimum(m['B'], 0))))
    return result


def load_v11(path):
    raw = {ast.literal_eval(k): v for k, v in json.loads(path.read_text()).items()}
    assert len(raw) == 3*4*3*16
    x = np.zeros((3, 4, 3, 16, 6))
    for (mo, s, mu, b), v in raw.items():
        x[MODELS.index(mo), SCALES.index(s), MUS.index(float(mu)), b] = v
    return x / (32 * 3 * 232**4)


def load_saved(path):
    raw = json.loads(path.read_text())
    names = sorted({r['image'] for r in raw['records']})
    seeds = sorted({r['seed'] for r in raw['records']})
    x = np.zeros((len(names), 3, len(seeds), 4, 3, 17, 8))
    dc = np.zeros((len(names), 3, len(seeds), 4, 3, 8))
    seen = set()
    for r in raw['records']:
        idx = (names.index(r['image']), MODELS.index(r['model']), seeds.index(r['seed']), SCALES.index(r['scale']), MUS.index(float(r['mu'])))
        assert idx not in seen
        seen.add(idx)
        x[idx] = r['bins']
        dc[idx] = r['dc']
    assert len(seen) == len(names)*3*len(seeds)*4*3
    assert len(names) == 14 and seeds == [1, 2]
    # Preserve paired images, average the two fixed training seeds.
    return x.mean(axis=2), dc.mean(axis=2), names


def bootstrap(x, n):
    rng = np.random.default_rng(20261023)
    vals = {p: [] for p in PAIRS}
    for begin in range(0, n, 100):
        k = min(100, n-begin)
        indices = rng.integers(0, len(x), size=(k, len(x)))
        weights = np.stack([np.bincount(row, minlength=len(x)) for row in indices]) / len(x)
        pooled = np.tensordot(weights, x, axes=(1, 0))
        m = metrics(pooled)
        for pair in PAIRS:
            vals[pair].append(pair_terms(m, *pair).mean(axis=1))
    output = []
    for pair in PAIRS:
        data = np.concatenate(vals[pair], axis=0)
        for mi, mu in enumerate(MUS):
            d = data[:, mi, :]
            shares = d[:, :4] / d[:, 4:5]
            proxy_shares = d[:, :2] / d[:, 5:6]
            output.append(dict(start=pair[0], end=pair[1], mu=mu,
                terms_ci95=np.quantile(d, [.025, .975], axis=0).T.tolist(),
                shares_actual_ci95=np.quantile(shares, [.025, .975], axis=0).T.tolist(),
                R_plus_w_share_actual_ci95=np.quantile(shares[:,:2].sum(axis=1),[.025,.975]).tolist(),
                shares_proxy_ci95=np.quantile(proxy_shares, [.025, .975], axis=0).T.tolist(),
                delta_D_nonpositive_fraction=float(np.mean(d[:, 4]<=0)),
                delta_Q_nonpositive_fraction=float(np.mean(d[:, 5]<=0))))
    return dict(n=n, seed=20261023, unit='paired image cluster; fixed trained seeds', pairs=output)


def per_image_fit_sensitivity(x):
    """Refit w separately per image as a sensitivity check, not the paper's estimator."""
    m=metrics(x)
    rows=[]
    for si,s in enumerate(SCALES):
        for mi,mu in enumerate(MUS):
            row=dict(scale=s,mu=mu)
            for k in ['D','Q','G','H','epsilon_energy','epsilon_error_cross']:
                row[k]=float(m[k][...,si,mi].mean())
            row['Q_over_D']=row['Q']/row['D']
            rows.append(row)
    pairs=[]
    for s,t in PAIRS:
        vals=pair_terms(m,s,t).mean(axis=(0,1))
        for mi,mu in enumerate(MUS):
            v=vals[mi]
            pairs.append(dict(start=s,end=t,mu=mu,terms=v.tolist(),shares_actual=(v[:4]/v[4]).tolist()))
    return dict(estimator='fit w per image and model after averaging two fixed seeds; average derived quantities',scales=rows,pairs=pairs)


def figures(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.8), layout='constrained')
    d = report['DIV2K']
    w = np.array(d['band_diagnostics']['w_mean_models'])
    for si, s in enumerate(SCALES):
        ax[0,0].plot((np.arange(16)+.5)/32, w[si,2], label=f'x{s}')
    ax[0,0].set(xlabel='Output frequency r (cycles / HR pixel)', ylabel='Fitted transfer w', title='A. Filter strength changes with scale')
    ax[0,0].legend(ncol=4, fontsize=9)
    rows = [r for r in d['scales'] if r['model']=='mean_of_models' and r['mu']==2]
    ax[0,1].plot(SCALES, [r['D']*1e6 for r in rows], 'o-', color='#222222', label='Measured cost D')
    ax[0,1].plot(SCALES, [r['Q']*1e6 for r in rows], 's-', color='#2878a0', label='Recovery-filter proxy Q')
    for s, r in zip(SCALES, rows):
        ax[0,1].annotate(f"Q/D={r['Q_over_D']:.2f}", (s,r['D']*1e6), xytext=(3,8), textcoords='offset points', fontsize=9)
    ax[0,1].set(xlabel='Upsampling scale', ylabel='RGB MSE cost (x 1e-6)', title='B. Proxy coverage depends on scale', xticks=SCALES)
    ax[0,1].legend()
    colors = ['#248277','#527ca6','#d9a441','#c06866']
    labels = ['Recovery R','Filter w','Native-error correction G','Non-scalar remainder H']
    pairs = [(2,3),(3,4),(4,6),(2,6)]
    for ii, pair in enumerate(pairs):
        r = next(r for r in d['pairs'] if r['model']=='mean_of_models' and r['mu']==2 and (r['start'],r['end'])==pair)
        pos=neg=0
        for j, v in enumerate(r['shares_of_actual']):
            v*=100
            base=pos if v>=0 else neg
            ax[1,0].bar(ii,v,bottom=base,color=colors[j],label=labels[j] if ii==0 else None)
            if v>=0:pos+=v
            else:neg+=v
    ax[1,0].axhline(100,color='#777777',ls=':',lw=1)
    ax[1,0].set(xticks=range(4),xticklabels=['2 to 3','3 to 4','4 to 6','2 to 6'],ylabel='Share of measured MSE-cost decrease (%)',title='C. DIV2K: exact accounting of the decrease')
    ax[1,0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.14),ncol=2,frameon=False)
    if 'Set14' in report:
        r=next(r for r in report['Set14']['pairs'] if r['model']=='mean_of_models' and r['mu']==2 and (r['start'],r['end'])==(2,6))
        ci=next(r for r in report['Set14_bootstrap']['pairs'] if r['mu']==2 and (r['start'],r['end'])==(2,6))
        v=np.array(r['shares_of_actual'])*100
        bounds=np.array(ci['shares_actual_ci95'])*100
        ax[1,1].bar(range(4),v,color=colors)
        # Draw percentile interval endpoints faithfully even if they exclude the point estimate.
        ax[1,1].vlines(range(4),bounds[:,0],bounds[:,1],color='#222222')
        ax[1,1].hlines(bounds[:,0],np.arange(4)-.08,np.arange(4)+.08,color='#222222')
        ax[1,1].hlines(bounds[:,1],np.arange(4)-.08,np.arange(4)+.08,color='#222222')
        ax[1,1].set(xticks=range(4),xticklabels=['R','w','G','H'],ylabel='Share of measured decrease (%)',title='D. Set14: x2 to x6, image-bootstrap 95% CI')
    fig.suptitle('Cell doubled (m=2): two-factor explanation and its remainder', fontsize=14)
    fig.supxlabel('0 < r < 0.5; H includes residual energy and its correlation with native error. Shares are descriptive MSE accounting.',fontsize=9)
    fig.savefig(out/'decomposition.png',dpi=180)
    fig.savefig(out/'decomposition.pdf')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v11',type=Path,required=True)
    ap.add_argument('--saved',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--plot',action='store_true')
    args=ap.parse_args()
    args.out.mkdir(exist_ok=False)
    t0=time.perf_counter()
    x=load_v11(args.v11)
    report={'DIV2K':describe(x,'DIV2K v11, 32 images, seed 1, 0<r<0.5')}
    if args.saved:
        saved,dc,names=load_saved(args.saved)
        report['Set14']=describe(saved.mean(axis=0)[...,:16,:],'Set14 saved float16, two fixed seeds, 0<r<0.5')
        report['Set14_all_nonDC']=describe(saved.mean(axis=0),'Set14 saved float16, all non-DC including Nyquist outer ring')
        report['Set14_bootstrap']=bootstrap(saved[...,:16,:],args.bootstrap)
        report['Set14_per_image_fit_sensitivity']=per_image_fit_sensitivity(saved[...,:16,:])
        core=saved[...,:16,5].sum(axis=-1).mean(axis=(0,1))
        edge=saved[...,16,5].mean(axis=(0,1))
        dc_cost=dc[...,5].mean(axis=(0,1))
        report['Set14_scope_audit']={'images':names,'core_D':core.tolist(),'nyquist_D':edge.tolist(),'DC_D':dc_cost.tolist(),
            'full_spatial_D':(core+edge+dc_cost).tolist(),
            'omitted_fraction_of_full':((edge+dc_cost)/(core+edge+dc_cost)).tolist()}
    report['provenance']={'v11_sha256':sha(args.v11),'saved_sha256':sha(args.saved) if args.saved else None,'script_sha256':sha(__file__),
        'seconds':time.perf_counter()-t0,'training_calls':0,'inference_calls':0,'units':'RGB MSE; shares are MSE shares, not dB shares'}
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.plot:figures(report,args.out)
    for cohort in ['DIV2K','Set14']:
        if cohort not in report:continue
        print(cohort)
        for row in report[cohort]['scales']:
            if row['model']=='mean_of_models' and row['mu']==2:
                print('scale',row['scale'],'Q/D',round(row['Q_over_D'],4),'D',round(row['D'],9),'dB',round(row['cost_db'],5))
        for row in report[cohort]['pairs']:
            if row['model']=='mean_of_models' and row['mu']==2:
                print('pair',row['start'],row['end'],'R,w,G,H % actual',np.round(np.array(row['shares_of_actual'])*100,3).tolist(),'R,w % proxy',np.round(np.array(row['shares_of_proxy'])*100,3).tolist())


if __name__=='__main__':main()
