"""Independent NumPy audit of frozen sinc path intervention; no model inference.

CPU1, nice10, fixed 120s wall/cumulative limit. Never overwrites an audit.
Saved 16-query branch probes are independently decoded; all pixels are rescored.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
LIMIT = 120.0
SIGN = ('positive', 'negative', 'boundary')
MODES = ('p00', 'p01', 'p10', 'p11', 'uniform1', 'permuted1')
MAXERR = {}
COUNTS = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def close(actual, expected, name, tol=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    MAXERR[name] = max(MAXERR.get(name, 0.0), error)
    COUNTS[name] = COUNTS.get(name, 0) + 1
    assert error <= tol, (name, error, tol)


def compare(actual, expected, name):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (name, set(actual) ^ set(expected))
        for k, v in expected.items():
            compare(actual[k], v, name + '/' + k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), name
        for a, b in zip(actual, expected):
            compare(a, b, name)
    elif isinstance(expected, bool) or isinstance(expected, (str, int)):
        assert actual == expected, (name, actual, expected)
    elif isinstance(expected, float):
        close(actual, expected, name.split('/')[0] + '_float')
    else:
        assert actual == expected, name


def classify(value, threshold):
    if abs(value) <= threshold:
        return 'boundary'
    return 'positive' if value > 0 else 'negative'


def components(array, box):
    lo, hi = box
    x = np.array(array[lo:hi, lo:hi], dtype=np.float64)
    if x.ndim == 2:
        x = np.stack((x, x, x), axis=2)
    n = hi - lo
    freq = np.arange(n, dtype=float) / n
    freq[freq >= .5] -= 1
    radius = np.maximum(np.abs(freq[:, None]), np.abs(freq[None, :]))
    spectrum = np.fft.fft2(x, axes=(0, 1))
    result = [x]
    for left, right in ((0, .125), (.125, .25), (.25, .5001)):
        keep = (radius != 0) & (radius >= left) & (radius < right)
        filtered = spectrum.copy()
        filtered[~keep] = 0
        result.append(np.fft.ifft2(filtered, axes=(0, 1)).real)
    return result


def dense_numpy(z, layers):
    h = z.copy()
    for i, (weight, bias) in enumerate(layers):
        h = h @ weight.T + bias
        if i < len(layers)-1:
            h = np.maximum(h, 0.)
    return h


def metrics(error, step):
    a = float(np.mean(error * step))
    b = float(np.mean(step * step))
    g = 2*a-b
    reference_mse = float(np.mean(error*error))
    actual_mse = float(np.mean((error-step)**2))
    close(reference_mse-actual_mse, g, 'direct_mse_gain')
    return dict(a=a, b=b, g=g, mse=actual_mse, reference_mse=reference_mse,
                direction=classify(a, 1e-10), gain_sign=classify(g, 1e-8))


def execute(args, check):
    root, source = args.run.resolve(), args.source.resolve()
    protocol=read(HERE/'protocol.json'); freeze=read(root/'input_freeze.json')
    for key, filename in [('source_sha256','run.py'),('protocol_sha256','protocol.json'),('cohort_sha256','cohort.json')]:
        assert freeze[key]==digest(HERE/filename),key
    assert protocol['cohort_sha256']==digest(HERE/'cohort.json')
    assert freeze['old_manifests']==protocol['old_source_manifests']
    cohort=read(HERE/'cohort.json')
    if freeze['preflight']:cohort=[c for c in cohort if c['family'] in protocol['preflight_families']]
    assert cohort==read(root/'cohort.json')
    assert digest(root/'cohort.json')==freeze['run_cohort_sha256']
    obs={c[k] for c in cohort for k in ['eta0','eta1']};families={c['family'] for c in cohort}
    for f,h in protocol['old_source_manifests'].items():
        check();assert digest(source/f)==h,f
    for manifest in ['input_manifest.json','prediction_cohort.json']:
        for rel,h in read(source/manifest).items():
            p=Path(rel)
            selected=(p.parent.name=='observations' and p.stem in obs) or (p.parent.name=='truth' and p.stem in families) or (p.parent.name=='predictions' and p.stem.endswith('__sc-inr') and p.stem.removesuffix('__sc-inr') in obs)
            if selected:check();assert digest(source/rel)==h,rel
    manifest=read(root/'prediction_cohort.json')
    files={str(p.relative_to(root)) for d in ['predictions','probes'] for p in (root/d).iterdir()}
    assert set(manifest)==files
    assert {p.stem for p in (root/'predictions').iterdir()}==families
    for rel,h in manifest.items():check();assert digest(root/rel)==h,rel
    pf=read(root/'prediction_freeze.json'); complete=read(root/'COMPLETE.json')
    assert pf['manifest_sha256']==digest(root/'prediction_cohort.json')
    for k,f in [('rows_sha256','rows.json'),('summary_sha256','summary.json')]:assert complete[k]==digest(root/f)
    launch=read(root/'launch.json');res=read(root/'resources.json')
    assert launch['start_epoch']<=freeze['epoch']<=pf['epoch']<=complete['epoch']<=res['end_epoch']<=launch['deadline_epoch']
    budget=protocol['budget']['preflight_seconds' if freeze['preflight'] else 'formal_seconds']
    assert launch['deadline_epoch']-launch['start_epoch']<=budget+1e-6
    assert res['wall_seconds']<=budget and res['script_seconds']<=budget
    assert res['status']=='complete' and res['CPU_threads']==1 and res['nice']==10 and res['GPU']==res['training']==0
    journal=read(root/'inference_journal.json');assert len(journal)==len(cohort)
    jj={r['family']:r for r in journal};assert set(jj)==families
    oj=read(root/'observation_journal.json');assert {x['observation'] for x in oj}==obs
    for r in oj:
        assert r['endpoint_float32_maxdiff']<=2e-6 and r['feature_product_maxdiff']<=2e-6
    layers=[]
    with np.load(root/'probes/weights.npz') as w:
        layers=[(w[f'w{i}'],w[f'b{i}']) for i in range(4)]
    saved=read(root/'rows.json');lookup={(r['family'],r['roi'],r['band']):r for r in saved}
    assert len(lookup)==len(saved)==len(cohort)*8
    outrows=[]; idx=protocol['numeric']['probe_query_indices'];perm=np.random.default_rng(20261018).permutation(128)
    modes=['native0','native1','swap10','swap01','uniform1','permuted1']
    for c in cohort:
        check();family=c['family'];j=jj[family];assert j['status']=='complete' and j['rebuild_double_maxdiff']<=2e-6
        close(j['uniform_feature_rms'],j['target_feature_rms'],'recorded_uniform_norm')
        close(j['permuted_feature_rms'],j['target_feature_rms'],'recorded_permuted_norm')
        close(j['weighted_attenuation_change'],j['weighted_attenuation1']-j['weighted_attenuation0'],'recorded_attenuation')
        with np.load(root/'predictions'/(family+'.npz')) as f:z={k:f[k] for k in f.files}
        assert all(v.shape==(64,64,3) and np.isfinite(v).all() for v in z.values())
        for eta in [0,1]:
            with np.load(source/'predictions'/(c[f'eta{eta}']+'__sc-inr.npz')) as old:
                for m in ['point','native']:
                    close(z[f'old_{m}{eta}'],old[m],'original_endpoint',2e-6)
                    close(z[f'{m}{eta}'],old[m],'double_endpoint',2e-6)
        with np.load(root/'probes'/(family+'.npz')) as f:p={k:f[k] for k in f.files}
        assert np.array_equal(p['indices'],idx) and np.array_equal(p['permutation'],perm)
        U0,U1,W0,W1=(p[k] for k in ['U0','U1','W0','W1']); assert all(v.shape==(4,16,256) for v in [U0,U1,W0,W1])
        close(p['rel_cell'],np.array([.5,.5]),'relative_cell')
        for eta,W in [(0,W0),(1,W1)]:
            omega=p[f'omega{eta}'];assert omega.shape==(4,16,128,2)
            sinc=np.prod(np.sinc(omega.astype(np.float64)*p['rel_cell'].astype(np.float64)/2),axis=-1)
            close(W,np.concatenate([sinc,sinc],axis=-1),'float32_sinc_reconstruction',2e-6)
        weights=p['weights'];assert weights.shape==(4,16,1) and (weights>=0).all()
        close(weights.sum(0),np.ones((16,1)),'branch_weight_sum',1e-7)
        for k in ['uniform_t','permuted_gamma','target_feature_rms','U1_rms']:close(p[k],j[k],'probe_journal_'+k)
        close(p['uniform_t']*p['U1_rms'],p['target_feature_rms'],'uniform_factor')
        close(p['permuted_gamma']*p['permuted_unscaled_rms'],p['target_feature_rms'],'permuted_factor')
        uniform=np.full_like(W0,1-float(p['uniform_t']))
        paired=np.r_[perm,perm+128];permuted=1-float(p['permuted_gamma'])*(1-W0[:,:,paired])
        defs={'point0':(U0,None,p['skip0']),'point1':(U1,None,p['skip1']),'native0':(U0,W0,p['skip0']),'native1':(U1,W1,p['skip1']),'swap10':(U1,W0,p['skip1']),'swap01':(U0,W1,p['skip0']),'uniform1':(U1,uniform,p['skip1']),'permuted1':(U1,permuted,p['skip1'])}
        for name,(u,w,skip) in defs.items():
            reconstructed=skip*.5+.5
            for b in range(4):reconstructed=reconstructed+.5*weights[b]*dense_numpy(u[b] if w is None else u[b]*w[b],layers)
            close(reconstructed,z[name].reshape(-1,3)[idx],'numpy_probe_'+name,1e-10)
        if c['template']=='control_low':
            close(W0,W1,'sham_envelope',1e-10);close(z['swap10'],z['native1'],'sham_forward',1e-10);close(z['swap01'],z['native0'],'sham_reverse',1e-10)
        with np.load(source/'truth'/(family+'.npz')) as f:truth=f['target']
        for roi,box in [('center',(16,48)),('full',(0,64))]:
            pp={k:components(v,box) for k,v in z.items()};tt=components(truth,box)
            for band,T in enumerate(tt):
                r=dict(c,roi=roi,band=band);errors=[T-pp[f'point{i}'][band] for i in [0,1]];steps=[pp[f'native{i}'][band]-pp[f'point{i}'][band] for i in [0,1]]
                for eta,e in enumerate(errors):r[f'reference_mse{eta}']=float(np.mean(e**2))
                for name in modes:
                    eta=0 if name in ['native0','swap01'] else 1
                    m=metrics(errors[eta],pp[name][band]-pp[f'point{eta}'][band])
                    for k in ['a','b','g','mse']:r[f'{k}_{name}']=m[k]
                for e in [0,1]:
                    for d in [0,1]:r[f'cross_e{e}_d{d}']=float(np.mean(errors[e]*steps[d]))
                r['demand_swapped_mse']=float(np.mean((errors[1]-steps[0])**2));r['demand_swapped_gain']=r['reference_mse1']-r['demand_swapped_mse']
                r['alignment_change']=r['a_native1']-r['a_native0'];r['alignment_change_response']=r['cross_e0_d1']-r['cross_e0_d0'];r['alignment_change_demand']=r['cross_e1_d1']-r['cross_e0_d1']
                close(r['alignment_change'],r['alignment_change_response']+r['alignment_change_demand'],'alignment_identity')
                r['target_improvement']=r['mse_native1']-r['mse_swap10'];r['target_over_uniform']=r['mse_uniform1']-r['mse_swap10'];r['target_over_permuted']=r['mse_permuted1']-r['mse_swap10'];r['reverse_gain_decrease']=r['g_native0']-r['g_swap01'];r['rescued']=r['g_native1']< -1e-8 and r['g_swap10']>=-1e-8
                for eta in [0,1]:
                    e=T-pp[f'old_point{eta}'][band];d=pp[f'old_native{eta}'][band]-pp[f'old_point{eta}'][band];m=metrics(e,d)
                    r[f'old_a{eta}']=m['a'];r[f'old_g{eta}']=m['g'];r[f'sensitive_direction{eta}']=classify(m['a'],1e-10)!=classify(r[f'a_native{eta}'],1e-10);r[f'sensitive_gain{eta}']=classify(m['g'],1e-8)!=classify(r[f'g_native{eta}'],1e-8)
                compare(lookup[(family,roi,band)],r,'rows');outrows.append(r)
    summary=read(root/'summary.json')
    assert summary['rows']==len(outrows) and summary['families']==len(cohort) and summary['GPU']==summary['training']==0
    assert summary['limitations']==protocol['limitations'] and summary['scope']==protocol['cohort_scope']
    active={'all32':any(c['template'].startswith('mid_') for c in cohort),'focal16':any(c['focal'] for c in cohort),'nonflip16':any(c['template'].startswith('mid_') and not c['focal'] for c in cohort),'sham16':any(c['template']=='control_low' for c in cohort)}
    assert set(summary['groups'])=={f'{g}/{roi}/b{b}' for g,v in active.items() if v for roi in ['center','full'] for b in range(4)}
    for key,g in summary['groups'].items():
        group,roi,b=key.split('/');band=int(b[1:]); selected=[]
        for r in outrows:
            accept={'all32':r['template'].startswith('mid_'),'focal16':r['focal'],'nonflip16':r['template'].startswith('mid_') and not r['focal'],'sham16':r['template']=='control_low'}[group]
            if accept and r['roi']==roi and r['band']==band:selected.append(r)
        assert len(selected)==g['n']
        for k,predicate in [('rescued',lambda r:r['rescued']),('negative_native1',lambda r:r['g_native1']< -1e-8),('reverse_decreased',lambda r:r['reverse_gain_decrease']>1e-8),('sensitive_direction',lambda r:r['sensitive_direction0'] or r['sensitive_direction1']),('sensitive_gain',lambda r:r['sensitive_gain0'] or r['sensitive_gain1'])]:assert g[k]==sum(predicate(r) for r in selected)
        for agg,fn in [('mean',np.mean),('median',np.median)]:
            for k,v in g[agg].items():close(v,float(fn([r[k] for r in selected])),'summary_'+agg)
    focal=[r for r in outrows if r['focal'] and r['roi']=='center' and r['band']==2]
    gates={};premise=None
    if len(focal)==16:
        med=lambda k:float(np.median([r[k] for r in focal]))
        rgb=[r for r in outrows if r['focal'] and r['roi']=='center' and r['band']==0]
        gates=dict(rescue_majority=sum(r['rescued'] for r in focal)>8,targeted_improvement=med('target_improvement')>1e-8,beats_uniform=med('target_over_uniform')>1e-8,beats_permuted=med('target_over_permuted')>1e-8,reverse_median=med('reverse_gain_decrease')>1e-8,reverse_majority=sum(r['reverse_gain_decrease']>1e-8 for r in focal)>8,fullRGB_not_worse=float(np.median([r['target_improvement'] for r in rgb]))>=-1e-8)
        premise=float(np.median([jj[r['family']]['weighted_attenuation_change'] for r in focal]))>1e-10
    assert summary['gates']==gates and summary['major_envelope_explanation_supported']==(all(gates.values()) if gates else None) and summary['stronger_attenuation_premise']==premise
    return dict(run=str(root),families=len(cohort),rows=len(outrows),gates=gates,major_envelope_explanation_supported=summary['major_envelope_explanation_supported'],probe_queries_per_family=16,probe_branches=4,scope='Independent NumPy sinc envelope and MLP reconstruction at saved16 queries; complete pixel scoring. Full feature norm factors/attenuation validated from journals and static code, not independently reconstructed over all features. Remote inherited source bytes represented by frozen manifest; no independent local checkpoint replay.',source_sha256=freeze['source_sha256'],protocol_sha256=freeze['protocol_sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    output = args.out or HERE / (args.run.name + '-independent-audit.json')
    assert not output.exists(), 'Never overwrite an audit'
    started_wall, started_mono = time.time(), time.monotonic()
    def expired(*unused):
        raise TimeoutError('Independent audit fixed 120s absolute/cumulative budget exhausted')
    def check():
        if time.time() >= started_wall + LIMIT or time.monotonic() - started_mono >= LIMIT:
            expired()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    report = dict(status='running', start_epoch=started_wall, deadline_epoch=started_wall + LIMIT,
                  budget_seconds=LIMIT, audit_sha256=digest(__file__), CPU_threads=1, GPU=0, training=0)
    save(output, report)
    try:
        assert os.getpriority(os.PRIO_PROCESS, 0) == 10
        report.update(execute(args, check))
        check()
        report['status'] = 'passed'
    except BaseException:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        u = resource.getrusage(resource.RUSAGE_SELF)
        report.update(end_epoch=time.time(), wall_seconds=time.time() - started_wall,
                      script_seconds=time.monotonic() - started_mono,
                      cpu_seconds=u.ru_utime + u.ru_stime,
                      peak_rss_native=u.ru_maxrss, nice=os.getpriority(os.PRIO_PROCESS, 0),
                      check_counts=COUNTS, maximum_errors=MAXERR)
        save(output, report)
        print(json.dumps({k: report.get(k) for k in ('status', 'wall_seconds', 'rows', 'eta_pairs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
