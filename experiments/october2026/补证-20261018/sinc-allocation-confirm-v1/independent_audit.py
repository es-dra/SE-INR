"""Independent NumPy audit of native-norm allocation confirmation; no model inference.

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


def dense_fixed_numpy(u,delta,layers):
    h=u.copy();v=delta.copy()
    for i,(w,b) in enumerate(layers):
        h=h@w.T+b;v=v@w.T
        if i<len(layers)-1:
            mask=h>0;h=np.where(mask,h,0.);v=np.where(mask,v,0.)
    return v


def execute(args,check):
    root,source=args.run.resolve(),args.source.resolve()
    protocol=read(HERE/'protocol.json');freeze=read(root/'input_freeze.json')
    for key,filename in [('source_sha256','run.py'),('protocol_sha256','protocol.json'),('cohort_sha256','cohort.json')]:assert freeze[key]==digest(HERE/filename),key
    assert protocol['cohort_sha256']==digest(HERE/'cohort.json')
    assert freeze['old_manifests']==protocol['old_source_manifests']
    assert freeze['helper_sha256']==protocol['helper_sha256'] and freeze['helper_protocol_sha256']==protocol['helper_protocol_sha256']
    helper=HERE.parent/'sinc-path-patch-v1'
    assert digest(helper/'run.py')==protocol['helper_sha256'] and digest(helper/'protocol.json')==protocol['helper_protocol_sha256']
    cohort=read(HERE/'cohort.json')
    if freeze['preflight']:cohort=[c for c in cohort if c['family'] in protocol['preflight_families']]
    assert cohort==read(root/'cohort.json') and digest(root/'cohort.json')==freeze['run_cohort_sha256']
    families={c['family'] for c in cohort};obs={c['eta1'] for c in cohort}
    for f,h in protocol['old_source_manifests'].items():check();assert digest(source/f)==h,f
    for manifest in ['input_manifest.json','prediction_cohort.json']:
        for rel,h in read(source/manifest).items():
            p=Path(rel);selected=(p.parent.name=='observations' and p.stem in obs) or (p.parent.name=='truth' and p.stem in families) or (p.parent.name=='predictions' and p.stem.endswith('__sc-inr') and p.stem.removesuffix('__sc-inr') in obs)
            if selected:check();assert digest(source/rel)==h,rel
    manifest=read(root/'prediction_cohort.json');files={str(p.relative_to(root)) for d in ['predictions','probes'] for p in (root/d).iterdir()}
    assert set(manifest)==files and {p.stem for p in (root/'predictions').iterdir()}==families
    for rel,h in manifest.items():check();assert digest(root/rel)==h,rel
    pf=read(root/'prediction_freeze.json');complete=read(root/'COMPLETE.json')
    assert pf['manifest_sha256']==digest(root/'prediction_cohort.json')
    for k,f in [('rows_sha256','rows.json'),('summary_sha256','summary.json')]:assert complete[k]==digest(root/f)
    launch=read(root/'launch.json');res=read(root/'resources.json')
    assert launch['start_epoch']<=freeze['epoch']<=pf['epoch']<=complete['epoch']<=res['end_epoch']<=launch['deadline_epoch']
    budget=protocol['budget']['preflight_seconds' if freeze['preflight'] else 'formal_seconds']
    assert launch['deadline_epoch']-launch['start_epoch']<=budget+1e-6 and res['wall_seconds']<=budget and res['script_seconds']<=budget
    assert res['status']=='complete' and res['CPU_threads']==1 and res['nice']==10 and res['GPU']==res['training']==0
    journal=read(root/'inference_journal.json');assert len(journal)==len(cohort)
    jj={r['family']:r for r in journal};assert set(jj)==families
    with np.load(root/'probes/weights.npz') as f:layers=[(f[f'w{i}'],f[f'b{i}']) for i in range(4)]
    saved=read(root/'rows.json');lookup={(r['family'],r['roi'],r['band']):r for r in saved};assert len(lookup)==len(saved)==len(cohort)*8
    rows=[];idx=protocol['numeric']['query_indices'];modes=['native','uniform','fixed_native','fixed_uniform']
    for c in cohort:
        check();family=c['family'];j=jj[family];assert j['status']=='complete'
        for key in ['endpoint_float32_maxdiff','feature_product_maxdiff','rebuild_double_maxdiff']:assert j[key]<=2e-6,key
        close(j['native_feature_rms'],j['uniform_feature_rms'],'logged_feature_norm')
        assert j['norm_error']<=1e-12
        close(j['uniform_weight'],1-j['uniform_t'],'logged_uniform_weight')
        expected_checks=4 if freeze['preflight'] or c==cohort[0] else 0
        assert len(j['derivative_checks'])==expected_checks
        for dcheck in j['derivative_checks']:
            for values in dcheck['checks'].values():assert max(values.values())<=1e-10

        with np.load(root/'predictions'/(family+'.npz')) as f:z={k:f[k] for k in f.files}
        assert all(v.shape==(64,64,3) and np.isfinite(v).all() for v in z.values())
        with np.load(source/'predictions'/(c['eta1']+'__sc-inr.npz')) as old:
            for m in ['point','native']:close(z['old_'+m],old[m],'old_endpoint',2e-6);close(z[m],old[m],'double_endpoint',2e-6)
        with np.load(root/'probes'/(family+'.npz')) as f:p={k:f[k] for k in f.files}
        assert np.array_equal(p['indices'],idx);U,W=p['U'],p['W'];assert U.shape==W.shape==(4,16,256)
        omega=p['omega'];assert omega.shape==(4,16,128,2);close(p['rel_cell'],np.array([.5,.5]),'relative_cell')
        wcalc=np.prod(np.sinc(omega.astype(np.float64)*p['rel_cell'].astype(np.float64)/2),axis=-1)
        close(W,np.concatenate([wcalc,wcalc],-1),'float32_sinc',2e-6)
        weights=p['weights'];assert weights.shape==(4,16,1) and (weights>=0).all();close(weights.sum(0),np.ones((16,1)),'branch_weight_sum',1e-7)
        t=float(p['uniform_t']);assert 0<=t<=1
        close(t*p['U_rms'],p['native_feature_rms'],'native_norm_factor')
        for k in ['uniform_t','native_feature_rms','U_rms']:close(p[k],j[k],'probe_journal_'+k)
        reconstructed={k:p['skip']*.5+.5 for k in ['point','native','uniform']}
        deltas={'fixed_native':np.zeros((16,3)),'fixed_uniform':np.zeros((16,3))}
        for b in range(4):
            for name,u in [('point',U[b]),('native',U[b]*W[b]),('uniform',U[b]*(1-t))]:reconstructed[name]=reconstructed[name]+.5*weights[b]*dense_numpy(u,layers)
            for name,d in [('fixed_native',U[b]*(W[b]-1)),('fixed_uniform',-t*U[b])]:deltas[name]+=.5*weights[b]*dense_fixed_numpy(U[b],d,layers)
        for name,d in deltas.items():reconstructed[name]=reconstructed['point']+d
        for name,value in reconstructed.items():close(value,z[name].reshape(-1,3)[idx],'numpy_probe_'+name,1e-10)
        with np.load(source/'truth'/(family+'.npz')) as f:truth=f['target']
        for roi,box in [('center',(16,48)),('full',(0,64))]:
            pp={k:components(v,box) for k,v in z.items()};tt=components(truth,box)
            for band,T in enumerate(tt):
                row=dict(c,roi=roi,band=band);e=T-pp['point'][band];row['reference_mse']=float(np.mean(e**2))
                for name in modes:
                    m=metrics(e,pp[name][band]-pp['point'][band])
                    for key in ['a','b','g','mse']:row[f'{key}_{name}']=m[key]
                for prefix,n,u in [('full','native','uniform'),('fixed','fixed_native','fixed_uniform')]:
                    row[prefix+'_improvement']=row['mse_'+n]-row['mse_'+u];row[prefix+'_rescued']=row['g_'+n]<-1e-8 and row['g_'+u]>=-1e-8
                m=metrics(T-pp['old_point'][band],pp['old_native'][band]-pp['old_point'][band]);row['old_a']=m['a'];row['old_g']=m['g'];row['sensitive_direction']=classify(m['a'],1e-10)!=classify(row['a_native'],1e-10);row['sensitive_gain']=classify(m['g'],1e-8)!=classify(row['g_native'],1e-8)
                compare(lookup[(family,roi,band)],row,'rows');rows.append(row)
    summary=read(root/'summary.json')
    assert summary['rows']==len(rows) and summary['families']==len(cohort) and summary['GPU']==summary['training']==0
    assert summary['origin']==protocol['origin'] and summary['limitations']==protocol['limitations']
    active={'all32':any(c['template'].startswith('mid_') for c in cohort),'focal16':any(c['focal'] for c in cohort),'nonflip16':any(c['template'].startswith('mid_') and not c['focal'] for c in cohort),'sham16':any(c['template']=='control_low' for c in cohort)}
    assert set(summary['groups'])=={f'{g}/{roi}/b{b}' for g,v in active.items() if v for roi in ['center','full'] for b in range(4)}
    for key,group in summary['groups'].items():
        name,roi,band=key.split('/');band=int(band[1:]);selected=[]
        for r in rows:
            accept={'all32':r['template'].startswith('mid_'),'focal16':r['focal'],'nonflip16':r['template'].startswith('mid_') and not r['focal'],'sham16':r['template']=='control_low'}[name]
            if accept and r['roi']==roi and r['band']==band:selected.append(r)
        assert group['n']==len(selected)
        for key in ['full_rescued','fixed_rescued','sensitive_direction','sensitive_gain']:assert group[key]==sum(r[key] for r in selected)
        for agg,fn in [('median',np.median),('mean',np.mean)]:
            for key,value in group[agg].items():close(value,float(fn([r[key] for r in selected])),'summary_'+agg)
        for key,prefix,sign,tol in [('positive_direction','a_',1,1e-10),('negative_direction','a_',-1,1e-10),('positive_gain','g_',1,1e-8),('negative_gain','g_',-1,1e-8)]:
            assert set(group[key])==set(modes)
            for mode,count in group[key].items():assert count==sum(sign*r[prefix+mode]>tol for r in selected)
    focal=[r for r in rows if r['focal'] and r['roi']=='center' and r['band']==2];gates={}
    if len(focal)==16:
        rgb=[r for r in rows if r['focal'] and r['roi']=='center' and r['band']==0]
        gates=dict(full_rescue_majority=sum(r['full_rescued'] for r in focal)>8,fixed_rescue_majority=sum(r['fixed_rescued'] for r in focal)>8,full_median_improvement=float(np.median([r['full_improvement'] for r in focal]))>1e-8,fixed_median_improvement=float(np.median([r['fixed_improvement'] for r in focal]))>1e-8,fullRGB_not_worse=float(np.median([r['full_improvement'] for r in rgb]))>=-1e-8)
    assert summary['gates']==gates and summary['allocation_explanation_supported']==(all(gates.values()) if gates else None)
    return dict(run=str(root),families=len(cohort),rows=len(rows),gates=gates,allocation_explanation_supported=summary['allocation_explanation_supported'],probe_queries_per_family=16,probe_branches=4,scope='Independent NumPy sinc, full MLP and frozen-zero-gate derivative reconstruction at16 saved queries/family; complete pixel scoring. Native/uniform full feature RMS validated only by logged identities/static code, not independently replayed all features. Remote source/checkpoint represented by frozen manifests.',source_sha256=freeze['source_sha256'],protocol_sha256=freeze['protocol_sha256'])


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
