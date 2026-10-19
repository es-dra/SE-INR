"""Retrospective saved-output analysis. No torch, model load, fitting, or inference."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse, hashlib, json, resource, time, traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
C=P['cohort']; EPS=P['metrics']['harm_threshold_mse']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def projection(x,shave):
    x=np.asarray(x,dtype=np.float64)
    if shave: x=x[shave:-shave,shave:-shave]
    n=x.shape[0]; f=np.fft.fftfreq(n); radius=np.maximum(abs(f[:,None]),abs(f[None,:]))
    z=np.fft.fft2(x,axes=(0,1))
    return {'rgb':x,**{f'b{k+1}':np.fft.ifft2(z*((radius>=lo)&(radius<hi)&(radius>0))[:,:,None],axes=(0,1)).real for k,(lo,hi) in enumerate(P['metrics']['bands'])}}
def decomposition(t,p0,pn):
    e=t-p0; d=pn-p0
    e0=float(np.mean(e*e)); en=float(np.mean((t-pn)**2)); a=float(np.mean(e*d)); b=float(np.mean(d*d)); g=e0-en
    tol=64*np.finfo(np.float64).eps*max(e0,b,1e-8)
    label='benefit_or_tie'
    if g < -EPS:
        label='boundary' if b<=tol or abs(a)<=tol or abs(a-b/2)<=tol else ('direction' if a<0 else 'overshoot')
    assert abs(g-(2*a-b)) < 1e-12
    return dict(reference_mse=e0,native_mse=en,a=a,b=b,gain=g,label=label,db=float(10*np.log10(e0/en)),identity_residual=g-(2*a-b))
def summarize(rr):
    e0,en,a,b=(np.array([r[k] for r in rr]) for k in ['reference_mse','native_mse','a','b'])
    original=e0-en; neg=original < -EPS; pos=original > EPS
    lam=float(np.clip(a.mean()/b.mean(),0,1)) if b.mean()>0 else 0.
    lc=np.divide(a,b,out=np.zeros_like(a),where=b>0); lc=np.clip(lc,0,1)
    def outcome(l):
        mse=e0-2*l*a+l*l*b; gain=e0-mse
        return dict(mean_mse=float(mse.mean()),mse_change_vs_native_percent=float(100*(mse.mean()/en.mean()-1)),original_negative_rescued=int(np.sum(neg&(gain>EPS))),original_positive_becomes_negative=int(np.sum(pos&(gain < -EPS))),all_worse_than_native=int(np.sum(mse-en>EPS)),negative_total=int(np.sum(gain < -EPS)))
    return dict(n_records=len(rr),n_images=len({r['image'] for r in rr}),reference_mse=float(e0.mean()),native_mse=float(en.mean()),mean_a=float(a.mean()),mean_b=float(b.mean()),mean_db=float(np.mean([r['db'] for r in rr])),negative=int(neg.sum()),positive=int(pos.sum()),overshoot=sum(r['label']=='overshoot' for r in rr),direction=sum(r['label']=='direction' for r in rr),boundary=sum(r['label']=='boundary' for r in rr),harm_gt_001db=sum(r['db'] < -.01 for r in rr),harm_gt_005db=sum(r['db'] < -.05 for r in rr),min_db=min(r['db'] for r in rr),half=outcome(.5),global_oracle=dict(coefficient=lam,**outcome(lam)),per_case_oracle=dict(coefficient_mean=float(lc.mean()),coefficient_min=float(lc.min()),coefficient_max=float(lc.max()),**outcome(lc)))
def main(out):
    if out.exists(): raise RuntimeError('Run output exists: resume prohibited')
    out.mkdir(parents=True)
    started=time.time(); mono=time.monotonic(); r0=resource.getrusage(resource.RUSAGE_SELF)
    budget=dict(start_unix=started,deadline_unix=started+600,absolute_wall_seconds=600,cumulative_phase_seconds=600,cpu_threads=1,nice=10,resume=False)
    write(out/'budget.json',budget)
    def guard():
        if time.time()>budget['deadline_unix'] or time.monotonic()-mono>600: raise TimeoutError('Fixed budget exhausted')
    try:
        write(out/'source_freeze.json',{x:sha(HERE/x) for x in ['protocol.json','analyze.py']})
        src=(HERE/P['input_root']).resolve()
        assert sha(src/'prediction_cohort.json')==P['input_prediction_cohort_sha256']
        preds=json.loads((src/'prediction_cohort.json').read_text()); inp=json.loads((src/'input_manifest.json').read_text())
        assert len(preds)==256
        actual={p.name:sha(p) for p in sorted((src/'predictions').glob('*.npz'))}; assert actual==preds
        truths={f'truth/{i}.npy':sha(src/'truth'/f'{i}.npy') for i in C['images']}
        assert all(inp[k]==v for k,v in truths.items())
        write(out/'input_freeze.json',dict(prediction_manifest_sha256=sha(src/'prediction_cohort.json'),truth_hashes=truths,input_manifest_sha256=sha(src/'input_manifest.json'),prediction_files=256))
        rows=[]; native_rows=[]
        for i in C['images']:
            guard(); t0=np.load(src/'truth'/f'{i}.npy'); assert t0.shape==(128,128,3)
            targets={sh:projection(t0,sh) for sh in P['metrics']['shaves']}
            for scale in C['scales']:
                for model in C['models']:
                    for seed in C['seeds']:
                        with np.load(src/'predictions'/f'{i}_x{scale}__{model}_seed{seed}.npz') as zz:
                            saved={k:np.asarray(zz[k],dtype=np.float64) for k in ['zero','h3',C['native'][str(scale)]]}
                        assert all(z.shape==(128,128,3) and np.isfinite(z).all() for z in saved.values())
                        for transform in P['metrics']['endpoint_transforms']:
                            endpoints=saved if transform=='raw' else {k:np.clip(z,0,1) for k,z in saved.items()}
                            for shave in P['metrics']['shaves']:
                                guard(); metric=f'{transform}_shave{shave}'
                                projected={k:projection(z,shave) for k,z in endpoints.items()}; native=projected[C['native'][str(scale)]]
                                nm=float(np.mean((targets[shave]['rgb']-native['rgb'])**2))
                                native_rows.append(dict(metric=metric,image=i,scale=scale,model=model,seed=seed,mse=nm,psnr=float(-10*np.log10(nm))))
                                for ref in C['references']:
                                    for band,t in targets[shave].items():
                                        rows.append(dict(metric=metric,image=i,scale=scale,model=model,seed=seed,ref=ref,band=band,**decomposition(t,projected[ref][band],native[band])))
        assert len(rows)==P['expected']['rows']; assert len(native_rows)==1536
        write(out/'rows.json',rows); write(out/'native_rows.json',native_rows)
        groups={}
        for r in rows: groups.setdefault(tuple(r[k] for k in ['metric','model','scale','ref','band']),[]).append(r)
        rng=np.random.default_rng(P['bootstrap']['seed']); indices=rng.integers(0,32,size=(10000,32))
        summaries=[]; curves=[]
        for keys,rr in groups.items():
            guard(); identity=dict(zip(['metric','model','scale','ref','band'],keys)); s=summarize(rr)
            vals=np.array([[np.mean([r[k] for r in rr if r['image']==i]) for k in ['db','native_mse','reference_mse','a','b']] for i in C['images']])
            boot=vals[indices].mean(axis=1)
            s['mean_db_ci']=np.quantile(boot[:,0],[.025,.975]).tolist()
            half=boot[:,2]-boot[:,3]+.25*boot[:,4]
            s['half_mse_change_ci']=np.quantile(100*(half/boot[:,1]-1),[.025,.975]).tolist()
            summaries.append(dict(**identity,**s))
            if keys[0]!='raw_shave4': continue
            e0,en,a,b=(np.array([r[k] for r in rr]) for k in ['reference_mse','native_mse','a','b'])
            neg=e0-en < -EPS; pos=e0-en>EPS
            for j in range(101):
                lam=j/100; mse=e0-2*lam*a+lam*lam*b; gain=e0-mse
                curves.append(dict(**identity,coefficient=lam,mean_mse=float(mse.mean()),mse_change_vs_native_percent=float(100*(mse.mean()/en.mean()-1)),original_negative_rescued=int(np.sum(neg&(gain>EPS))),original_positive_becomes_negative=int(np.sum(pos&(gain < -EPS))),all_worse_than_native=int(np.sum(mse-en > EPS)),negative_total=int(np.sum(gain < -EPS))))
        assert len(curves)==3232
        comparisons=[]
        for metric in sorted({r['metric'] for r in native_rows}):
            for scale in C['scales']:
                for seed in [1,2,'mean_of_two_seeds_within_image']:
                    ar=[r for r in native_rows if r['metric']==metric and r['scale']==scale and (isinstance(seed,str) or r['seed']==seed)]
                    vals=np.array([[np.mean([r['psnr'] for r in ar if r['image']==i and r['model']==model]) for model in C['models']] for i in C['images']])
                    delta=vals[:,1]-vals[:,0]; boot=vals[indices].mean(axis=1); bd=boot[:,1]-boot[:,0]
                    comparisons.append(dict(metric=metric,scale=scale,seed=seed,n_images=32,lte_mean_psnr=float(vals[:,0].mean()),sc_inr_mean_psnr=float(vals[:,1].mean()),lte_mean_psnr_ci=np.quantile(boot[:,0],[.025,.975]).tolist(),sc_inr_mean_psnr_ci=np.quantile(boot[:,1],[.025,.975]).tolist(),paired_sc_minus_lte_db=float(delta.mean()),paired_delta_ci=np.quantile(bd,[.025,.975]).tolist(),sc_wins=int(np.sum(delta>1e-12)),ties=int(np.sum(abs(delta)<=1e-12)),sc_losses=int(np.sum(delta < -1e-12)),image_deltas=[dict(image=i,sc_minus_lte_db=float(d)) for i,d in zip(C['images'],delta)]))
        examples=[]
        for model in C['models']:
            ar=[r for r in rows if r['metric']=='raw_shave4' and r['model']==model and r['ref']=='zero' and r['band']=='rgb' and r['gain'] < -EPS]
            for label in ['overshoot','direction']:
                rr=[r for r in ar if r['label']==label]
                if not rr: examples.append(dict(model=model,selection='median',label=label,status='missing_class')); continue
                median=float(np.median([-r['db'] for r in rr]))
                pick=min(rr,key=lambda r:(abs(-r['db']-median),r['image'],r['seed'],r['scale']))
                examples.append(dict(selection='median',class_count=len(rr),class_median_psnr_loss=median,status='selected',**pick))
            if ar:
                pick=min(ar,key=lambda r:(r['db'],r['image'],r['seed'],r['scale']))
                examples.append(dict(selection='tail',status='selected',**pick))
            else: examples.append(dict(model=model,selection='tail',status='missing_harm'))
        summary=dict(protocol_sha256=sha(HERE/'protocol.json'),source_sha256=sha(HERE/'analyze.py'),rows=len(rows),primary='raw_shave4',interpretation=P['analysis_status'],model_scope=P['model_scope'],groups=summaries,native_comparison=comparisons)
        write(out/'curves.json',curves); write(out/'summary.json',summary); write(out/'selected_examples.json',examples)
        guard(); r1=resource.getrusage(resource.RUSAGE_SELF)
        cost=dict(wall_seconds=time.monotonic()-mono,cpu_user_seconds=r1.ru_utime-r0.ru_utime,cpu_sys_seconds=r1.ru_stime-r0.ru_stime,peak_rss_platform_units=r1.ru_maxrss,platform=os.uname().sysname,nice=os.getpriority(os.PRIO_PROCESS,0),cpu_threads=1,gpu=0,inference=0,training=0)
        write(out/'cost.json',cost)
        write(out/'COMPLETE.json',dict(status='complete',rows=len(rows),curves=len(curves),source_freeze_sha256=sha(out/'source_freeze.json'),outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
        print(json.dumps(dict(status='complete',rows=len(rows),cost=cost)))
    except BaseException:
        write(out/'FAILURE.json',dict(traceback=traceback.format_exc(),wall_seconds=time.monotonic()-mono)); raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args(); main(Path(args.output).resolve())
