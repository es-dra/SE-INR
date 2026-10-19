"""Independent CPU audit of saved readout inference; never import the experiment/models.
Usage: python readout-audit.py --out OUT --source-root FROZEN_V4_ROOT --report FILE
A downloaded checkpoint may be addressed with --checkpoint-root ROOT/name/epoch-last.pth.
Timestamp evidence documents ordering of artifacts, not an OS-level proof of all reads.
"""
import argparse, ast, hashlib, json, os, time
from pathlib import Path
for _threadvar in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_threadvar] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

KEYS = ['band_pred_gain', 'footprint_fit_pred_gain', 'scaling_pred_gain', 'quadrature_pred_gain']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def ranks(a):
    a=np.asarray(a,dtype=float); order=np.argsort(a,kind='stable'); r=np.empty(len(a))
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]: j+=1
        r[order[i:j]]=(i+j-1)/2+1; i=j
    return r

def rho(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(a)==0 or np.ptp(b)==0: return None
    x=ranks(a);y=ranks(b);x-=x.mean();y-=y.mean()
    return float(x@y/np.sqrt((x@x)*(y@y)))

def grid(n,cut=None):
    v=(np.arange(n,dtype=float)+.5)/n
    if cut: v=v[cut[0]:cut[1]]
    return np.array([(y,x) for y in v for x in v])

def mse(a,b): return float(np.square(a-b).mean())

def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,(np.bool_,)): return bool(v)
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(float,np.floating)) and not np.isfinite(v): return None
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True);ap.add_argument('--source-root',required=True)
    ap.add_argument('--checkpoint-root');ap.add_argument('--report',required=True);ap.add_argument('--budget',type=float,default=120)
    a=ap.parse_args(); start=time.monotonic(); cpu_start=time.process_time(); out=Path(a.out); src=Path(a.source_root)
    failures=[]; differences=[]; nonfinite=[]; checked=0
    def guard():
        if time.monotonic()-start>a.budget or time.process_time()-cpu_start>a.budget: raise TimeoutError('independent audit CPU/wall budget exceeded')
    def check(ok,what):
        if not ok: failures.append(what)
    def read(name): return json.loads((out/name).read_text())
    def num(actual,expected,what,tol=2e-9):
        if actual is None or expected is None:
            check(actual is None and expected is None,what+' null mismatch');return
        if isinstance(actual,bool) or isinstance(expected,bool): check(actual==expected,what);return
        if not np.isfinite(float(expected)):
            nonfinite.append(what); failures.append(what+' nonfinite original');return
        delta=abs(float(actual)-float(expected)); differences.append(delta)
        check(delta<=tol*(1+abs(float(expected))),what+' numerical mismatch '+str(delta))
    report={'scope':'Complete frozen synthetic grid and retrospective barbara; bootstrap over 18 fixed k/direction grid clusters, not a population or natural-image generalization CI.'}
    try:
        frozen=read('frozen.json'); P=json.loads((src/'protocol.json').read_text()); cases=read('cases.json')
        check((out/'COMPLETE').exists(),'COMPLETE missing');check(not (out/'failure.txt').exists(),'failure artifact exists')
        check(len(cases)==74,'74 cases');check(len({c['id'] for c in cases})==74,'duplicate cases')
        expected={(n,k,tuple(d),i) for n in P['input_sizes'] for k in P['frequencies'] for d in P['directions'] for i in range(2)}
        actual={(c['n'],c['k'],tuple(c['direction']),c['phase_index']) for c in cases if c['kind']=='synthetic'}
        check(actual==expected and len(actual)==72,'synthetic fixed cohort')
        for c in cases:
            if c['kind']=='synthetic':
                check(c['phase']==P['phases'][c['phase_index']],'declared phase '+c['id'])
                check(c['id']==f'n{c["n"]}_k{c["k"]}_d{c["direction"][0]}{c["direction"][1]}_p{c["phase_index"]}','synthetic case identifier')
            else: check(c['n']==256//c['scale'] and c['scale'] in [2,4],'natural case dimensions')
        check({c['id'] for c in cases if c['kind']=='natural'}=={'barbara_x2','barbara_x4'},'natural cohort')
        check(digest(src/'protocol.json')==frozen['protocol_sha256'],'protocol SHA')
        for rel,h in frozen['source_sha256'].items():
            guard(); fp=src/rel;check(fp.is_file(),'missing source '+rel)
            if fp.is_file(): check(digest(fp)==h,'source SHA '+rel)
        # Original predictor is forbidden from loading target truth. This is source/dataflow evidence.
        tree=ast.parse((src/'experiment.py').read_text())
        predict=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
        loads=[n for n in ast.walk(predict) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load']
        check(all("'truth'" not in ast.unparse(n) and '"truth"' not in ast.unparse(n) for n in loads),'predict truth load')
        for rel,h in read('input_manifest.json').items():
            guard();check(digest(out/rel)==h,'input SHA '+rel);checked+=1
        im=read('input_manifest.json');check(len(im)==148,'148 input files')
        check(set(im)=={f'{d}/{c["id"]}.npy' for d in ['observations','truth'] for c in cases},'complete input manifest')
        check(im['truth/barbara_x2.npy']==im['truth/barbara_x4.npy'],'shared natural target truth')
        cohort=read('prediction_cohort.json'); freeze=read('prediction_freeze.json')
        check(len(cohort)==152 and freeze['files']==152,'152 prediction files');check(digest(out/'prediction_cohort.json')==freeze['cohort_sha256'],'cohort SHA')
        expected_npz={f'predictions/{c["id"]}__{name}.npz' for c in cases for name in (P['models'] if c['kind']=='synthetic' else P['models']+P['ablated_models'])}
        check(set(cohort)==expected_npz,'prediction cohort completeness')
        check(set(cohort)=={str(p.relative_to(out)) for p in (out/'predictions').glob('*.npz')},'unlisted prediction file')
        for rel,h in cohort.items(): guard();check(digest(out/rel)==h,'prediction SHA '+rel);checked+=1
        checkpoints=read('checkpoint_manifest.json')
        for name,meta in checkpoints.items():
            fp=Path(a.checkpoint_root)/name/'epoch-last.pth' if a.checkpoint_root else Path(meta['path'])
            check(fp.is_file(),'missing checkpoint '+name)
            if fp.is_file():check(digest(fp)==meta['sha256'],'checkpoint SHA '+name)
            check(meta['epoch']==1000,'checkpoint epoch '+name)
        check(set(checkpoints)==set(P['models']+P['ablated_models']),'four checkpoints')
        phase=read('phase_costs.json'); resources=read('resources.json'); journal=read('inference_journal.json')
        check([x['phase'] for x in phase]==['prepare','predict','score'],'phase order');check(len(journal)==74,'journal completeness')
        check([x['case'] for x in journal]==[c['id'] for c in cases],'journal case order')
        cumulative=0
        for c,j in zip(cases,journal):
            if c['kind']=='synthetic':
                nq=(P['query_crop'][1]-P['query_crop'][0])**2
                count=nq*len(P['models'])*(1+len(P['target_footprints'])*(1+sum(o*o for o in P['quadrature_orders'])))
            else: count=256*256*(2*len(P['models'])+len(P['ablated_models']))
            cumulative+=count;num(cumulative,j['query_rgb_pixels_cumulative'],'journal query pixel count '+c['id'],tol=0)
        num(cumulative,freeze['query_rgb_pixels'],'freeze total query pixels',tol=0)
        check(resources['status']=='complete' and resources['training_runs']==0 and resources['gpu_processes']==0,'CPU inference resource status')
        check(resources['wall_seconds']<=P['wall_budget_seconds'],'original fixed budget');check(sum(x['seconds'] for x in phase)<=resources['in_script_wall_seconds']+1e-5,'phase cost total')
        check(frozen['absolute_deadline_epoch']-frozen['start_epoch']<=P['wall_budget_seconds']+.1,'absolute deadline')
        freeze_epoch=time.mktime(time.strptime(freeze['time_utc'],'%Y-%m-%dT%H:%M:%SZ'))-time.timezone
        check(frozen['start_epoch']<=freeze_epoch+1<=frozen['absolute_deadline_epoch']+1,'freeze timestamp budget')
        # cp/tar must preserve original mtimes; raw hash freeze remains checked independently.
        freeze_mtime=(out/'prediction_freeze.json').stat().st_mtime
        timestamp_order={name:(out/name).stat().st_mtime>=freeze_mtime-1e-6 for name in ['synthetic_rows.json','natural_rows.json','summary.json']}
        check(all(timestamp_order.values()),'score artifacts older than freeze (or download mtime not preserved)')
        report['freeze_evidence']={'source_predict_no_truth_load':all("'truth'" not in ast.unparse(n) and '"truth"' not in ast.unparse(n) for n in loads),'score_artifact_mtime_order':timestamp_order,'limit':'Artifact ordering and inspected source, not instrumented filesystem-read chronology.'}
        rows=[]; natural=[]; natural_mse=[]; q=grid(P['hr_reference'],P['query_crop'])
        for c in cases:
            guard(); y=np.load(out/'observations'/(c['id']+'.npy')); truth=np.load(out/'truth'/(c['id']+'.npy'))
            check(np.isfinite(y).all() and np.isfinite(truth).all(),'input finite '+c['id'])
            if c['kind']=='synthetic':
                ky,kx=np.asarray(c['direction'])*c['k']; phaseangle=c['phase']
                point_truth=.5+.25*np.cos(2*np.pi*(q[:,0]*ky+q[:,1]*kx)+phaseangle)
                check(np.max(abs(truth-point_truth))<=1e-14,'synthetic truth '+c['id'])
                cc=grid(c['n']); yy=.5+.25*np.sinc(ky/c['n'])*np.sinc(kx/c['n'])*np.cos(2*np.pi*(cc[:,0]*ky+cc[:,1]*kx)+phaseangle)
                yy=np.repeat(yy.reshape(1,c['n'],c['n']),3,axis=0).astype('float32')
                check(np.array_equal(y,yy),'synthetic observation '+c['id'])
                for name in P['models']:
                    with np.load(out/'predictions'/(c['id']+'__'+name+'.npz')) as z:
                        check(set(z.files)=={'point'}|{prefix+'_'+tag for prefix in ['native','quad4','quad8'] for tag in P['target_footprints']},'npz keys '+c['id']+name)
                        check(all(np.isfinite(z[k]).all() and z[k].shape==(len(q),3) for k in z.files),'npz finite/shape '+c['id']+name)
                        p0=z['point'].astype(float)
                        for tag,fac in P['target_footprints'].items():
                            wh=np.asarray(fac)/64; T=float(np.sinc(ky*wh[0])*np.sinc(kx*wh[1])); target=(.5+T*(point_truth-.5))[:,None]
                            S=float(np.square(target-.5).mean()); ph=z['native_'+tag].astype(float); q4=z['quad4_'+tag].astype(float);q8=z['quad8_'+tag].astype(float)
                            e,en,eq=mse(p0,target),mse(ph,target),mse(q8,target); d=ph-p0
                            change=mse(d,0)/S;cross=2*float(((p0-target)*d).mean())/S;gain=(e-en)/S
                            comm=mse(ph,q8);ref=mse(q4,q8)
                            rows.append(dict(case=c['id'],model=name,n=c['n'],k=c['k'],direction=c['direction'],phase_index=c['phase_index'],footprint=tag,band=1 if c['k']<8 else 2 if c['k']<16 else 3,T=T,signal_energy=S,native_mse=en,point_mse=e,quadrature_mse=eq,gain=gain,footprint_score=(1-T)**2,scaling_pred_gain=(e-mse(.5+T*(p0-.5),target))/S,quadrature_pred_gain=(e-eq)/S,commutation_mse=comm,quadrature_refinement_mse=ref,change_energy=change,alignment=cross,identity_residual=gain+change+cross,clamped_native_mse=mse(np.clip(ph,0,1),target),out_of_range_fraction=float(((ph<0)|(ph>1)).mean()),quadrature_resolved=bool(ref<=1e-5 and ref<=.1*comm)))
            else:
                t=truth.transpose(1,2,0); sig=t.mean(2); X=np.fft.fft2(sig-sig.mean());f=np.fft.fftfreq(256); rad=np.maximum(abs(f[:,None]),abs(f[None,:]))
                check(y.shape==(3,c['n'],c['n']),'natural observation shape')
                for name,abl in zip(P['models'],P['ablated_models']):
                    with np.load(out/'predictions'/(c['id']+'__'+name+'.npz')) as z,np.load(out/'predictions'/(c['id']+'__'+abl+'.npz')) as za:
                        full=z['native'].reshape(256,256,3).astype(float);point=z['point'].reshape(256,256,3).astype(float);ab=za['native'].reshape(256,256,3).astype(float)
                    check(all(np.isfinite(x).all() for x in [full,point,ab]),'natural finite')
                    for clamp in [False,True]:
                        fg,pg,ag=[np.clip(x,0,1) if clamp else x for x in [full,point,ab]]
                        natural_mse.append(dict(case=c['id'],model=name,clamped=clamp,native_rgb_mse=mse(fg,t),point_rgb_mse=mse(pg,t),retrained_rgb_mse=mse(ag,t)))
                        E,Ep,Ea,D=[np.fft.fft2(v) for v in [sig-fg.mean(2),sig-pg.mean(2),sig-ag.mean(2),fg.mean(2)-pg.mean(2)]]
                        for b,(lo,hi) in enumerate([(0,.125),(.125,.25),(.25,.5001)],1):
                            mask=(rad>=lo)&(rad<hi)&(rad>0); S=float(np.square(abs(X[mask])).sum())
                            gain=float((np.square(abs(Ep[mask]))-np.square(abs(E[mask]))).sum()/S);ce=float(np.square(abs(D[mask])).sum()/S);align=2*float((Ep[mask].conj()*D[mask]).real.sum()/S)
                            natural.append(dict(case=c['id'],model=name,band=b,clamped=clamp,paired_retrained_gain=float((np.square(abs(Ea[mask]))-np.square(abs(E[mask]))).sum()/S),fixed_checkpoint_gain=gain,change_energy=ce,alignment=align,identity_residual=gain+ce-align))
        oldrows=read('synthetic_rows.json');oldnat=read('natural_rows.json');oldsummary=read('summary.json')
        check(len(rows)==len(oldrows)==576,'576 synthetic rows');check(len(natural)==len(oldnat)==24,'24 natural rows')
        def compare_row(new,old,i):
            for k,v in new.items():
                if isinstance(v,(int,float,bool)) and not isinstance(v,str):num(v,old.get(k),i+' '+k)
                else:check(v==old.get(k),i+' metadata '+k)
        groups=[]; rng=np.random.default_rng(20261011)
        for name in P['models']:
            for n in P['input_sizes']:
                cal=[r for r in rows if r['model']==name and r['n']==n and r['phase_index']==0];test=[r for r in rows if r['model']==name and r['n']==n and r['phase_index']==1]
                cf=np.linalg.lstsq(np.array([[1,r['footprint_score']] for r in cal]),np.array([r['gain'] for r in cal]),rcond=None)[0]
                for r in test:
                    r['band_pred_gain']=float(np.median([x['gain'] for x in cal if x['band']==r['band']]))
                    r['footprint_fit_pred_gain']=float(cf@[1,r['footprint_score']])
                actualg=np.array([r['gain'] for r in test]);active=abs(actualg)>P['sign_margin'];negative=active&(actualg<0);stats={}
                for key in KEYS:
                    pred=np.array([r[key] for r in test]); stats[key]=dict(mae=float(abs(pred-actualg).mean()),spearman=rho(pred,actualg),sign_accuracy=float((np.sign(pred[active])==np.sign(actualg[active])).mean()) if active.any() else None,negative_recall=float((pred[negative]<0).mean()) if negative.any() else None,within_band_spearman={str(b):rho([r[key] for r in test if r['band']==b],[r['gain'] for r in test if r['band']==b]) if key!='band_pred_gain' else None for b in [1,2,3]})
                g=dict(model=name,n=n,test_rows=len(test),active_sign_rows=int(active.sum()),negative_rows=int(negative.sum()),predictors=stats,commutation_median=float(np.median([r['commutation_mse']/r['signal_energy'] for r in test])),quadrature_refinement_max=float(max(r['quadrature_refinement_mse'] for r in test)),quadrature_resolved_rows=sum(r['quadrature_resolved'] for r in test),candidate_gate={})
                for key in KEYS[2:]:
                    st=stats[key];base=min(stats[k]['mae'] for k in KEYS[:2]);g['candidate_gate'][key]=bool(st['mae']<=.8*base and st['sign_accuracy'] is not None and st['sign_accuracy']>=.75 and negative.sum()>=8 and st['negative_recall']>=.75 and (key!=KEYS[3] or all(r['quadrature_resolved'] for r in test)))
                # k/direction joint clusters retain their four correlated footprint rows.
                clusters=sorted({(r['k'],tuple(r['direction'])) for r in test});check(len(clusters)==18,'18 bootstrap grid clusters')
                ix=np.array([[i for i,r in enumerate(test) if (r['k'],tuple(r['direction']))==c] for c in clusters]);check(ix.shape==(18,4),'4 footprint rows per cluster')
                sampled=rng.integers(0,18,size=(2000,18)); sample_rows=ix[sampled].reshape(2000,72); boot={}
                def ci(v):
                    v=np.asarray(v,dtype=float); valid=np.isfinite(v)
                    return dict(low=float(np.quantile(v[valid],.025)) if valid.any() else None,high=float(np.quantile(v[valid],.975)) if valid.any() else None,undefined_replicates=int((~valid).sum()))
                for key in KEYS:
                    pred=np.array([r[key] for r in test]);am=active[sample_rows];nm=negative[sample_rows]
                    with np.errstate(invalid='ignore',divide='ignore'):
                        sign=((np.sign(pred[sample_rows])==np.sign(actualg[sample_rows]))&am).sum(1)/am.sum(1)
                        rec=((pred[sample_rows]<0)&nm).sum(1)/nm.sum(1)
                    boot[key]={'sign_accuracy':ci(sign),'negative_recall':ci(rec)}
                    if key in KEYS[2:]:
                        boot[key]['paired_mae_difference_candidate_minus_baseline']={base:ci((abs(pred-actualg)-abs(np.array([r[base] for r in test])-actualg))[sample_rows].mean(1)) for base in KEYS[:2]}
                g['bootstrap_fixed_grid_clusters']=dict(replicates=2000,clusters=18,seed=20261011,interval_interpretation='Sensitivity to resampling fixed synthetic k/direction grid; no population generalization claim.',predictors=boot);groups.append(g)
        for i,(new,old) in enumerate(zip(rows,oldrows)): compare_row(new,old,'synthetic '+str(i))
        for i,(new,old) in enumerate(zip(natural,oldnat)): compare_row(new,old,'natural '+str(i))
        def compare_nested(new,old,what):
            if isinstance(new,dict):
                for k,v in new.items(): compare_nested(v,old.get(k),what+'.'+k)
            elif isinstance(new,(float,int,bool)) or new is None:num(new,old,what)
            else:check(new==old,what)
        for i,g in enumerate(groups):compare_nested({k:v for k,v in g.items() if k!='bootstrap_fixed_grid_clusters'},oldsummary['groups'][i],'group '+str(i))
        maxid=max(abs(r['identity_residual']) for r in rows+natural);check(maxid<2e-10,'both squared-error identities');num(maxid,oldsummary['max_identity_residual'],'summary identity')
        num(len(rows),oldsummary['synthetic_rows'],'summary count');num(len(natural),oldsummary['natural_rows'],'natural summary count');check(oldsummary['model_training_runs']==0,'summary training count')
        report.update(synthetic_rows=len(rows),natural_rows=len(natural),groups=groups,natural_rgb_mse=natural_mse,maximum_recomputed_difference=max(differences or [0]),max_identity_residual=maxid,hashed_input_and_prediction_files=checked,original_cost=resources,natural_input_limit='Natural bicubic observation generation not independently reimplemented; original observation/truth hashes and array shapes checked.')
    except Exception as e:
        failures.append(type(e).__name__+': '+str(e))
    if time.monotonic()-start>a.budget or time.process_time()-cpu_start>a.budget: failures.append("independent audit CPU/wall budget exceeded")
    report.update(cpu_seconds=time.process_time()-cpu_start,threads=4,passed=not failures,failures=failures,nonfinite_original_fields=nonfinite,wall_seconds=time.monotonic()-start,cpu_only=True,training_runs=0,audit_source_sha256=digest(Path(__file__)))
    Path(a.report).write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':not failures,'failure_count':len(failures),'wall_seconds':report['wall_seconds'],'report':a.report}))
    if failures:raise SystemExit(1)

if __name__=='__main__': main()
