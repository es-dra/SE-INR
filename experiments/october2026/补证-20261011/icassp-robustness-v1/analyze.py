"""Frozen retrospective sensitivity analysis; no models, inference or fitting."""
import json, pathlib, hashlib, time, resource, datetime
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
BASE = ROOT.parents[1]
L1ROOT = BASE / '服务器结果-20261007/claude-20261005-L12/out'
G1ROOT = BASE / '补证-20261008/g1/out'
NEG = BASE / '补证-20261011/negative-collected/codex-20261011-negative-formal-v3'
E3 = BASE / '补证-20261008/paper-baseline-20261008/e3-audit/record-recalculation.json'

def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def rank(x):
    x=np.asarray(x); order=np.argsort(x,kind='stable'); r=np.empty(len(x))
    j=0
    while j<len(x):
        k=j+1
        while k<len(x) and x[order[k]]==x[order[j]]: k+=1
        r[order[j:k]]=(j+k-1)/2; j=k
    return r
def rho(x,y): return float(np.corrcoef(rank(x),rank(y))[0,1])

def l1_analysis():
    rows=[]; K=np.arange(121); edges=np.arange(.25,2.001,.25)
    for ds in ['Set14','BSD100','Urban100']:
        d=load(L1ROOT/f'l1_inr_{ds}.json'); names=sorted(d['S']); S=np.array([d['S'][n] for n in names])
        assert S.shape==(len(names),121)
        for model in ['liif','lte','sc-inr']:
          for suffix in ['', '@s2','@s3']:
            mn=model+suffix; seed=1 if not suffix else int(suffix[-1])
            for pooling in ['energy_pool','uniform_image_mean']:
                curves={}
                for axis in ['u','nu']:
                    for s in [2,3,4]:
                        E=np.array([d['E'][mn][str(s)][n] for n in names]); assert E.shape==S.shape
                        vals=[]
                        x=2*s*K/240 if axis=='u' else 6*K/240
                        for i in range(7):
                            mask=(x>=edges[i]-1e-12)&((x<edges[i+1]-1e-12) if i<6 else (x<=edges[i+1]+1e-12))&(K>0)
                            se=S[:,mask].sum(1); ee=E[:,mask].sum(1)
                            assert np.all(se>0) and np.isfinite(ee).all()
                            vals.append(float(1-ee.sum()/se.sum()) if pooling=='energy_pool' else float(np.mean(1-ee/se)))
                        curves[axis,s]=np.array(vals)
                for scales in [[2,3,4],[2,3],[2,4],[3,4]]:
                  for domain,bins in [('all',range(7)),('below_nyquist',range(3)),('above_nyquist',range(3,7))]:
                    b=list(bins); vals={a:float(np.median(np.ptp(np.array([curves[a,s] for s in scales])[:,b],axis=0))) for a in ['u','nu']}
                    rows.append(dict(dataset=ds,model=model,seed=seed,images=len(names),pooling=pooling,scales=scales,domain=domain,D_u=vals['u'],D_nu=vals['nu'],normalisation_improves=vals['u']<vals['nu']))
    summary=[]
    for pool in ['energy_pool','uniform_image_mean']:
      for scales in [[2,3,4],[2,3],[2,4],[3,4]]:
       for domain in ['all','below_nyquist','above_nyquist']:
        rr=[r for r in rows if r['pooling']==pool and r['scales']==scales and r['domain']==domain]
        summary.append(dict(pooling=pool,scales=scales,domain=domain,groups=len(rr),improved=sum(r['normalisation_improves'] for r in rr),D_u_range=[min(r['D_u'] for r in rr),max(r['D_u'] for r in rr)],D_nu_range=[min(r['D_nu'] for r in rr),max(r['D_nu'] for r in rr)],strict_005=sum(r['D_u']<=.05 for r in rr)))
    return dict(rows=rows,summary=summary)

def g1_analysis():
    rows=[]; differences=[]; perimage=[]
    for seed in [2,3]:
      data={s:load(G1ROOT/f'seed{seed}/f1s_x{s}.json') for s in [2,3,4]}
      for model,abl in [('lte','lte-nocellphase'),('sc-inr','sc-inr-nosinc')]:
       for pool in ['saved_energy_pool','uniform_image_mean']:
        cells=[]
        for s,d in data.items():
         for ds in ['Set14','BSD100','Urban100']:
          rr=[r for r in d['per_image'] if r['set']==ds]
          assert len(rr)==(13 if ds=='Set14' else 30)
          ll=[]
          for b in [1,2,3]:
            value=d['pooled'][f'{ds}|{model}|b{b}']-d['pooled'][f'{ds}|{abl}|b{b}'] if pool=='saved_energy_pool' else np.mean([r[f'{model}_b{b}']-r[f'{abl}_b{b}'] for r in rr])
            cells.append(dict(scale=s,dataset=ds,band=b,loss=float(value)));ll.append(float(value))
          differences.append(dict(model=model,seed=seed,pooling=pool,dataset=ds,scale=s,L3_minus_L2=ll[2]-ll[1],ordering=ll[2]>ll[1]))
        rows.append(dict(model=model,seed=seed,pooling=pool,rho=rho([r['band'] for r in cells],[r['loss'] for r in cells]),ordering_pass=sum(r['ordering'] for r in differences if r['model']==model and r['seed']==seed and r['pooling']==pool),ordering_total=9,cells=cells))
       for s,d in data.items():
        for r in d['per_image']:
          l2=r[f'{model}_b2']-r[f'{abl}_b2'];l3=r[f'{model}_b3']-r[f'{abl}_b3']
          perimage.append(dict(model=model,seed=seed,scale=s,dataset=r['set'],image=r['img'],L2=l2,L3=l3,ordering=l3>l2))
    return dict(groups=rows,differences=differences,per_image=perimage,limitation='Original per-image G1 R lacks E/S weights; saved pooled values are not independently reconstructed. Alternate uniform-image estimand is descriptive.')

def negative_analysis():
    original=load(NEG/'test_scores.json'); lookup={(r['case'],r['model'],r['band']):r for r in original}
    rows=[]; max_original_error=0.; signs=[]
    cuts=[('original',1/8,1/4),('narrower',1/10,1/5),('wider',1/6,1/3)]
    for p in sorted((NEG/'predictions').glob('test*.npz')):
        case,model=p.stem.split('__');pred=np.load(p);truth=np.load(NEG/'truth'/f'{case}.npy').transpose(1,2,0).astype(np.float64)
        point=pred['point'].reshape(128,128,3).astype(np.float64); native=pred['native'].reshape(128,128,3).astype(np.float64)
        image=int(case.split('_')[1]);scale=int(case[-1])
        for shave in [0,4,8]:
            sl=slice(shave,128-shave);t=truth[sl,sl];f0=point[sl,sl];fn=native[sl,sl]
            mse0=float(np.mean((f0-t)**2));msen=float(np.mean((fn-t)**2)); fullgain=mse0-msen
            base=lookup[case,model,0]
            policy={k:msen if value>0 else mse0 for k,value in base['forecasts'].items()}
            rows.append(dict(case=case,image=image,scale=scale,model=model,shave=shave,bands='fullRGB',band=0,gain=fullgain,point_mse=mse0,native_mse=msen,policy_mse=policy))
            if shave==4: max_original_error=max(max_original_error,abs(fullgain-base['gain']),abs(mse0-base['point_mse']),abs(msen-base['native_mse']))
            h=t.shape[0];nu=np.maximum(*np.meshgrid(np.abs(np.fft.fftfreq(h)),np.abs(np.fft.fftfreq(h)),indexing='ij'))
            e0=np.fft.fft2(f0-t,axes=(0,1));en=np.fft.fft2(fn-t,axes=(0,1));energy0=np.abs(e0)**2;energyn=np.abs(en)**2
            for label,c1,c2 in cuts:
                for band,(lo,hi) in enumerate([(0,c1),(c1,c2),(c2,.5001)],1):
                    mask=(nu>=lo)&(nu<hi)&(nu>0)
                    m0=float(energy0[mask].sum()/(3*h**4));mn=float(energyn[mask].sum()/(3*h**4));gain=m0-mn
                    rows.append(dict(case=case,image=image,scale=scale,model=model,shave=shave,bands=label,band=band,gain=gain,point_mse=m0,native_mse=mn))
                    if shave==4 and label=='original':max_original_error=max(max_original_error,abs(gain-lookup[case,model,band]['gain']))
    summary=[]
    rng=np.random.default_rng(20261011); boot=rng.integers(0,12,size=(10000,12))
    for model in ['lte','sc-inr']:
      for shave in [0,4,8]:
       for label in ['fullRGB','original','narrower','wider']:
        rr=[r for r in rows if r['model']==model and r['shave']==shave and r['bands']==label]
        harms=[r for r in rr if r['gain']< -1e-8]
        summary.append(dict(model=model,shave=shave,bands=label,rows=len(rr),harmful=len(harms),harmful_images=len(set(r['image'] for r in harms)),mean_gain=float(np.mean([r['gain'] for r in rr])),band_mean_gain={str(b):float(np.mean([r['gain'] for r in rr if r['band']==b])) for b in sorted(set(r['band'] for r in rr))}))
       rr=[r for r in rows if r['model']==model and r['shave']==shave and r['bands']=='fullRGB']
       for policy in ['proxy','norm_ridge','directional_ridge']:
        byimage=np.array([np.mean([r['native_mse']-r['policy_mse'][policy] for r in rr if r['image']==im]) for im in range(825,837)])
        ci=np.percentile(byimage[boot].mean(1),[2.5,97.5]).tolist()
        signs.append(dict(model=model,shave=shave,policy=policy,MSE_reduction_vs_native=float(np.mean(byimage)),image_cluster_95_sensitivity=ci))
    flip=[]
    for model in ['lte','sc-inr']:
      for shave in [0,8]:
        for label in ['fullRGB','original']:
          rr=[r for r in rows if r['model']==model and r['shave']==shave and r['bands']==label]
          ref={(r['case'],r['band']):r for r in rows if r['model']==model and r['shave']==4 and r['bands']==label}
          flip.append(dict(model=model,shave=shave,bands=label,sign_disagreements=sum((r['gain']< -1e-8)!=(ref[r['case'],r['band']]['gain']< -1e-8) for r in rr),rows=len(rr)))
    return dict(rows=rows,summary=summary,fixed_policy=signs,sign_changes=flip,original_reproduction_max_error=max_original_error,limitation='Existing, already-scored evaluation images. Alternative crop/bands change the estimand and are retrospective sensitivity only; no calibration re-fit, new forecasts, gate reassessment or holdout claim. Fixed original full-RGB policies are applied unchanged to crop sensitivity.')

def thresholds():
    cells=load(E3)['cells'];out=[]
    for lo in [.05,.1,.15]:
     for hi in [.7,.8,.9]:
        low=[r for r in cells if r['prior']<lo]; high=[r for r in cells if r['prior']>=hi]
        out.append(dict(low=lo,high=hi,total=len(cells),C_count=len(low),C_network_range_le003=sum(r['net_range']<=.03 for r in low),C_all_networks_above_oracle=sum(r['netmin_minus_oracle']>0 for r in low),A_count=len(high),A_best_network_within005=sum(abs(r['netmax_minus_oracle'])<=.05 for r in high)))
    return dict(rows=out,limitation='Saved E3 scalar records only; threshold labels are not network causal regimes. Pixel/energy/true solver residual gaps remain. This audit is not a primary submission contribution.')

def main():
    freeze=load(ROOT/'freeze.json'); protocol=load(ROOT/'protocol.json')
    assert sha(ROOT/'analyze.py')==freeze['source_sha256']
    assert sha(ROOT/'protocol.json')==freeze['protocol_sha256']
    for r in freeze['inputs']: assert sha(pathlib.Path(r['path']))==r['sha256']
    wall=time.perf_counter();cpu=time.process_time();t0=datetime.datetime.now(datetime.timezone.utc)
    assert t0.timestamp()<protocol['absolute_deadline_unix']
    result=dict(scope=protocol['scope'],l1=l1_analysis(),g1=g1_analysis(),negative=negative_analysis(),thresholds=thresholds())
    duration=time.perf_counter()-wall;cpu_seconds=time.process_time()-cpu
    assert duration<=protocol['cumulative_analysis_wall_budget_seconds'] and datetime.datetime.now(datetime.timezone.utc).timestamp()<protocol['absolute_deadline_unix']
    result['cost']=dict(start_utc=t0.isoformat(),end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=duration,cpu_seconds=cpu_seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CPU_threads=1,GPU=0,training=0,inference=0)
    save(ROOT/'results.json',result)
    save(ROOT/'output_manifest.json',dict(results_sha256=sha(ROOT/'results.json'),freeze_sha256=sha(ROOT/'freeze.json'),source_sha256=sha(ROOT/'analyze.py')))
    print(json.dumps(dict(cost=result['cost'],l1=result['l1']['summary'],g1=[{k:v for k,v in r.items() if k!='cells'} for r in result['g1']['groups']],negative=result['negative']['summary'],policies=result['negative']['fixed_policy'],original_error=result['negative']['original_reproduction_max_error']),indent=2))

if __name__=='__main__':main()
