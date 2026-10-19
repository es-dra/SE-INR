"""Independent numpy-only audit; does not import experiment or execute SR models.
Checks saved crop/observation inputs. Does not claim original DIV2K PNG preprocessing
or model inference replication, for which torch/server inputs would be required.
"""
import argparse,json,hashlib,time,os,resource
from pathlib import Path
import numpy as np

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(p.read_text())
def grid(n,start=0,end=None):
 a=(np.arange(start,n if end is None else end)+.5)/n
 return np.array([(x,y) for x in a for y in a])
def spectral_planes(a):
 x=np.asarray(a,dtype=np.float64).reshape(128,128,3)[4:124,4:124]
 f=np.fft.fftfreq(120); rr=np.maximum(np.abs(f)[:,None],np.abs(f)[None,:]); coeff=np.fft.fft2(x,axes=(0,1))
 return [x]+[np.fft.ifft2(coeff*((rr>=lo)&(rr<hi)&(rr!=0))[...,None],axes=(0,1)).real for lo,hi in [(0,.125),(.125,.25),(.25,.5001)]]
def alias_ref(y,query,width):
 # Independent inverse via sequential separable Fourier evaluation, not experiment matrix.
 n=len(y[0]); m=np.arange(n); m=np.where(m<=n//2,m,m-n); keep=np.abs(m)<n/2
 modes=m[keep]; c=np.fft.fft2(y.mean(0))/n**2; c=c[np.ix_(keep,keep)]
 c=c/(np.sinc(modes[:,None]/n)*np.sinc(modes[None,:]/n))
 c*=np.sinc(width*modes[:,None])*np.sinc(width*modes[None,:])
 e0=np.exp(2j*np.pi*(query[:,0,None]-.5/n)*modes)
 e1=np.exp(2j*np.pi*(query[:,1,None]-.5/n)*modes)
 ans=np.einsum('qi,ij,qj->q',e0,c,e1).real
 return np.repeat(ans[:,None],3,axis=1)
def fit_ridge(X,y,lam):
 x=np.array(X); mu=np.mean(x,0); sd=np.sqrt(np.mean((x-mu)**2,0)); sd[sd<1e-12]=1
 z=(x-mu)/sd; # intercept handled via centering; equivalent but independent solve
 y=np.array(y); yc=y-y.mean(); zc=z-z.mean(0)
 beta=np.linalg.solve(zc.T@zc+lam*np.eye(z.shape[1]),zc.T@yc)
 intercept=y.mean()-z.mean(0)@beta
 return dict(mu=mu.tolist(),sd=sd.tolist(),beta=np.r_[intercept,beta].tolist())
def value(f,x): return float(f['beta'][0]+((np.array(x)-f['mu'])/f['sd'])@np.array(f['beta'][1:]))
def rates(rows,key,eps):
 labels=np.array([r['gain'] for r in rows]); pred=np.array([r['forecasts'][key] for r in rows]); neg=labels<-eps; pos=labels>eps; harm=pred<0
 n=int(neg.sum()); p=int(pos.sum()); tp=int((neg&harm).sum()); fp=int((pos&harm).sum()); cost=np.array([r['native_mse'] if v>0 else r['point_mse'] for r,v in zip(rows,pred)])
 return dict(rows=len(rows),active=n+p,negative=n,negative_images=len({r['image'] for r,z in zip(rows,neg) if z}),positive=p,ties=len(rows)-n-p,recall=tp/n if n else None,precision=tp/(tp+fp) if tp+fp else None,FPR=fp/p if p else None,BA=.5*(tp/n+1-fp/p) if n and p else None,MAE_normalized=float(np.mean([abs(v-r['gain'])/r['Z'] for r,v in zip(rows,pred)])) if key not in ['always_enable','always_disable'] else None,policy_mse=float(cost.mean()),oracle_regret=float(np.mean(cost-np.array([min(r['native_mse'],r['point_mse']) for r in rows]))))
def bootstrap(rows,candidate,baselines,P):
 ids=sorted({r['image'] for r in rows}); rng=np.random.default_rng(P['bootstrap_seed']); ix=rng.choice(len(ids),size=(P['bootstrap_replicates'],len(ids)),replace=True)
 # Sum per-draw sufficient statistics by indexing clusters directly instead of weight matrix.
 groups=[[r for r in rows if r['image']==i] for i in ids]; total=rows[0]['band']==0
 keys=[candidate]+baselines
 if total:
  denom=np.array([len(g) for g in groups])[ix].sum(1)
  vals={k:np.array([sum(r['native_mse'] if r['forecasts'][k]>0 else r['point_mse'] for r in g) for g in groups])[ix].sum(1)/denom for k in keys}; valid=np.ones(len(ix),bool)
 else:
  n=np.array([sum(r['gain']<-P['sign_epsilon'] for r in g) for g in groups])[ix].sum(1); p=np.array([sum(r['gain']>P['sign_epsilon'] for r in g) for g in groups])[ix].sum(1); valid=(n>0)&(p>0); vals={}
  for k in keys:
   tp=np.array([sum(r['gain']<-P['sign_epsilon'] and r['forecasts'][k]<0 for r in g) for g in groups])[ix].sum(1)
   fp=np.array([sum(r['gain']>P['sign_epsilon'] and r['forecasts'][k]<0 for r in g) for g in groups])[ix].sum(1)
   vals[k]=.5*(tp[valid]/n[valid]+1-fp[valid]/p[valid])
 return {k:dict(ci95=np.quantile(vals[k]-vals[candidate] if total else vals[candidate]-vals[k],[.025,.975]).tolist() if valid.any() else None,valid=int(valid.sum()),missing_class_replicates=int((~valid).sum())) for k in baselines}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--source-root',default=str(Path(__file__).parent)); ap.add_argument('--snapshot-root'); ap.add_argument('--result'); a=ap.parse_args(); out=Path(a.out); source=Path(a.source_root); P=read(source/'protocol.json'); started=time.monotonic(); checks=[]; errors=[]; diffs={}; counter={}
 def check(ok,label):
  checks.append(label)
  if not ok: errors.append(label)
 def close(x,y,label,tol=1e-10):
  if x is None or y is None: check(x is y,label); return
  if isinstance(x,dict):
   check(set(x)==set(y),label+' keys')
   for k in x:
    if k in y: close(x[k],y[k],label+'.'+k,tol)
  elif isinstance(x,(list,tuple)):
   check(len(x)==len(y),label+' length')
   for i,(xx,yy) in enumerate(zip(x,y)): close(xx,yy,label+f'[{i}]',tol)
  elif isinstance(x,(int,float,np.number)) and not isinstance(x,bool):
   d=abs(float(x)-float(y)); diffs[label.split('.')[0]]=max(d,diffs.get(label.split('.')[0],0)); check(np.isfinite(d) and d<=tol*(1+abs(float(x))),label)
  else: check(x==y,label)
 check((out/'COMPLETE').exists(),'complete sentinel'); check(not (out/'failure.txt').exists(),'no failure'); frozen=read(out/'frozen.json'); check(not frozen['smoke'],'formal not smoke'); check(frozen['device']=='cpu' and frozen['threads']==P['cpu_threads'] and frozen['nice']==10 and frozen['SR_training'] is False,'resource settings')
 sources=frozen['sources']; inherited=P['inherited_root']; snapshot=Path(a.snapshot_root) if a.snapshot_root else source.parent/'collected/codex-20261011-readout-probe-v4/snapshot'
 for name,h in sources.items():
  p=Path(name)
  if name.startswith(inherited+'/snapshot/'):
   local=snapshot/name.split('/snapshot/',1)[1]
  else: local=source/p.name
  check(local.exists() and digest(local)==h,'source '+name)
 check(any(Path(k).name=='experiment.py' for k in sources),'experiment frozen'); check(any(Path(k).name=='protocol.json' for k in sources),'protocol frozen')
 for model,cp in read(out/'checkpoint_manifest.json').items():
  local=snapshot/'checkpoints'/model/'epoch-last.pth'; check(local.exists() and digest(local)==cp['sha256'],'checkpoint '+model); check(cp['epoch']==1000,'epoch '+model)
 for manifest in ['input_manifest.json','prediction_cohort.json','forecast_cohort.json']:
  items=read(out/manifest)
  for f,h in items.items(): check((out/f).exists() and digest(out/f)==h,manifest+' '+f)
  counter[manifest]=len(items)
 pf=read(out/'prediction_freeze.json'); ff=read(out/'forecast_freeze.json'); check(pf['sha256']==digest(out/'prediction_cohort.json'),'prediction freeze digest'); check(ff['sha256']==digest(out/'forecast_cohort.json'),'forecast freeze digest'); check(ff['test_truth_opened'] is False,'claimed prediction truth isolation'); check(pf['time_epoch']<=ff['time_epoch'],'freeze chronological order')
 cases=read(out/'cases.json'); expected=[]
 for split,ids in [('calibration',P['calibration_ids']),('test',P['test_ids'])]:
  for i in ids:
   for s in [2,4]: expected.append(f'{split}_{i:04d}_x{s}')
 expected += [f'alias_n{n}_k{k}_a{ax}_p{ph}' for n,ks in [(16,[3,5]),(32,[9,13])] for k in ks for ax in [0,1] for ph in [0,1]]
 check([c['id'] for c in cases]==expected,'complete fixed cohort'); check(len(set(expected))==len(expected),'unique case IDs'); check(set(P['calibration_ids']).isdisjoint(P['test_ids']),'image split isolation'); check(len(read(out/'dataset_manifest.json'))==44,'44 source PNG hashes recorded, not independently preprocessed')
 allrows=[]; aliases=[]; caches={}
 for c in cases:
  y=np.load(out/'observations'/(c['id']+'.npy')); check(np.isfinite(y).all() and y.shape==(3,c['n'],c['n']),'observation '+c['id']);
  if c['kind']=='natural':
   t=np.load(out/'truth'/(c['id']+'.npy')); check(t.shape==(3,128,128) and np.isfinite(t).all() and t.min()>=0 and t.max()<=1,'crop truth '+c['id'])
   ym=y.astype(float).mean(0); spectrum=np.abs(np.fft.fft2(ym-ym.mean()))**2; weights=spectrum/spectrum.sum() if spectrum.sum()>0 else spectrum; m=np.fft.fftfreq(c['n']); radii=np.hypot(m[:,None],m[None,:]); centroid=float(np.sum(weights*radii)); w=weights[weights>0]; entropy=float(-np.sum(w*np.log(w))/np.log(weights.size)); yf=y.astype(float); grad=float(np.mean(np.diff(yf,axis=1)**2)+np.mean(np.diff(yf,axis=2)**2)); tt=spectral_planes(t.transpose(1,2,0))
  for model in P['models']:
   z=dict(np.load(out/'predictions'/(c['id']+'__'+model+'.npz'))); check(all(np.isfinite(v).all() for v in z.values()),'finite prediction '+c['id']+model)
   if c['kind']=='natural':
    check(set(z)=={'point','native','proxy'},'prediction fields '+c['id']+model); check(all(v.shape==(16384,3) for v in z.values()),'prediction shape '+c['id']+model); pp,hh,bb=spectral_planes(z['point']),spectral_planes(z['native']),spectral_planes(z['proxy']); Z=max(float(np.mean((bb[0]-bb[0].mean((0,1),keepdims=True))**2)),1e-8)
    caches[(c['id'],model)]=(z,tt)
    for band,(p,h,b,tb) in enumerate(zip(pp,hh,bb,tt)):
     d=h-p; v=float(np.mean(d*d)); align=float(2*np.mean((b-p)*d)); ep=float(np.mean(p*p)) if band else float(np.mean((p-p.mean((0,1),keepdims=True))**2)); eb=float(np.mean(b*b)) if band else Z; feat=[float(c['scale']==2),float(c['scale']==4)]+[float(band==j) for j in range(4)]+[float(np.log1p(v/Z)),float(np.log1p(ep/Z)),float(np.log1p(eb/Z)),float(np.log1p(grad/Z)),centroid,entropy]; e0=float(np.mean((p-tb)**2)); eh=float(np.mean((h-tb)**2))
     r=dict(case=c['id'],image=c['image'],split=c['split'],scale=c['scale'],model=model,band=band,Z=Z,change_energy=v,proxy_alignment=align,proxy_gain=align-v,energy_gain=-v,features=feat,gain=e0-eh,point_mse=e0,native_mse=eh); allrows.append(r)
     close((e0-eh)-(align-v),2*float(np.mean((tb-b)*d)),'natural_identity',1e-12)
   else:
    q=grid(64,16,48); p=z['point'].astype(float); check(p.shape==(1024,3),'alias prediction shape')
    obsback=alias_ref(y,grid(c['n']),1/c['n']); resid=float(np.max(abs(obsback-y.transpose(1,2,0).reshape(-1,3)))); close(resid,float(z['observation_consistency_max']),'alias_observation_residual',1e-12)
    for fac in [1,2]:
     h=z['native'+str(fac)].astype(float); b=alias_ref(y,q,fac/64); close(b.tolist(),z['proxy'+str(fac)].tolist(),'alias_proxy',1e-11); aliases.append(dict(case=c['id'],model=model,fac=fac,proxy_gain=float(np.mean((p-b)**2)-np.mean((h-b)**2)),change_energy=float(np.mean((h-p)**2)),observation_consistency_max=resid))
 cal=[r for r in allrows if r['split']=='calibration']; test=[r for r in allrows if r['split']=='test']; close(cal,read(out/'calibration_rows.json'),'calibration_rows',1e-11)
 fit={}; forecast=[]
 for model in P['models']:
  cc=[r for r in cal if r['model']==model]; fit[model]={}
  for kind in ['norm_ridge','directional_ridge']:
   xx=[r['features']+([r['proxy_alignment']/r['Z']] if kind=='directional_ridge' else []) for r in cc]; fit[model][kind]=fit_ridge(xx,[r['gain']/r['Z'] for r in cc],P['ridge_lambda'])
  for r in [r for r in test if r['model']==model]:
   t={k:v for k,v in r.items() if k not in ['gain','point_mse','native_mse']}; group=[rr for rr in cc if rr['scale']==r['scale'] and rr['band']==r['band']]; f=dict(always_enable=1.,always_disable=-1.,band_median=float(np.median([rr['gain']/rr['Z'] for rr in group]))*r['Z'],energy_only=r['energy_gain'],proxy=r['proxy_gain'])
   for kind in ['norm_ridge','directional_ridge']: f[kind]=value(fit[model][kind],r['features']+([r['proxy_alignment']/r['Z']] if kind=='directional_ridge' else []))*r['Z']
   t['forecasts']=f; forecast.append(t)
 close(fit,read(out/'calibration_fits.json'),'ridge_fits',1e-9); close(forecast,read(out/'forecasts.json'),'forecasts',1e-10); close(aliases,read(out/'alias_forecasts.json'),'alias_forecasts',1e-11)
 lookup={(r['case'],r['model'],r['band']):r for r in test}; scored=[]
 for f in forecast:
  t=dict(f); raw=lookup[(f['case'],f['model'],f['band'])]; t.update({k:raw[k] for k in ['gain','point_mse','native_mse']}); scored.append(t)
 close(scored,read(out/'test_scores.json'),'test_scores',1e-10)
 baselines=['always_enable','always_disable','band_median','energy_only','norm_ridge']; candidates=['proxy','directional_ridge']; modelsummary={}
 for model in P['models']:
  br=[r for r in scored if r['model']==model and r['band']>0]; total=[r for r in scored if r['model']==model and r['band']==0]; bm={k:rates(br,k,P['sign_epsilon']) for k in baselines+candidates}; policies={k:rates(total,k,P['sign_epsilon']) for k in baselines+candidates}; gates={}
  for k in candidates:
   cb=bootstrap(br,k,baselines,P); pb=bootstrap(total,k,baselines,P); m=bm[k]; best=max([bm[b]['BA'] for b in baselines if bm[b]['BA'] is not None],default=None)
   cg=bool(m['negative']>=12 and m['negative_images']>=6 and m['recall'] is not None and m['recall']>=.75 and m['precision'] is not None and m['precision']>=.6 and m['FPR'] is not None and m['FPR']<=.25 and best is not None and m['BA']>=best+.05 and all(v['ci95'] and v['ci95'][0]>0 for v in cb.values())); pg=all(v['ci95'] and v['ci95'][0]>0 for v in pb.values()); gates[k]=dict(classification_pass=cg,RGB_policy_pass=pg,combined_pass=cg and pg,paired_image_cluster_BA=cb,paired_image_cluster_policy_mse=pb)
  modelsummary[model]=dict(band_metrics=bm,RGB_policies=policies,gates=gates)
 close(modelsummary,read(out/'summary.json')['models'],'model_summary',1e-9)
 specs=read(out/'truth/alias_specs.json'); alias_scores=[]; flips=0; maxobs=0.; maxlow=0.; maxid=0.
 for r in aliases:
  c=next(c for c in cases if c['id']==r['case']); spec=specs[r['case']]; toks=c['id'].split('_'); n=int(toks[1][1:]); k=int(toks[2][1:]); ax=int(toks[3][1:]); ph=int(toks[4][1:])*np.pi/4
  close(spec,dict(k_low=k,k_high=n-k,axis=ax,phase_low=ph,phase_high=np.pi-ph,amplitude_low=.25*k/(n-k),amplitude_high=.25),'alias_specs')
  gobs=grid(n); lo=.5+.25*k/(n-k)*np.sinc(k/n)*np.cos(2*np.pi*k*gobs[:,ax]+ph); hi=.5+.25*np.sinc((n-k)/n)*np.cos(2*np.pi*(n-k)*gobs[:,ax]+np.pi-ph); maxobs=max(maxobs,float(np.max(abs(lo-hi)))); close(np.tile(lo.reshape(n,n)[None],(3,1,1)).astype('float32').tolist(),np.load(out/'observations'/(r['case']+'.npy')).tolist(),'alias_exact_observation',1e-12)
  z=dict(np.load(out/'predictions'/(r['case']+'__'+r['model']+'.npz'))); p=z['point'].astype(float); h=z['native'+str(r['fac'])].astype(float); b=z['proxy'+str(r['fac'])]; d=h-p; gs=[]; q=grid(64,16,48)
  for which in ['low','high']:
   kk=spec['k_'+which]; t=(.5+spec['amplitude_'+which]*np.sinc(kk*r['fac']/64)*np.cos(2*np.pi*kk*q[:,ax]+spec['phase_'+which]))[:,None]; e0=float(np.mean((p-t)**2)); eh=float(np.mean((h-t)**2)); gain=e0-eh; residual=gain-r['proxy_gain']-2*float(np.mean((t-b)*d)); maxid=max(maxid,abs(residual)); gs.append(gain)
   if which=='low': maxlow=max(maxlow,float(np.max(abs(t-b))))
   alias_scores.append(dict(r,truth=which,gain=gain,identity_residual=residual,point_mse=e0,native_mse=eh))
  flips+=int(gs[0]*gs[1]<0 and min(abs(np.array(gs)))>P['sign_epsilon'])
 close(alias_scores,read(out/'alias_scores.json'),'alias_scores',1e-10); summary=read(out/'summary.json')
 close(dict(test_rows=len(scored),alias_pairs=len(alias_scores)//2,alias_sign_flips=flips,alias_observation_pair_max_error=maxobs,alias_low_proxy_max_error=maxlow,alias_identity_max_error=maxid,model_training_runs=0,scope=P['scope']),{k:summary[k] for k in summary if k!='models'},'summary',1e-10)
 journal=read(out/'inference_journal.json'); check([r['case'] for r in journal]==expected,'inference journal complete'); count=0
 for c,j in zip(cases,journal):
  count+=len(P['models'])*(2*16384 if c['kind']=='natural' else 3*1024); check(j['queries_cumulative']==count,'journal query accounting '+c['id']); check(j['seconds']>=0,'nonnegative inference cost')
 check(pf['queries']==count,'total queries'); cost=read(out/'resources.json'); phases=read(out/'phase_costs.json'); close(phases,cost['phases'],'phase_costs'); check([r['phase'] for r in phases]==['prepare','predict','forecast','score'],'full phases'); launch=frozen['launch']; check(cost['status']=='complete' and cost['gpu_processes']==0 and cost['SR_training_runs']==0,'complete zero GPU zero SR train'); check(cost['script_wall_seconds']>=sum(p['seconds'] for p in phases)-1e-4,'phase cost bounded by script'); check(cost['wall_seconds']>=cost['script_wall_seconds']-1e-4,'launcher includes script'); check(cost['wall_seconds']<=P['wall_budget_seconds'] and cost['script_wall_seconds']<=P['wall_budget_seconds'],'absolute and cumulative budget'); check(launch['start_epoch']+cost['wall_seconds']<=launch['absolute_deadline_epoch']+.01,'fixed absolute deadline'); check(ff['time_epoch']<launch['start_epoch']+cost['wall_seconds'],'forecast froze before end');
 result=dict(status='passed' if not errors else 'failed',checks=len(checks),failed_checks=errors,max_differences=diffs,complete_rows=dict(calibration=len(cal),test=len(scored),aliases=len(alias_scores),cases=len(cases),model_queries=count),manifest_counts=counter,statistics=modelsummary,alias_sign_flips=flips,resources=cost,independent_audit_seconds=time.monotonic()-started,cpu_user=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system=resource.getrusage(resource.RUSAGE_SELF).ru_stime,limitations=['No SR model inference replicated.','Original source PNG preprocessing not independently rerun; hashes recorded and frozen cropped truth checked.','Natural bicubic proxy values retained as saved; numpy does not replicate torch bicubic interpolation.','Timestamp freeze and source flow evidence cannot prove absence of hidden historical truth access.','12 image clusters; band/scale rows are correlated, not independent trials.'])
 dest=Path(a.result) if a.result else source/'audit.json'; dest.write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps({k:result[k] for k in ['status','checks','failed_checks','max_differences','complete_rows','independent_audit_seconds']},indent=2)); return 0 if not errors else 1
if __name__=='__main__': raise SystemExit(main())
