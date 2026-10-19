"""Bounded LR-only self-reconstruction probe; frozen SR models, development OOF only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,argparse,resource,traceback,shutil,faulthandler
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
faulthandler.enable();torch.backends.mkldnn.enabled=False
torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent; P=json.loads((HERE/'protocol.json').read_text())
OLD=Path(P['inherited_output_root']); MODEL=Path(P['inherited_model_root'])
sys.path.insert(0,str(MODEL/'snapshot/src'))
import models
BASE=['always_enable','always_disable','band_median','energy_only','norm_ridge','proxy','directional_ridge']
CAND=['internal_transfer','internal_ridge']; METHODS=BASE+CAND; EPS=1e-8

def sha(f):
 h=hashlib.sha256()
 with open(f,'rb') as fd:
  for b in iter(lambda:fd.read(1048576),b''): h.update(b)
 return h.hexdigest()
def save(p,v):
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(np.bool_,)):return bool(v)
 return v

def planes(a,n,shave):
 a=np.asarray(a,dtype=np.float64).reshape(n,n,3)[shave:n-shave,shave:n-shave]
 f=np.fft.fftfreq(len(a));r=np.maximum(abs(f[:,None]),abs(f[None,:]));ff=np.fft.fft2(a,axes=(0,1))
 ans=[a]
 for lo,hi in [(0,.125),(.125,.25),(.25,.5001)]:
  mask=(r>=lo)&(r<hi)&(r>0);ans.append(np.fft.ifft2(ff*mask[:,:,None],axes=(0,1)).real)
 return ans

def cases(smoke):
 return [dict(id=f'calibration_{i:04d}_x{s}',image=i,scale=s,n=128//s) for i in (P['calibration_ids'][:1] if smoke else P['calibration_ids']) for s in P['scales']]

def prepare(out,cs,guard):
 for folder in ['observations','predictions','inherited_predictions']: (out/folder).mkdir()
 oldf=json.loads((OLD/'frozen.json').read_text());cps=json.loads((OLD/'checkpoint_manifest.json').read_text()); checked={}
 for f,v in oldf['sources'].items():
  guard(); assert sha(f)==v,(f,'inherited source drift');checked[f]=v
 for c in cps.values():assert sha(c['path'])==c['sha256'];checked[c['path']]=c['sha256']
 manifest={}
 for c in cs:
  guard(); f=OLD/'observations'/(c['id']+'.npy');dest=out/'observations'/f.name;shutil.copy2(f,dest)
  manifest[str(f)]=sha(f);assert sha(dest)==manifest[str(f)];dest.chmod(0o444)
  for name in P['models']:
   f=OLD/'predictions'/(c['id']+'__'+name+'.npz');manifest[str(f)]=sha(f)
   dest=out/'inherited_predictions'/f.name;shutil.copy2(f,dest);assert sha(dest)==manifest[str(f)];dest.chmod(0o444)
 # Hash calibration labels without parsing or using them. Actual labels only opened after lower feature freeze.
 manifest[str(OLD/'calibration_rows.json')]=sha(OLD/'calibration_rows.json')
 dest=out/'calibration_rows.json';shutil.copy2(OLD/'calibration_rows.json',dest);assert sha(dest)==manifest[str(OLD/'calibration_rows.json')];dest.chmod(0o444)
 save(out/'input_manifest.json',manifest);save(out/'inherited_manifest.json',checked);save(out/'cases.json',cs)
 save(out/'frozen.json',{'source_sha256':sha(HERE/'run.py'),'protocol_sha256':sha(HERE/'protocol.json'),'inputs_sha256':sha(out/'input_manifest.json'),'inherited_sha256':sha(out/'inherited_manifest.json'),'time_epoch':time.time()})

def infer(out,cs,guard):
 nets={}
 for name in P['models']:
  ck=torch.load(MODEL/'snapshot/checkpoints'/name/'epoch-last.pth',map_location='cpu',weights_only=True);assert ck['epoch']==1000
  m=models.make(ck['model'],load_sd=True).cpu().eval()
  for par in m.parameters():par.requires_grad_(False)
  nets[name]=m
 rows=[];journal=[];queries=0
 for c in cs:
  guard();tic=time.monotonic();y=np.load(out/'observations'/(c['id']+'.npy'));n=c['n'];s=c['scale']
  low=F.interpolate(torch.from_numpy(y)[None],size=(n//s,n//s),mode='bicubic',antialias=True,align_corners=False).clamp(0,1)
  v=(np.arange(n)+.5)/n;q=np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)
  qt=torch.tensor((2*q-1).astype(np.float32))[None]
  tp=planes(y.transpose(1,2,0),n,4//s);Z=max(float(np.mean((tp[0]-tp[0].mean((0,1),keepdims=True))**2)),1e-8)
  for name,m in nets.items():
   pp={}
   with torch.inference_mode():
    m.gen_feat(low*2-1)
    for typ,cell in [('point',0.),('native',2/n)]:
     ct=torch.full_like(qt,cell);parts=[]
     for k in range(0,n*n,4096):
      guard();a=m.query_rgb(qt[:,k:k+4096],ct[:,k:k+4096])[0].numpy()*.5+.5;assert np.isfinite(a).all();parts.append(a);queries+=len(a)
     pp[typ]=np.concatenate(parts)
   np.savez_compressed(out/'predictions'/(c['id']+'__'+name+'.npz'),**pp,lower_input=low[0].numpy())
   p0=planes(pp['point'],n,4//s);ph=planes(pp['native'],n,4//s)
   for band,(p,h,t) in enumerate(zip(p0,ph,tp)):
    e0=float(np.mean((p-t)**2));eh=float(np.mean((h-t)**2));gain=e0-eh
    rows.append(dict(case=c['id'],image=c['image'],scale=s,model=name,band=band,n=n,shave=4//s,Z_lower=Z,gain_lower=gain,transfer_feature=gain/Z,point_lower_mse=e0,native_lower_mse=eh))
  journal.append({'case':c['id'],'seconds':time.monotonic()-tic,'queries_cumulative':queries});save(out/'inference_journal.json',journal)
  print('LOWER',len(journal),len(cs),c['id'],round(journal[-1]['seconds'],3),flush=True)
 save(out/'lower_features.json',rows)
 manifest={str(f.relative_to(out)):sha(f) for f in sorted((out/'predictions').glob('*.npz'))};manifest['lower_features.json']=sha(out/'lower_features.json')
 save(out/'lower_cohort.json',manifest);save(out/'lower_freeze.json',{'cohort_sha256':sha(out/'lower_cohort.json'),'time_epoch':time.time(),'original_target_labels_parsed':False,'queries':queries})
 return rows

def fit(X,y):
 X=np.asarray(X);mu=X.mean(0);sd=X.std(0);sd[sd<1e-12]=1
 A=np.column_stack([np.ones(len(X)),(X-mu)/sd]);reg=np.eye(A.shape[1]);reg[0,0]=0
 beta=np.linalg.solve(A.T@A+reg,A.T@np.asarray(y));return dict(mu=mu.tolist(),sd=sd.tolist(),beta=beta.tolist())
def pred(f,x):return float(np.r_[1,(np.asarray(x)-f['mu'])/f['sd']]@f['beta'])
def feat(r,method):return r['features']+([r['proxy_alignment']/r['Z']] if method=='directional_ridge' else [r['transfer_feature']] if method=='internal_ridge' else [])

def forecast(out,low,guard):
 # Only here parse existing calibration HR outcomes; none of the old evaluation files are used.
 manifest=json.loads((out/'input_manifest.json').read_text());assert sha(out/'calibration_rows.json')==manifest[str(OLD/'calibration_rows.json')]
 old=json.loads((out/'calibration_rows.json').read_text());lm={(r['case'],r['model'],r['band']):r for r in low};rows=[]
 for r in old:
  key=(r['case'],r['model'],r['band'])
  if key in lm:
   t=dict(r);t['transfer_feature']=lm[key]['transfer_feature'];t['gain_lower']=lm[key]['gain_lower'];t['fold']=(t['image']-1)%8;rows.append(t)
 assert len(rows)==512
 fits={}
 for model in P['models']:
  mr=[r for r in rows if r['model']==model]
  for fold in range(8):
   guard();train=[r for r in mr if r['fold']!=fold];hold=[r for r in mr if r['fold']==fold];assert len(train)==224 and len(hold)==32
   fs={method:fit([feat(r,method) for r in train],[r['gain']/r['Z'] for r in train]) for method in ['norm_ridge','directional_ridge','internal_ridge']}
   fits[f'{model}_{fold}']={'train_images':sorted(set(r['image'] for r in train)),'held_images':sorted(set(r['image'] for r in hold)),'fits':fs}
   for r in hold:
    group=[t for t in train if t['scale']==r['scale'] and t['band']==r['band']]
    forecasts={'always_enable':1.,'always_disable':-1.,'band_median':float(np.median([t['gain']/t['Z'] for t in group]))*r['Z'],'energy_only':r['energy_gain'],'proxy':r['proxy_gain'],'internal_transfer':r['transfer_feature']*r['Z']}
    for method,f in fs.items():forecasts[method]=pred(f,feat(r,method))*r['Z']
    r['forecasts']=forecasts
 save(out/'oof_fits.json',fits);save(out/'oof_rows.json',rows)
 save(out/'oof_freeze.json',{'time_epoch':time.time(),'rows_sha256':sha(out/'oof_rows.json'),'fits_sha256':sha(out/'oof_fits.json'),'scope':'development OOF using prior calibration labels; not new blinded test'})
 return rows

def metric_dist(rs,method,W):
 # W: bootstrap/image multiplicities, one original-cohort row first.
 iw=W[:,np.array([r['image']-1 for r in rs])];g=np.array([r['gain'] for r in rs]);pr=np.array([r['forecasts'][method] for r in rs]);negative=g<-EPS;positive=g>EPS;harm=pr<0
 n=iw@negative;p=iw@positive;tp=iw@(negative&harm);fp=iw@(positive&harm)
 with np.errstate(divide='ignore',invalid='ignore'):
  recall=tp/n;precision=tp/(tp+fp);fpr=fp/p;ba=.5*(recall+1-fpr)
 return dict(negative=n,positive=p,TP=tp,FP=fp,recall=recall,precision=precision,FPR=fpr,BA=ba)

def auc_dist(rs,method,W,eligible):
 ans=[]
 for s,b in eligible:
  rr=[r for r in rs if r['scale']==s and r['band']==b];neg=[r for r in rr if r['gain']<-EPS];pos=[r for r in rr if r['gain']>EPS]
  a=np.array([r['forecasts'][method] for r in neg]);bval=np.array([r['forecasts'][method] for r in pos]);v=(a[:,None]<bval[None,:]).astype(float)+.5*(a[:,None]==bval[None,:])
  wn=W[:,[r['image']-1 for r in neg]];wp=W[:,[r['image']-1 for r in pos]]
  with np.errstate(divide='ignore',invalid='ignore'): ans.append(np.sum((wn@v)*wp,axis=1)/(wn.sum(1)*wp.sum(1)))
 return np.mean(ans,axis=0) if ans else np.full(len(W),np.nan)

def ci(x):
 x=np.asarray(x);v=x[np.isfinite(x)];return {'ci95':np.quantile(v,[.025,.975]).tolist() if len(v) else None,'valid':len(v),'missing':len(x)-len(v)}

def analyze(out,rows,guard):
 rng=np.random.default_rng(20261012);draws=rng.integers(0,32,size=(2000,32));W=np.vstack([np.ones(32),np.array([np.bincount(a,minlength=32) for a in draws])]);save(out/'bootstrap_draws.json',draws.tolist());result={}
 for model in P['models']:
  guard();rr=[r for r in rows if r['model']==model];br=[r for r in rr if r['band']>0];rgb=[r for r in rr if r['band']==0];eligible=[];cells=[]
  for s in P['scales']:
   for band in [1,2,3]:
    cc=[r for r in br if r['scale']==s and r['band']==band];ng=sum(r['gain']<-EPS for r in cc);po=sum(r['gain']>EPS for r in cc)
    if ng and po:eligible.append((s,band))
    cells.append({'scale':s,'band':band,'negative':ng,'positive':po,'ties':32-ng-po,'lower_positive_upper_negative':sum(r['gain']<-EPS and r['gain_lower']>0 for r in cc),'lower_negative_upper_positive':sum(r['gain']>EPS and r['gain_lower']<0 for r in cc),'metrics':{k:{x:clean(v[0]) for x,v in metric_dist(cc,k,W[:1]).items()} for k in METHODS}})
  stats={k:metric_dist(br,k,W) for k in METHODS};aucs={k:auc_dist(br,k,W,eligible) for k in METHODS}
  iw=W[:,[r['image']-1 for r in rgb]];den=iw.sum(1);pol={};mse0=np.array([r['point_mse'] for r in rgb]);mseh=np.array([r['native_mse'] for r in rgb]);oracle=np.minimum(mse0,mseh)
  for method in METHODS:
   native=np.array([r['forecasts'][method]>0 for r in rgb]);err=np.where(native,mseh,mse0)
   pol[method]={'dist':(iw@err)/den,'mse':float(np.mean(err)),'mean_PSNR':float(np.mean(-10*np.log10(err))),'oracle_regret':float(np.mean(err-oracle)),'disable_cases':int(np.sum(~native)),'harmful_native_kept':int(np.sum(native&(mse0-mseh<-EPS))),'beneficial_native_disabled':int(np.sum((~native)&(mse0-mseh>EPS)))}
  negimages=len(set(r['image'] for r in br if r['gain']<-EPS));gates={}
  for method in CAND:
   ba={k:ci((stats[method]['BA']-stats[k]['BA'])[1:]) for k in BASE};ad=ci((aucs[method]-aucs['norm_ridge'])[1:]);pm={k:ci((pol[k]['dist']-pol[method]['dist'])[1:]) for k in BASE}
   st=stats[method];class_ok=st['negative'][0]>=12 and negimages>=6 and st['recall'][0]>=.75 and st['precision'][0]>=.6 and st['FPR'][0]<=.25
   class_ok=class_ok and all(st['BA'][0]-stats[k]['BA'][0]>=.05 and ba[k]['valid']>=1000 and ba[k]['ci95'][0]>0 for k in BASE)
   auc_ok=len(eligible)>=3 and aucs[method][0]-aucs['norm_ridge'][0]>=.05 and ad['valid']>=1000 and ad['ci95'][0]>0
   policy_ok=all(pm[k]['valid']>=1000 and pm[k]['ci95'][0]>0 for k in BASE)
   gates[method]={'classification_pass':bool(class_ok),'within_cell_AUC_pass':bool(auc_ok),'RGB_policy_pass':bool(policy_ok),'combined_pass':bool(class_ok and auc_ok and policy_ok),'BA_improvement_vs_controls':ba,'within_cell_AUC_improvement_vs_norm':ad,'policy_MSE_improvement_vs_controls':pm}
  result[model]={'band_rows':len(br),'RGB_rows':len(rgb),'negative_images':negimages,'eligible_AUC_cells':eligible,'cells':cells,'band_metrics':{k:{a:clean(v[0]) for a,v in stats[k].items()} for k in METHODS},'macro_within_cell_AUC':{k:clean(aucs[k][0]) for k in METHODS},'RGB_policies':{k:{a:v for a,v in pol[k].items() if a!='dist'} for k in METHODS},'gates':gates}
 save(out/'summary.json',clean({'scope':P['scope'],'models':result,'bootstrap':'conditional fixed OOF forecasts; missing-class handling frozen','SR_training':0,'GPU':0,'new_lower_prediction_files':128,'old_original_predictions_reused':128}))
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
 launch={'start_epoch':time.time(),'absolute_deadline_epoch':time.time()+(120 if a.smoke else 1800),'cumulative_phase_budget_seconds':120 if a.smoke else 900,'CPU_threads':1,'nice':os.getpriority(os.PRIO_PROCESS,0),'GPU':0,'smoke':a.smoke}
 save(out/'launch.json',launch);start=time.monotonic();cpu=time.process_time();costs={};status='failed'
 def guard():
  if time.time()>=launch['absolute_deadline_epoch'] or time.monotonic()-start>=launch['cumulative_phase_budget_seconds']:raise TimeoutError('fixed budget exhausted')
 def phase(name,fn):
  tic=time.monotonic();val=fn();costs[name]=time.monotonic()-tic;save(out/'phase_costs.json',costs);return val
 try:
  cs=cases(a.smoke);phase('prepare',lambda:prepare(out,cs,guard));low=phase('lower_inference_and_features',lambda:infer(out,cs,guard))
  if not a.smoke:
   rows=phase('OOF',lambda:forecast(out,low,guard));phase('analysis',lambda:analyze(out,rows,guard))
  guard();status='complete';(out/'COMPLETE').write_text('complete development only; no SR training\n')
 except Exception as e:
  save(out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
 finally:
  save(out/'resources.json',{'status':status,'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'phase_seconds':costs,'CPU_threads':1,'nice':os.getpriority(os.PRIO_PROCESS,0),'GPU':0,'SR_training':0,'end_epoch':time.time(),'cost_scope':'incremental lower inference+OOF+scoring+IO; inherited original inference cost separately recorded in old resources, not zero'})
if __name__=='__main__':main()
