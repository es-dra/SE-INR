"""Observation-only forecasts for frozen native versus zero-cell readout.
Calibration truth is permitted only for calibration. Test truth is opened after forecast freeze.
No SR optimiser/gradient. Conventional proxy/ridge tools, not an unbiased-risk theorem.
"""
import os, sys, json, hashlib, time, argparse, traceback, resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
torch.backends.mkldnn.enabled=False
torch.set_num_threads(4); torch.set_num_interop_threads(1)
sys.path.insert(0,str(Path(P['inherited_root'])/'snapshot/src'))
import models

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False))
def coords(n,crop=None):
 v=(np.arange(n,dtype=float)+.5)/n
 if crop: v=v[crop[0]:crop[1]]
 return np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)
def cases(smoke):
 rows=[]
 for split,ids in [('calibration',P['calibration_ids']),('test',P['test_ids'])]:
  if smoke: ids=ids[:1] if split=='calibration' else []
  for i in ids:
   for s in [2,4]: rows.append(dict(id=f'{split}_{i:04d}_x{s}',split=split,image=i,n=128//s,scale=s,kind='natural'))
 for n,ks in [(16,[3,5]),(32,[9,13])]:
  for k in ks:
   for ax in [0,1]:
    for ph in [0,1]: rows.append(dict(id=f'alias_n{n}_k{k}_a{ax}_p{ph}',split='alias',n=n,kind='alias'))
 return rows if not smoke else rows[:2]+[r for r in rows if r['kind']=='alias'][:1]
def prepare(out,cs):
 for d in ['observations','truth','predictions']: (out/d).mkdir()
 manifest={}; alias_specs={}
 for c in cs:
  if c['kind']=='natural':
   split='train' if c['split']=='calibration' else 'valid'
   fp=Path(P['dataset_root'])/f'DIV2K_{split}_HR'/f"{c['image']:04d}.png"
   im=Image.open(fp).convert('RGB'); w,h=im.size; assert min(w,h)>=128
   t=np.array(im.crop(((w-128)//2,(h-128)//2,(w-128)//2+128,(h-128)//2+128)),dtype='float32').transpose(2,0,1)/255
   y=F.interpolate(torch.from_numpy(t)[None],size=(c['n'],c['n']),mode='bicubic',antialias=True,align_corners=False).clamp(0,1)[0].numpy()
   np.save(out/'truth'/(c['id']+'.npy'),t)
   manifest[str(fp)]=sha(fp)
  else:
   _,nn,kk,aa,pp=c['id'].split('_'); n=int(nn[1:]); k=int(kk[1:]); ax=int(aa[1:]); ph=int(pp[1:])*np.pi/4
   a=.25*k/(n-k); q=coords(n)
   y=.5+a*np.sinc(k/n)*np.cos(2*np.pi*k*q[:,ax]+ph)
   y=np.tile(y.reshape(n,n)[None],(3,1,1)).astype('float32')
   alias_specs[c['id']]=dict(k_low=k,k_high=n-k,axis=ax,phase_low=ph,phase_high=np.pi-ph,amplitude_low=a,amplitude_high=.25)
  np.save(out/'observations'/(c['id']+'.npy'),y)
 write(out/'cases.json',cs); write(out/'truth/alias_specs.json',alias_specs)
 write(out/'dataset_manifest.json',manifest)
 write(out/'input_manifest.json',{str(p.relative_to(out)):sha(p) for d in ['observations','truth'] for p in sorted((out/d).glob('*'))})
def alias_proxy(y,q,wh):
 n=y.shape[-1]; m=np.fft.fftfreq(n)*n; my,mx=np.meshgrid(m,m,indexing='ij')
 ok=(abs(my)<=((n-1)//2))&(abs(mx)<=((n-1)//2))
 coeff=np.fft.fft2(y.mean(0))/n**2
 yy=my[ok]; xx=mx[ok]; z=coeff[ok]/(np.sinc(yy/n)*np.sinc(xx/n))*np.sinc(yy*wh[0])*np.sinc(xx*wh[1])
 e=np.exp(2j*np.pi*(q[:,0,None]*yy+q[:,1,None]*xx-(yy+xx)[None,:]/(2*n)))
 b=(e@z).real
 return np.tile(b[:,None],(1,3))
def predict(out,cs,guard):
 nets={}; cpmeta={}
 for name in P['models']:
  cp=Path(P['inherited_root'])/'snapshot/checkpoints'/name/'epoch-last.pth'
  ck=torch.load(cp,map_location='cpu',weights_only=True); assert ck['epoch']==1000
  m=models.make(ck['model'],load_sd=True).cpu().eval()
  for p in m.parameters(): p.requires_grad_(False)
  nets[name]=m; cpmeta[name]=dict(path=str(cp),sha256=sha(cp),epoch=1000)
 write(out/'checkpoint_manifest.json',cpmeta); journal=[]; nq=0
 @torch.inference_mode()
 def query(m,q,wh):
  nonlocal nq
  qt=torch.from_numpy((2*q-1).astype('float32'))[None]; ct=torch.tensor(wh,dtype=torch.float32).view(1,1,2).expand_as(qt)*2
  arr=[]
  for i in range(0,len(q),4096):
   guard(); a=m.query_rgb(qt[:,i:i+4096],ct[:,i:i+4096])[0].numpy()*.5+.5
   assert np.isfinite(a).all(); arr.append(a); nq+=len(a)
  return np.concatenate(arr)
 for idx,c in enumerate(cs):
  guard(); tic=time.monotonic(); y=np.load(out/'observations'/(c['id']+'.npy'))
  q=coords(128) if c['kind']=='natural' else coords(64,[16,48])
  for name,m in nets.items():
   with torch.inference_mode(): m.gen_feat(torch.from_numpy(y)[None]*2-1)
   z={'point':query(m,q,(0,0))}
   if c['kind']=='natural':
    z['native']=query(m,q,(1/128,1/128))
    z['proxy']=F.interpolate(torch.from_numpy(y)[None],size=(128,128),mode='bicubic',align_corners=False)[0].numpy().transpose(1,2,0).reshape(-1,3)
   else:
    for fac in [1,2]:
     z['native'+str(fac)]=query(m,q,(fac/64,fac/64))
     z['proxy'+str(fac)]=alias_proxy(y,q,(fac/64,fac/64))
    yy=alias_proxy(y,coords(c['n']),(1/c['n'],1/c['n']))
    z['observation_consistency_max']=np.array(np.max(abs(yy-y.transpose(1,2,0).reshape(-1,3))))
   np.savez_compressed(out/'predictions'/(c['id']+'__'+name+'.npz'),**z)
  journal.append(dict(case=c['id'],seconds=time.monotonic()-tic,queries_cumulative=nq)); write(out/'inference_journal.json',journal)
  print('PRED',idx+1,len(cs),c['id'],round(journal[-1]['seconds'],3),flush=True)
 write(out/'prediction_cohort.json',{str(p.relative_to(out)):sha(p) for p in sorted((out/'predictions').glob('*'))})
 write(out/'prediction_freeze.json',dict(sha256=sha(out/'prediction_cohort.json'),time_epoch=time.time(),queries=nq))
def planes(a):
 a=a.reshape(128,128,3)[4:-4,4:-4].astype('float64'); f=np.fft.fftfreq(120); r=np.maximum(abs(f[:,None]),abs(f[None,:])); ff=np.fft.fft2(a,axes=(0,1))
 bands=[a]
 for lo,hi in [(0,.125),(.125,.25),(.25,.5001)]:
  mask=(r>=lo)&(r<hi)&(r>0); bands.append(np.fft.ifft2(ff*mask[:,:,None],axes=(0,1)).real)
 return bands
def observable_rows(out,cs):
 rows=[]
 for c in cs:
  if c['kind']!='natural': continue
  y=np.load(out/'observations'/(c['id']+'.npy')).astype('float64'); ym=y.mean(0); yf=abs(np.fft.fft2(ym-ym.mean()))**2
  f=np.fft.fftfreq(c['n']); rr=np.sqrt(f[:,None]**2+f[None,:]**2); w=yf/yf.sum() if yf.sum()>0 else yf
  cent=float((w*rr).sum()); entropy=float(-(w[w>0]*np.log(w[w>0])).sum()/np.log(w.size))
  gradient=float(np.mean(np.diff(y,axis=1)**2)+np.mean(np.diff(y,axis=2)**2))
  for name in P['models']:
   z=np.load(out/'predictions'/(c['id']+'__'+name+'.npz')); pa=planes(z['point']); ha=planes(z['native']); ba=planes(z['proxy'])
   Z=max(float(np.mean((ba[0]-ba[0].mean((0,1),keepdims=True))**2)),1e-8)
   for band,(p,h,b) in enumerate(zip(pa,ha,ba)):
    d=h-p; v=float(np.mean(d*d)); alignment=float(2*np.mean((b-p)*d)); ep=float(np.mean(p*p)) if band else float(np.mean((p-p.mean((0,1),keepdims=True))**2)); eb=float(np.mean(b*b)) if band else Z
    features=[float(c['scale']==2),float(c['scale']==4)]+[float(band==j) for j in range(4)]+[np.log1p(v/Z),np.log1p(ep/Z),np.log1p(eb/Z),np.log1p(gradient/Z),cent,entropy]
    rows.append(dict(case=c['id'],image=c['image'],split=c['split'],scale=c['scale'],model=name,band=band,Z=Z,change_energy=v,proxy_alignment=alignment,proxy_gain=alignment-v,energy_gain=-v,features=features))
 return rows
def truth_rows(out,rs):
 ans=[]; cache={}
 for r in rs:
  key=(r['case'],r['model']); band=r['band']
  if key not in cache:
   z=np.load(out/'predictions'/(key[0]+'__'+key[1]+'.npz')); truth=np.load(out/'truth'/(key[0]+'.npy')).transpose(1,2,0)
   cache[key]=(planes(truth.reshape(-1,3)),planes(z['point']),planes(z['native']))
  tt,pp,hh=cache[key]; e0=float(np.mean((pp[band]-tt[band])**2)); eh=float(np.mean((hh[band]-tt[band])**2)); t=dict(r); t.update(gain=e0-eh,point_mse=e0,native_mse=eh); ans.append(t)
 return ans
def ridge_fit(X,y):
 X=np.array(X); mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1
 A=np.column_stack([np.ones(len(X)),(X-mu)/sd]); penalty=np.eye(A.shape[1])*P['ridge_lambda']; penalty[0,0]=0
 beta=np.linalg.solve(A.T@A+penalty,A.T@np.array(y)); return dict(mu=mu.tolist(),sd=sd.tolist(),beta=beta.tolist())
def ridge_eval(f,x): return float(np.r_[1,(np.array(x)-f['mu'])/f['sd']]@f['beta'])
def forecast(out,cs):
 rs=observable_rows(out,cs); cal=truth_rows(out,[r for r in rs if r['split']=='calibration']); fits={}; forecasts=[]
 for name in P['models']:
  cc=[r for r in cal if r['model']==name]; fits[name]={}
  for kind in ['norm_ridge','directional_ridge']:
   def feat(r): return r['features']+([r['proxy_alignment']/r['Z']] if kind=='directional_ridge' else [])
   fits[name][kind]=ridge_fit([feat(r) for r in cc],[r['gain']/r['Z'] for r in cc])
  for r in [r for r in rs if r['model']==name and r['split']=='test']:
   rr=dict(r); group=[a for a in cc if a['scale']==r['scale'] and a['band']==r['band']]
   pred={'always_enable':1.,'always_disable':-1.,'band_median':float(np.median([a['gain']/a['Z'] for a in group]))*r['Z'],'energy_only':r['energy_gain'],'proxy':r['proxy_gain']}
   for kind in ['norm_ridge','directional_ridge']:
    xx=r['features']+([r['proxy_alignment']/r['Z']] if kind=='directional_ridge' else [])
    pred[kind]=ridge_eval(fits[name][kind],xx)*r['Z']
   rr['forecasts']=pred; forecasts.append(rr)
 alias=[]
 for c in cs:
  if c['kind']!='alias': continue
  for name in P['models']:
   z=np.load(out/'predictions'/(c['id']+'__'+name+'.npz')); p=z['point'].astype('float64')
   for fac in [1,2]:
    h=z['native'+str(fac)].astype('float64'); b=z['proxy'+str(fac)]; d=h-p
    alias.append(dict(case=c['id'],model=name,fac=fac,proxy_gain=float(np.mean((p-b)**2)-np.mean((h-b)**2)),change_energy=float(np.mean(d*d)),observation_consistency_max=float(z['observation_consistency_max'])))
 write(out/'calibration_rows.json',cal); write(out/'calibration_fits.json',fits); write(out/'forecasts.json',forecasts); write(out/'alias_forecasts.json',alias)
 files=['prediction_cohort.json','prediction_freeze.json','calibration_rows.json','calibration_fits.json','forecasts.json','alias_forecasts.json']
 write(out/'forecast_cohort.json',{f:sha(out/f) for f in files})
 write(out/'forecast_freeze.json',dict(sha256=sha(out/'forecast_cohort.json'),time_epoch=time.time(),test_truth_opened=False))
def metrics(rs,key):
 active=[r for r in rs if abs(r['gain'])>P['sign_epsilon']]; neg=[r for r in active if r['gain']<0]; pos=[r for r in active if r['gain']>0]
 tp=sum(r['forecasts'][key]<0 for r in neg); fp=sum(r['forecasts'][key]<0 for r in pos)
 return dict(rows=len(rs),active=len(active),negative=len(neg),negative_images=len(set(r['image'] for r in neg)),positive=len(pos),ties=len(rs)-len(active),recall=tp/len(neg) if neg else None,precision=tp/(tp+fp) if tp+fp else None,FPR=fp/len(pos) if pos else None,BA=.5*(tp/len(neg)+(len(pos)-fp)/len(pos)) if neg and pos else None,MAE_normalized=float(np.mean([abs(r['forecasts'][key]-r['gain'])/r['Z'] for r in rs])) if key not in ['always_enable','always_disable'] else None,policy_mse=float(np.mean([r['native_mse'] if r['forecasts'][key]>0 else r['point_mse'] for r in rs])),oracle_regret=float(np.mean([(r['native_mse'] if r['forecasts'][key]>0 else r['point_mse'])-min(r['native_mse'],r['point_mse']) for r in rs])))
def bootstrap(rs,kind,keys):
 ids=sorted(set(r['image'] for r in rs)); rng=np.random.default_rng(P['bootstrap_seed'])
 draws=rng.choice(len(ids),size=(P['bootstrap_replicates'],len(ids)),replace=True)
 weights=np.zeros((len(draws),len(ids)),dtype=float)
 for j in range(len(ids)): weights[:,j]=(draws==j).sum(1)
 by=[[r for r in rs if r['image']==i] for i in ids]
 if rs[0]['band']==0:
  count=weights@np.array([len(bag) for bag in by])
  def calc(key):
   costs=np.array([sum(r['native_mse'] if r['forecasts'][key]>0 else r['point_mse'] for r in bag) for bag in by])
   return (weights@costs)/count
  cm=calc(kind); valid=np.ones(len(draws),dtype=bool)
 else:
  neg=weights@np.array([sum(r['gain'] < -P['sign_epsilon'] for r in bag) for bag in by]); pos=weights@np.array([sum(r['gain'] > P['sign_epsilon'] for r in bag) for bag in by]); valid=(neg>0)&(pos>0)
  def calc(key):
   tp=weights@np.array([sum(r['gain'] < -P['sign_epsilon'] and r['forecasts'][key]<0 for r in bag) for bag in by])
   fp=weights@np.array([sum(r['gain'] > P['sign_epsilon'] and r['forecasts'][key]<0 for r in bag) for bag in by])
   return .5*(tp[valid]/neg[valid]+1-fp[valid]/pos[valid])
  cm=calc(kind)
 result={}
 for key in keys:
  v=calc(key)-cm if rs[0]['band']==0 else cm-calc(key)
  result[key]=dict(ci95=np.quantile(v,[.025,.975]).tolist() if len(v) else None,valid=int(valid.sum()),missing_class_replicates=int((~valid).sum()))
 return result
def score(out,cs):
 for fn,h in json.loads((out/'forecast_cohort.json').read_text()).items(): assert sha(out/fn)==h
 for fn,h in json.loads((out/'prediction_cohort.json').read_text()).items(): assert sha(out/fn)==h
 rs=truth_rows(out,json.loads((out/'forecasts.json').read_text())); write(out/'test_scores.json',rs)
 summary={}; baselines=['always_enable','always_disable','band_median','energy_only','norm_ridge']; candidates=['proxy','directional_ridge']
 for name in P['models']:
  bands=[r for r in rs if r['model']==name and r['band']>0]; total=[r for r in rs if r['model']==name and r['band']==0]
  st={k:metrics(bands,k) for k in baselines+candidates}; policies={k:metrics(total,k) for k in baselines+candidates}; gates={}
  for k in candidates:
   cb=bootstrap(bands,k,baselines); pb=bootstrap(total,k,baselines); m=st[k]; best=max(st[b]['BA'] for b in baselines if st[b]['BA'] is not None) if m['BA'] is not None else None
   classgate=bool(m['negative']>=12 and m['negative_images']>=6 and m['recall'] is not None and m['recall']>=.75 and m['precision'] is not None and m['precision']>=.6 and m['FPR'] is not None and m['FPR']<=.25 and best is not None and m['BA']>=best+.05 and all(v['ci95'] and v['ci95'][0]>0 for v in cb.values()))
   policygate=all(v['ci95'] and v['ci95'][0]>0 for v in pb.values())
   gates[k]=dict(classification_pass=classgate,RGB_policy_pass=policygate,combined_pass=classgate and policygate,paired_image_cluster_BA=cb,paired_image_cluster_policy_mse=pb)
  summary[name]=dict(band_metrics=st,RGB_policies=policies,gates=gates)
 asp=json.loads((out/'truth/alias_specs.json').read_text()); ar=[]; q=coords(64,[16,48]); flips=0; identitymax=0; lowerr=0; obsmax=0
 for r in json.loads((out/'alias_forecasts.json').read_text()):
  a=asp[r['case']]; c=next(c for c in cs if c['id']==r['case']); z=np.load(out/'predictions'/(r['case']+'__'+r['model']+'.npz')); p=z['point'].astype('float64'); h=z['native'+str(r['fac'])].astype('float64'); b=z['proxy'+str(r['fac'])]; d=h-p; gs=[]
  lowobs=.5+a['amplitude_low']*np.sinc(a['k_low']/c['n'])*np.cos(2*np.pi*a['k_low']*coords(c['n'])[:,a['axis']]+a['phase_low'])
  highobs=.5+a['amplitude_high']*np.sinc(a['k_high']/c['n'])*np.cos(2*np.pi*a['k_high']*coords(c['n'])[:,a['axis']]+a['phase_high']); obsmax=max(obsmax,float(np.max(abs(lowobs-highobs))))
  for which in ['low','high']:
   k=a['k_'+which]; t=(.5+a['amplitude_'+which]*np.sinc(k*r['fac']/64)*np.cos(2*np.pi*k*q[:,a['axis']]+a['phase_'+which]))[:,None]
   gain=float(np.mean((p-t)**2)-np.mean((h-t)**2)); residual=gain-r['proxy_gain']-2*float(np.mean((t-b)*d)); identitymax=max(identitymax,abs(residual)); gs.append(gain)
   if which=='low': lowerr=max(lowerr,float(np.max(abs(t-b))))
   ar.append(dict(r,truth=which,gain=gain,identity_residual=residual,point_mse=float(np.mean((p-t)**2)),native_mse=float(np.mean((h-t)**2))))
  flips+=int(gs[0]*gs[1]<0 and min(abs(np.array(gs)))>P['sign_epsilon'])
 write(out/'alias_scores.json',ar); write(out/'summary.json',dict(models=summary,test_rows=len(rs),alias_pairs=len(ar)//2,alias_sign_flips=flips,alias_observation_pair_max_error=obsmax,alias_low_proxy_max_error=lowerr,alias_identity_max_error=identitymax,model_training_runs=0,scope=P['scope']))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--launch-record',required=True); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); out=Path(a.out); assert not out.exists(); out.mkdir(); launch=json.loads(Path(a.launch_record).read_text()); start=time.monotonic(); phases=[]; status='failed'
 limit=120 if a.smoke else P['wall_budget_seconds']
 def guard():
  if time.monotonic()-start>limit or time.time()>launch['absolute_deadline_epoch']: raise TimeoutError('fixed absolute/cumulative budget exceeded')
 deps=Path(P['inherited_root'])/'snapshot/src'; sources={str(p):sha(p) for p in sorted(deps.rglob('*.py'))}; sources.update({str(p):sha(p) for p in HERE.glob('*') if p.suffix in ['.py','.json']})
 write(out/'frozen.json',dict(sources=sources,launch=launch,wall_and_phase_budget=limit,device='cpu',threads=4,nice=os.getpriority(os.PRIO_PROCESS,0),smoke=a.smoke,torch=torch.__version__,SR_training=False))
 try:
  cs=cases(a.smoke)
  for phase,fun in [('prepare',lambda:prepare(out,cs)),('predict',lambda:predict(out,cs,guard)),('forecast',lambda:forecast(out,cs)),('score',lambda:score(out,cs))]:
   guard(); t=time.monotonic();
   if a.smoke and phase=='score': continue
   fun(); guard(); phases.append(dict(phase=phase,seconds=time.monotonic()-t)); write(out/'phase_costs.json',phases)
  status='complete'; (out/'COMPLETE').write_text('complete; frozen SR models, calibration coefficients only\n')
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc()); raise
 finally:
  r=resource.getrusage(resource.RUSAGE_SELF); write(out/'resources.json',dict(status=status,wall_seconds=time.time()-launch['start_epoch'],script_wall_seconds=time.monotonic()-start,cpu_user=r.ru_utime,cpu_system=r.ru_stime,maxrss_kib=r.ru_maxrss,gpu_processes=0,SR_training_runs=0,phases=phases))
if __name__=='__main__': main()
