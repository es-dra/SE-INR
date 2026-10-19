"""Independent NumPy-only full archived development audit. Never imports run.py/models/torch."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,time,resource,datetime,tarfile
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
R=H/'collected/codex-20261012-internal-transfer-development-v1'
O=H.parent.parent/'补证-20261011/negative-collected/codex-20261011-negative-formal-v3'
BASE=['always_enable','always_disable','band_median','energy_only','norm_ridge','proxy','directional_ridge'];CAND=['internal_transfer','internal_ridge'];M=BASE+CAND
START=time.monotonic();CPU=time.process_time();N=0;FAIL=[];ERR={};EPS=1e-8

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(p):return json.loads(p.read_text())
def ck(ok,label,detail=None):
 global N
 N+=1
 if not bool(ok):FAIL.append({'label':label,'detail':str(detail)[:300]})
 if time.monotonic()-START>170:raise TimeoutError('independent audit170s stop')
def near(a,b,label,tol=1e-9):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 if a.shape!=b.shape:ck(False,label,'shape');return
 finite=np.isfinite(a)&np.isfinite(b);diff=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0.
 ERR[label.split(':')[0]]=max(ERR.get(label.split(':')[0],0),diff)
 ck(np.array_equal(np.isfinite(a),np.isfinite(b)) and diff<=tol,label,diff)
def recurse(a,b,label):
 if isinstance(a,dict):
  for k,v in a.items():
   ck(k in b,label+':key:'+k)
   if k in b:recurse(v,b[k],label+':'+k)
 elif isinstance(a,list):
  ck(len(a)==len(b),label+':length')
  if len(a)==len(b):
   for i,(x,y) in enumerate(zip(a,b)):recurse(x,y,label+':'+str(i))
 elif a is None:ck(b is None,label)
 elif isinstance(a,(str,bool)):ck(a==b,label,(a,b))
 else:near(a,b,label)
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,(np.bool_,)):return bool(x)
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x
# Parseval computation uses no inverse FFT of band images.
def crop(a,n,k):return np.asarray(a,dtype=np.float64).reshape(n,n,3)[k:n-k,k:n-k]
def masks(n):
 f=np.fft.fftfreq(n);r=np.maximum(abs(f[:,None]),abs(f[None,:]));return [(r>=a)&(r<b)&(r>0) for a,b in [(0,.125),(.125,.25),(.25,.5001)]]
def energy(a):
 n=len(a);F=np.fft.fft2(a,axes=(0,1));v=abs(F)**2
 return [float(np.mean(a*a))]+[float(v[z].sum()/(3*n**4)) for z in masks(n)]
def prod(a,b):
 n=len(a);A=np.fft.fft2(a,axes=(0,1));B=np.fft.fft2(b,axes=(0,1));v=(A.conj()*B).real
 return [float(np.mean(a*b))]+[float(v[z].sum()/(3*n**4)) for z in masks(n)]
def variance(a):return max(float(np.mean((a-a.mean((0,1),keepdims=True))**2)),1e-8)

def audit():
 archive=H/'collected/codex-20261012-internal-transfer-complete-v1.tar.gz'
 ck(sha(archive)=='70f8e65368d85130c973afb0ebfe693fbacc6aebe22f1e49c96f07a54b923a3c','archive SHA')
 with tarfile.open(archive) as tf: archive_modes={m.name:m.mode for m in tf.getmembers()}
 p=rd(H/'protocol.json');freeze=rd(H/'freeze.json');f=rd(R/'frozen.json');man=rd(R/'input_manifest.json');inh=rd(R/'inherited_manifest.json')
 for name in ['source','protocol']:
  h=sha(H/('run.py' if name=='source' else 'protocol.json'));ck(h==freeze[name+'_sha256']==f[name+'_sha256'],'sourcehash:'+name)
 ck(sha(R/'input_manifest.json')==f['inputs_sha256'],'inputmanifesthash');ck(sha(R/'inherited_manifest.json')==f['inherited_sha256'],'inheritedhash')
 for path,h in man.items():
  q=Path(path)
  dest=R/'calibration_rows.json' if q.name=='calibration_rows.json' else R/('observations' if q.suffix=='.npy' else 'inherited_predictions')/q.name
  ck(sha(dest)==h,'inputbytehash:'+dest.name);ck((archive_modes[R.name+'/'+str(dest.relative_to(R))]&0o222)==0,'archive readonly:'+dest.name)
 for name,c in rd(O/'checkpoint_manifest.json').items():ck(inh.get(c['path'])==c['sha256'],'checkpoint_manifest:'+name)
 for name,h in rd(O/'frozen.json')['sources'].items():ck(inh.get(name)==h,'inherited_source_manifest:'+name)
 ck(sha(R/'calibration_rows.json')==sha(O/'calibration_rows.json'),'calibration original bytes')
 lo=rd(R/'lower_features.json');rows=rd(R/'oof_rows.json');old=rd(R/'calibration_rows.json');oldmap={(r['case'],r['model'],r['band']):r for r in old}
 ck(len(lo)==len(rows)==512,'complete row counts');ck(len(list((R/'predictions').glob('*.npz')))==128,'128 lower predictions')
 lm={(r['case'],r['model'],r['band']):r for r in lo};ck(len(lm)==512,'unique lower keys');rebuilt=[]
 for i in range(1,33):
  for s in [2,4]:
   name=f'calibration_{i:04d}_x{s}';n=128//s;sh=4//s;y=np.load(R/'observations'/(name+'.npy'));lt=crop(y.transpose(1,2,0),n,sh);zl=variance(lt)
   truth=np.load(O/'truth'/(name+'.npy'));target=crop(truth.transpose(1,2,0),128,4)
   ym=y.astype(float).mean(0);pw=abs(np.fft.fft2(ym-ym.mean()))**2;w=pw/pw.sum() if pw.sum()>0 else pw;freq=np.fft.fftfreq(n);rad=np.sqrt(freq[:,None]**2+freq[None,:]**2)
   cent=float((w*rad).sum());entropy=float(-(w[w>0]*np.log(w[w>0])).sum()/np.log(w.size));grad=float(np.mean(np.diff(y.astype(float),axis=1)**2)+np.mean(np.diff(y.astype(float),axis=2)**2))
   for model in ['lte','sc-inr']:
    z=np.load(R/'predictions'/(name+'__'+model+'.npz'));l0=crop(z['point'],n,sh);lh=crop(z['native'],n,sh);e0=energy(l0-lt);eh=energy(lh-lt)
    ck(z['lower_input'].shape==(3,n//s,n//s),'lower input shape');ck(np.isfinite(z['lower_input']).all(),'lower input finite')
    u=np.load(R/'inherited_predictions'/(name+'__'+model+'.npz'));a=crop(u['point'],128,4);b=crop(u['native'],128,4);proxy=crop(u['proxy'],128,4);zu=variance(proxy)
    E0=energy(a-target);EH=energy(b-target);v=energy(b-a);align=np.asarray(prod(proxy-a,b-a))*2;epa=energy(a);epb=energy(proxy);epa[0]=float(np.mean((a-a.mean((0,1),keepdims=True))**2));epb[0]=zu
    for band in range(4):
     key=(name,model,band);rec=lm[key];gain=e0[band]-eh[band];t=gain/zl
     for k,val in {'point_lower_mse':e0[band],'native_lower_mse':eh[band],'Z_lower':zl,'gain_lower':gain,'transfer_feature':t}.items():near(val,rec[k],'lower:'+k,tol=1e-10)
     r=dict(oldmap[key]);F=[float(s==2),float(s==4)]+[float(band==j) for j in range(4)]+[np.log1p(v[band]/zu),np.log1p(epa[band]/zu),np.log1p(epb[band]/zu),np.log1p(grad/zu),cent,entropy]
     vals={'point_mse':E0[band],'native_mse':EH[band],'gain':E0[band]-EH[band],'Z':zu,'change_energy':v[band],'proxy_alignment':align[band],'proxy_gain':align[band]-v[band],'energy_gain':-v[band],'features':F}
     for k,val in vals.items():near(val,r[k],'upper:'+k,tol=1e-9)
     r.update(vals);r.update(transfer_feature=t,gain_lower=gain,fold=(i-1)%8);rebuilt.append(r)
 cohort=rd(R/'lower_cohort.json')
 for name,h in cohort.items():ck(sha(R/name)==h,'lower_cohort:'+name)
 lf=rd(R/'lower_freeze.json');of=rd(R/'oof_freeze.json');launch=rd(R/'launch.json');res=rd(R/'resources.json')
 ck(sha(R/'lower_cohort.json')==lf['cohort_sha256'],'lower freeze hash');ck(sha(R/'oof_rows.json')==of['rows_sha256'],'OOF rows hash');ck(sha(R/'oof_fits.json')==of['fits_sha256'],'OOF fit hash')
 ck(launch['start_epoch']<=f['time_epoch']<=lf['time_epoch']<=of['time_epoch']<=res['end_epoch'],'chronological freezes');ck(res['wall_seconds']<=900 and res['end_epoch']<=launch['absolute_deadline_epoch'],'budget');ck(res['CPU_threads']==1 and res['GPU']==0 and res['SR_training']==0 and res['nice']==10,'resource constraints')
 fits=rd(R/'oof_fits.json');om={(r['case'],r['model'],r['band']):r for r in rows}
 def features(r,k):return r['features']+([r['proxy_alignment']/r['Z']] if k=='directional_ridge' else [r['transfer_feature']] if k=='internal_ridge' else [])
 for model in ['lte','sc-inr']:
  rr=[r for r in rebuilt if r['model']==model]
  for fold in range(8):
   train=[r for r in rr if r['fold']!=fold];held=[r for r in rr if r['fold']==fold];TI=sorted({r['image'] for r in train});HI=sorted({r['image'] for r in held});saved=fits[f'{model}_{fold}']
   ck(not set(TI)&set(HI) and len(TI)==28 and len(HI)==4,'OOF disjoint');recurse(TI,saved['train_images'],'train images');recurse(HI,saved['held_images'],'held images');ff={}
   for k in ['norm_ridge','directional_ridge','internal_ridge']:
    X=np.asarray([features(r,k) for r in train]);y=np.asarray([r['gain']/r['Z'] for r in train]);mu=X.mean(0);sd=X.std(0);sd[sd<1e-12]=1;A=np.c_[np.ones(len(X)),(X-mu)/sd];L=np.eye(A.shape[1]);L[0,0]=0
    beta=np.linalg.lstsq(np.concatenate([A,L]),np.r_[y,np.zeros(len(L))],rcond=None)[0];ff[k]=(mu,sd,beta)
    for key,val in [('mu',mu),('sd',sd),('beta',beta)]:near(val,saved['fits'][k][key],'OOFfit:'+key,1e-8)
   for r in held:
    group=[q['gain']/q['Z'] for q in train if q['scale']==r['scale'] and q['band']==r['band']]
    forecasts={'always_enable':1.,'always_disable':-1.,'band_median':np.median(group)*r['Z'],'energy_only':r['energy_gain'],'proxy':r['proxy_gain'],'internal_transfer':r['transfer_feature']*r['Z']}
    for k,(mu,sd,beta) in ff.items():forecasts[k]=np.dot(np.r_[1,(np.asarray(features(r,k))-mu)/sd],beta)*r['Z']
    r['forecasts']=forecasts;actual=om[(r['case'],r['model'],r['band'])]
    recurse(clean(forecasts),actual['forecasts'],'OOFforecast');ck(actual['fold']==fold,'storedfold')
 draws=np.random.default_rng(20261012).integers(0,32,(2000,32));ck(np.array_equal(draws,np.asarray(rd(R/'bootstrap_draws.json'))),'bootstrap draw bytes');W=np.vstack([np.ones(32),np.stack([np.bincount(d,minlength=32) for d in draws])]);saved=rd(R/'summary.json')['models'];summary={}
 def interval(x):
  good=x[np.isfinite(x)];return {'ci95':np.percentile(good,[2.5,97.5]).tolist() if len(good) else None,'valid':len(good),'missing':len(x)-len(good)}
 def stat(rr,k):
  g=np.asarray([r['gain'] for r in rr]);pr=np.asarray([r['forecasts'][k] for r in rr]);iw=W[:,[r['image']-1 for r in rr]];a=g<-EPS;b=g>EPS;h=pr<0;n=iw@a;p=iw@b;tp=iw@(a&h);fp=iw@(b&h)
  with np.errstate(invalid='ignore',divide='ignore'):return {'negative':n,'positive':p,'TP':tp,'FP':fp,'recall':tp/n,'precision':tp/(tp+fp),'FPR':fp/p,'BA':(tp/n+1-fp/p)/2}
 def auc(rr,k,cells):
  pieces=[]
  for s,b in cells:
   neg=[r for r in rr if r['scale']==s and r['band']==b and r['gain']<-EPS];pos=[r for r in rr if r['scale']==s and r['band']==b and r['gain']>EPS];wn=W[:,[r['image']-1 for r in neg]];wp=W[:,[r['image']-1 for r in pos]]
   A=np.asarray([r['forecasts'][k] for r in neg])[:,None];B=np.asarray([r['forecasts'][k] for r in pos])[None,:];rank=(A<B).astype(float)+(A==B)*.5
   num=np.einsum('bi,ij,bj->b',wn,rank,wp,optimize=True)
   with np.errstate(invalid='ignore',divide='ignore'):pieces.append(num/(wn.sum(1)*wp.sum(1)))
  return np.asarray(pieces).mean(0) if pieces else np.full(2001,np.nan)
 for model in ['lte','sc-inr']:
  rr=[r for r in rebuilt if r['model']==model];br=[r for r in rr if r['band']>0];rgb=[r for r in rr if r['band']==0];elig=[];cells=[]
  for s in [2,4]:
   for b in [1,2,3]:
    cc=[r for r in br if r['scale']==s and r['band']==b];neg=sum(r['gain']<-EPS for r in cc);pos=sum(r['gain']>EPS for r in cc)
    if neg and pos:elig.append([s,b])
    cells.append({'scale':s,'band':b,'negative':neg,'positive':pos,'ties':32-neg-pos,'lower_positive_upper_negative':sum(r['gain']<-EPS and r['gain_lower']>0 for r in cc),'lower_negative_upper_positive':sum(r['gain']>EPS and r['gain_lower']<0 for r in cc),'metrics':{k:{a:v[0] for a,v in stat(cc,k).items()} for k in M}})
  st={k:stat(br,k) for k in M};aa={k:auc(br,k,elig) for k in M};iw=W[:,[r['image']-1 for r in rgb]];p0=np.asarray([r['point_mse'] for r in rgb]);ph=np.asarray([r['native_mse'] for r in rgb]);pol={};dist={}
  for k in M:
   n=np.asarray([r['forecasts'][k]>0 for r in rgb]);err=np.where(n,ph,p0);dist[k]=(iw@err)/iw.sum(1);pol[k]={'mse':float(err.mean()),'mean_PSNR':float((-10*np.log10(err)).mean()),'oracle_regret':float((err-np.minimum(p0,ph)).mean()),'disable_cases':int((~n).sum()),'harmful_native_kept':int((n&(p0-ph<-EPS)).sum()),'beneficial_native_disabled':int(((~n)&(p0-ph>EPS)).sum())}
  ni=len({r['image'] for r in br if r['gain']<-EPS});gates={}
  for k in CAND:
   ba={b:interval((st[k]['BA']-st[b]['BA'])[1:]) for b in BASE};ad=interval((aa[k]-aa['norm_ridge'])[1:]);pm={b:interval((dist[b]-dist[k])[1:]) for b in BASE};z=st[k]
   c=bool(z['negative'][0]>=12 and ni>=6 and z['recall'][0]>=.75 and z['precision'][0]>=.6 and z['FPR'][0]<=.25 and all(z['BA'][0]-st[b]['BA'][0]>=.05 and ba[b]['valid']>=1000 and ba[b]['ci95'][0]>0 for b in BASE))
   a=bool(len(elig)>=3 and aa[k][0]-aa['norm_ridge'][0]>=.05 and ad['valid']>=1000 and ad['ci95'][0]>0);po=all(pm[b]['valid']>=1000 and pm[b]['ci95'][0]>0 for b in BASE)
   gates[k]={'classification_pass':c,'within_cell_AUC_pass':a,'RGB_policy_pass':po,'combined_pass':c and a and po,'BA_improvement_vs_controls':ba,'within_cell_AUC_improvement_vs_norm':ad,'policy_MSE_improvement_vs_controls':pm}
  summary[model]={'band_rows':len(br),'RGB_rows':len(rgb),'negative_images':ni,'eligible_AUC_cells':elig,'cells':cells,'band_metrics':{k:{a:v[0] for a,v in st[k].items()} for k in M},'macro_within_cell_AUC':{k:aa[k][0] for k in M},'RGB_policies':pol,'gates':gates}
  recurse(clean(summary[model]),saved[model],'summary:'+model)
 return {'independent_summary':clean(summary),'run_resources':res,'source_hash':sha(H/'run.py'),'protocol_hash':sha(H/'protocol.json')}

if __name__=='__main__':
 result={}
 try:result=audit()
 except Exception as e:
  import traceback
  FAIL.append({'exception':repr(e),'traceback':traceback.format_exc()})
 report={'status':'passed' if not FAIL else 'failed','checks':N,'failures':FAIL,'max_abs_errors':ERR,'wall_seconds':time.monotonic()-START,'CPU_seconds':time.process_time()-CPU,'CPU_threads':1,'script_sha256':sha(Path(__file__)),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent NumPy Parseval all saved pixel gains and old calibration truth; augmented LS all OOF fits; full2000draw metrics/gates; hash/chronology/budget checks. No test or alias truth read.','limitations':['No model inference or model reimplementation. Lower_input bicubicAA actual numbers not rederived without torch; only shape/finite checked.','Checkpoint/source byte verification performed at frozen production prepare; this local audit verifies inherited manifests against prior records, not remote checkpoint bytes.','Fixed-forecast OOF bootstrap does not refit selectors or establish unseen-SR-model-image generalization.'],**result}
 report['audit_correction']={'previous_failed_report_sha256':sha(H/'audit.json'),'reason':'First audit passed every numeric/hash/OOF/metric check but193 local readonly checks failed. Verified complete archive SHA and original member mode0444; local extraction produced0644. V2checks original archived mode, preserves local files and failed audit.','original_failed_checks':193,'numeric_tolerance_changed':False}
 target=H/'audit-v2.json'
 if target.exists():raise RuntimeError('audit.json exists; preserve and use explicit new version')
 target.write_text(json.dumps(clean(report),ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':N,'failures':FAIL[:3],'wall_seconds':report['wall_seconds']}))
