import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time,datetime,math,resource
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
start=time.monotonic(); cpu=time.process_time(); checks=0; errors=[]; maxerr=0.
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(ok,label):
 global checks
 checks+=1
 if not ok:errors.append(label)
def close(x,y,label):
 global maxerr
 if x is None or y is None:check(x is y,label);return
 if isinstance(x,(str,bool)):check(x==y,label);return
 err=abs(x-y);maxerr=max(maxerr,err);check(err<=1e-11,label)
def recurse(x,y,label):
 if isinstance(x,dict):
  check(set(x)==set(y),label+'/keys')
  for k in x:recurse(x[k],y[k],label+'/'+str(k))
 elif isinstance(x,list):
  check(len(x)==len(y),label+'/length')
  for i,(a,b) in enumerate(zip(x,y)):recurse(a,b,label+'/'+str(i))
 else:close(x,y,label)
p=load(H/'protocol.json'); root=Path(p['input_root']); rows=load(H/'rows.json'); summary=load(H/'summary.json'); freeze=load(H/'freeze.json'); inputs=load(H/'input_manifest.json'); outs=load(H/'output_manifest.json'); eps=p['harm_epsilon']; lambdas=p['lambdas']
check(sha(H/'analyze.py')==freeze['source_sha256'],'source SHA');check(sha(H/'protocol.json')==freeze['protocol_sha256'],'protocol SHA')
check(sha(H/'input_manifest.json')==load(H/'input_freeze.json')['sha256'],'input manifest SHA')
expected={f'truth/calibration_{i:04d}_x{s}.npy' for i in range(1,33) for s in (2,4)}|{f'predictions/calibration_{i:04d}_x{s}__{m}.npz' for i in range(1,33) for s in (2,4) for m in p['models']}|{'calibration_rows.json','input_manifest.json','prediction_cohort.json'}
check(set(inputs)==expected,'only exact calibration inputs')
for f,digest in inputs.items():check(sha(root/f)==digest,'input '+f)
for f,digest in outs.items():check(sha(H/f)==digest,'output '+f)
oldinput=load(root/'input_manifest.json');oldpred=load(root/'prediction_cohort.json')
for f in expected:
 if f.startswith('truth/'):check(inputs[f]==oldinput[f],'original truth hash '+f)
 if f.startswith('predictions/'):check(inputs[f]==oldpred[f],'original prediction hash '+f)
f=np.fft.fftfreq(120); radial=np.maximum(abs(f[:,None]),abs(f[None,:])); masks=[np.ones((120,120),bool)]+[(radial>=lo)&(radial<hi)&(radial>0) for lo,hi in ((0,.125),(.125,.25),(.25,.5001))]
# Parseval directly on the cropped raw arrays, independent of spatial band reconstruction.
def spec(a):return np.fft.fft2(np.asarray(a,dtype=float).reshape(128,128,3)[4:-4,4:-4],axes=(0,1))
lookup={(r['image'],r['scale'],r['model'],r['band']):r for r in rows}; rebuilt=[]; den=3*120**4
check(len(rows)==512 and len(lookup)==512,'full cohort')
for i in range(1,33):
 for scale in (2,4):
  case=f'calibration_{i:04d}_x{scale}';t=spec(np.load(root/'truth'/f'{case}.npy').transpose(1,2,0))
  for model in p['models']:
   with np.load(root/'predictions'/f'{case}__{model}.npz') as z:F0=spec(z['point']);Fc=spec(z['native'])
   residual=t-F0;delta=Fc-F0
   for band,mask in enumerate(masks):
    rr=residual[mask];dd=delta[mask]; e0=float(np.vdot(rr,rr).real/den);a=float(np.vdot(rr,dd).real/den);b=float(np.vdot(dd,dd).real/den);ec=float(np.vdot(rr-dd,rr-dd).real/den);g=2*a-b
    tol=64*np.finfo(float).eps*max(e0,b,1e-8);star=a/b if b>tol else None;lam=0 if star is None else float(np.clip(star,0,1)); oracle=float(np.vdot(rr-lam*dd,rr-lam*dd).real/den)
    cls='not_harmful'
    if g < -eps:
     if b<=tol or abs(a)<=tol or abs(a-b/2)<=tol:cls='numerical_boundary'
     elif a<0:cls='wrong_direction'
     elif 0<a<b/2:cls='overshoot'
     else:cls='INVALID';check(False,'invalid partition')
    r={'case':case,'image':i,'scale':scale,'model':model,'band':band,'band_name':p['bands'][band],'point_mse':e0,'native_mse':ec,'gain_native':g,'a':a,'b':b,'lambda_star':star,'oracle_lambda_0to1':lam,'oracle_shrink_mse':oracle,'residual_alignment_cosine':a/math.sqrt(e0*b) if e0*b>0 else None,'correction_to_residual_norm':math.sqrt(b/e0) if e0>0 else None,'numerical_tol':tol,'failure_class':cls}
    actual=lookup[(i,scale,model,band)]
    for k,v in r.items():close(v,actual[k],f'{case}/{model}/{band}/{k}')
    r['fixed_lambda']=[]
    for j,l in enumerate(lambdas):
     mse=float(np.vdot(rr-l*dd,rr-l*dd).real/den);gain=2*l*a-l*l*b
     z={'lambda':l,'mse_direct':mse,'gain_direct':gain,'gain_quadratic':gain,'identity_abs_error':0.}
     recurse(z,actual['fixed_lambda'][j],f'{case}/{model}/{band}/lambda{l}');r['fixed_lambda'].append(z)
    rebuilt.append(r)
def aggregate(rs):
 neg=[r for r in rs if r['gain_native'] < -eps];pos=[r for r in rs if r['gain_native']>eps];loss=math.fsum(-r['gain_native'] for r in neg);classes={}
 for c in ('wrong_direction','overshoot','numerical_boundary'):
  subset=[r for r in neg if r['failure_class']==c];closs=math.fsum(-r['gain_native'] for r in subset)
  classes[c]={'rows':len(subset),'images':len(set(r['image'] for r in subset)),'negative_loss_sum':closs,'fraction_negative_rows':len(subset)/len(neg) if neg else None,'fraction_negative_loss':closs/loss if loss else None}
 fixed={};mean=lambda v:math.fsum(v)/len(rs)
 for j,l in enumerate(lambdas):
  gain=lambda r:r['fixed_lambda'][j]['gain_direct'];mse=lambda r:r['fixed_lambda'][j]['mse_direct'];gn=lambda r:r['gain_native']
  fixed[str(l)]={'rows':len(rs),'negative_rows_after':sum(gain(r)<-eps for r in rs),'original_negative_strictly_rescued':sum(gain(r)>eps for r in neg),'original_negative_now_nonharmful':sum(gain(r)>=-eps for r in neg),'original_negative_still_harmful':sum(gain(r)<-eps for r in neg),'original_positive_now_harmful':sum(gain(r)<-eps for r in pos),'original_positive_lost_some_benefit':sum(gain(r)-gn(r)<-eps for r in pos),'original_positive_loss_sum':math.fsum(max(gn(r)-gain(r),0) for r in pos),'original_negative_improvement_sum':math.fsum(gain(r)-gn(r) for r in neg),'mean_mse':mean(mse(r) for r in rs),'mean_psnr':mean(-10*math.log10(mse(r)) for r in rs),'mean_mse_improvement_vs_native':mean(gain(r)-gn(r) for r in rs),'mean_mse_improvement_vs_zero':mean(gain(r) for r in rs),'mean_regret_to_oracle_shrink':mean(mse(r)-r['oracle_shrink_mse'] for r in rs)}
 return {'rows':len(rs),'images':len(set(r['image'] for r in rs)),'negative_rows':len(neg),'negative_images':len(set(r['image'] for r in neg)),'positive_rows':len(pos),'ties':len(rs)-len(neg)-len(pos),'negative_loss_sum':loss,'failure_types':classes,'point_mse_mean':mean(r['point_mse'] for r in rs),'native_mse_mean':mean(r['native_mse'] for r in rs),'lambda_results':fixed}
for key,value in summary['groups'].items():
 pieces=key.split('/');rs=[r for r in rebuilt if r['model']==pieces[0]]
 if pieces[1]=='pooled_bands_descriptive':rs=[r for r in rs if r['band']>0]
 else:
  rs=[r for r in rs if r['band_name']==pieces[2]]
  if pieces[1]!='xpooled':rs=[r for r in rs if r['scale']==int(pieces[1][1:])]
 recurse(aggregate(rs),value,'summary/'+key)
resources=load(H/'resources.json');launcher=load(H/'launcher.json');end=datetime.datetime.fromisoformat(resources['end_utc']);deadline=datetime.datetime.fromisoformat(p['absolute_deadline_utc'])
check(end<deadline,'completed before deadline');check(launcher['wall_seconds_including_import']<180 and launcher['returncode']==0,'launcher timeout and success');check(resources['CPU_threads']==1 and resources['nice']==10 and resources['GPU']==resources['training']==resources['inference']==0,'resources')
check(datetime.datetime.fromisoformat(freeze['frozen_utc']).timestamp()<load(H/'input_freeze.json')['time_epoch']<end.timestamp(),'freeze precedes analysis')
result={'status':'passed' if not errors else 'failed','checks':checks,'errors':errors,'max_absolute_error':maxerr,'rows':len(rebuilt),'groups':len(summary['groups']),'full_input_hashes_checked':len(inputs),'review_source_sha256':sha(Path(__file__)),'analysis_source_sha256':sha(H/'analyze.py'),'protocol_sha256':sha(H/'protocol.json'),'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,'nice':os.getpriority(os.PRIO_PROCESS,0),'CPU_threads':1,'limitations':['Retrospective calibration only; oracle HR alignment is not observation-only prediction.','Output delta scaling is not a cell-parameter intervention.','No original model inference or checkpoint validation was performed.','Direction and overshoot are exact error geometry classifications, not causal network mechanisms.']}
(H/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
