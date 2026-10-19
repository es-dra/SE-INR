"""Frozen-output oracle direction/overshoot analysis. No model or torch import."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib,resource,datetime,traceback
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;P=json.loads((H/'protocol.json').read_text());O=Path(P['input_root']);L=P['lambdas'];EPS=P['harm_epsilon'];START=time.monotonic();CPU=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,v):(H/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def guard():
 if time.monotonic()-START>=175 or datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(P['absolute_deadline_utc']):raise TimeoutError('fixed stage budget expired')
def planes(z):
 v=np.asarray(z,dtype=np.float64).reshape(128,128,3)[4:124,4:124];f=np.fft.fftfreq(120);k=np.maximum(abs(f[:,None]),abs(f[None,:]));F=np.fft.fft2(v,axes=(0,1));a=[v]
 for lo,hi in [(0,.125),(.125,.25),(.25,.5001)]:a.append(np.fft.ifft2(F*((k>=lo)&(k<hi)&(k>0))[:,:,None],axes=(0,1)).real)
 return a

def summarise(rows):
 negative=[r for r in rows if r['gain_native']<-EPS];positive=[r for r in rows if r['gain_native']>EPS];loss=sum(-r['gain_native'] for r in negative);classes={}
 for cls in ['wrong_direction','overshoot','numerical_boundary']:
  rr=[r for r in negative if r['failure_class']==cls];v=sum(-r['gain_native'] for r in rr)
  classes[cls]={'rows':len(rr),'images':len({r['image'] for r in rr}),'negative_loss_sum':v,'fraction_negative_rows':len(rr)/len(negative) if negative else None,'fraction_negative_loss':v/loss if loss else None}
 lambda_stats={}
 for j,l in enumerate(L):
  mse=np.array([r['fixed_lambda'][j]['mse_direct'] for r in rows]);gain=np.array([r['fixed_lambda'][j]['gain_direct'] for r in rows]);gn=np.array([r['gain_native'] for r in rows]);n=gn<-EPS;b=gn>EPS
  lambda_stats[str(l)]={'rows':len(rows),'negative_rows_after':int(sum(gain<-EPS)),'original_negative_strictly_rescued':int(sum(n&(gain>EPS))),'original_negative_now_nonharmful':int(sum(n&(gain>=-EPS))),'original_negative_still_harmful':int(sum(n&(gain<-EPS))),'original_positive_now_harmful':int(sum(b&(gain<-EPS))),'original_positive_lost_some_benefit':int(sum(b&((gain-gn)<-EPS))),'original_positive_loss_sum':float(np.maximum(gn[b]-gain[b],0).sum()),'original_negative_improvement_sum':float((gain[n]-gn[n]).sum()),'mean_mse':float(mse.mean()),'mean_psnr':float((-10*np.log10(mse)).mean()),'mean_mse_improvement_vs_native':float((gain-gn).mean()),'mean_mse_improvement_vs_zero':float(gain.mean()),'mean_regret_to_oracle_shrink':float(np.mean(mse-np.array([r['oracle_shrink_mse'] for r in rows])))}
 return {'rows':len(rows),'images':len({r['image'] for r in rows}),'negative_rows':len(negative),'negative_images':len({r['image'] for r in negative}),'positive_rows':len(positive),'ties':len(rows)-len(negative)-len(positive),'negative_loss_sum':loss,'failure_types':classes,'point_mse_mean':float(np.mean([r['point_mse'] for r in rows])),'native_mse_mean':float(np.mean([r['native_mse'] for r in rows])),'lambda_results':lambda_stats}

def main():
 frozen=read(H/'freeze.json');assert frozen['source_sha256']==sha(H/'analyze.py') and frozen['protocol_sha256']==sha(H/'protocol.json')
 im=read(O/'input_manifest.json');pm=read(O/'prediction_cohort.json');manifest={}
 for i in P['images']:
  for s in P['scales']:
   guard();case=f'calibration_{i:04d}_x{s}';name='truth/'+case+'.npy';h=sha(O/name);assert h==im[name];manifest[name]=h
   for model in P['models']:
    name='predictions/'+case+'__'+model+'.npz';h=sha(O/name);assert h==pm[name];manifest[name]=h
 for name in ['calibration_rows.json','input_manifest.json','prediction_cohort.json']:manifest[name]=sha(O/name)
 save('input_manifest.json',manifest);save('input_freeze.json',{'sha256':sha(H/'input_manifest.json'),'time_epoch':time.time(),'files':len(manifest),'truth_subset':'calibration0001-0032 only'})
 original=read(O/'calibration_rows.json');old={(r['case'],r['model'],r['band']):r for r in original};rows=[];maxidentity=0.;maxold=0.;maxparseval=0.;nearcount=0
 for i in P['images']:
  for s in P['scales']:
   guard();case=f'calibration_{i:04d}_x{s}';truth=np.load(O/'truth'/(case+'.npy')).transpose(1,2,0);t=planes(truth)
   for model in P['models']:
    z=np.load(O/'predictions'/(case+'__'+model+'.npz'));base=planes(z['point']);native=planes(z['native'])
    for band,(tt,f0,fc) in enumerate(zip(t,base,native)):
     d=fc-f0;res=tt-f0;e0=float(np.mean(res*res));ec=float(np.mean((fc-tt)**2));a=float(np.mean(res*d));b=float(np.mean(d*d));g=e0-ec;tol=64*np.finfo(np.float64).eps*max(e0,b,1e-8)
     # Independent direct FFT Parseval inner products confirm spatial a,b.
     rf=np.fft.fft2(res,axes=(0,1));df=np.fft.fft2(d,axes=(0,1));den=3*120**4;ap=float((rf.conj()*df).real.sum()/den);bp=float((abs(df)**2).sum()/den);maxparseval=max(maxparseval,abs(ap-a),abs(bp-b))
     fixed=[]
     for l in L:
      err=float(np.mean((f0+l*d-tt)**2));gd=e0-err;gq=2*l*a-l*l*b;gap=abs(gd-gq);maxidentity=max(maxidentity,gap);assert gap<1e-11
      fixed.append({'lambda':l,'mse_direct':err,'gain_direct':gd,'gain_quadratic':gq,'identity_abs_error':gap})
     star=a/b if b>tol else None;lam=min(1.,max(0.,star)) if star is not None else 0.;oracle=float(np.mean((f0+lam*d-tt)**2))
     if g<-EPS:
      if b<=tol or abs(a)<=tol or abs(a-b/2)<=tol:cls='numerical_boundary';nearcount+=1
      elif a<0:cls='wrong_direction'
      elif 0<a<b/2:cls='overshoot'
      else:raise AssertionError(('negative outside exact partition',case,model,band,a,b,g))
     else:cls='not_harmful'
     prev=old[(case,model,band)];maxold=max(maxold,abs(e0-prev['point_mse']),abs(ec-prev['native_mse']),abs(g-prev['gain']));assert maxold<1e-11
     rows.append({'case':case,'image':i,'model':model,'scale':s,'band':band,'band_name':P['bands'][band],'point_mse':e0,'native_mse':ec,'gain_native':g,'a':a,'b':b,'lambda_star':star,'oracle_lambda_0to1':lam,'oracle_shrink_mse':oracle,'residual_alignment_cosine':a/(np.sqrt(e0*b)) if e0*b>0 else None,'correction_to_residual_norm':float(np.sqrt(b/e0)) if e0>0 else None,'numerical_tol':tol,'failure_class':cls,'fixed_lambda':fixed})
 assert len(rows)==512 and len({(r['case'],r['model'],r['band']) for r in rows})==512
 groups={}
 for model in P['models']:
  for scale in ['pooled',2,4]:
   for band in range(4):
    rr=[r for r in rows if r['model']==model and r['band']==band and (scale=='pooled' or r['scale']==scale)];groups[f'{model}/x{scale}/{P["bands"][band]}']=summarise(rr)
  rr=[r for r in rows if r['model']==model and r['band']>0];groups[f'{model}/pooled_bands_descriptive']=summarise(rr)
 save('rows.json',rows);save('summary.json',{'scope':P['scope'],'groups':groups,'checks':{'rows':len(rows),'fixed_lambda_direct_checks':len(rows)*4,'max_quadratic_identity_error':maxidentity,'max_Parseval_inner_product_error':maxparseval,'max_reused_original_label_error':maxold,'negative_numerical_boundaries':nearcount},'limitations':['Same image appears across scale/band;512rows not independent trials.','This HRoracle diagnostic cannot predict unknown-target benefit; lambda scaling is not cell alteration.','Original calibration images, existing model training split; no new generalization test.','Raw image interpolation can lose detail and does not define a new trained model.','Direction means alignment relative to true residual, not an absolute claim about the correction vector.']})
 save('output_manifest.json',{n:sha(H/n) for n in ['rows.json','summary.json','input_manifest.json','input_freeze.json']});(H/'COMPLETE').write_text('complete retrospective oracle analysis\n')
 print(json.dumps({'status':'complete','rows':len(rows),'max_identity_error':maxidentity,'max_label_error':maxold}))
if __name__=='__main__':
 status='failed'
 try:main();status='complete'
 except Exception as e:save('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
 finally:save('resources.json',{'status':status,'wall_seconds':time.monotonic()-START,'CPU_seconds':time.process_time()-CPU,'CPU_threads':1,'nice':os.getpriority(os.PRIO_PROCESS,0),'GPU':0,'inference':0,'training':0,'max_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cost_scope':'compute+input hash+IO after Python imports; external launcher separately times full process'})
