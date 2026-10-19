#!/usr/bin/env python3
"""Independent audit: read only protocol and saved pixels/results, never analyze.py.
Uses orthonormal Fourier-domain Parseval sums instead of filtered spatial images.
Bounded single-process CPU work; alarm and external launcher enforce 120 seconds.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,hashlib,time,signal,resource,traceback
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
RUN=BASE/'run-v1'; ROOT=BASE.parent/'cell-reference-v2/collected/codex-20261018-cell-reference-formal-v2'
START=time.time(); CLOCK=time.monotonic(); signal.alarm(120)
report={'status':'running','start_unix':START,'deadline_unix':START+120,'cpu_threads':1,'gpu':0,'inference':0,'training':0,'independence':'No import/read of analyze.py beyond SHA; orthonormal Parseval recomputation from every saved endpoint and truth.'}
checks=0; maxima={}; failures=[]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def eq(a,b,tag):
 global checks
 checks+=1
 if a!=b: raise AssertionError(f'{tag}: {a!r} != {b!r}')
def close(a,b,tag,tol=2e-10):
 global checks
 checks+=1
 delta=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
 maxima[tag.split(':')[0]]=max(maxima.get(tag.split(':')[0],0),delta)
 if not np.isfinite(delta) or delta>tol: raise AssertionError(f'{tag}: diff {delta} > {tol}; {a!r} != {b!r}')
def key(r): return tuple(r[k] for k in ['metric','image','scale','model','seed','ref','band'])
def gkey(r):return tuple(r[k] for k in ['metric','model','scale','ref','band'])
def measure(e,d,nerr,mask=None):
 if mask is None:return [float(np.mean(e*e)),float(np.mean(nerr*nerr)),float(np.mean(e*d)),float(np.mean(d*d))]
 return [float(np.sum(np.abs(e[mask])**2)/e.size),float(np.sum(np.abs(nerr[mask])**2)/e.size),float(np.real(np.sum(e[mask].conj()*d[mask]))/e.size),float(np.sum(np.abs(d[mask])**2)/e.size)]
def classify(r):
 if r['gain']>=-1e-8:return 'benefit_or_tie'
 tol=64*np.finfo(float).eps*max(r['reference_mse'],r['b'],1e-8)
 if r['b']<=tol or abs(r['a'])<=tol or abs(r['a']-.5*r['b'])<=tol:return 'boundary'
 return 'direction' if r['a']<0 else 'overshoot'
def stats(rr,lam):
 v=np.array([[r[k] for k in ['reference_mse','native_mse','a','b','gain']] for r in rr]);ref,nat,a,b,g=v.T
 mse=ref-2*lam*a+lam*lam*b;gain=ref-mse
 return dict(mean_mse=float(mse.mean()),mse_change_vs_native_percent=float(100*(mse.mean()/nat.mean()-1)),original_negative_rescued=int(np.sum((g<-1e-8)&(gain>1e-8))),original_positive_becomes_negative=int(np.sum((g>1e-8)&(gain<-1e-8))),all_worse_than_native=int(np.sum(mse-nat>1e-8)),negative_total=int(np.sum(gain<-1e-8)))
def verify_fields(actual,expected,tag):
 for k,v in expected.items():
  if isinstance(v,int):eq(actual[k],v,tag+':'+k)
  else:close(actual[k],v,tag+':'+k)
try:
 p=read(BASE/'protocol.json');fr=read(RUN/'input_freeze.json');done=read(RUN/'COMPLETE.json')
 for name,h in done['outputs'].items():eq(sha(RUN/name),h,'output SHA '+name)
 sf=read(RUN/'source_freeze.json')
 for name,h in sf.items():eq(sha(BASE/name),h,'source SHA '+name)
 eq(sha(ROOT/'prediction_cohort.json'),p['input_prediction_cohort_sha256'],'prediction cohort SHA')
 eq(sha(ROOT/'prediction_cohort.json'),fr['prediction_manifest_sha256'],'input freeze pred SHA')
 eq(sha(ROOT/'input_manifest.json'),fr['input_manifest_sha256'],'input freeze input SHA')
 manifest=read(ROOT/'prediction_cohort.json');im=read(ROOT/'input_manifest.json')
 eq(len(manifest),256,'prediction cohort size');eq(len(fr['truth_hashes']),32,'truth cohort size')
 for name,h in manifest.items():eq(sha(ROOT/'predictions'/name),h,'prediction SHA')
 for name,h in fr['truth_hashes'].items():eq(h,im[name],'truth original SHA');eq(sha(ROOT/name),h,'truth actual SHA')
 original=read(RUN/'rows.json');orig={key(r):r for r in original};eq(len(orig),12288,'unique rows')
 recomputed={};native={};masks={}
 for image in p['cohort']['images']:
  truth=np.load(ROOT/f'truth/{image}.npy').astype(float)
  for scale in [2,4]:
   for model in ['lte','sc-inr']:
    for seed in [1,2]:
     with np.load(ROOT/f'predictions/{image}_x{scale}__{model}_seed{seed}.npz') as z:
      endpoints={k:z[k].astype(float) for k in ['zero','h3',p['cohort']['native'][str(scale)]]}
     for transform in ['raw','clip']:
      ep={k:(np.clip(v,0,1) if transform=='clip' else v) for k,v in endpoints.items()}
      for shave in [0,4,8]:
       metric=f'{transform}_shave{shave}';sl=slice(shave,-shave) if shave else slice(None)
       t=truth[sl,sl]; n=ep[p['cohort']['native'][str(scale)]][sl,sl];nm=float(np.mean((t-n)**2));native[(metric,image,scale,model,seed)]=(nm,-10*np.log10(nm))
       size=t.shape[0]
       if size not in masks:
        f=np.fft.fftfreq(size);rad=np.maximum(abs(f[:,None]),abs(f[None,:]));masks[size]=[(rad>=lo)&(rad<hi)&(rad>0) for lo,hi in p['metrics']['bands']]
       for ref in ['zero','h3']:
        y=ep[ref][sl,sl];e=t-y;d=n-y;ne=t-n
        es,ds,ns=[np.fft.fft2(v,axes=(0,1),norm='ortho') for v in [e,d,ne]]
        for band,mask in [('rgb',None)]+list(zip(['b1','b2','b3'],masks[size])):
         vals=measure(e,d,ne) if mask is None else measure(es,ds,ns,mask)
         r=dict(metric=metric,image=image,scale=scale,model=model,seed=seed,ref=ref,band=band)
         r.update(zip(['reference_mse','native_mse','a','b'],vals));r['gain']=vals[0]-vals[1];r['db']=10*np.log10(vals[0]/vals[1]);r['identity_residual']=r['gain']-(2*r['a']-r['b']);r['label']=classify(r)
         o=orig[key(r)]
         for k in ['reference_mse','native_mse','a','b','gain','identity_residual']:close(o[k],r[k],'row_numeric:'+k,2e-14)
         close(o['db'],r['db'],'row_db',2e-10);eq(o['label'],r['label'],'row_label');recomputed[key(r)]=r
 eq(len(recomputed),12288,'recomputed rows')
 for r in read(RUN/'native_rows.json'):
  v=native[tuple(r[k] for k in ['metric','image','scale','model','seed'])];close(r['mse'],v[0],'native_mse',2e-14);close(r['psnr'],v[1],'native_psnr')
 groups={}
 for r in recomputed.values():groups.setdefault(gkey(r),[]).append(r)
 rng=np.random.default_rng(2026101802);sample=rng.integers(0,32,size=(10000,32))
 def ci(v):return np.percentile(np.asarray(v)[sample].mean(axis=1),[2.5,97.5])
 summary=read(RUN/'summary.json')
 for g in summary['groups']:
  rr=groups[gkey(g)];v=np.array([[r[k] for k in ['reference_mse','native_mse','a','b','db']] for r in rr]);ref,nat,a,b,db=v.T
  expected=dict(n_records=64,n_images=32,reference_mse=ref.mean(),native_mse=nat.mean(),mean_a=a.mean(),mean_b=b.mean(),mean_db=db.mean(),negative=sum(r['gain']<-1e-8 for r in rr),positive=sum(r['gain']>1e-8 for r in rr),overshoot=sum(r['label']=='overshoot' for r in rr),direction=sum(r['label']=='direction' for r in rr),boundary=sum(r['label']=='boundary' for r in rr),harm_gt_001db=sum(r['db']<-.01 for r in rr),harm_gt_005db=sum(r['db']<-.05 for r in rr),min_db=db.min())
  verify_fields(g,expected,'group')
  opt=float(np.clip(a.mean()/b.mean(),0,1));per=np.clip(a/b,0,1)
  close(g['global_oracle']['coefficient'],opt,'oracle_coefficient')
  for label,coef in [('half',.5),('global_oracle',opt),('per_case_oracle',per)]:verify_fields(g[label],stats(rr,coef),'oracle_stats')
  verify_fields(g['per_case_oracle'],dict(coefficient_mean=per.mean(),coefficient_min=per.min(),coefficient_max=per.max()),'oracle_distribution')
  imgdb=[];imgh=[];imgn=[]
  for image in p['cohort']['images']:
   ir=[r for r in rr if r['image']==image];imgdb.append(np.mean([r['db'] for r in ir]));imgh.append(np.mean([r['reference_mse']-r['a']+.25*r['b'] for r in ir]));imgn.append(np.mean([r['native_mse'] for r in ir]))
  if 'mean_db_ci' in g:close(g['mean_db_ci'],ci(imgdb),'group_ci')
  if 'half_mse_change_ci' in g:close(g['half_mse_change_ci'],np.percentile(100*(np.asarray(imgh)[sample].mean(axis=1)/np.asarray(imgn)[sample].mean(axis=1)-1),[2.5,97.5]),'group_half_ci')
 curves=read(RUN/'curves.json');eq(len(curves),3232,'curve points')
 seen=set()
 for c in curves:
  rr=groups[gkey(c)];seen.add((gkey(c),round(c['coefficient'],2)));verify_fields(c,stats(rr,c['coefficient']),'curve')
 eq(len(seen),3232,'curve uniqueness')
 for n in summary['native_comparison']:
  metric,scale,seed=n['metric'],n['scale'],n['seed'];seeds=[seed] if isinstance(seed,int) else [1,2]
  lte=np.array([np.mean([native[(metric,i,scale,'lte',s)][1] for s in seeds]) for i in p['cohort']['images']]);sc=np.array([np.mean([native[(metric,i,scale,'sc-inr',s)][1] for s in seeds]) for i in p['cohort']['images']]);delta=sc-lte
  verify_fields(n,dict(n_images=32,lte_mean_psnr=lte.mean(),sc_inr_mean_psnr=sc.mean(),paired_sc_minus_lte_db=delta.mean(),sc_wins=int(np.sum(delta>1e-12)),ties=int(np.sum(abs(delta)<=1e-12)),sc_losses=int(np.sum(delta < -1e-12))),'native_comparison')
  for name,v in [('lte_mean_psnr_ci',lte),('sc_inr_mean_psnr_ci',sc),('paired_delta_ci',delta)]:close(n[name],ci(v),'native_bootstrap')
  for j,i in enumerate(p['cohort']['images']):eq(n['image_deltas'][j]['image'],i,'native image order');close(n['image_deltas'][j]['sc_minus_lte_db'],delta[j],'native perimage')
 # Use verified stored dB to preserve exact published tie arithmetic; independent
 # pixel recomputation already checked dB to 2e-10 and labels exactly.
 for s in read(RUN/'selected_examples.json'):
  rr=[r for r in original if r['metric']=='raw_shave4' and r['band']=='rgb' and r['ref']=='zero' and r['model']==s['model'] and r['gain'] < -1e-8]
  if s['selection']=='median':
   rr=[r for r in rr if r['label']==s['label']]
   if not rr:eq(s['status'],'missing_class','missing example');continue
   med=float(np.median([-r['db'] for r in rr]));selected=min(rr,key=lambda r:(abs(-r['db']-med),r['image'],r['seed'],r['scale']))
   eq(s['class_count'],len(rr),'example class count');close(s['class_median_psnr_loss'],med,'example median')
  else:selected=min(rr,key=lambda r:(r['db'],r['image'],r['seed'],r['scale']))
  eq(key(s),key(selected),'selected example identity')
 budget=read(RUN/'budget.json');cost=read(RUN/'cost.json')
 close(budget['deadline_unix']-budget['start_unix'],600,'budget duration')
 eq(cost['nice'],10,'nice');eq(cost['cpu_threads'],1,'threads');eq(cost['gpu'],0,'gpu');eq(cost['inference'],0,'inference')
 if not (0<cost['wall_seconds']<600):raise AssertionError('analysis exceeds budget')
 report.update(status='pass',checks=checks,max_absolute_differences=maxima,recomputed_rows=12288,curve_points=3232,group_summaries=len(summary['groups']),native_comparisons=len(summary['native_comparison']),examples=6,input_prediction_files=256,input_truth_files=32,source_sha256=sha(pathlib.Path(__file__)),protocol_sha256=sha(BASE/'protocol.json'),result_complete_sha256=sha(RUN/'COMPLETE.json'))
except BaseException as e:
 report.update(status='fail',checks=checks,error=repr(e),traceback=traceback.format_exc(),max_absolute_differences=maxima)
finally:
 report['wall_seconds']=time.monotonic()-CLOCK;report['end_unix']=time.time();report['resource']=dict(user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,sys_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (BASE/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if report['status']!='pass':raise SystemExit(1)
