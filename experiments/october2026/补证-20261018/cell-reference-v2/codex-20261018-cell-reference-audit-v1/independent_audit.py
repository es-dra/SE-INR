"""Independent saved-output audit. Never imports experiment code or neural models.
Run after COMPLETE only, under: timeout 120s nice -n 10 env CUDA_VISIBLE_DEVICES='' ...
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, json, time, traceback, resource, signal
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1)
START=time.monotonic();CHECKS=0;MAX={};FAIL=[]
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def test(ok,msg):
    global CHECKS
    CHECKS+=1
    if not bool(ok):raise AssertionError(msg)
def close(x,y,msg,atol=1e-11,rtol=1e-10):
    x=np.asarray(x);y=np.asarray(y)
    test(x.shape==y.shape,msg+' shape')
    err=float(np.max(np.abs(x-y))) if x.size else 0.
    MAX[msg.split('/')[0]]=max(MAX.get(msg.split('/')[0],0.),err)
    test(np.all(np.isfinite(x)) and np.all(np.isfinite(y)),msg+' finite')
    test(np.allclose(x,y,atol=atol,rtol=rtol),f'{msg}: max error {err}')
def compare(expected,actual,path='summary'):
    if isinstance(expected,dict):
        test(set(expected)==set(actual),path+' keys')
        for k in expected:compare(expected[k],actual[k],path+'/'+str(k))
    elif isinstance(expected,list):
        test(len(expected)==len(actual),path+' length')
        for i,(x,y) in enumerate(zip(expected,actual)):compare(x,y,path+'/'+str(i))
    elif expected is None or isinstance(expected,(str,bool,int,np.integer)):
        test(expected==actual,path+' equality')
    else:close(expected,actual,path,atol=1e-9,rtol=1e-9)
def alarm(*args):raise TimeoutError('independent audit 120 second hard limit')
signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
def make_summary(rows,P,ids):
    groups=[]
    for model in ['lte','sc-inr']:
      for seed in [1,2]:
       for scale in P['scales']:
        for ref in P['relative_cells']:
         rr=[r for r in rows if (r['model'],r['seed'],r['scale'],r['ref'])==(model,seed,scale,ref)]
         if not rr:continue
         metrics={}
         for band in ['rgb','b1','b2','b3','bands']:
            ar=[r for r in rr if (r['band']!='rgb' if band=='bands' else r['band']==band)]
            neg=[r for r in ar if r['gain'] < -P['harm_threshold_mse']]
            loss=sum(-r['gain'] for r in neg)
            metrics[band]=dict(n=len(ar),negative=len(neg),overshoot=sum(r['label']=='overshoot' for r in neg),direction=sum(r['label']=='direction' for r in neg),boundary=sum(r['label']=='boundary' for r in neg),direction_harm_share=sum(-r['gain'] for r in neg if r['label']=='direction')/loss if loss else None,mean_db=float(np.mean([r['db'] for r in ar])),min_db=min(r['db'] for r in ar),max_db=max(r['db'] for r in ar),harm_gt_001db=sum(r['db']<-.01 for r in ar),harm_gt_005db=sum(r['db']<-.05 for r in ar),half_rescues=sum(r['half_gain']>P['harm_threshold_mse'] for r in neg),half_all_mse_change_percent=100*(sum(r['half_mse'] for r in ar)/sum(r['native_mse'] for r in ar)-1))
         groups.append(dict(model=model,seed=seed,scale=scale,ref=ref,metrics=metrics))
    # A single common image-cluster index matrix preserves all paired conditions.
    index=np.random.default_rng(P['bootstrap_seed']).integers(len(ids),size=(P['bootstrap_reps'],len(ids)))
    aggregate=[];cross_seed=[];cross_reference=[]
    for model in ['lte','sc-inr']:
      for ref in P['primary_references']:
        for scale in [2,4,'pooled_secondary']:
          rr=[r for r in rows if r['model']==model and r['ref']==ref and r['band']=='rgb' and (scale=='pooled_secondary' or r['scale']==scale)]
          per_image=[]
          for i in ids:
            local=[r for r in rr if r['image']==i]
            test(len(local)==(4 if scale=='pooled_secondary' else 2),'bootstrap cluster size')
            per_image.append([np.mean([r[k] for r in local]) for k in ['db','reference_mse','native_mse','half_mse']])
          v=np.asarray(per_image);samples=v[index].mean(axis=1)
          aggregate.append(dict(model=model,ref=ref,scale=scale,n_images=len(ids),n_records=len(rr),mean_db=float(v[:,0].mean()),mean_db_ci=np.percentile(samples[:,0],[2.5,97.5]).tolist(),half_mse_change_percent=100*(v[:,3].mean()/v[:,2].mean()-1),half_mse_change_ci=np.percentile(100*(samples[:,3]/samples[:,2]-1),[2.5,97.5]).tolist()))
        for band in ['rgb','bands']:
          rr=[r for r in rows if r['model']==model and r['ref']==ref and (r['band']!='rgb' if band=='bands' else r['band']=='rgb')]
          pairs={}
          for r in rr:pairs.setdefault((r['image'],r['scale'],r['band']),{})[r['seed']]=r
          vals=list(pairs.values());both=0;either=0;equal=0;equal_negative=0
          for v in vals:
            test(set(v)=={1,2},'cross seed pair complete')
            one=v[1]['gain'] < -P['harm_threshold_mse'];two=v[2]['gain'] < -P['harm_threshold_mse']
            both+=int(one and two);either+=int(one or two);equal+=int(v[1]['label']==v[2]['label']);equal_negative+=int(one and two and v[1]['label']==v[2]['label'])
          cross_seed.append(dict(model=model,ref=ref,band=band,n_pairs=len(vals),negative_either=either,negative_both=both,negative_jaccard=both/either if either else None,same_label=equal,same_label_among_both_negative=equal_negative))
    for model in ['lte','sc-inr']:
      for seed in [1,2]:
       for scale in [2,4]:
        for band in ['rgb','b1','b2','b3']:
          lookup={(r['image'],r['ref']):r for r in rows if (r['model'],r['seed'],r['scale'],r['band'])==(model,seed,scale,band) and r['ref'] in ['zero','h3']}
          counts=dict(both=0,zero_only=0,h3_only=0,neither=0);agreement=0
          for i in ids:
            a,b=lookup[(i,'zero')],lookup[(i,'h3')];z=a['gain'] < -P['harm_threshold_mse'];h=b['gain'] < -P['harm_threshold_mse']
            key={(True,True):'both',(True,False):'zero_only',(False,True):'h3_only',(False,False):'neither'}[(z,h)]
            counts[key]+=1;agreement+=int(z and h and a['label']==b['label'])
          cross_reference.append(dict(model=model,seed=seed,scale=scale,band=band,n_images=len(ids),counts=counts,same_label_both_harmful=agreement))
    return dict(rows=len(rows),groups=groups,aggregate=aggregate,cross_seed=cross_seed,cross_reference=cross_reference,max_identity_residual=max(abs(r['identity_residual']) for r in rows),training_runs=0)

def audit(root,code):
    P=read(code/'protocol.json');ids=P['cohort'];complete=read(root/'COMPLETE.json');budget=read(root/'budget.json')
    test(complete['status']=='complete','completed formal root required')
    test(len(ids)==32 and len(set(ids))==32,'32 fixed unique images')
    test(read(root/'cohort.json')==ids,'fixed cohort')
    source=read(root/'source_manifest.json')
    for rel,h in source.items():test(digest(code/rel)==h,'source hash '+rel)
    test(source['protocol.json']==digest(code/'protocol.json'),'protocol source hash')
    for name,h in P['checkpoints'].items():test(digest(code/'snapshot'/'checkpoints'/(name+'.pth'))==h,'checkpoint '+name)
    raw=read(root/'raw_image_manifest.json');test(len(raw)==32,'raw PNG count')
    inputs=read(root/'input_manifest.json');test(len(inputs)==96,'32 HR plus64 LR inputs')
    for rel,h in inputs.items():test(digest(root/rel)==h,'input hash '+rel)
    for i in ids:
        p=Path(P['data_root'])/f'{i:04}.png';test(digest(p)==raw[str(p)],'raw PNG hash')
        im=Image.open(p).convert('RGB');w,h=im.size
        arr=np.asarray(im,dtype=np.uint8)
        expected=arr[(h-128)//2:(h-128)//2+128,(w-128)//2:(w-128)//2+128].astype(np.float32)/np.float32(255)
        close(expected,np.load(root/'truth'/f'{i}.npy'),'preprocess HR',atol=0,rtol=0)
        for scale in P['scales']:
            tensor=torch.from_numpy(expected.transpose(2,0,1).copy())[None]
            low=F.interpolate(tensor,size=(128//scale,128//scale),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)[0].numpy()
            close(low,np.load(root/'observations'/f'{i}_x{scale}.npy'),'preprocess LR',atol=0,rtol=0)
    frozen=read(root/'prediction_cohort.json');freeze=read(root/'prediction_freeze.json')
    test(freeze['sha256']==digest(root/'prediction_cohort.json'),'prediction cohort freeze hash')
    test(freeze['files']==256 and len(frozen)==256,'256 frozen prediction files')
    expected_files={f'{i}_x{s}__{m}.npz' for i in ids for s in P['scales'] for m in P['checkpoints']}
    test(set(frozen)==expected_files,'complete unique prediction key set')
    test({p.name for p in (root/'predictions').glob('*')}==expected_files,'no missing extra prediction files')
    for rel,h in frozen.items():test(digest(root/'predictions'/rel)==h,'prediction file hash')
    journal=read(root/'inference_journal.json')
    test(len(journal)==256,'inference journal count')
    test({(r['image'],r['scale'],r['model']) for r in journal}=={(i,s,m) for i in ids for s in P['scales'] for m in P['checkpoints']},'journal exact fixed cohort')
    saved=read(root/'rows.json');test(len(saved)==3072,'3072 rows')
    rowkey=lambda r:(r['image'],r['scale'],r['model'],r['seed'],r['ref'],r['band'])
    lookup={rowkey(r):r for r in saved};test(len(lookup)==3072,'rows unique')
    f=np.fft.fftfreq(120);rad=np.maximum(abs(f[:,None]),abs(f[None,:]));masks=[(rad>=a)&(rad<b)&(rad>0) for a,b in P['bands']]
    test(np.array_equal(sum(mask.astype(int) for mask in masks),(rad>0).astype(int)),'frequency masks partition nonDC')
    regenerated=[];output_count=0;den=(120*120)**2*3;dc_residuals=[]
    for i in ids:
      truth=np.load(root/'truth'/f'{i}.npy').astype(np.float64)[4:124,4:124]
      T=np.fft.fft2(truth,axes=(0,1))
      for s in P['scales']:
       for name in P['checkpoints']:
        with np.load(root/'predictions'/f'{i}_x{s}__{name}.npz') as z:
          test(set(z.files)==set(P['relative_cells']),'four cell output keys')
          pred={}
          for tag in z.files:
            a=z[tag];test(a.shape==(128,128,3) and a.dtype==np.float32 and np.isfinite(a).all(),'finite float32 RGB prediction');output_count+=1
            pred[tag]=a.astype(np.float64)[4:124,4:124]
        native=P['native'][str(s)];pn=pred[native];model,seedtext=name.rsplit('_seed',1);seed=int(seedtext)
        spectra={tag:np.fft.fft2(x,axes=(0,1)) for tag,x in pred.items()}
        for ref in pred:
          if ref==native:continue
          spatial_e=truth-pred[ref];spatial_d=pn-pred[ref];E=T-spectra[ref];D=spectra[native]-spectra[ref]
          gains=[]
          for k,band in enumerate(['rgb','b1','b2','b3']):
            if k==0:
              e0=float(np.mean(spatial_e**2));en=float(np.mean((truth-pn)**2));a=float(np.mean(spatial_e*spatial_d));b=float(np.mean(spatial_d**2));half=float(np.mean((spatial_e-.5*spatial_d)**2))
            else:
              e=E[masks[k-1]];d=D[masks[k-1]]
              e0=float(np.sum(abs(e)**2)/den);en=float(np.sum(abs(e-d)**2)/den);a=float(np.real(np.vdot(e,d))/den);b=float(np.sum(abs(d)**2)/den);half=float(np.sum(abs(e-.5*d)**2)/den)
            gain=e0-en;tol=64*np.finfo(np.float64).eps*max(e0,b,1e-8)
            label='benefit_or_tie'
            if gain < -P['harm_threshold_mse']:
              if b<=tol or abs(a)<=tol or abs(a-b/2)<=tol:label='boundary'
              elif a<0:label='direction'
              else:label='overshoot'
            row=dict(image=i,scale=s,model=model,seed=seed,ref=ref,band=band,reference_mse=e0,native_mse=en,a=a,b=b,gain=gain,identity_residual=gain-(2*a-b),half_mse=half,half_gain=e0-half,label=label,db=float(10*np.log10(e0/en)))
            compare(row,lookup[rowkey(row)],'rows');regenerated.append(row);gains.append(gain)
          dc_gain=float(np.mean(spatial_e.mean((0,1))**2)-np.mean((truth-pn).mean((0,1))**2))
          close(gains[0]-sum(gains[1:]),dc_gain,'RGB minus bands equals DC',atol=1e-12)
          dc_residuals.append(abs(gains[0]-sum(gains[1:])-dc_gain))
    test(output_count==1024,'1024 RGB arrays independently audited')
    test(max(abs(r['identity_residual']) for r in regenerated)<1e-12,'independent gain identity bound')
    # Match only declared order, not saved arithmetic, before independent summary calculation.
    rlookup={rowkey(r):r for r in regenerated};ordered=[rlookup[rowkey(r)] for r in saved]
    expected=make_summary(ordered,P,ids);actual=read(root/'summary.json');compare(expected,actual)
    test(complete['gpu']==0 and complete['threads']==1 and complete['training_runs']==0,'CPU-only no-training completion')
    test(budget['wall_limit_seconds']==1800 and budget['cumulative_limit_seconds']==1800 and budget['resume_allowed'] is False,'fixed budget')
    close(budget['deadline_unix']-budget['start_unix'],1800,'budget absolute duration',atol=1e-6)
    test(budget['start_unix']<=freeze['unix_time']<=complete['end_unix']<=budget['deadline_unix'],'freeze and end inside launch deadline')
    test(0<=complete['seconds']<=1800,'script cumulative wall budget')
    phases=read(root/'phases.json');compare(phases,complete['phases'],'phase ledger')
    test(set(phases)=={'prepare','inference','score'} and all(v>=0 for v in phases.values()),'all three phases complete')
    test(sum(phases.values())<=complete['seconds']+1e-3,'phase sum within complete cost')
    test(sum(r['seconds'] for r in journal)<=phases['inference']+1e-3,'journal time within inference')
    return dict(prediction_files=256,rgb_arrays=1024,rows=3072,preprocessed_pngs=32,bootstrap_repetitions=10000,source_sha256=digest(code/'run.py'),protocol_sha256=digest(code/'protocol.json'),prediction_cohort_sha256=digest(root/'prediction_cohort.json'),max_rgb_band_dc_error=max(dc_residuals),formal_cost=complete)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--code',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path(a.root).resolve();code=Path(a.code).resolve();out=Path(a.out).resolve()
    test(not out.exists(),'new independent audit output required');test(root not in out.parents,'audit output must be outside formal root');out.mkdir(parents=True)
    record=dict(status='failed',audit_source_sha256=digest(Path(__file__)),formal_root=str(root),code_root=str(code),gpu=0,model_inference_runs=0,training_runs=0,threads=1,hard_limit_seconds=120,nice=os.getpriority(os.PRIO_PROCESS,0))
    try:
      test(record['nice']>=10,'run audit with nice10 or lower priority')
      record.update(audit(root,code));record['status']='passed'
    except BaseException:
      record['error']=traceback.format_exc();raise
    finally:
      record.update(checks=CHECKS,max_errors=MAX,wall_seconds=time.monotonic()-START,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
      (out/'audit.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
      print(json.dumps({k:record[k] for k in ['status','checks','wall_seconds']},indent=2),flush=True)
      signal.alarm(0)
if __name__=='__main__':main()
