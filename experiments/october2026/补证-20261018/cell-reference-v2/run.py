"""Bounded inference-only cell-reference replication; frozen protocol, no fitting."""
import argparse, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
P = json.loads((HERE/'protocol.json').read_text())
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
sys.path.insert(0,str(HERE/'snapshot'/'src'))
import models

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for z in iter(lambda:f.read(1048576),b''): h.update(z)
    return h.hexdigest()

def write(p,d):
    p.write_text(json.dumps(d,indent=2,allow_nan=False,default=lambda v: v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__)))+'\n')

def manifest(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}

def load(name):
    p=HERE/'snapshot'/'checkpoints'/(name+'.pth')
    assert sha(p)==P['checkpoints'][name]
    ck=torch.load(p,map_location='cpu',weights_only=True)
    assert ck['epoch']==1000
    m=models.make(ck['model'],load_sd=True).cpu().eval()
    for a in m.parameters(): a.requires_grad_(False)
    return m

def prepare(out,ids,guard):
    (out/'truth').mkdir();(out/'observations').mkdir()
    raw={}
    for i in ids:
        guard(); p=Path(P['data_root'])/f'{i:04}.png';raw[str(p)]=sha(p)
        im=Image.open(p).convert('RGB');w,h=im.size;assert min(w,h)>=128
        x=np.array(im.crop(((w-128)//2,(h-128)//2,(w-128)//2+128,(h-128)//2+128)),dtype='float32').transpose(2,0,1)/255
        np.save(out/'truth'/f'{i}.npy',x.transpose(1,2,0))
        for s in P['scales']:
            y=F.interpolate(torch.from_numpy(x)[None],size=(128//s,128//s),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)[0].numpy()
            np.save(out/'observations'/f'{i}_x{s}.npy',y)
    write(out/'raw_image_manifest.json',raw)
    write(out/'input_manifest.json',{str(p.relative_to(out)):sha(p) for d in ['truth','observations'] for p in sorted((out/d).glob('*'))})

@torch.inference_mode()
def query(m,q,c,chunk,guard):
    z=[]
    for j in range(0,q.shape[1],chunk):
        guard();a=q[:,j:j+chunk];z.append(m.query_rgb(a,torch.full_like(a,c))[0].numpy()*.5+.5)
    z=np.concatenate(z).reshape(128,128,3)
    assert np.isfinite(z).all()
    return z

@torch.inference_mode()
def predict(out,ids,guard,preflight):
    (out/'predictions').mkdir(); journal=[]; checks=[]
    v=(np.arange(128)+.5)/128
    q=torch.from_numpy((2*np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)-1).astype('float32'))[None]
    for name in P['checkpoints']:
        m=load(name)
        for i in ids:
            for s in P['scales']:
                guard();t=time.monotonic();y=np.load(out/'observations'/f'{i}_x{s}.npy')
                m.gen_feat(torch.from_numpy(y)[None]*2-1)
                preds={tag:query(m,q,r/(128//s),P['query_chunk'],guard) for tag,r in P['relative_cells'].items()}
                if preflight:
                    native=P['native'][str(s)]
                    full=query(m,q,2/128,128*128,guard)
                    diff=float(np.max(abs(full-preds[native])))
                    assert diff<=2e-6, (name,s,diff)
                    checks.append(dict(model=name,scale=s,max_native_chunk_difference=diff))
                np.savez_compressed(out/'predictions'/f'{i}_x{s}__{name}.npz',**preds)
                journal.append(dict(image=i,scale=s,model=name,seconds=time.monotonic()-t))
                write(out/'inference_journal.json',journal)
                print('PRED',name,i,s,round(journal[-1]['seconds'],3),flush=True)
        del m
    cohort=manifest(out/'predictions');write(out/'prediction_cohort.json',cohort)
    write(out/'prediction_freeze.json',dict(sha256=sha(out/'prediction_cohort.json'),files=len(cohort),unix_time=time.time()))
    if preflight:write(out/'preflight_checks.json',checks)

def project(x):
    x=x.astype('float64')[4:124,4:124]
    f=np.fft.fftfreq(120);r=np.maximum(abs(f[:,None]),abs(f[None,:]));X=np.fft.fft2(x,axes=(0,1))
    return {'rgb':x,**{f'b{k+1}':np.fft.ifft2(X*((r>=a)&(r<b)&(r>0))[:,:,None],axes=(0,1)).real for k,(a,b) in enumerate(P['bands'])}}

def score(out,ids,guard):
    frozen=json.loads((out/'prediction_cohort.json').read_text())
    assert manifest(out/'predictions')==frozen
    rows=[]
    for i in ids:
        target=project(np.load(out/'truth'/f'{i}.npy'))
        for s in P['scales']:
            for name in P['checkpoints']:
                guard();z=np.load(out/'predictions'/f'{i}_x{s}__{name}.npz')
                pr={k:project(z[k]) for k in z.files};native=P['native'][str(s)]
                for ref in pr:
                    if ref==native:continue
                    for band,t in target.items():
                        p0=pr[ref][band];pn=pr[native][band];d=pn-p0;e=t-p0
                        e0=float(np.mean(e*e));en=float(np.mean((t-pn)**2));a=float(np.mean(e*d));b=float(np.mean(d*d))
                        g=e0-en;tol=64*np.finfo('float64').eps*max(e0,b,1e-8)
                        label='benefit_or_tie'
                        if g < -P['harm_threshold_mse']:
                            label='boundary' if b<=tol or abs(a)<=tol or abs(a-b/2)<=tol else ('direction' if a<0 else 'overshoot')
                        half=float(np.mean((t-(p0+.5*d))**2))
                        rows.append(dict(image=i,scale=s,model=name.split('_')[0],seed=int(name[-1]),ref=ref,band=band,reference_mse=e0,native_mse=en,a=a,b=b,gain=g,identity_residual=g-(2*a-b),half_mse=half,half_gain=e0-half,label=label,db=10*np.log10(e0/en)))
    assert max(abs(r['identity_residual']) for r in rows)<1e-12
    write(out/'rows.json',rows)
    groups=[]
    for model in ['lte','sc-inr']:
      for seed in [1,2]:
       for s in P['scales']:
        for ref in P['relative_cells']:
         rr=[r for r in rows if r['model']==model and r['seed']==seed and r['scale']==s and r['ref']==ref]
         if not rr:continue
         g=dict(model=model,seed=seed,scale=s,ref=ref,metrics={})
         for band in ['rgb','b1','b2','b3','bands']:
            ar=[r for r in rr if (r['band']!='rgb' if band=='bands' else r['band']==band)]
            neg=[r for r in ar if r['gain'] < -1e-8];loss=sum(-r['gain'] for r in neg)
            g['metrics'][band]=dict(n=len(ar),negative=len(neg),overshoot=sum(r['label']=='overshoot' for r in neg),direction=sum(r['label']=='direction' for r in neg),boundary=sum(r['label']=='boundary' for r in neg),direction_harm_share=sum(-r['gain'] for r in neg if r['label']=='direction')/loss if loss else None,mean_db=float(np.mean([r['db'] for r in ar])),min_db=min(r['db'] for r in ar),max_db=max(r['db'] for r in ar),harm_gt_001db=sum(r['db']<-.01 for r in ar),harm_gt_005db=sum(r['db']<-.05 for r in ar),half_rescues=sum(r['half_gain']>1e-8 for r in neg),half_all_mse_change_percent=100*(sum(r['half_mse'] for r in ar)/sum(r['native_mse'] for r in ar)-1))
         groups.append(g)
    rng=np.random.default_rng(P['bootstrap_seed']);ix=rng.integers(0,len(ids),(P['bootstrap_reps'],len(ids)))
    aggregate=[];stability=[];reference_stability=[]
    for model in ['lte','sc-inr']:
      for ref in P['primary_references']:
        ar=[r for r in rows if r['model']==model and r['ref']==ref and r['band']=='rgb']
        for scale in [2,4,'pooled_secondary']:
          sr=[r for r in ar if scale=='pooled_secondary' or r['scale']==scale]
          vals=np.array([[np.mean([r[k] for r in sr if r['image']==i]) for k in ['db','reference_mse','native_mse','half_mse']] for i in ids])
          boot=vals[ix].mean(1)
          ratio=100*(boot[:,3]/boot[:,2]-1)
          aggregate.append(dict(model=model,ref=ref,scale=scale,n_images=len(ids),n_records=len(sr),mean_db=float(vals[:,0].mean()),mean_db_ci=np.quantile(boot[:,0],[.025,.975]).tolist(),half_mse_change_percent=100*(vals[:,3].mean()/vals[:,2].mean()-1),half_mse_change_ci=np.quantile(ratio,[.025,.975]).tolist()))
        for band in ['rgb','bands']:
          ar=[r for r in rows if r['model']==model and r['ref']==ref and (r['band']!='rgb' if band=='bands' else r['band']=='rgb')]
          pairs={}
          for r in ar:pairs.setdefault((r['image'],r['scale'],r['band']),{})[r['seed']]=r
          p=list(pairs.values());u=sum(x[1]['gain'] < -1e-8 or x[2]['gain'] < -1e-8 for x in p)
          both=sum(x[1]['gain'] < -1e-8 and x[2]['gain'] < -1e-8 for x in p)
          stability.append(dict(model=model,ref=ref,band=band,n_pairs=len(p),negative_either=u,negative_both=both,negative_jaccard=both/u if u else None,same_label=sum(x[1]['label']==x[2]['label'] for x in p),same_label_among_both_negative=sum(x[1]['gain'] < -1e-8 and x[2]['gain'] < -1e-8 and x[1]['label']==x[2]['label'] for x in p)))
    for model in ['lte','sc-inr']:
      for seed in [1,2]:
       for scale in [2,4]:
        for band in ['rgb','b1','b2','b3']:
          ar=[r for r in rows if r['model']==model and r['seed']==seed and r['scale']==scale and r['band']==band and r['ref'] in ['zero','h3']]
          pairs={}
          for r in ar:pairs.setdefault(r['image'],{})[r['ref']]=r
          counts={'both':0,'zero_only':0,'h3_only':0,'neither':0};agree=0
          for pair in pairs.values():
            z=pair['zero']['gain'] < -1e-8; h=pair['h3']['gain'] < -1e-8
            counts['both' if z and h else 'zero_only' if z else 'h3_only' if h else 'neither']+=1
            agree+=int(z and h and pair['zero']['label']==pair['h3']['label'])
          reference_stability.append(dict(model=model,seed=seed,scale=scale,band=band,n_images=len(pairs),counts=counts,same_label_both_harmful=agree))
    write(out/'summary.json',dict(rows=len(rows),groups=groups,aggregate=aggregate,cross_seed=stability,cross_reference=reference_stability,max_identity_residual=max(abs(r['identity_residual']) for r in rows),training_runs=0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=False)
    start=time.time();mono=time.monotonic();budget=P['budget']['preflight_wall_seconds' if args.preflight else 'formal_wall_seconds']
    deadline=start+budget;write(out/'budget.json',dict(start_unix=start,deadline_unix=deadline,wall_limit_seconds=budget,cumulative_limit_seconds=budget,resume_allowed=False))
    def guard():
        if time.time()>=deadline or time.monotonic()-mono>=budget:raise TimeoutError('fixed budget exhausted')
    write(out/'source_manifest.json',manifest(HERE))
    ids=P['preflight_ids'] if args.preflight else P['cohort'];write(out/'cohort.json',ids)
    phases={}
    try:
        for name,fn in [('prepare',lambda:prepare(out,ids,guard)),('inference',lambda:predict(out,ids,guard,args.preflight)),('score',lambda:score(out,ids,guard))]:
            t=time.monotonic();fn();guard();phases[name]=time.monotonic()-t;write(out/'phases.json',phases)
        if not args.preflight:
            assert len(json.loads((out/'prediction_cohort.json').read_text()))==P['expected']['prediction_files']
            assert len(json.loads((out/'rows.json').read_text()))==P['expected']['contrast_rows']
        write(out/'COMPLETE.json',dict(status='complete',seconds=time.monotonic()-mono,end_unix=time.time(),phases=phases,cpu_user=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system=resource.getrusage(resource.RUSAGE_SELF).ru_stime,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu=0,threads=1,training_runs=0))
    except Exception:
        write(out/'FAILED.json',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-mono,phases=phases));raise

if __name__=='__main__':main()
