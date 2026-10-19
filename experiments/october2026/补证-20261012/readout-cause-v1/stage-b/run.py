"""Finite fixed-model readout-cause intervention. No optimiser or model training."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,argparse,hashlib,resource,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False
HERE=Path(__file__).resolve().parent;P=json.loads((HERE/'protocol.json').read_text())
sys.path.insert(0,str(Path(P['model_root'])/'snapshot/src'))
import models

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def coords(n):
 v=(np.arange(n)+.5)/n
 return np.stack(np.meshgrid(v,v,indexing='ij'),-1)
def wave(q,modes,phases,shift):
 z=np.zeros(q.shape[:2],np.float64)
 for (ky,kx),ph in zip(modes,phases):z+=P['amplitude_per_mode']*np.cos(2*np.pi*(ky*(q[:,:,0]+shift[0])+kx*(q[:,:,1]+shift[1]))+ph)
 return z
def periodic_operator():
 # Three tiled periods, take the middle LR period. No outer-boundary weights reach it.
 eye=np.eye(64,dtype=np.float64)
 z=np.tile(eye[:,None,None,:],(1,1,3,3))
 out=F.interpolate(torch.from_numpy(z),size=(3,48),mode='bicubic',antialias=True,align_corners=False)
 return out[:,0,1,16:32].numpy().T.copy()
def prepare(root,smoke,guard):
 for d in ['observations','truth','predictions']:(root/d).mkdir()
 inherited=json.loads((HERE/'inherited_manifest.json').read_text())
 for path,h in inherited.items():guard();assert sha(path)==h,('inherited drift',path)
 write(root/'inherited_manifest.json',inherited)
 M=periodic_operator();np.save(root/'operator.npy',M)
 assert np.max(abs(M.sum(1)-1))<1e-12
 q=coords(64);rng=np.random.default_rng(P['phase_seed'])
 phases=np.vstack([np.zeros(3),rng.uniform(0,2*np.pi,(3,3))]);write(root/'phases.json',phases.tolist())
 templates=P['templates'] if not smoke else {'preflight':P['preflight']['template']}
 cases=[];families=[];pool={};checks=[]
 for template,modes in templates.items():
  usedphases=phases if not smoke else np.array([P['preflight']['phases']])
  for pi,ph in enumerate(usedphases):
   for si,delta in enumerate(P['shifts_LR_pixels'] if not smoke else [P['preflight']['shift_LR_pixels']]):
    guard();shift=np.array(delta)/16
    L=.5+wave(q,P['low_modes'],[0,0,0],shift);H=wave(q,modes,ph,shift);T=L+H
    assert T.min()>0 and T.max()<1
    YL=M@L@M.T;YH=M@H@M.T
    tiled=np.tile(T,(3,3));direct=F.interpolate(torch.tensor(tiled)[None,None],size=(48,48),mode='bicubic',antialias=True,align_corners=False)[0,0,16:32,16:32].numpy()
    err=float(np.max(abs(direct-(YL+YH))));assert err<1e-12
    fid=f'{template}_p{pi}_s{si}'
    np.savez_compressed(root/'truth'/(fid+'.npz'),target=T,low=L,high=H,Ylow=YL,Yhigh=YH)
    families.append(dict(id=fid,template=template,phase=pi,phase_values=ph.tolist(),shift=si,shift_LR=delta,modes=modes))
    checks.append(dict(family=fid,direct_operator_error=err,target_range=[float(T.min()),float(T.max())],high_energy=float(np.mean(H*H)),target_variance=float(np.var(T)),alias_energy=float(np.mean(YH*YH))))
    for eta in P['etas']:
     effective=1. if template=='control_low' else eta
     y=(YL+effective*YH).astype(np.float32);assert y.min()>0 and y.max()<1
     h=arrsha(y)
     if h not in pool:
      pool[h]=f'o{len(pool):04d}';np.save(root/'observations'/(pool[h]+'.npy'),y)
     cases.append(dict(id=f'{fid}_e{eta:g}',family=fid,template=template,phase=pi,shift=si,eta=eta,effective_eta=effective,observation=pool[h],observation_array_sha256=h,input_mean=float(y.mean(dtype=np.float64)),input_variance=float(y.var(dtype=np.float64)),input_periodic_gradient=float(np.mean((y-np.roll(y,1,0))**2+(y-np.roll(y,1,1))**2)),high_path_energy=float(np.mean(YH**2))))
 write(root/'families.json',families);write(root/'cases.json',cases);write(root/'construction_checks.json',checks)
 manifest={str(f.relative_to(root)):sha(f) for d in ['truth','observations'] for f in sorted((root/d).glob('*'))}
 for f in ['operator.npy','phases.json','families.json','cases.json','construction_checks.json']:manifest[f]=sha(root/f)
 write(root/'input_manifest.json',manifest)
 write(root/'frozen.json',dict(time_epoch=time.time(),source_sha256=sha(HERE/'run.py'),protocol_sha256=sha(HERE/'protocol.json'),inputs_sha256=sha(root/'input_manifest.json'),inherited_sha256=sha(root/'inherited_manifest.json'),conditions=len(cases),unique_observations=len(pool),smoke=smoke))
 return cases
def infer(root,cases,guard):
 journal=[];qt=torch.tensor(2*coords(64).reshape(-1,2)-1,dtype=torch.float32)[None];queries=0
 for name in P['models']:
  ck=torch.load(Path(P['model_root'])/'snapshot/checkpoints'/name/'epoch-last.pth',map_location='cpu',weights_only=True);assert ck['epoch']==1000
  net=models.make(ck['model'],load_sd=True).cpu().eval()
  for par in net.parameters():par.requires_grad_(False)
  for i,f in enumerate(sorted((root/'observations').glob('*.npy'))):
   guard();tic=time.monotonic();y=np.load(f);inp=torch.from_numpy(np.tile(y[None],(3,1,1)))[None]
   pp={}
   with torch.inference_mode():
    net.gen_feat(inp*2-1)
    for typ,c in [('point',0.),('native',2/64)]:
     a=net.query_rgb(qt,torch.full_like(qt,c))[0].numpy()*.5+.5
     assert np.isfinite(a).all();pp[typ]=a.reshape(64,64,3);queries+=len(a)
   np.savez_compressed(root/'predictions'/(f.stem+'__'+name+'.npz'),**pp)
   journal.append(dict(observation=f.stem,model=name,seconds=time.monotonic()-tic,queries_cumulative=queries));write(root/'inference_journal.json',journal)
   if i%16==0:print('INFER',name,i+1,round(journal[-1]['seconds'],3),flush=True)
  del net,ck
 manifest={str(f.relative_to(root)):sha(f) for f in sorted((root/'predictions').glob('*'))}
 write(root/'prediction_cohort.json',manifest);write(root/'prediction_freeze.json',dict(time_epoch=time.time(),cohort_sha256=sha(root/'prediction_cohort.json'),queries=queries))
def planes(z,roi):
 lo,hi=roi;z=np.asarray(z,dtype=np.float64)[lo:hi,lo:hi]
 if z.ndim==2:z=np.repeat(z[:,:,None],3,2)
 f=np.fft.fftfreq(len(z));r=np.maximum(abs(f[:,None]),abs(f[None,:]));ff=np.fft.fft2(z,axes=(0,1));ans=[z]
 for low,high in [(0,.125),(.125,.25),(.25,.5001)]:
  mask=(r>=low)&(r<high)&(r>0);ans.append(np.fft.ifft2(ff*mask[:,:,None],axes=(0,1)).real)
 return ans
def measure(t,p,h):
 d=h-p;e=t-p;a=float(np.mean(e*d));b=float(np.mean(d*d));e0=float(np.mean(e*e));en=float(np.mean((t-h)**2));g=e0-en
 assert abs(g-(2*a-b))<1e-12
 typ='beneficial' if g>1e-8 else 'tie'
 if g< -1e-8:typ='tiny_change' if b<=1e-12 else 'wrong_direction' if a< -1e-10 else 'overshoot' if a>1e-10 else 'direction_boundary'
 gains={str(lam):float(e0-np.mean((t-(p+lam*d))**2)) for lam in P['evaluation']['lambdas']}
 assert max(abs(gains[str(lam)]-(2*lam*a-lam*lam*b)) for lam in P['evaluation']['lambdas'])<1e-12
 return dict(gain=g,alignment=a,change_energy=b,lambda_star=a/b if b>1e-12 else None,cos_alignment=a/np.sqrt(e0*b) if e0*b>1e-24 else None,point_mse=e0,native_mse=en,category=typ,mixture_gains=gains)
def aggregate(rs):
 neg=[r for r in rs if r['gain']< -1e-8];loss=sum(-r['gain'] for r in neg)
 return dict(rows=len(rs),negative=len(neg),positive=sum(r['gain']>1e-8 for r in rs),ties=sum(abs(r['gain'])<=1e-8 for r in rs),mean_gain=float(np.mean([r['gain'] for r in rs])),min_gain=min(r['gain'] for r in rs),max_gain=max(r['gain'] for r in rs),categories={k:sum(r['category']==k for r in neg) for k in ['wrong_direction','overshoot','direction_boundary','tiny_change']},negative_loss=loss,wrong_direction_loss_fraction=sum(-r['gain'] for r in neg if r['category']=='wrong_direction')/loss if loss else None)
def score(root,cases,guard):
 # Called only after entire predictions SHA freeze. Truth is first read here after preparation.
 rows=[]
 for c in cases:
  guard();t=np.load(root/'truth'/(c['family']+'.npz'))['target']
  for name in P['models']:
   z=np.load(root/'predictions'/(c['observation']+'__'+name+'.npz'))
   for roi,box in P['evaluation']['ROIs'].items():
    for band,(tp,p,h) in enumerate(zip(planes(t,box),planes(z['point'],box),planes(z['native'],box))):
     rows.append(dict(**c,model=name,roi=roi,band=band,**measure(tp,p,h)))
 write(root/'rows.json',rows)
 keys=lambda r:(r['family'],r['model'],r['roi'],r['band'])
 lookup={(keys(r),r['eta']):r for r in rows};pairs=[]
 for r0 in rows:
  if r0['eta']!=0:continue
  half=lookup[(keys(r0),.5)];r1=lookup[(keys(r0),1)]
  pairs.append(dict(family=r0['family'],template=r0['template'],phase=r0['phase'],shift=r0['shift'],model=r0['model'],roi=r0['roi'],band=r0['band'],gain0=r0['gain'],gainhalf=half['gain'],gain1=r1['gain'],delta_alias=r1['gain']-r0['gain'],delta_alignment=r1['alignment']-r0['alignment'],delta_energy=r1['change_energy']-r0['change_energy'],sign_flip=r0['gain']*r1['gain']<0 and min(abs(r0['gain']),abs(r1['gain']))>1e-8,category0=r0['category'],category1=r1['category']))
 write(root/'alias_pairs.json',pairs)
 groups={};effects={};phase=[]
 for model in P['models']:
  for roi in P['evaluation']['ROIs']:
   for template in sorted(set(r['template'] for r in rows)):
    for band in range(4):
     subset=[r for r in rows if (r['model'],r['roi'],r['template'],r['band'])==(model,roi,template,band)]
     for eta in P['etas']:
      rs=[r for r in subset if r['eta']==eta];groups[f'{model}/{roi}/{template}/b{band}/e{eta:g}']=aggregate(rs)
     pr=[r for r in pairs if (r['model'],r['roi'],r['template'],r['band'])==(model,roi,template,band)]
     effects[f'{model}/{roi}/{template}/b{band}']=dict(pairs=len(pr),sign_flips=sum(r['sign_flip'] for r in pr),mean_alias_delta=float(np.mean([r['delta_alias'] for r in pr])),min_alias_delta=min(r['delta_alias'] for r in pr),max_alias_delta=max(r['delta_alias'] for r in pr),positive_alias_delta=sum(r['delta_alias']>1e-8 for r in pr),negative_alias_delta=sum(r['delta_alias']< -1e-8 for r in pr))
     for shift in sorted(set(r['shift'] for r in subset)):
      for eta in P['etas']:
       rr=[r for r in subset if r['shift']==shift and r['eta']==eta];gs=[r['gain'] for r in rr]
       phase.append(dict(model=model,roi=roi,template=template,band=band,shift=shift,eta=eta,phase_gains=gs,range_gain=max(gs)-min(gs),both_signs=min(gs)< -1e-8 and max(gs)>1e-8))
 write(root/'phase_effects.json',phase)
 pairmap={(r['template'],r['phase'],r['shift'],r['model'],r['roi'],r['band']):r for r in pairs};interactions=[]
 for r in pairs:
  if r['phase']==0:continue
  r0=pairmap[(r['template'],0,r['shift'],r['model'],r['roi'],r['band'])]
  interactions.append(dict(template=r['template'],phase=r['phase'],shift=r['shift'],model=r['model'],roi=r['roi'],band=r['band'],alias_by_phase=r['delta_alias']-r0['delta_alias'],alignment_interaction=r['delta_alignment']-r0['delta_alignment'],energy_interaction=r['delta_energy']-r0['delta_energy']))
 write(root/'interactions.json',interactions)
 write(root/'summary.json',dict(groups=groups,alias_effects=effects,rows=len(rows),conditions=len(cases),prediction_files=len(list((root/'predictions').glob('*.npz'))),scope=P['purpose'],SR_training=0,GPU=0))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--launch',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 root=Path(args.out);root.mkdir();launch=json.loads(Path(args.launch).read_text());write(root/'launch.json',launch);start=time.monotonic();phasecost={};status='failed'
 def guard():
  if time.time()>launch['deadline_epoch'] or time.monotonic()-start>P['budget']['run_cumulative_seconds']:raise TimeoutError('frozen budget exhausted')
 try:
  tic=time.monotonic();cases=prepare(root,args.smoke,guard);phasecost['prepare']=time.monotonic()-tic
  tic=time.monotonic();infer(root,cases,guard);phasecost['inference']=time.monotonic()-tic
  tic=time.monotonic();score(root,cases,guard);phasecost['score']=time.monotonic()-tic;guard();status='complete';(root/'COMPLETE').write_text('complete\n')
 except BaseException:
  (root/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(root/'resources.json',dict(status=status,wall_seconds=time.time()-launch['start_epoch'],script_seconds=time.monotonic()-start,cpu_seconds=u.ru_utime+u.ru_stime,peak_rss_kib=u.ru_maxrss,phase_seconds=phasecost,CPU_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),GPU=0,SR_training=0,end_epoch=time.time()))
if __name__=='__main__':main()
