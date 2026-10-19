"""Independent NumPy StageB artifact audit. No torch/model/run.py import."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,time,datetime,resource,traceback
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H/'collected/codex-20261012-readout-cause-run-v1';P=json.loads((H/'protocol.json').read_text());START=time.monotonic();CPU=time.process_time();N=0;FAIL=[];ERR={};E=1e-8

def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(ok,k,detail=None):
 global N
 N+=1
 if not bool(ok):FAIL.append({'label':k,'detail':str(detail)[:200]})
 if time.monotonic()-START>170:raise TimeoutError('audit170sbudget')
def near(a,b,k,atol=1e-10,rtol=1e-9):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 if a.shape!=b.shape:ck(False,k,'shape');return
 finite=np.isfinite(a)&np.isfinite(b);dif=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0.;ERR[k.split(':')[0]]=max(ERR.get(k.split(':')[0],0),dif)
 ck(np.array_equal(np.isfinite(a),np.isfinite(b)) and np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True),k,dif)
def rec(a,b,k):
 if isinstance(a,dict):
  for x,v in a.items():
   ck(x in b,k+':key:'+x)
   if x in b:rec(v,b[x],k+':'+x)
 elif isinstance(a,(list,tuple)):
  ck(len(a)==len(b),k+':length')
  if len(a)==len(b):
   for i,(v,w) in enumerate(zip(a,b)):rec(v,w,k+':'+str(i))
 elif a is None or isinstance(a,(str,bool)):ck(a==b,k,(a,b))
 else:near(a,b,k)
def clean(a):
 if isinstance(a,dict):return {k:clean(v) for k,v in a.items()}
 if isinstance(a,(list,tuple)):return [clean(v) for v in a]
 if isinstance(a,np.ndarray):return clean(a.tolist())
 if isinstance(a,np.bool_):return bool(a)
 if isinstance(a,np.integer):return int(a)
 if isinstance(a,(float,np.floating)):return float(a) if np.isfinite(a) else None
 return a

def synth(modes,phases,delta):
 n=P['HR'];grid=np.arange(n)+.5;result=np.zeros((n,n))
 for (ky,kx),phi in zip(modes,phases):
  ey=np.exp(2j*np.pi*ky*(grid+4*delta[0])/n);ex=np.exp(2j*np.pi*kx*(grid+4*delta[1])/n)
  result+=P['amplitude_per_mode']*(np.outer(ey,ex)*np.exp(1j*phi)).real
 return result

def Fcrop(v,roi):
 lo,hi=roi;a=np.asarray(v,dtype=float)[lo:hi,lo:hi]
 if a.ndim==2:a=np.repeat(a[:,:,None],3,2)
 return np.fft.fft2(a,axes=(0,1))
def filter_masks(n):
 f=np.fft.fftfreq(n);rr=np.maximum(abs(f[:,None]),abs(f[None,:]));return [np.ones((n,n),bool)]+[(rr>=a)&(rr<b)&(rr>0) for a,b in [(0,.125),(.125,.25),(.25,.5001)]]
def calc(T,F,C,band):
 n=len(T);mask=filter_masks(n)[band];err=T-F;delta=C-F;den=3*n**4
 inner=lambda x,y:float((x[mask].conj()*y[mask]).real.sum()/den)
 a=inner(err,delta);b=inner(delta,delta);e0=inner(err,err);en=inner(err-delta,err-delta);g=e0-en
 typ='beneficial' if g>E else 'tie'
 if g<-E:typ='tiny_change' if b<=1e-12 else 'wrong_direction' if a<-1e-10 else 'overshoot' if a>1e-10 else 'direction_boundary'
 mix={str(l):e0-inner(err-l*delta,err-l*delta) for l in P['evaluation']['lambdas']}
 for l in P['evaluation']['lambdas']:near(mix[str(l)],2*l*a-l*l*b,'quadratic_identity',1e-12,0)
 return dict(gain=g,alignment=a,change_energy=b,lambda_star=a/b if b>1e-12 else None,cos_alignment=a/(e0*b)**.5 if e0*b>1e-24 else None,point_mse=e0,native_mse=en,category=typ,mixture_gains=mix)

def aggregate(rs):
 neg=[r for r in rs if r['gain']<-E];loss=sum(-r['gain'] for r in neg)
 return dict(rows=len(rs),negative=len(neg),positive=sum(r['gain']>E for r in rs),ties=sum(abs(r['gain'])<=E for r in rs),mean_gain=np.mean([r['gain'] for r in rs]),min_gain=min(r['gain'] for r in rs),max_gain=max(r['gain'] for r in rs),categories={k:sum(r['category']==k for r in neg) for k in ['wrong_direction','overshoot','direction_boundary','tiny_change']},negative_loss=loss,wrong_direction_loss_fraction=sum(-r['gain'] for r in neg if r['category']=='wrong_direction')/loss if loss else None)

def audit():
 frozen=rd(R/'frozen.json');manifest=rd(R/'input_manifest.json');cohort=rd(R/'prediction_cohort.json');pf=rd(R/'prediction_freeze.json');launch=rd(R/'launch.json');res=rd(R/'resources.json')
 ck(sha(H/'run.py')==frozen['source_sha256'],'sourceSHA');ck(sha(H/'protocol.json')==frozen['protocol_sha256'],'protocolSHA');ck(sha(R/'input_manifest.json')==frozen['inputs_sha256'],'inputManifestSHA');ck(sha(R/'inherited_manifest.json')==frozen['inherited_sha256'],'inheritedSHA');ck(rd(R/'inherited_manifest.json')==rd(H/'inherited_manifest.json'),'inherited manifest bytes')
 for n,h in {**manifest,**cohort}.items():ck(sha(R/n)==h,'fileSHA:'+n)
 ck(sha(R/'prediction_cohort.json')==pf['cohort_sha256'],'predictionCohortSHA');ck(launch['start_epoch']<=frozen['time_epoch']<=pf['time_epoch']<=res['end_epoch'],'freeze order');ck(res['end_epoch']<=launch['deadline_epoch'] and res['script_seconds']<=600 and res['wall_seconds']<=900,'budgets');ck(res['CPU_threads']==1 and res['nice']==10 and res['GPU']==0 and res['SR_training']==0,'resource restrictions')
 ck(res['status']=='complete' and (R/'COMPLETE').exists(),'completion')
 M=np.load(R/'operator.npy');ck(M.shape==(16,64),'M shape');near(M.sum(1),np.ones(16),'M constants',1e-12,0)
 for j in range(16):near(M[j],np.roll(M[0],4*j),'M periodic4shift',1e-12,0)
 phase=np.vstack([np.zeros(3),np.random.default_rng(P['phase_seed']).uniform(0,2*np.pi,(3,3))]);near(phase,rd(R/'phases.json'),'phases',0,0)
 fam=rd(R/'families.json');cases=rd(R/'cases.json');ck(len(fam)==80 and len(cases)==240,'cohort sizes');ck(len({c['id'] for c in cases})==240,'unique conditions')
 truth={};lowobs={};spectra={};controls={};checks=rd(R/'construction_checks.json');checkmap={c['family']:c for c in checks}
 for f in fam:
  fid=f['id'];modes=P['templates'][f['template']];delta=P['shifts_LR_pixels'][f['shift']];ph=phase[f['phase']];rec(modes,f['modes'],'mode IDs');near(delta,f['shift_LR'],'shifts',0,0);near(ph,f['phase_values'],'phase values',0,0)
  if f['template']=='control_low':ck(all(max(abs(k) for k in v)<8 for v in modes),'sham belowNyquist')
  else:ck(all(max(abs(k) for k in v)>8 for v in modes),'supraNyquist modes')
  low=.5+synth(P['low_modes'],[0,0,0],delta);hi=synth(modes,ph,delta);t=low+hi;YL=M.dot(low).dot(M.T);YH=M.dot(hi).dot(M.T);z=np.load(R/'truth'/(fid+'.npz'))
  for k,v in [('target',t),('low',low),('high',hi),('Ylow',YL),('Yhigh',YH)]:near(v,z[k],'construct:'+k,1e-12,0)
  ck(t.min()>0 and t.max()<1,'target noclamp');truth[fid]=t;lowobs[fid]=(YL,YH)
  chk=checkmap[fid];near([t.min(),t.max()],chk['target_range'],'construct range');near(np.mean(hi**2),chk['high_energy'],'high energy');near(np.var(t),chk['target_variance'],'target variance');near(np.mean(YH**2),chk['alias_energy'],'path energy');ck(chk['direct_operator_error']<1e-12,'root operator direct check')
  for rn,box in P['evaluation']['ROIs'].items():
   lo,h=box;patch=t[lo:h,lo:h];spec=abs(np.fft.fft2(patch));key=(f['template'],rn)
   if key in spectra:near(spec,spectra[key],'per_frequency_phase_shift_invariance',1e-10,1e-12)
   else:spectra[key]=spec
   # Explicit expected per-mode magnitude, including all zero bins, in both ROIs.
   n=len(patch);expected=np.zeros((n,n));expected[0,0]=.5*n*n
   for ky,kx in P['low_modes']+modes:
    ij=(ky*n//64,kx*n//64);expected[ij[0]%n,ij[1]%n]+=P['amplitude_per_mode']*n*n/2;expected[-ij[0]%n,-ij[1]%n]+=P['amplitude_per_mode']*n*n/2
   near(spec,expected,'per_frequency_expected_magnitude',1e-10,1e-12)
   near(np.var(patch),6*P['amplitude_per_mode']**2/2,'ROIenergy',1e-12,0)
 obs={};eta0={};sham={}
 for c in cases:
  YL,YH=lowobs[c['family']];effective=1. if c['template']=='control_low' else c['eta'];near(effective,c['effective_eta'],'effective eta',0,0);y=(YL+effective*YH).astype(np.float32);stored=np.load(R/'observations'/(c['observation']+'.npy'))
  near(y,stored,'observation_float32',0,0);h=hashlib.sha256(np.ascontiguousarray(stored).tobytes()).hexdigest();ck(h==c['observation_array_sha256'],'observation arrayhash');ck(stored.min()>0 and stored.max()<1,'observation noclamp');obs[c['observation']]=stored
  near(stored.mean(dtype=float),c['input_mean'],'LRmean');near(stored.var(dtype=float),c['input_variance'],'LRvariance');near(np.mean((stored-np.roll(stored,1,0))**2+(stored-np.roll(stored,1,1))**2),c['input_periodic_gradient'],'LRgradient');near(np.mean(YH**2),c['high_path_energy'],'LRhighenergy')
  if c['template']=='control_low':
   if c['family'] in sham:ck(c['observation']==sham[c['family']],'sham shared')
   else:sham[c['family']]=c['observation']
  elif c['eta']==0:
   if c['shift'] in eta0:ck(c['observation']==eta0[c['shift']],'eta0 shared')
   else:eta0[c['shift']]=c['observation']
 ck(len(obs)==148==frozen['unique_observations'],'unique input count');expected_pred={f'predictions/{oid}__{m}.npz' for oid in obs for m in P['models']};ck(set(cohort)==expected_pred,'allpredictioncorrespondence')
 journal=rd(R/'inference_journal.json');ck(len(journal)==296,'journal count');ck({(r['observation'],r['model']) for r in journal}=={(o,m) for o in obs for m in P['models']},'journal pairs');ck(pf['queries']==296*64*64*2==journal[-1]['queries_cumulative'],'query count')
 saved=rd(R/'rows.json');savedmap={(r['id'],r['model'],r['roi'],r['band']):r for r in saved};ck(len(saved)==len(savedmap)==3840,'complete3840 rows');cached={};rows=[]
 for c in cases:
  for model in P['models']:
   pk=(c['observation'],model)
   if pk not in cached:
    z=np.load(R/'predictions'/(pk[0]+'__'+pk[1]+'.npz'));ck(set(z.files)=={'point','native'},'predictionkeys')
    for v in z.values():ck(v.shape==(64,64,3) and np.isfinite(v).all(),'prediction finite shape')
    cached[pk]={rn:(Fcrop(z['point'],box),Fcrop(z['native'],box)) for rn,box in P['evaluation']['ROIs'].items()}
   for rn,box in P['evaluation']['ROIs'].items():
    T=Fcrop(truth[c['family']],box);F,C=cached[pk][rn]
    for b in range(4):
     r=dict(**c,model=model,roi=rn,band=b,**calc(T,F,C,b));rec(clean(r),savedmap[(c['id'],model,rn,b)],'row');rows.append(r)
 lookup={(r['family'],r['model'],r['roi'],r['band'],r['eta']):r for r in rows};pairs=[]
 for z in rows:
  if z['eta']!=0:continue
  base=(z['family'],z['model'],z['roi'],z['band']);half=lookup[base+(.5,)];one=lookup[base+(1,)];a=one['alignment']-z['alignment'];b=one['change_energy']-z['change_energy'];d=one['gain']-z['gain'];near(d,2*a-b,'aliasdecomposition',1e-12,0)
  pairs.append(dict(family=z['family'],template=z['template'],phase=z['phase'],shift=z['shift'],model=z['model'],roi=z['roi'],band=z['band'],gain0=z['gain'],gainhalf=half['gain'],gain1=one['gain'],delta_alias=d,delta_alignment=a,delta_energy=b,sign_flip=(z['gain']<-E and one['gain']>E) or (z['gain']>E and one['gain']<-E),category0=z['category'],category1=one['category']))
 rec(clean(pairs),rd(R/'alias_pairs.json'),'aliaspairs');groups={};effects={};phases=[]
 for m in P['models']:
  for rn in P['evaluation']['ROIs']:
   for template in sorted(P['templates']):
    for b in range(4):
     subset=[r for r in rows if (r['model'],r['roi'],r['template'],r['band'])==(m,rn,template,b)];pr=[r for r in pairs if (r['model'],r['roi'],r['template'],r['band'])==(m,rn,template,b)]
     for eta in P['etas']:groups[f'{m}/{rn}/{template}/b{b}/e{eta:g}']=aggregate([r for r in subset if r['eta']==eta])
     ds=[r['delta_alias'] for r in pr];effects[f'{m}/{rn}/{template}/b{b}']=dict(pairs=len(pr),sign_flips=sum(r['sign_flip'] for r in pr),mean_alias_delta=np.mean(ds),min_alias_delta=min(ds),max_alias_delta=max(ds),positive_alias_delta=sum(d>E for d in ds),negative_alias_delta=sum(d<-E for d in ds))
     for shift in range(4):
      for eta in P['etas']:
       gs=[r['gain'] for r in subset if r['shift']==shift and r['eta']==eta];phases.append(dict(model=m,roi=rn,template=template,band=b,shift=shift,eta=eta,phase_gains=gs,range_gain=max(gs)-min(gs),both_signs=min(gs)<-E and max(gs)>E))
 rec(clean(phases),rd(R/'phase_effects.json'),'phaseeffects');pm={(r['template'],r['phase'],r['shift'],r['model'],r['roi'],r['band']):r for r in pairs};inter=[]
 for r in pairs:
  if r['phase']==0:continue
  q=pm[(r['template'],0,r['shift'],r['model'],r['roi'],r['band'])];inter.append(dict(template=r['template'],phase=r['phase'],shift=r['shift'],model=r['model'],roi=r['roi'],band=r['band'],alias_by_phase=r['delta_alias']-q['delta_alias'],alignment_interaction=r['delta_alignment']-q['delta_alignment'],energy_interaction=r['delta_energy']-q['delta_energy']))
 rec(clean(inter),rd(R/'interactions.json'),'interactions');rec(clean(dict(groups=groups,alias_effects=effects,rows=3840,conditions=240,prediction_files=296,scope=P['purpose'],SR_training=0,GPU=0)),rd(R/'summary.json'),'summary')
 for r in pairs:
  if r['template']=='control_low':near([r['delta_alias'],r['delta_alignment'],r['delta_energy']],np.zeros(3),'sham outcome',1e-14,0)
 return {'sourceSHA':sha(H/'run.py'),'protocolSHA':sha(H/'protocol.json'),'actual_counts':{'families':len(fam),'conditions':len(cases),'unique_observations':len(obs),'prediction_files':len(cohort),'rows':len(rows),'alias_pairs':len(pairs),'phase_rows':len(phases),'interactions':len(inter)},'resources':res,'root_kernel_audit_note':'Independent NumPy Keys kernel comparison to M is supplied separately by root; this audit checks constant preservation, circular shifts and every M*x*M.T input.'}

if __name__=='__main__':
 out={}
 try:out=audit()
 except Exception as e:FAIL.append({'exception':repr(e),'traceback':traceback.format_exc()})
 report={'status':'passed' if not FAIL else 'failed','checks':N,'failures':FAIL,'max_absolute_errors':ERR,'wall_seconds':time.monotonic()-START,'CPU_seconds':time.process_time()-CPU,'CPU_threads':1,'nice':os.getpriority(os.PRIO_PROCESS,0),'audit_sourceSHA':sha(Path(__file__)),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limitations':['No model reinference or independent network implementation; validates archived results and exact synthetic construction.','Finite related synthetic phases/modes/shifts, no population CI or natural-image generalization.','Eta toggles full high-frequency observation path and jointly changes LR statistics. It does not isolate a unique neural alias mechanism.','Eta0 different high-phase targets share observations/predictions; different scores can reflect unobserved target ambiguity.'],**out}
 target=H/'audit.json';assert not target.exists(),'preserve prior audit'
 target.write_text(json.dumps(clean(report),ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':N,'failures':FAIL[:2],'wall_seconds':report['wall_seconds']}))
