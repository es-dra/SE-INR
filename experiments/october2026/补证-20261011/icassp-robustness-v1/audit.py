"""Independent formulas for complete saved-output sensitivity verification."""
import pathlib,json,hashlib,time,resource,datetime
import numpy as np
R=pathlib.Path(__file__).resolve().parent; B=R.parents[1]
read=lambda p:json.loads(p.read_text())
hashof=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
start=time.perf_counter();cpu=time.process_time();D=read(R/'results.json');F=read(R/'freeze.json');P=read(R/'protocol.json')
assert datetime.datetime.now(datetime.timezone.utc).timestamp()<P['absolute_deadline_unix']
checks=0; maxerr={'L1':0.,'G1':0.,'negative':0.,'bootstrap':0.}
def check(a,b,k,tol=1e-12):
 global checks
 a=float(a);b=float(b);error=abs(a-b);maxerr[k]=max(maxerr[k],error);checks+=1
 assert error<tol,(k,a,b,error)
assert hashof(R/'analyze.py')==F['source_sha256'] and hashof(R/'protocol.json')==F['protocol_sha256'];checks+=2
for item in F['inputs']:
 assert hashof(pathlib.Path(item['path']))==item['sha256'];checks+=1

# Direct per-image arithmetic, independently formed ring memberships.
cache={ds:read(B/f'服务器结果-20261007/claude-20261005-L12/out/l1_inr_{ds}.json') for ds in ['Set14','BSD100','Urban100']}
for row in D['l1']['rows']:
 d=cache[row['dataset']];mn=row['model']+('' if row['seed']==1 else '@s'+str(row['seed']))
 bins={'all':list(range(7)),'below_nyquist':list(range(3)),'above_nyquist':list(range(3,7))}[row['domain']]
 for axis in ['u','nu']:
  differences=[]
  for j in bins:
   recoveries=[]
   for scale in row['scales']:
    rate=2*scale/240 if axis=='u' else 6/240
    inds=[k for k in range(1,121) if k*rate>=(j+1)*.25-1e-12 and (k*rate<(j+2)*.25-1e-12 if j<6 else k*rate<=(j+2)*.25+1e-12)]
    pairs=[(sum(d['S'][name][k] for k in inds),sum(d['E'][mn][str(scale)][name][k] for k in inds)) for name in sorted(d['S'])]
    recoveries.append(1-sum(e for s,e in pairs)/sum(s for s,e in pairs) if row['pooling']=='energy_pool' else sum(1-e/s for s,e in pairs)/len(pairs))
   differences.append(max(recoveries)-min(recoveries))
  check(np.median(differences),row['D_'+axis],'L1')
 assert row['normalisation_improves']==(row['D_u']<row['D_nu']);checks+=1
for summary in D['l1']['summary']:
 rr=[r for r in D['l1']['rows'] if r['pooling']==summary['pooling'] and r['domain']==summary['domain'] and r['scales']==summary['scales']]
 assert len(rr)==summary['groups'] and sum(r['D_u']<r['D_nu'] for r in rr)==summary['improved'];checks+=2

# Mid-ranks formed by counting strictly smaller/equal values rather than sorting.
def midrank(v):return np.array([sum(x<z for x in v)+(sum(x==z for x in v)-1)/2 for z in v])
for group in D['g1']['groups']:
 model=group['model'];abl={'lte':'lte-nocellphase','sc-inr':'sc-inr-nosinc'}[model];loss=[];bands=[];ordering=0
 for scale in [2,3,4]:
  raw=read(B/f'补证-20261008/g1/out/seed{group["seed"]}/f1s_x{scale}.json')
  for ds in ['Set14','BSD100','Urban100']:
   cur=[]
   for band in [1,2,3]:
    value=raw['pooled'][f'{ds}|{model}|b{band}']-raw['pooled'][f'{ds}|{abl}|b{band}'] if group['pooling']=='saved_energy_pool' else sum(r[f'{model}_b{band}']-r[f'{abl}_b{band}'] for r in raw['per_image'] if r['set']==ds)/sum(r['set']==ds for r in raw['per_image'])
    cur.append(value);loss.append(value);bands.append(band)
   ordering+=cur[2]>cur[1]
 x=midrank(bands);y=midrank(loss);x-=x.mean();y-=y.mean()
 check(x@y/np.sqrt((x@x)*(y@y)),group['rho'],'G1');assert ordering==group['ordering_pass'];checks+=1

# Parseval reconstruction uses separate transforms of signal and predictions;
# raw gain is recomputed via the exact error/change inner-product identity.
N=B/'补证-20261011/negative-collected/codex-20261011-negative-formal-v3';forecasts={(r['case'],r['model']):r for r in read(N/'forecasts.json') if r['band']==0}
cutoffs={'original':(.125,.25),'narrower':(.1,.2),'wider':(1/6,1/3)};cache={};computed={}
for row in D['negative']['rows']:
 key=(row['case'],row['model'],row['shave'])
 if key not in cache:
  z=np.load(N/'predictions'/f'{row["case"]}__{row["model"]}.npz');t=np.load(N/'truth'/f'{row["case"]}.npy').astype('float64').transpose(1,2,0)
  f0=z['point'].astype('float64').reshape(128,128,3);fc=z['native'].astype('float64').reshape(128,128,3)
  sh=row['shave'];sl=slice(sh,128-sh);t=t[sl,sl];f0=f0[sl,sl];fc=fc[sl,sl];h=len(t)
  delta=fc-f0;error=f0-t;g=-np.mean(2*error*delta+delta*delta)
  # Split transforms differ computationally from fft(error) in analyze.py.
  a=np.fft.rfft2(f0,axes=(0,1))-np.fft.rfft2(t,axes=(0,1));b=np.fft.rfft2(fc,axes=(0,1))-np.fft.rfft2(t,axes=(0,1))
  fy=np.abs(np.fft.fftfreq(h));fx=np.fft.rfftfreq(h);radius=np.maximum(fy[:,None],fx[None,:]);w=np.ones(len(fx));w[1:-1]=2
  cache[key]=(h,g,float(np.mean(error**2)),float(np.mean((fc-t)**2)),radius,(np.abs(a)**2)*(w[None,:,None]),(np.abs(b)**2)*(w[None,:,None]))
 h,g,m0,m1,radius,a,b=cache[key]
 if row['band']==0:
  gain=g;p0=m0;p1=m1
  for policy,value in forecasts[row['case'],row['model']]['forecasts'].items():check(p1 if value>0 else p0,row['policy_mse'][policy],'negative')
 else:
  c1,c2=cutoffs[row['bands']];lo,hi=[(0,c1),(c1,c2),(c2,.5001)][row['band']-1];mask=(radius>=lo)&(radius<hi)&(radius!=0)
  p0=a[mask].sum()/(3*h**4);p1=b[mask].sum()/(3*h**4);gain=p0-p1
 check(gain,row['gain'],'negative');check(p0,row['point_mse'],'negative');check(p1,row['native_mse'],'negative');computed[key,row['bands'],row['band']]=float(gain)
for summary in D['negative']['summary']:
 rr=[r for r in D['negative']['rows'] if r['model']==summary['model'] and r['shave']==summary['shave'] and r['bands']==summary['bands']]
 vals=[computed[(r['case'],r['model'],r['shave']),r['bands'],r['band']] for r in rr]
 assert sum(v< -1e-8 for v in vals)==summary['harmful'] and len(vals)==summary['rows'];checks+=2
 check(np.mean(vals),summary['mean_gain'],'negative')
indices=np.random.default_rng(20261011).integers(0,12,size=(10000,12))
for summary in D['negative']['fixed_policy']:
 model=summary['model'];shave=summary['shave'];policy=summary['policy'];vec=[]
 for image in range(825,837):
  effects=[]
  for scale in [2,4]:
   case=f'test_{image:04d}_x{scale}';h,g,m0,m1,_,_,_=cache[case,model,shave]
   effects.append(0 if forecasts[case,model]['forecasts'][policy]>0 else -g)
  vec.append(np.mean(effects))
 vec=np.array(vec);check(vec.mean(),summary['MSE_reduction_vs_native'],'bootstrap')
 interval=np.quantile(np.take(vec,indices).mean(axis=1),[.025,.975])
 for a,b in zip(interval,summary['image_cluster_95_sensitivity']):check(a,b,'bootstrap')
duration=time.perf_counter()-start
assert duration+D['cost']['wall_seconds']<P['cumulative_analysis_wall_budget_seconds'] and datetime.datetime.now(datetime.timezone.utc).timestamp()<P['absolute_deadline_unix']
out={'status':'complete independent saved-output sensitivity audit passed','checks':checks,'max_absolute_error':maxerr,'audit_wall_seconds':duration,'audit_cpu_seconds':time.process_time()-cpu,'total_analysis_plus_audit_wall_seconds':duration+D['cost']['wall_seconds'],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_sha256':hashof(R/'audit.py'),'analyze_sha256':hashof(R/'analyze.py'),'results_sha256':hashof(R/'results.json'),'protocol_sha256':hashof(R/'protocol.json'),'CPU_threads':1,'GPU':0,'training':0,'inference':0,'limits':'No model reimplementation, new images or forecasts. E3 threshold table is descriptive saved-record arithmetic, not independently re-run linear inference. Input and source hashes verified.'}
(R/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
