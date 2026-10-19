"""Bounded independent standard-library audit; no model import, inference or source mutation."""
import json, hashlib, random, statistics, math, time, resource
from pathlib import Path
start=time.monotonic()
BASE=Path(__file__).resolve().parents[2]
OLD=BASE/'服务器结果-20261007'
L=OLD/'claude-20261005-L12'
F=OLD/'claude-20261005-f1s'
inputs={}
def read(p):
 inputs[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text())
def source(p): inputs[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
UE=[.25,.5,.75,1,1.25,1.5,1.75,2]
SCALES=[2,3,4]; NETS=['liif','lte','sc-inr']
def ids(s,i,absolute=False):
 lo,hi=UE[i:i+2]
 return [k for k in range(1,121) if (k/40 if absolute else 2*s*k/240)>=lo-1e-12 and ((k/40 if absolute else 2*s*k/240)<hi-1e-12 if i<6 else (k/40 if absolute else 2*s*k/240)<=hi+1e-12)]
def pooled(d,m,s,i,imgs,absolute=False):
 kk=ids(s,i,absolute)
 sig=sum(sum(d['S'][n][k] for k in kk) for n in imgs)
 err=sum(sum(d['E'][m][str(s)][n][k] for k in kk) for n in imgs)
 return 1-err/sig
# Precompute each image/bin sum for 200 diagnostic draws, which preserves pool-then-divide.
def per_image(d,m):
 return {(s,i,n):(sum(d['S'][n][k] for k in ids(s,i)),sum(d['E'][m][str(s)][n][k] for k in ids(s,i))) for s in SCALES for i in range(7) for n in d['S']}
def cached_D(c,imgs):
 spread=[]
 for i in range(7):
  rr=[1-sum(c[s,i,n][1] for n in imgs)/sum(c[s,i,n][0] for n in imgs) for s in SCALES]
  spread.append(max(rr)-min(rr))
 return statistics.median(spread)
def percent(v,q):
 v=sorted(v); ix=(len(v)-1)*q; k=int(ix); return v[k]+(ix-k)*(v[min(k+1,len(v)-1)]-v[k])
D={ds:read(L/f'out/l1_inr_{ds}.json') for ds in ['Set14','BSD100','Urban100']}
old=read(L/'l12_result.json')
L1={}
for ds,d in D.items():
 for m in NETS:
  imgs=sorted(d['S']); u=[pooled(d,m,s,i,imgs) for s in SCALES for i in range(7)]
  du=statistics.median(max(u[j*7+i] for j in range(3))-min(u[j*7+i] for j in range(3)) for i in range(7))
  v=[pooled(d,m,s,i,imgs,True) for s in SCALES for i in range(7)]
  dv=statistics.median(max(v[j*7+i] for j in range(3))-min(v[j*7+i] for j in range(3)) for i in range(7))
  L1[f'{ds}|{m}']={'n_images':len(imgs),'D_u':du,'D_nu':dv,'normalization_improves':dv>du,'collapse_threshold_pass':du<=.05,'delta_from_original':max(abs(du-old['L1_stats'][f'{ds}|{m}']['Din_med']),abs(dv-old['L1_stats'][f'{ds}|{m}']['Dnu_med']))}
random_gen=random.Random(0); bd=sorted(D['BSD100']['S']); draws=[random_gen.sample(bd,14) for _ in range(200)]
DIAG={}
for m in NETS+['lmmse_generic']:
 s14=per_image(D['Set14'],m); bc=per_image(D['BSD100'],m); names=sorted(D['Set14']['S']); obs=cached_D(s14,names)
 loo={n:cached_D(s14,[x for x in names if x!=n]) for n in names}; v=[cached_D(bc,n) for n in draws]
 DIAG[m]={'full_Set14_D':obs,'without_barbara_D':loo['barbara.png'],'leave_one_out_min_image':min(loo,key=loo.get),'all_leave_one_out_D':loo,'BSD_first30_n14_200draws':{'median':statistics.median(v),'p95':percent(v,.95),'max':max(v),'n_ge_Set14':sum(x>=obs for x in v),'n_gt_0_05':sum(x>.05 for x in v)},'draw_values':v}
SC=read(F/'f1s_result_1.json')['1|sc']
LTE=read(F/'f1s_result_1.json')['1|lte']
max_cell=0.; counts={}
f1_set14_images=set()
for s in [2,3,4,8]:
 d=read(F/f'out/seed1/f1s_x{s}.json'); counts[str(s)]={ds:sum(x['set']==ds for x in d['per_image']) for ds in ['Set14','BSD100','Urban100']}
 if s==2: f1_set14_images={x['img'] for x in d['per_image'] if x['set']=='Set14'}
 for c in (x for x in SC if x['s']==s):
  key=f"{c['ds']}|sc-inr|{c['b']}"; full=d['pooled'][key]; loss=full-d['pooled'][f"{c['ds']}|sc-inr-nosinc|{c['b']}"]
  max_cell=max(max_cell,abs(loss-c['L']),abs(full-c['Rnet']))
def ranks(x):
 ix=sorted(range(len(x)),key=lambda i:x[i]); r=[0.]*len(x); a=0
 while a<len(ix):
  b=a+1
  while b<len(ix) and x[ix[b]]==x[ix[a]]: b+=1
  for j in range(a,b):r[ix[j]]=(a+b-1)/2+1
  a=b
 return r
def corr(x,y):
 x=ranks(x); y=ranks(y); xm=statistics.mean(x); ym=statistics.mean(y)
 return sum((a-xm)*(b-ym) for a,b in zip(x,y))/math.sqrt(sum((a-xm)**2 for a in x)*sum((b-ym)**2 for b in y))
def g(r): return r*(1-r)**2
C=[x for x in SC if x['s'] in SCALES]; T=[x for x in LTE if x['s'] in SCALES]
F1={'definition':'L = pooled R(sc-inr) - pooled R(sc-inr-nosinc), seed1 epoch-last, correct encoder family; not sc-inr-eq','source_imaging':'center256 gray RGBmean, torch bicubic antialias, clamped network predictions; no FFT Hann window in f1s_h6c.py','n_images':counts,'saved_cells_vs_pooled_max_error':max_cell,'H_F1':corr([g(c['Rlin']) for c in C],[c['L'] for c in C]),'H_F2':corr([g(c['Rnet']) for c in C],[c['L'] for c in C]),'cross_arch_H_F4':corr([c['L'] for c in C],[c['L'] for c in T]),'A_Rlin_ge0_8_meanL':statistics.mean(c['L'] for c in C if c['Rlin']>=.8),'x3b2_meanL':statistics.mean(c['L'] for c in C if c['s']==3 and c['b']=='b2'),'overall_band_meanL':{b:statistics.mean(c['L'] for c in C if c['b']==b) for b in ['b1','b2','b3']},'scale_band_meanL':{f'x{s}|{b}':statistics.mean(c['L'] for c in C if c['s']==s and c['b']==b) for s in SCALES for b in ['b1','b2','b3']},'negative_cells':[c for c in C if c['L']<0]}
for p in [L/'common.py',L/'ana_l12.py',L/'l1.py',L/'diag/diag_set14.py',L/'diag/diag_set14b.py',L/'diag/diag_set14_A2-4.log',L/'diag/diag_set14_A3-5.log',L/'预注册-L1L2-20261005.md',F/'ana_f1s.py',F/'f1s_h6c.py',F/'f1s_analysis_seed1.log']:source(p)
result={'status':'independent saved-array recomputation complete; no GPU/FF/checkpoint rerun','input_sha256':inputs,'L1_definition':{'R':'1 - sum_image,sum_radius E / sum_image,sum_radius S; exclude DC; RGBmean; Chebyshev radius nu=k/240','u':'u=2*s*nu; LR Nyquist u=1','bins':UE,'D':'median across 7 bins of max-min R over x2/x3/x4; absolute nu bins frozen at u-bins/6','not_per_image_mean_R':True,'n_coordinate_improves':sum(x['normalization_improves'] for x in L1.values()),'n_collapse_pass':sum(x['collapse_threshold_pass'] for x in L1.values())},'L1_recomputed':L1,'Set14_posthoc_diagnostics':DIAG,'F1_SC_correct_pair':F1,'limitations':['Saved frequency arrays/pooled summaries are recomputed; original RGB/checkpoint predictions not rerun.','BSD100 random-subset diagnostic uses first30 frozen L1 images, not whole dataset; 200 dependent draws and posthoc barbara deletion are descriptive.','Deleting barbara does not repair failed preregistered 8/9 collapse criterion (original pass 6/9).','F1 corrected pair is explicit in inference source and JSON labels; binary checkpoint identity not independently reloaded here.','Negative L cells and negative aggregate b2/A values retained; cannot state every ablation worsens.','JPEG/blockiness diagnosis was not recomputed from original HR PNGs; supplied logs show heuristic only, no causal explanation.'], 'resources':{'wall_seconds':time.monotonic()-start,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'cpu_budget_seconds':120,'gpu_calls':0}}
assert max(x['delta_from_original'] for x in L1.values())<1e-12
assert max_cell<1e-12
result['F1_SC_correct_pair']['Set14_excluded_vs_L1']=sorted(set(D['Set14']['S'])-f1_set14_images)
result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out=Path(__file__).with_name('evidence_audit.json'); out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'L1':L1,'diag':{k:{z:v for z,v in q.items() if z not in ['all_leave_one_out_D','draw_values']} for k,q in DIAG.items()},'F1':F1,'resources':result['resources']},ensure_ascii=False,indent=2))
