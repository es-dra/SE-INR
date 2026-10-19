"""Independent scalar-loop audit of frozen saved ring-energy diagnostics.
Does not import the analysis implementation or run model/preprocessing code.
"""
import json, pathlib, hashlib, statistics, itertools, time, resource, datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
START=time.perf_counter(); CPU=time.process_time(); checks=0; maxdiff=0.; maxdimensionless=0.; maxrelativeenergy=0.; failures=[]
def read(p): return json.loads(pathlib.Path(p).read_text())
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ck(a,b,label):
 global checks,maxdiff,maxdimensionless,maxrelativeenergy
 checks+=1
 if isinstance(a,(int,float,np.floating)) and not isinstance(a,bool):
  diff=abs(float(a)-float(b));maxdiff=max(maxdiff,diff)
  rawenergy=('.energy.' in label or label.endswith('.emin') or label.endswith('.emedian'))
  magnitude=max(abs(float(a)),abs(float(b)))
  tolerance=1e-11+(1e-12*magnitude if rawenergy else 0.)
  if rawenergy: maxrelativeenergy=max(maxrelativeenergy,diff/magnitude if magnitude>0 else 0.)
  else: maxdimensionless=max(maxdimensionless,diff)
  if not np.isfinite(diff) or diff>tolerance: failures.append(dict(label=label,expected=a,observed=b))
 elif a!=b: failures.append(dict(label=label,expected=a,observed=b))
def arr(a,b,label):
 if isinstance(a,(list,tuple)):
  ck(len(a),len(b),label+'.length')
  for k,(x,y) in enumerate(zip(a,b)): arr(x,y,f'{label}.{k}')
 else: ck(a,b,label)
def desc(v):
 return dict(min=min(v),q25=float(np.percentile(v,25)),median=statistics.median(v),q75=float(np.percentile(v,75)),q90=float(np.percentile(v,90)),max=max(v))
def D(c): return statistics.median([max(c[s][b] for s in range(3))-min(c[s][b] for s in range(3)) for b in range(7)])
p=read(ROOT/'protocol.json'); freeze=read(ROOT/'freeze.json'); result=read(ROOT/'results.json'); manifest=read(ROOT/'output_manifest.json')
ck(sha(ROOT/'analyze.py'),freeze['source_sha256'],'source');ck(sha(ROOT/'protocol.json'),freeze['protocol_sha256'],'protocol')
for item in freeze['inputs']: ck(sha(item['path']),item['sha256'],'input:'+item['path'])
for file,key in [('results.json','results_sha256'),('analyze.py','source_sha256'),('protocol.json','protocol_sha256'),('freeze.json','freeze_sha256')]:ck(sha(ROOT/file),manifest[key],'output:'+file)
ck(len(result['groups']),27,'groups');ck(len(result['pairs']),567,'pairs')
pairlookup={(r['dataset'],r['model'],r['seed'],tuple(r['scale_pair']),r['bin']):r for r in result['pairs']}
all_summary=[]
for ds in p['datasets']:
 data=read(pathlib.Path(p['l1_root'])/f'l1_inr_{ds}.json');names=sorted(data['S']);n=len(names); geo=next(r for r in result['weight_geometry'] if r['dataset']==ds)
 ck(names,geo['images'],ds+'.names'); edges=p['u_edges']; scales=p['scales']; rings={}
 for axis in ['u','abs']:
  rings[axis]=[[[k for k in range(1,121) if ((2*s*k/240 if axis=='u' else 6*k/240)>=edges[b]-1e-12) and (((2*s*k/240 if axis=='u' else 6*k/240)<edges[b+1]-1e-12) if b<6 else ((2*s*k/240 if axis=='u' else 6*k/240)<=edges[b+1]+1e-12))] for b in range(7)] for s in scales]
 S=[[[sum(data['S'][im][k] for k in rings['u'][j][b]) for b in range(7)] for im in names] for j in range(3)]
 totals=[[sum(S[j][i][b] for i in range(n)) for b in range(7)] for j in range(3)]
 W=[[[S[j][i][b]/totals[j][b] for b in range(7)] for i in range(n)] for j in range(3)]
 F=[[sum(W[j][i][b] for j in range(3))/3 for b in range(7)] for i in range(n)]
 for j,s in enumerate(scales):
  for b in range(7):
   g=next(r for r in geo['bins'] if r['scale']==s and r['bin']==b); weights=[W[j][i][b] for i in range(n)];en=[S[j][i][b] for i in range(n)];order=sorted(range(n),key=lambda i:-weights[i])
   ck(1/sum(x*x for x in weights),g['effective_images'],ds+'.ess');ck(min(en),g['energy_min'],ds+'.emin');ck(statistics.median(en),g['energy_median'],ds+'.emedian');ck(names[order[0]],g['top1']['image'],ds+'.topimage');ck(weights[order[0]],g['top1']['weight'],ds+'.topweight');ck(sum(weights[i] for i in order[:3]),g['top3_weight'],ds+'.top3');arr(en,g['energy'],ds+'.energy');arr(weights,g['weights'],ds+'.weights')
 boot=np.random.default_rng(p['bootstrap_seed']).integers(0,n,size=(p['bootstrap_replicates'],n));countsets=[]
 for sample in boot:
  counts=[0]*n
  for i in sample:counts[int(i)]+=1
  countsets.append([(i,count) for i,count in enumerate(counts) if count])
 summaries=[]
 for model in p['models']:
  for seed in p['seeds']:
   assert time.perf_counter()-START<115,'independent audit budget';assert time.time()<p['absolute_deadline_unix'],'absolute deadline'
   tag=model if seed==1 else f'{model}@s{seed}';g=next(r for r in result['groups'] if r['dataset']==ds and r['model']==model and r['seed']==seed)
   E=[[[sum(data['E'][tag][str(s)][im][k] for k in rings['u'][j][b]) for b in range(7)] for im in names] for j,s in enumerate(scales)]
   R=[[[1-E[j][i][b]/S[j][i][b] for b in range(7)] for i in range(n)] for j in range(3)]
   pool=[[1-sum(E[j][i][b] for i in range(n))/totals[j][b] for b in range(7)] for j in range(3)]
   eq=[[sum(R[j][i][b] for i in range(n))/n for b in range(7)] for j in range(3)]
   fixed=[[sum(F[i][b]*R[j][i][b] for i in range(n)) for b in range(7)] for j in range(3)]
   cov=[]
   for j in range(3):
    covrow=[]
    for b in range(7):
     sm=totals[j][b]/n;cv=sum((S[j][i][b]-sm)*(R[j][i][b]-eq[j][b]) for i in range(n))/(n*sm);covrow.append(cv);ck(eq[j][b]+cv,pool[j][b],'covariance_identity')
    cov.append(covrow)
   for lab,curve in [('pooled',pool),('equal',eq),('fixed',fixed),('covariance',cov)]:arr(curve,g['curves'][lab],ds+'.'+lab)
   dvs={'original_energy_pool':D(pool),'equal_image':D(eq),'fixed_cross_scale_energy_weights':D(fixed)}
   for k,v in dvs.items():ck(v,g['D'][k],ds+'.D.'+k)
   du=[];da=[];loo=[];improved=0
   for i,im in enumerate(names):
    individual=D([[R[j][i][b] for b in range(7)] for j in range(3)]);absolute=[]
    for j,s in enumerate(scales):
     absolute.append([1-sum(data['E'][tag][str(s)][im][k] for k in rings['abs'][j][b])/sum(data['S'][im][k] for k in rings['abs'][j][b]) for b in range(7)])
    ab=D(absolute);du.append(individual);da.append(ab);improved+=individual<ab;r=g['per_image'][i];ck(im,r['image'],'image');ck(individual,r['D_u'],'imageD');ck(ab,r['D_absolute'],'imageabsD');ck(individual<ab,r['normalization_improves'],'imageimprovement');ck(individual<=.05,r['strict_005'],'imageDthreshold');ck(sum(F[i])/7,r['mean_energy_weight'],'imageweight');ck(max(W[j][i][b] for j in range(3) for b in range(7)),r['max_energy_weight'],'imagemaxweight');arr([[R[j][i][b] for b in range(7)] for j in range(3)],r['R'],'imageR')
    lp=[[1-sum(E[j][ii][b] for ii in range(n) if ii!=i)/sum(S[j][ii][b] for ii in range(n) if ii!=i) for b in range(7)] for j in range(3)];le=[[sum(R[j][ii][b] for ii in range(n) if ii!=i)/(n-1) for b in range(7)] for j in range(3)];ld=D(lp);row={'image_removed':im,'D_original':ld,'D_equal':D(le),'delta_D_original':ld-D(pool)};loo.append(row)
    for k,v in row.items():ck(v,g['leave_one_out'][i][k],'LOO.'+k)
   for k,v in desc(du).items():ck(v,g['per_image_summary'][k],'Ddistribution.'+k)
   for k,v in desc([r['D_original'] for r in loo]).items():ck(v,g['leave_one_out_D_range'][k],'LOOdistribution.'+k)
   ck(improved,g['per_image_normalization_improves'],'individualcount');ck(sum(v<=.05 for v in du),g['per_image_strict_005'],'strictcount')
   for i,r in enumerate(sorted(loo,key=lambda r:abs(r['delta_D_original']),reverse=True)[:3]):
    for k,v in r.items():ck(v,g['largest_deletion_effect'][i][k],'largestLOO.'+k)
   current=[]
   for a,b in itertools.combinations(range(3),2):
    for band in range(7):
     r=pairlookup[ds,model,seed,(scales[a],scales[b]),band];wi=[];mi=[];gi=[]
     for i in range(n):
      wi.append((W[a][i][band]+W[b][i][band])*(R[a][i][band]-R[b][i][band])/2);mi.append((W[a][i][band]-W[b][i][band])*(R[a][i][band]+R[b][i][band])/2);gi.append(R[a][i][band]-R[b][i][band]);ck(names[i],r['image_components'][i]['image'],'componentimage');ck(wi[-1],r['image_components'][i]['within'],'componentwithin');ck(mi[-1],r['image_components'][i]['weight_migration'],'componentmigration')
     actual={'pooled_gap':pool[a][band]-pool[b][band],'equal_gap':eq[a][band]-eq[b][band],'covariance_gap':cov[a][band]-cov[b][band],'within_image_component':sum(wi),'weight_migration_component':sum(mi),'mean_abs_individual_gap':sum(abs(x) for x in gi)/n,'mean_signed_individual_gap':sum(gi)/n,'positive_image_gaps':sum(x>0 for x in gi),'negative_image_gaps':sum(x<0 for x in gi)}
     for k,v in actual.items():ck(v,r[k],'pair.'+k)
     ck(sum(wi)+sum(mi),actual['pooled_gap'],'symmetric_identity');ck(actual['equal_gap']+actual['covariance_gap'],actual['pooled_gap'],'equalcov_identity');current.append(actual)
   for outkey,inkey in [('sum_abs_within','within_image_component'),('sum_abs_migration','weight_migration_component'),('sum_abs_pooled','pooled_gap')]:ck(sum(abs(r[inkey]) for r in current),g['signed_gap_totals'][outkey],outkey)
   bp=[];be=[]
   for cs in countsets:
    pc=[];ec=[]
    for j in range(3):
     pc.append([1-sum(count*E[j][i][b] for i,count in cs)/sum(count*S[j][i][b] for i,count in cs) for b in range(7)])
     ec.append([sum(count*R[j][i][b] for i,count in cs)/n for b in range(7)])
    bp.append(D(pc));be.append(D(ec))
   arr(np.percentile(bp,[2.5,97.5]).tolist(),g['bootstrap']['pool_D_interval'],'bootstrap.pool');arr(np.percentile(be,[2.5,97.5]).tolist(),g['bootstrap']['equal_D_interval'],'bootstrap.equal');arr(np.percentile([x-y for x,y in zip(bp,be)],[2.5,97.5]).tolist(),g['bootstrap']['pool_minus_equal_D_interval'],'bootstrap.diff')
   summaries.append(dict(D=dvs,individual=du,improved=improved,n=n));
 summary=next(r for r in result['summary'] if r['dataset']==ds)
 expected={'groups':9,'original_strict_pass':sum(r['D']['original_energy_pool']<=.05 for r in summaries),'equal_strict_pass':sum(r['D']['equal_image']<=.05 for r in summaries),'fixed_strict_pass':sum(r['D']['fixed_cross_scale_energy_weights']<=.05 for r in summaries),'fixed_D_below_original':sum(r['D']['fixed_cross_scale_energy_weights']<r['D']['original_energy_pool'] for r in summaries)}
 for k,v in expected.items():ck(v,summary[k],ds+'.summary.'+k)
 for key,vals in [('median_image_D',[statistics.median(r['individual']) for r in summaries]),('individual_strict_fraction',[sum(x<=.05 for x in r['individual'])/n for r in summaries]),('individual_normalization_improvement_fraction',[r['improved']/n for r in summaries])]:
  for k,v in desc(vals).items():ck(v,summary[key][k],ds+'.summary.'+key+'.'+k)
 all_summary.append({'dataset':ds,**expected})
cost=dict(wall_seconds=time.perf_counter()-START,cpu_seconds=time.process_time()-CPU,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CPU_threads=1,GPU=0,inference=0,training=0)
ck(cost['wall_seconds']<=120,True,'audit120s');ck(time.time()<p['absolute_deadline_unix'],True,'deadline')
v1=read(ROOT/'audit.json')
audit=dict(prior_audit=dict(passed=v1['passed'],failure_count=len(v1['failures']),source_sha256=sha(ROOT/'audit.py'),results_sha256=sha(ROOT/'audit.json'),reason='Initial audit used absolute 1e-11 tolerance on unnormalised FFT energies around 1e7. All 752 differences are scalar-vs-vector summation rounding; initial failure preserved unchanged.',wall_seconds=v1['cost']['wall_seconds'],cpu_seconds=v1['cost']['cpu_seconds']),tolerances=dict(raw_energy='abs1e-11 + relative1e-12 * max(abs(expected),abs(observed)); zero-safe',dimensionless='absolute1e-11'),max_dimensionless_abs_error=maxdimensionless,max_raw_energy_relative_error=maxrelativeenergy,cumulative_audit_wall_seconds=v1['cost']['wall_seconds']+cost['wall_seconds'],scope='Independent from original three JSON ring arrays with scalar summation loops; no import analyze.py, no image/model preprocessing audit.',passed=not failures,checks=checks,max_abs_error=maxdiff,failures=failures,groups=27,pairs=567,summary=all_summary,bootstrap='All 2000 paired draws independently recomputed with integer multiplicity and scalar sums, all 27 group intervals checked.',hashes={'source_sha256':sha(ROOT/'analyze.py'),'protocol_sha256':sha(ROOT/'protocol.json'),'results_sha256':sha(ROOT/'results.json'),'audit_source_sha256':sha(ROOT/'audit-v2.py')},cost=cost,scientific_limits=p['limits'])
(ROOT/'audit-v2.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({k:audit[k] for k in ['passed','checks','max_abs_error','failures','summary','cost']},ensure_ascii=False,indent=2))
