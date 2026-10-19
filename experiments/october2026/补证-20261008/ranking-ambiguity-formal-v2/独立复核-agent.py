"""Independent audit: never imports experiment.py and never changes frozen records.
Only numpy and Python standard library are needed. Chi-square CDF is evaluated
via the finite gamma series for this cohort's even degrees of freedom.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, math, hashlib, itertools, collections, sys
from pathlib import Path
import numpy as np

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
def read(name): return json.loads((ROOT/name).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
P=read('protocol.json'); meta=read('case_metadata.json'); sel=read('selections.json')
truth=read('evaluation_truth.json'); scores=read('scores.json'); exact=read('exact_diagnostics.json')
frozen=read('frozen.json'); cohort=read('selection_cohort.json'); summary=read('summary.json')
out={'audit_kind':'saved observations and results independently recomputed; no new experiment or training',
     'experiment_module_imported':False,'gpu_used':False}
byid={r['id']:r for r in sel}; sid={r['id']:r for r in scores}; ids={f'case{i:04d}' for i in range(1920)}
tuples=collections.Counter((tuple(r['phases']),r['noise_variance'],truth[r['id']]['name'],r['N'],r['repeat']) for r in meta)
expected=set(itertools.product(map(tuple,P['phase_sets']),P['noise_variance'],P['truth_spectra'],P['observational_replicates_N'],range(P['mc_repeats'])))
out['cohort']={'counts':dict(metadata=len(meta),selections=len(sel),scores=len(scores),truths=len(truth)),
 'all_ids_complete_unique': all(set(x)==ids and len(x)==1920 for x in [[r['id'] for r in meta],[r['id'] for r in sel],[r['id'] for r in scores],list(truth)]),
 'full_factorial_complete_once':set(tuples)==expected and all(v==1 for v in tuples.values()),
 'selection_sha256':sha(ROOT/'selections.json'),'selection_sha_matches':sha(ROOT/'selections.json')==cohort['selections_sha256'],
 'protocol_sha_matches':sha(ROOT/'protocol.json')==frozen['protocol_sha256']}
version=ROOT.parent/'ranking-ambiguity-v1/versions/formal-v2'
out['cohort']['version_protocol_identical']=(version/'protocol.json').read_bytes()==(ROOT/'protocol.json').read_bytes()
out['cohort']['source_sha_matches']=sha(version/'experiment.py')==frozen['source_sha256']
out['cohort']['journal_matches']={}
for filename,records in [('selection_journal.jsonl',sel),('score_journal.jsonl',scores)]:
 lines=[json.loads(s) for s in (ROOT/filename).read_text().splitlines()]
 out['cohort']['journal_matches'][filename]=lines==records

t=np.arange(8); basis=[]
for k in (1,2,3):
 F=np.column_stack((np.cos(np.pi*k*t/4),np.sin(np.pi*k*t/4)))*.5
 basis.append(F@F.T)
f=np.cos(np.pi*t)/np.sqrt(8);basis.append(np.outer(f,f));Q=np.array(basis)
mult=np.array([2.,2.,2.,1.]); arms=P['candidate_spectra']
def spectral(name):
 p=np.array([k**(-3. if name=='power3' else -2.) for k in range(1,5)])
 if name=='power2_swap_k1_k3':p=p[[2,1,0,3]]
 if name=='middle_spike':p=np.array([.1,3.,.1,.8])
 return p*8/(mult@p)
def covariance(p):return sum(p[k]*Q[k] for k in range(4))
def setup(phases,nu,W=None):
 A=np.eye(8)[[q for h in phases for q in (h,h+4)]];m=len(A)
 dirs=list(np.eye(m))
 for i,j in itertools.combinations(range(m),2):dirs.extend([np.eye(m)[i]+np.eye(m)[j],np.eye(m)[i]-np.eye(m)[j]])
 D=np.array(dirs);M=np.array([[d@A@q@A.T@d for q in Q] for d in D]);W=np.eye(8) if W is None else W
 coeff=[];Bs=[];Ss=[]
 for name in arms:
  C=covariance(spectral(name));S=A@C@A.T+nu*np.eye(m);B=C@A.T@np.linalg.inv(S);E=W@(np.eye(8)-B@A)
  coeff.append((np.array([np.trace(E@q@E.T) for q in Q]),nu*np.sum((W@B)**2)));Bs.append(B);Ss.append(S)
 return A,D,M,coeff,Ss
def chi_cdf(x,n):
 z=x/2;m=n//2
 if z<=0:return 0.
 terms=[j*math.log(z)-math.lgamma(j+1) for j in range(m)]
 mx=max(terms);log_sf=-z+mx+math.log(sum(math.exp(a-mx) for a in terms))
 return -math.expm1(min(log_sf,0.))
quantiles={}
def quantile(prob,n):
 key=(prob,n)
 if key not in quantiles:
  lo=0.;hi=float(n)*4
  while chi_cdf(hi,n)<prob:hi*=2
  for _ in range(90):
   mid=(lo+hi)/2
   if chi_cdf(mid,n)<prob:lo=mid
   else:hi=mid
  quantiles[key]=(lo+hi)/2
 return quantiles[key]

metrics={k:[] for k in ['risk_error','lower_error','upper_error','eb_score_error','truth_spectrum_error']}
coverage=[];false=[];cert=[];group=collections.defaultdict(list);tie_rows=[];eb_mismatch=[]
diagnostics={s:{'witness_count':0,'pair_count':0,'max_primal_residual':0.,'max_objective_record_error':0.,'max_recomputed_expanded_gap':0.,'max_stored_gap_error':0.,'max_recorded_residual_error':0.} for s in ['selection','exact_full','exact_projected']}
def witness_audit(interval,M,lo,hi,coeff,label):
 if interval['status']!='ok':return
 diag=diagnostics[label]
 for pair in interval['pairs']:
  diag['pair_count']+=1;j,k=pair['j'],pair['k'];c=coeff[j][0]-coeff[k][0];offset=coeff[j][1]-coeff[k][1];res=[]
  for e,(w,d) in enumerate(zip(pair['witnesses'],pair['dualities'])):
   p=np.array(w);residual=max(0.,max(M@p-hi),max(lo-M@p),abs(mult@p-8),max(-p),max(p-8));res.append(residual)
   signed=(1 if e==0 else -1)*float(c@p)
   outer=(pair['lower']-offset) if e==0 else -(pair['upper']-offset)
   gap=signed-outer
   diag['witness_count']+=1;diag['max_primal_residual']=max(diag['max_primal_residual'],residual)
   diag['max_objective_record_error']=max(diag['max_objective_record_error'],abs(signed-d['primal']))
   diag['max_recomputed_expanded_gap']=max(diag['max_recomputed_expanded_gap'],gap)
   diag['max_stored_gap_error']=max(diag['max_stored_gap_error'],abs(gap-d['expanded_duality_gap']))
  diag['max_recorded_residual_error']=max(diag['max_recorded_residual_error'],abs(max(res)-pair['max_residual']))

with np.load(ROOT/'observations.npz') as observations:
 out['cohort']['observations_ids_complete']=set(observations.files)==ids
 for r in meta:
  cid=r['id'];s=byid[cid];sc=sid[cid];nu=r['noise_variance'];N=r['N'];A,D,M,coef,Ss=setup(r['phases'],nu);y=observations[cid]
  assert y.shape==(N,len(A))
  variance=np.sum((y@D.T)**2,axis=0)/N;alpha=.05/len(D);noise=nu*np.sum(D*D,axis=1)
  lower=N*variance/quantile(1-alpha/2,N)-noise;upper=N*variance/quantile(alpha/2,N)-noise
  metrics['lower_error'].append(max(abs(lower-np.array(s['direction_lower']))));metrics['upper_error'].append(max(abs(upper-np.array(s['direction_upper']))))
  p=spectral(truth[cid]['name']);metrics['truth_spectrum_error'].append(max(abs(p-np.array(truth[cid]['p']))));risk=np.array([a@p+b for a,b in coef]);metrics['risk_error'].append(max(abs(risk-np.array(sc['risk']))))
  covered=bool(np.all(M@p>=lower-1e-7) and np.all(M@p<=upper+1e-7));coverage.append(covered)
  assert covered==sc['confidence_contains_truth']
  ev=np.array([np.linalg.slogdet(S)[1]+sum(np.sum(y*np.linalg.solve(S,y.T).T,axis=1))/N for S in Ss]);metrics['eb_score_error'].append(max(abs(ev-np.array(s['eb_scores']))))
  choices=np.argsort(ev); gap=float(ev[choices[1]]-ev[choices[0]])
  if gap<1e-12:tie_rows.append({'id':cid,'phases':r['phases'],'truth':truth[cid]['name'],'frozen_choice':s['eb_arm'],'audit_choice':int(choices[0]),'score_gap':gap,'risk':risk.tolist()})
  if int(np.argmin(ev))!=s['eb_arm']:eb_mismatch.append(cid)
  U=np.zeros((3,3))
  if s['intervals']['status']=='ok':
   for pair in s['intervals']['pairs']:U[pair['j'],pair['k']]=pair['upper'];U[pair['k'],pair['j']]=-pair['lower']
   admissible=[i for i in range(3) if max(U[i])<=1e-6]
  else:admissible=[]
  selected=admissible[0] if admissible else None;assert selected==s['certified_arm']
  regret=None if selected is None else float(risk[selected]-min(risk));f=False if regret is None else regret>1e-6+1e-10
  assert f==sc['false_certificate'];cert.append(selected is not None);false.append(f)
  group[(tuple(r['phases']),N)].append((covered,selected is not None,f,float(risk[s['eb_arm']]-min(risk)),float(risk[0]-min(risk))))
  witness_audit(s['intervals'],M,lower,upper,coef,'selection')

exact_groups=[];counterexamples=[]
for key,e in exact.items():
 phases,nu,name=json.loads(key);p=spectral(name);A,D,M,coef,_=setup(phases,nu);v=M@p
 assert int(np.linalg.matrix_rank(M,tol=1e-9))==e['covariance_map_rank']
 witness_audit(e['full_intervals'],M,v-1e-10,v+1e-10,coef,'exact_full')
 W=np.linalg.pinv(A)@A;_,_,_,pc,_=setup(phases,nu,W)
 witness_audit(e['projected_intervals'],M,v-1e-10,v+1e-10,pc,'exact_projected')
 fullwidth=max(x['upper']-x['lower'] for x in e['full_intervals']['pairs']);projwidth=max(x['upper']-x['lower'] for x in e['projected_intervals']['pairs'])
 exact_groups.append({'phases':phases,'nu':nu,'truth':name,'rank':e['covariance_map_rank'],'max_full_width':fullwidth,'max_projected_width':projwidth})
 if phases==[0,0] and nu==.01 and name=='power2':
  pair=next(x for x in e['full_intervals']['pairs'] if (x['j'],x['k'])==(0,2));w0,w1=map(np.array,pair['witnesses'])
  counterexamples.append({'key':key,'pair':[0,2],'interval':[pair['lower'],pair['upper']],'witnesses':pair['witnesses'],'observation_covariance_difference':float(np.max(abs(A@(covariance(w0)-covariance(w1))@A.T))),'witness_risks':[[float(a@w+b) for a,b in coef] for w in [w0,w1]]})

out['max_recompute_errors']={k:max(v) for k,v in metrics.items()};out['coverage_count']=sum(coverage);out['coverage_rate']=float(np.mean(coverage));out['certified_count']=sum(cert);out['false_certificate_count']=sum(false)
out['groups']={str(k):{'cases':len(v),'coverage':float(np.mean([r[0] for r in v])),'certified':sum(r[1] for r in v),'false':sum(r[2] for r in v),'frozen_EB_regret_mean':float(np.mean([r[3] for r in v])),'static_regret_mean':float(np.mean([r[4] for r in v]))} for k,v in group.items()}
out['EB']={'near_minimum_ties_1e-12':len(tie_rows),'recomputed_argmin_disagreements':len(eb_mismatch),'disagreements_ids':eb_mismatch,'tie_phase_counts':dict(collections.Counter(str(r['phases']) for r in tie_rows)),'examples':tie_rows[:8], 'note':'Frozen EB selections kept. Recomputed floating ties may choose different aliases; no retrospective tie-break replacement.'}
out['EB']['indistinguishable_aliases_by_phase']={}
out['phase_aggregates']={}
for phases in P['phase_sets']:
 ss=[r for r in sel if r['phases']==phases];sc=[r for r in scores if r['phases']==phases]
 delta=np.array([r['eb_scores'][0]-r['eb_scores'][2] for r in ss])
 out['EB']['indistinguishable_aliases_by_phase'][str(phases)]={'cases':len(ss),'candidate0_2_max_score_difference':float(max(abs(delta))),'candidate0_2_exact_score_equal':int(sum(delta==0)),'frozen_choices':dict(collections.Counter(r['eb_arm'] for r in ss))}
 out['phase_aggregates'][str(phases)]={'cases':len(ss),'certified':sum(r['certified_arm'] is not None for r in ss),'frozen_EB_regret_mean':float(np.mean([r['eb_regret'] for r in sc])),'static_regret_mean':float(np.mean([r['static_power2_regret'] for r in sc])),'observations_per_realization':2*len(phases),'comparison_cost_note':'Only [0,0] and [0,1] have equal measurement counts. N multiplies measurement count.'}
ok_cases=sum(r['intervals']['status']=='ok' for r in sel);empty_cases=sum(r['intervals']['status']=='empty' for r in sel)
out['cost_record_checks']={'selection_statuses':dict(collections.Counter(r['intervals']['status'] for r in sel)),'selection_lp_calls_reconstructed':ok_cases*6+empty_cases,'exact_lp_calls_reconstructed':len(exact)*12,'selection_seconds_sum_recomputed':sum(r['selection_seconds'] for r in sel),'note':'Wall times are original records, not independent timings of original run; LP call reconstruction assumes each empty case failed its first endpoint as implemented.'}
out['exact_rank_counts']=dict(collections.Counter(r['rank'] for r in exact_groups))
out['witness_diagnostics']=diagnostics;out['exact_groups']=exact_groups;out['counterexamples']=counterexamples
out['limits']=['No dual multipliers saved: expanded dual bound values and gap identities checked, but weak-duality validity cannot be fully reconstructed from saved JSON alone.','Raw observations and population covariance family synthetic; no natural-image or new-method reliability claim.','0 false among 197 certified is empirical only; one-sided binomial 95% upper bound approximately 1.51%, with further dependence/heterogeneity caveats.','Certification uses original stored outward bounds. Independently recomputed primitive constraints and objective consistency do not constitute interval arithmetic proof.']
out['all_primal_witnesses_pass_1e-7']=all(v['max_primal_residual']<=1e-7 for v in diagnostics.values())
out['all_recomputed_expanded_gaps_pass_1e-6']=all(v['max_recomputed_expanded_gap']<=1e-6 for v in diagnostics.values())
out['summary_counts_match']=sum(cert)==summary['certified'] and sum(false)==summary['false_certificates'] and abs(np.mean(coverage)-summary['confidence_set_coverage'])<1e-14
(ROOT/'独立复核-agent.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['groups','exact_groups']},ensure_ascii=False,indent=2,default=lambda x:x.item()))
