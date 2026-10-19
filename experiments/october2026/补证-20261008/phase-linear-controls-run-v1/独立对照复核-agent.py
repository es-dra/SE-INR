"""Independent post-run CPU audit. Never imports the experiment or trains.
One full independent LU solve per case plus fixed-RFF primal check.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,hashlib,time,math,itertools,ast
from pathlib import Path
import numpy as np
from scipy.linalg import solve
import torch
torch.set_num_threads(1)
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
BASE=ROOT.parent;CODE=BASE/'phase-linear-controls-v1';INPUT=BASE/'inr-phase-development-run-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
began=time.monotonic();deadline=began+120
def budget():
 if time.monotonic()>deadline:raise TimeoutError('Independent audit wall120 seconds exceeded')
P=read(ROOT/'protocol.json');frozen=read(ROOT/'frozen.json');cohort=read(ROOT/'prediction_cohort.json');summary=read(ROOT/'summary.json');resources=read(ROOT/'resources.json');scores=read(ROOT/'scores.json');inputs=read(ROOT/'input_hashes.json')
expected={f'{im}_ph{"".join(map(str,ph))}_sigma{s}_{m}' for im,ph,s,m in itertools.product(P['cohort']['images'],P['cohort']['phase_sets'],P['cohort']['sigma'],P['cohort']['methods'])}
out={'stage':'independent audit of all saved linear controls; no new predictions or training written','GPU_used':False,'cuda_initialized_before':torch.cuda.is_initialized(),'budget_seconds':120,'completed_cases':0,'failures':[]}
out['identity_checks']={'source_sha_matches':sha(CODE/'experiment.py')==frozen['source_sha256'],'protocol_sha_matches':sha(CODE/'protocol.json')==sha(ROOT/'protocol.json')==frozen['protocol_sha256'],'source_sha':sha(CODE/'experiment.py'),'protocol_sha':sha(ROOT/'protocol.json'),'cohort_sha':sha(ROOT/'prediction_cohort.json'),'cohort_sha_matches':sha(ROOT/'prediction_cohort.json')==summary['cohort_sha256'],'all_input_hashes_match':all(sha(p)==h for p,h in inputs.items()),'input_file_count':len(inputs),'original_source_sha_matches':sha(frozen['original_source']['path'])==frozen['original_source']['sha256'],'full_24_prediction_ids':set(cohort['records'])==expected and cohort['n_cases']==24,'full_24_score_ids':{r['id'] for r in scores}==expected and len(scores)==24,'complete':(ROOT/'COMPLETE').exists(),'no_incomplete':not (ROOT/'INCOMPLETE.json').exists()}
sc={r['id']:r for r in scores};records={cid:read(ROOT/'predictions'/cid/'record.json') for cid in expected}
out['journal_checks']={'solve_equals_frozen_records':[json.loads(s) for s in (ROOT/'solve_journal.jsonl').read_text().splitlines()]==[records[r['id']] for r in [json.loads(s) for s in (ROOT/'solve_journal.jsonl').read_text().splitlines()]],'solve_24_unique':len((ROOT/'solve_journal.jsonl').read_text().splitlines())==24,'score_equals_scores':[json.loads(s) for s in (ROOT/'score_journal.jsonl').read_text().splitlines()]==scores,'all_score_cohort_hash_match':all(r['prediction_cohort_sha256']==sha(ROOT/'prediction_cohort.json') for r in scores)}
run=next(n for n in ast.parse((CODE/'experiment.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
fl=next(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save' and 'prediction_cohort.json' in ast.unparse(n));tl=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='obs' and isinstance(n.slice,ast.Constant) and n.slice.value in ['clean','truth']]
out['freeze_gate']={'cohort_line':fl,'truth_lines':tl,'all_truth_access_after_cohort':min(tl)>fl,'frozen_timestamp_in_invocation':frozen['started_unix']<=cohort['frozen_before_truth_score_unix']<=resources['ended_unix'],'note':'Source ordering plus immutable hashes/journals checked; retrospective artifacts alone do not timestamp every read.'}
grid=np.array([[x,y] for y in range(64) for x in range(64)]);target=(grid+.5)/64;crop=(grid[:,0]>=4)&(grid[:,0]<60)&(grid[:,1]>=4)&(grid[:,1]<60);mask_regions={'full':np.ones(4096,dtype=bool),'shave4':crop}
def phi(x,B):return np.concatenate([np.cos(2*np.pi*x@B.T),np.sin(2*np.pi*x@B.T)],axis=1)/16
def rbf(x,z,s):return np.exp(-2*np.pi**2*s*s*np.sum((x[:,None,:]-z[None,:,:])**2,axis=-1))
maxima={k:0. for k in ['MSE_error','PSNR_error','risk_term_error','coefficient_true_residual','response_true_residual','LU_coeff_relative_difference','LU_W_relative_difference','LU_raw_max_difference','posterior_variance_max_difference','expected_noise_max_difference','RFF_primal_dual_raw_max_difference','RFF_primal_dual_W_relative_difference','RFF_primal_true_residual']}
def mm(key,v):maxima[key]=max(maxima[key],float(v))
results=[];geometry_cache={};old_ff_seen=set();old_ff_max_error=0.
for cid in sorted(expected):
 budget();rec=records[cid];geometry=rec['geometry'];geompath=INPUT/'observations'/(geometry+'.npz')
 assert rec['source_observations_sha256']==sha(geompath)
 cp=INPUT/'fits'/(geometry+'_sigma'+str(rec['sigma']))/'checkpoint.pth';assert rec['source_ff_checkpoint_sha256']==sha(cp)
 dest=ROOT/'predictions'/cid;assert sha(dest/'record.json')==cohort['records'][cid]['record_sha256'];assert sha(dest/'arrays.npz')==cohort['records'][cid]['arrays_sha256']==rec['arrays_sha256']
 with np.load(geompath,allow_pickle=False) as f:y=f['y'].astype(float);coords=f['coords'].astype(float);xy=f['xy'];clean=f['clean'].astype(float);truth=f['truth'].astype(float)
 with np.load(dest/'arrays.npz',allow_pickle=False) as f:raw=f['unclamped'];clamp=f['clamped'];W=f['linear_response'];coeff=f['posterior_coefficients'];B=f['frequency_B'];post=f['posterior_variance'];noise_saved=f['expected_noise_mse_by_pixel'];assert np.array_equal(f['observation_xy'],xy) and np.array_equal(f['evaluation_xy'],grid) and np.array_equal(f['crop_mask'],crop)
 assert np.array_equal(clamp,np.clip(raw,0,1));payload=torch.load(cp,map_location='cpu',weights_only=True);assert np.array_equal(B,payload['model']['B'].numpy().astype(float));del payload
 unique=np.unique(xy,axis=0)
 if geometry not in geometry_cache:
  d=np.min(np.sum((grid[:,None]-unique[None,:,:])**2,axis=-1),axis=1);val,cnt=np.unique(d[crop],return_counts=True);hist=[{'squared_distance_pixels':int(v),'count':int(c)} for v,c in zip(val,cnt)];reported=rec['coverage']['nearest_observed_shave4']['histogram'];assert hist==[{'squared_distance_pixels':r['squared_distance_pixels'],'count':r['count']} for r in reported]
  assert len(y)==512;assert len(unique)==rec['coverage']['unique_observed_pixels'];assert sum(cnt)==3136
  geometry_cache[geometry]={'nearest_distance_histogram':hist,'n_points':512,'n_unique':len(unique),'obs_sha':sha(geompath),'shared_sigma_method_inputs':True}
 s=rec['sigma'];method=rec['method'];prior_phi=phi(coords,B) if method=='fixed_rf' else None;target_phi=phi(target,B) if method=='fixed_rf' else None
 K=prior_phi@prior_phi.T if method=='fixed_rf' else rbf(coords,coords,s);Kt=target_phi@prior_phi.T if method=='fixed_rf' else rbf(target,coords,s);S=K+1e-4*np.eye(512)
 # General LU, independent of original Cholesky; both RHS solved together once.
 rhs=np.concatenate([y-.5,Kt.T],axis=1);ind=solve(S,rhs,assume_a='gen',check_finite=True);a=ind[:,:3];wi=ind[:,3:].T
 cr=np.linalg.norm(S@coeff-(y-.5))/np.linalg.norm(y-.5);wr=np.linalg.norm(S@W.T-Kt.T)/np.linalg.norm(Kt.T);mm('coefficient_true_residual',cr);mm('response_true_residual',wr)
 mm('LU_coeff_relative_difference',np.linalg.norm(a-coeff)/max(np.linalg.norm(coeff),1e-30));mm('LU_W_relative_difference',np.linalg.norm(wi-W)/np.linalg.norm(W));pred=(.5+Kt@a).T.reshape(3,64,64);mm('LU_raw_max_difference',np.max(abs(pred-raw)))
 diag=np.ones(4096) if method=='rbf' else np.sum(target_phi**2,axis=1);pi=diag-np.sum(wi*Kt,axis=1);mm('posterior_variance_max_difference',np.max(abs(pi-post)));nv=1e-4*np.sum(wi**2,axis=1);mm('expected_noise_max_difference',np.max(abs(nv-noise_saved)))
 primal={}
 if method=='fixed_rf':
  G=prior_phi.T@prior_phi+1e-4*np.eye(512);prhs=np.concatenate([prior_phi.T@(y-.5),prior_phi.T],axis=1);pa=solve(G,prhs,assume_a='gen',check_finite=True);beta=pa[:,:3];prw=target_phi@pa[:,3:];prpred=.5+target_phi@beta;prres=np.linalg.norm(G@beta-prhs[:,:3])/np.linalg.norm(prhs[:,:3]);mm('RFF_primal_true_residual',prres);mm('RFF_primal_dual_raw_max_difference',np.max(abs(prpred-pred.reshape(3,-1).T)));mm('RFF_primal_dual_W_relative_difference',np.linalg.norm(prw-wi)/np.linalg.norm(wi));primal={'primal_true_residual':float(prres),'raw_max_difference':float(np.max(abs(prpred-pred.reshape(3,-1).T)))}
 bias=(.5+wi@(clean-.5)).T-truth.reshape(3,-1);noise=(wi@(y-clean)).T;risk={};metric={}
 for name,mask in mask_regions.items():
  metric[name]={}
  for n,x in [('unclamped',raw),('clamped',clamp)]:
   mse=float(np.mean((x.reshape(3,-1)[:,mask]-truth.reshape(3,-1)[:,mask])**2));psnr=-10*math.log10(mse);metric[name][n]={'mse':mse,'psnr':psnr};mm('MSE_error',abs(mse-sc[cid]['metrics'][name][n]['mse']));mm('PSNR_error',abs(psnr-sc[cid]['metrics'][name][n]['psnr']))
  b=float(np.mean(bias[:,mask]**2));n=float(np.mean(noise[:,mask]**2));cross=float(2*np.mean(bias[:,mask]*noise[:,mask]));en=float(nv[mask].mean());risk[name]={'bias_squared_mse':b,'expected_noise_mse':en,'expected_deterministic_raw_mse':b+en,'realized_noise_mse':n,'realized_bias_noise_cross':cross,'realized_decomposition_mse':b+n+cross}
  for key,value in risk[name].items():mm('risk_term_error',abs(value-sc[cid]['deterministic_risk'][name][key]))
  mm('risk_term_error',abs(b+n+cross-metric[name]['unclamped']['mse']))
 fkey=geometry+'_sigma'+str(s)
 if fkey not in old_ff_seen:
  with np.load(INPUT/'fits'/fkey/'predictions.npz',allow_pickle=False) as ff:
   for region,mask in mask_regions.items():
    for n in ['unclamped','clamped']:
     mse=float(np.mean((ff[n].reshape(3,-1)[:,mask].astype(float)-truth.reshape(3,-1)[:,mask])**2));old_ff_max_error=max(old_ff_max_error,abs(mse-sc[cid]['old_ff_metrics'][region][n]['mse']))
  old_ff_seen.add(fkey)
 results.append({'id':cid,'geometry':geometry,'method':method,'sigma':s,'metrics':metric,'risk':risk,'posterior_shave4':float(pi[crop].mean()),'posterior_min':float(pi.min()),'coefficient_residual':float(cr),'W_residual':float(wr),'RFF_primal':primal});out['completed_cases']=len(results)

ri={r['id']:r for r in results};comparisons=[]
for im,s,method in itertools.product(P['cohort']['images'],P['cohort']['sigma'],P['cohort']['methods']):
 rr=[ri[f'{im}_ph{p}_sigma{s}_{method}'] for p in ['00','01','02']];a,b=rr[1],rr[2]
 pv=b['posterior_shave4']-a['posterior_shave4'];en=b['risk']['shave4']['expected_noise_mse']-a['risk']['shave4']['expected_noise_mse'];er=b['risk']['shave4']['expected_deterministic_raw_mse']-a['risk']['shave4']['expected_deterministic_raw_mse']
 comparisons.append({'image':im,'sigma':s,'method':method,'PSNR_ph00_ph01_ph02':[r['metrics']['shave4']['clamped']['psnr'] for r in rr],'PSNR02_minus01':b['metrics']['shave4']['clamped']['psnr']-a['metrics']['shave4']['clamped']['psnr'],'raw_expected_risk_ph00_ph01_ph02':[r['risk']['shave4']['expected_deterministic_raw_mse'] for r in rr],'posterior_variance_ph00_ph01_ph02':[r['posterior_shave4'] for r in rr],'expected_noise_risk_ph00_ph01_ph02':[r['risk']['shave4']['expected_noise_mse'] for r in rr],'phase02_minus01':{'model_posterior_variance':pv,'true_conditional_expected_noise_risk':en,'true_conditional_raw_expected_risk':er},'posterior_agrees_with_conditional_noise_order':bool(pv*en>=0),'posterior_agrees_with_conditional_total_order':bool(pv*er>=0)})
for comparison in comparisons:
 posterior_order=list(map(int,np.argsort(comparison['posterior_variance_ph00_ph01_ph02'])))
 noise_order=list(map(int,np.argsort(comparison['expected_noise_risk_ph00_ph01_ph02'])))
 total_order=list(map(int,np.argsort(comparison['raw_expected_risk_ph00_ph01_ph02'])))
 comparison['three_phase_order_indices_low_to_high']={'posterior':posterior_order,'conditional_noise':noise_order,'conditional_total':total_order}
 comparison['three_phase_posterior_noise_order_agrees']=posterior_order==noise_order
 comparison['three_phase_posterior_total_order_agrees']=posterior_order==total_order
solve_journal=[json.loads(s) for s in (ROOT/'solve_journal.jsonl').read_text().splitlines()]
out['journal_checks']['solve_all_24_ids_unique']=len(solve_journal)==24 and {r['id'] for r in solve_journal}==expected
out['max_errors']=maxima;out['old_FF_MSE_max_error']=old_ff_max_error;out['geometry_checks']=geometry_cache;out['cases']=results;out['phase_comparisons']=comparisons
out['budget_checks']={'original_resources':resources,'within_deadline':resources['ended_unix']<=frozen['deadline_unix'],'phase_wall_under600':resources['cumulative_cpu_phase_wall_seconds']<=600,'full_wall_under600':resources['full_wall_seconds']<=600,'solved_scored24':resources['solved']==resources['scored']==24,'zero_failed_records':all(r['status']=='complete' for r in records.values()),'no_failure_artifacts':not (ROOT/'INCOMPLETE.json').exists() and not (ROOT/'prediction_failures.json').exists()}
out['limits']=['Fixed mean .5 and prior variance1/lambda1e-4; finite development controls, not calibrated image priors.','Posterior variance comparisons only within a fixed method/sigma across phases; not a cross-prior selector.','Bias/noise expected raw risk uses saved clean/truth and new iid noise; not observable risk estimator.','Two previously seen images and one shared saved noise draw; no confirmatory generalization.','Phase observation sets nonnested, nonperiodic integer distance histogram differs from continuous periodic maxgap.','Prior diagonal variance and noise propagation variance are distinct quantities.','Independent numerical LU checks are floating arithmetic, not a machine proof.']
out['cuda_initialized_after']=torch.cuda.is_initialized();out['audit_wall_seconds']=time.monotonic()-began;out['complete']=len(results)==24
(ROOT/'独立对照复核-agent.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['cases','geometry_checks']},ensure_ascii=False,indent=2,default=lambda x:x.item()))
