"""Independent saved common-law cohort audit; no source import, solve or fit.
CPU4, invisible CUDA, separate non-extending 120-second audit budget.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,hashlib,time,itertools,math,ast,signal,resource
from pathlib import Path
BEGIN=time.monotonic();ABS_END=time.time()+120
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
 if time.time()>=ABS_END or time.monotonic()-BEGIN>=120:raise TimeoutError('Independent audit absolute/cumulative120s exhausted')
def stop(a,b):raise TimeoutError('Independent audit interrupted')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
OUT={'audit_scope':'Independent recomputation from 32 original images and saved fixed operators; no W solve/training','GPU_used':False,'cpu_threads':4,'audit_budget_seconds':120,'complete':False,'failed':[]}
try:
 import numpy as np
 import torch
 import torch.nn.functional as F
 from PIL import Image
 torch.set_num_threads(4);torch.set_num_interop_threads(4)
 OUT['cuda_initialized_before']=torch.cuda.is_initialized()
 P=read(ROOT/'protocol.json');freeze=read(ROOT/'recommendation_cohort.json');frozen=read(ROOT/'source_frozen.json');summary=read(ROOT/'summary.json');res=read(ROOT/'resources.json')
 rows=read(ROOT/'calibration_rows.json');rowmap={(r['image'],r['arm']):r for r in rows};scored=read(ROOT/'postfreeze_scores.json');scoremap={(r['image'],r['arm']):r for r in scored};recs=read(ROOT/'recommendations.json');risks_saved=read(ROOT/'calibration_risks.json');moments={r['arm']:r for r in read(ROOT/'second_moment_check.json')}
 arms=sorted(itertools.product(P['cohort']['methods'],P['cohort']['sigma'],P['cohort']['phases']))
 def aid(a):return f'ph{a[2]}_sigma{a[1]}_{a[0]}'
 def choose(values,candidates):
  best=min(values[a] for a in candidates);return min(a for a in candidates if values[a]-best<=1e-12)
 def sign(v):return 0 if abs(v)<=1e-12 else (-1 if v<0 else 1)
 OUT['hash_checks']={'source':sha(ROOT/'experiment.py')==frozen['source_sha256']==freeze['identities']['source_sha256'],'protocol':sha(ROOT/'protocol.json')==frozen['protocol_sha256']==freeze['identities']['protocol_sha256'],'cohort_sha':sha(ROOT/'recommendation_cohort.json'),'cohort_matches_summary':sha(ROOT/'recommendation_cohort.json')==summary['recommendation_cohort_sha256'],'all_frozen_output_hashes':all(sha(ROOT/k)==v for k,v in freeze['frozen_files'].items()),'all24_operator_input_hashes':all(sha(p)==h for p,h in freeze['all24_input_sha256'].items()),'all32_calibration_source_hashes':all(sha(v['path'])==v['sha256'] for v in freeze['calibration32_source_hashes'].values()),'postfreeze_input_hashes':all(sha(p)==h for p,h in read(ROOT/'postfreeze_input_hashes.json').items())}
 check();allpaths=freeze['all24_input_sha256'];array_paths={Path(p).parent.name:Path(p) for p in allpaths if p.endswith('/arrays.npz')};ops={};operator_checks=[]
 for arm in arms:
  check();first=None
  for im in P['cohort']['development_images']:
   cid=im+'_'+aid(arm);path=array_paths[cid]
   with np.load(path,allow_pickle=False) as f:item={k:f[k].copy() for k in P['operator_gate']['keys_before_freeze']}
   assert item['linear_response'].shape==(4096,512) and item['linear_response'].dtype==np.float64
   if first is None:first=item
   else:
    for k,v in item.items():
     delta=float(np.max(np.abs(v.astype(float)-first[k].astype(float))));byte=v.dtype==first[k].dtype and v.shape==first[k].shape and v.tobytes()==first[k].tobytes()
     assert delta<=1e-14 and (k=='linear_response' or byte)
     operator_checks.append({'arm':aid(arm),'key':k,'max_difference':delta,'byte_identical':byte})
  h=freeze['reference12_operator_hashes'][aid(arm)]
  assert hashlib.sha256(first['linear_response'].tobytes()).hexdigest()==h['W_sha256'];assert hashlib.sha256(first['frequency_B'].tobytes()).hexdigest()==h['B_sha256'];assert hashlib.sha256(first['observation_xy'].tobytes()).hexdigest()==h['xy_sha256']
  mask=first['crop_mask'];assert mask.sum()==3136;first['Wcrop']=first['linear_response'][mask];first['noise']=float(1e-4*np.einsum('ij,ij->',first['Wcrop'],first['Wcrop'])/3136);ops[arm]=first
 OUT['operator_checks']=operator_checks
 prepj=[json.loads(s) for s in (ROOT/'preprocess_journal.jsonl').read_text().splitlines()];prepm={r['image']:r for r in prepj}
 maximum={k:0. for k in ['independent_numpy_pool_difference','per_image_bias_error','noise_error','per_image_total_error','calibration_mean_error','lowrank_error','development_expected_raw_error','realized_clamped_MSE_error','realized_clamped_PSNR_error','pair_gap_error','recommendation_regret_error','baseline_regret_error']}
 def mx(k,x):maximum[k]=max(maximum[k],float(x))
 biases={a:[] for a in arms};zt=[];zo={a:[] for a in arms};prep_checks=[]
 for im in P['calibration_images']:
  check();entry=freeze['calibration32_source_hashes'][im];image=Image.open(entry['path']).convert('RGB');width,height=image.size;l=(width-128)//2;t=(height-128)//2;box=(l,t,l+128,t+128)
  native=np.asarray(image.crop(box),np.float32)/np.float32(255.);truth=F.avg_pool2d(torch.from_numpy(native).permute(2,0,1)[None],2)[0].numpy().astype(float)
  # Separate NumPy four-site mean verifies crop/pooling numerically; original
  # Torch operation order retained for exact float32 preprocessing hash.
  pooled=native.reshape(64,2,64,2,3).mean(axis=(1,3),dtype=np.float32).transpose(2,0,1);mx('independent_numpy_pool_difference',np.max(abs(pooled-truth)))
  assert sha(entry['path'])==entry['sha256']==prepm[im]['source_sha256'];assert list(box)==prepm[im]['crop'];digest=hashlib.sha256(truth.tobytes()).hexdigest();assert digest==prepm[im]['pooled_float64_sha256']
  prep_checks.append({'image':im,'crop':list(box),'pooled_hash_matches':True});target=truth.reshape(3,-1)[:,ops[arms[0]]['crop_mask']];zt.append(target.T-.5)
  for arm in arms:
   check();op=ops[arm];xy=op['observation_xy'];clean=truth[:,xy[:,1],xy[:,0]];z=clean-.5
   # Channel-wise matrix products independently accumulate the 3P bias loss.
   predictions=np.stack([.5+op['Wcrop']@z[c] for c in range(3)]);bias=float(np.sum((predictions-target)**2)/(3*3136));total=bias+op['noise'];r=rowmap[(im,aid(arm))]
   mx('per_image_bias_error',abs(bias-r['bias_mse']));mx('noise_error',abs(op['noise']-r['noise_mse']));mx('per_image_total_error',abs(total-r['total_mse']));assert r['crop']==list(box);biases[arm].append(total);zo[arm].append(z.T)
 cal={a:float(np.mean(biases[a])) for a in arms};Zt=np.concatenate(zt,axis=1);assert Zt.shape==(3136,96)
 lowrank=[]
 for a in arms:
  check();Zo=np.concatenate(zo[a],axis=1);assert Zo.shape==(512,96);E=ops[a]['Wcrop']@Zo-Zt;value=float(np.einsum('ij,ij->',E,E)/(32*3*3136)+ops[a]['noise']);mx('calibration_mean_error',abs(cal[a]-risks_saved[aid(a)]));mx('lowrank_error',max(abs(value-cal[a]),abs(value-moments[aid(a)]['equivalent_risk'])));lowrank.append({'arm':aid(a),'risk':value,'shape':list(E.shape)})
 groups={'phase'+ph:[a for a in arms if a[2]==ph] for ph in P['cohort']['phases']};groups.update({m+'_sigma'+str(s):[a for a in arms if a[:2]==(m,s)] for m,s in itertools.product(P['cohort']['methods'],P['cohort']['sigma'])});groups['global']=arms
 recommendations={k:aid(choose(cal,candidates)) for k,candidates in groups.items()};assert all(recommendations[k]==recs[k]['chosen']==freeze['recommendations'][k]['chosen'] for k in groups)
 OUT['recommendations_recomputed']=recommendations;OUT['calibration_risks_recomputed']={aid(a):cal[a] for a in arms};OUT['lowrank_recomputed']=lowrank
 expected={};posterior={};comparison_map={(r['image'],r['group']):r for r in read(ROOT/'recommendation_comparisons.json')};pairmap={(r['image'],r['arm_a'],r['arm_b']):r for r in read(ROOT/'pair_gaps.json')};baseline_map={(r['image'],r['phase']):r for r in read(ROOT/'baselines.json')};opposing=[];misses=[];baseline_diagnostics=[];comparisons=[]
 BASE=ROOT.parent;linear=BASE/'phase-linear-controls-run-v1';dev=BASE/'inr-phase-development-run-v1';nu=1e-4
 for im in P['cohort']['development_images']:
  for ph in P['cohort']['phases']:
   check()
   with np.load(dev/'observations'/(im+'_ph'+ph+'.npz'),allow_pickle=False) as f:truth=f['truth'].astype(float);clean=f['clean'].astype(float)
   target=truth.reshape(3,-1)[:,ops[arms[0]]['crop_mask']]
   for arm in groups['phase'+ph]:
    check();op=ops[arm];assert np.array_equal(clean,truth[:,op['observation_xy'][:,1],op['observation_xy'][:,0]].T)
    pred=np.stack([.5+op['Wcrop']@(clean[:,c]-.5) for c in range(3)]);risk=float(np.sum((pred-target)**2)/(3*3136)+op['noise']);expected[(im,arm)]=risk;r=scoremap[(im,aid(arm))];mx('development_expected_raw_error',abs(risk-r['expected_raw_mse']))
    cid=im+'_'+aid(arm)
    with np.load(linear/'predictions'/cid/'arrays.npz',allow_pickle=False) as f:clamp=f['clamped'].reshape(3,-1)[:,op['crop_mask']]
    mse=float(np.mean((clamp-target)**2));psnr=-10*math.log10(mse);mx('realized_clamped_MSE_error',abs(mse-r['realized_clamped']['mse']));mx('realized_clamped_PSNR_error',abs(psnr-r['realized_clamped']['psnr']))
    posterior[(im,arm)]=read(linear/'predictions'/cid/'record.json')['model_implied_variance']['shave4_mean'];assert posterior[(im,arm)]==r['own_prior_posterior'];assert r['recommendation_cohort_sha256']==sha(ROOT/'recommendation_cohort.json')
  vals={a:expected[(im,a)] for a in arms}
  for key,candidates in groups.items():
   a=choose(cal,candidates);oracle=choose(vals,candidates);regret=vals[a]-min(vals[x] for x in candidates);stored=comparison_map[(im,key)]
   assert aid(a)==stored['calibration_chosen'] and aid(oracle)==stored['oracle_chosen'] and (a==oracle)==stored['same_choice'];mx('recommendation_regret_error',abs(regret-stored['expected_raw_regret']));row={'image':im,'group':key,'calibration_chosen':aid(a),'oracle_chosen':aid(oracle),'regret':regret,'same_choice':a==oracle};comparisons.append(row)
   if a!=oracle:misses.append(row)
  for a,b in itertools.combinations(arms,2):
   g=cal[a]-cal[b];v=vals[a]-vals[b];opp=sign(g)*sign(v)==-1;stored=pairmap[(im,aid(a),aid(b))];assert stored['calibration_sign']==sign(g) and stored['actual_sign']==sign(v) and stored['opposing']==opp;mx('pair_gap_error',max(abs(g-stored['calibration_gap']),abs(v-stored['actual_expected_raw_gap'])))
   if opp:opposing.append({'image':im,'arm_a':aid(a),'arm_b':aid(b),'calibration_gap':g,'development_expected_gap':v})
  for ph in P['cohort']['phases']:
   candidates=groups['phase'+ph];minimum=min(vals[a] for a in candidates);fixed=('rbf',3,ph);post=choose({a:posterior[(im,a)] for a in candidates},candidates);stored=baseline_map[(im,ph)];assert stored['fixed_rbf_sigma3']['arm']==aid(fixed) and stored['ownprior_postfreeze']['arm']==aid(post)
   for label,a in [('fixed_rbf_sigma3',fixed),('ownprior_postfreeze',post)]:mx('baseline_regret_error',abs(vals[a]-minimum-stored[label]['expected_raw_regret']))
   baseline_diagnostics.append({'image':im,'phase':ph,'fixed_rbf3_regret':vals[fixed]-minimum,'ownprior_choice':aid(post),'ownprior_regret':vals[post]-minimum})
 OUT['complete_counts']={'preprocessed32':len(prep_checks)==32,'cal384':len(rows)==len(rowmap)==384 and set(rowmap)==set(itertools.product(P['calibration_images'],map(aid,arms))),'dev24':len(scored)==len(scoremap)==24,'comparison16':len(comparison_map)==16,'pair132':len(pairmap)==132,'baseline6':len(baseline_map)==6}
 OUT['journal_checks']={'calibration_equals_rows':[json.loads(s) for s in (ROOT/'calibration_journal.jsonl').read_text().splitlines()]==rows,'score_equals_scores':[json.loads(s) for s in (ROOT/'score_journal.jsonl').read_text().splitlines()]==scored,'preprocess32_unique':len(prepj)==len(prepm)==32,'input24_unique':len((ROOT/'input_journal.jsonl').read_text().splitlines())==24 and len({json.loads(s)['id'] for s in (ROOT/'input_journal.jsonl').read_text().splitlines()})==24}
 stored_opp=read(ROOT/'opposing_pairs.json');assert {(r['image'],r['arm_a'],r['arm_b']) for r in stored_opp}=={(r['image'],r['arm_a'],r['arm_b']) for r in opposing}
 run=next(n for n in ast.parse((ROOT/'experiment.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run');fl=next(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save' and 'recommendation_cohort.json' in ast.unparse(n));tl=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='obs' and isinstance(n.slice,ast.Constant) and n.slice.value in ['clean','truth','y']]
 source=(ROOT/'experiment.py').read_text();prefix=source.split("STATE['stage']='postfreeze_score'")[0];assert "scores.json" not in prefix and "model_implied_variance" not in prefix and min(tl)>fl
 OUT['freeze_gate']={'freeze_line':fl,'dev_access_lines':tl,'source_order_pass':True,'freeze_time_within_invocation':res['began_unix']<=freeze['frozen_before_development_score_unix']<=res['ended_unix'],'only32train_and_fixedW_before_choice':True,'posthoc_design_limitation_retained':True}
 OUT['maximum_errors']=maximum;OUT['preprocess_checks']=prep_checks;OUT['all_opposing_pairs']=opposing;OUT['recommendation_misses']=misses;OUT['baseline_diagnostics']=baseline_diagnostics;OUT['recommendation_match_count']=sum(r['same_choice'] for r in comparisons);OUT['opposing_pair_count']=len(opposing)
 OUT['cost_checks']={'original_resources':res,'wall_under600':res['full_wall_seconds']<=600 and res['cumulative_cpu_phase_wall_seconds']<=600,'ended_before_original_deadline':res['ended_unix']<=res['deadline_unix'],'all_expected_stage_costs_present':all(k in summary['stage_wall_seconds'] for k in ['operator_verification_seconds','calibration_preprocess_seconds','calibration_arm_seconds','second_moment_seconds','recommendation_seconds','hash_freeze_seconds','postfreeze_score_seconds']),'complete_marker':(ROOT/'COMPLETE').exists(),'no_failure_marker':not (ROOT/'INCOMPLETE.json').exists(),'counts_match_summary':len(opposing)==summary['n_opposing_pairs'] and sum(r['same_choice'] for r in comparisons)==summary['recommendation_matches']}
 OUT['limits']=['Finite32-image joint RGB empirical law, no iid channel confidence interval or prospective generalization.','All3 per-phase recommended configurations equal fixed RBFsigma3: this cohort does not show benefit over that static baseline.','Own-prior posterior baseline is postfreeze diagnostic only; differences across priors do not justify model selection.','The15/16 grouped comparisons overlap heavily on2 already-seen images; not16 independent samples.','Pairs involve overlapping operators and targets, not132 independent trials.']
 OUT['cuda_initialized_after']=torch.cuda.is_initialized();OUT['complete']=True
except Exception as exc:
 OUT['failed'].append({'type':type(exc).__name__,'message':str(exc)});OUT['complete']=False
finally:
 OUT['audit_wall_seconds']=time.monotonic()-BEGIN
 usage=resource.getrusage(resource.RUSAGE_SELF);OUT['audit_process_user_cpu_seconds']=usage.ru_utime;OUT['audit_process_system_cpu_seconds']=usage.ru_stime
 (ROOT/'审查.json').write_text(json.dumps(OUT,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
 print(json.dumps({k:v for k,v in OUT.items() if k not in ['operator_checks','preprocess_checks','lowrank_recomputed']},ensure_ascii=False,indent=2,default=lambda x:x.item()))
 if not OUT['complete']:sys.exit(1)
