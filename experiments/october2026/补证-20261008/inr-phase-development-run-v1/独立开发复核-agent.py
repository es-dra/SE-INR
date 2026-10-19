"""Read-only independent CPU audit of saved INR development artifacts.
Never imports or executes experiment.py/original-e1ff.py. Writes audit JSON only.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib,ast,math,itertools,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
torch.set_num_threads(1)
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
CODE=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT.parent/'inr-phase-development-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=read(CODE/'protocol.json');frozen=read(ROOT/'frozen.json');device=read(ROOT/'device.json');resource=read(ROOT/'resources.json');summary=read(ROOT/'summary.json');ident=frozen['identities']
began=time.monotonic();out={'scope':'Saved predictions/checkpoints and observed data independently checked on CPU; no GPU operation or new fit','cuda_initialized_before':torch.cuda.is_initialized()}
out['identities']={k:sha(p)==ident[k] for k,p in [('protocol_sha256',CODE/'protocol.json'),('source_sha256',CODE/'experiment.py'),('original_source_sha256',CODE/'original-e1ff.py')]}
original=Path('/workspace/SE-INR/claude-20261004-e1/e1ff.py')
out['identities']['matches_original_server_Claude_e1ff']=sha(original)==ident['original_source_sha256'] if original.exists() else None
tree=ast.parse((CODE/'original-e1ff.py').read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['FF','gauss_B']]
out['AST']={'two_original_nodes_present':{n.name for n in nodes}=={'FF','gauss_B'},'source_training_not_executed':True,'new_selector':False}
expected={f'{im}_ph{"".join(map(str,ph))}_sigma{s}' for im,ph,s in itertools.product(P['cohort']['images'],P['cohort']['phase_sets'],P['model']['sigma'])}
actual={p.name for p in (ROOT/'fits').iterdir() if p.is_dir()};records={r['id']:r for r in summary['fits']};out['cohort']={'expected':len(expected),'actual':len(actual),'full_unique_12':expected==actual==set(records) and len(summary['fits'])==12,'complete_marker':(ROOT/'COMPLETE').exists(),'incomplete_marker':(ROOT/'INCOMPLETE.json').exists()}
obs={};observation_checks=[];fits=[];max_mse_error=0.;max_psnr_error=0.;max_prediction_obs_loss_error=0.;max_crop_error=0.
for im in P['cohort']['images']:
 src=Path(ident['data'][im]['path']);assert sha(src)==ident['data'][im]['sha256']
 image=Image.open(src).convert('RGB');w,h=image.size;box=((w-128)//2,(h-128)//2,(w-128)//2+128,(h-128)//2+128)
 native=np.asarray(image.crop(box),dtype=np.float32)/255.;native_chw=torch.from_numpy(native).permute(2,0,1)[None]
 expected_truth=torch.nn.functional.avg_pool2d(native_chw,2)[0].numpy()
 for ph in P['cohort']['phase_sets']:
  geom=im+'_ph'+''.join(map(str,ph));path=ROOT/'observations'/(geom+'.npz');m=read(path.with_suffix('.json'))
  with np.load(path,allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
  xy=np.concatenate([np.array([[x,y] for y in range(0,64,4) for x in range(phase,64,4)]) for phase in ph]);clean=np.array([expected_truth[:,y,x] for x,y in xy]);coords=(xy.astype(np.float32)+.5)/64
  noise=torch.randn((512,3),generator=torch.Generator().manual_seed(1001)).numpy()*.01
  err=float(np.max(abs(expected_truth-z['truth'])));max_crop_error=max(max_crop_error,err)
  check={'geometry':geom,'source_sha_match':sha(src)==m['source']['sha256'],'crop_matches':list(box)==m['crop'],'observation_sha_match':sha(path)==m['observation_sha256'],'n_points':len(z['y']),'unique_locations':len(np.unique(xy,axis=0)),'xy_exact':np.array_equal(xy,z['xy']),'coords_exact':np.array_equal(coords,z['coords']),'truth_avgpool_max_error':err,'clean_max_error':float(np.max(abs(clean-z['clean']))),'fixed_noise_y_exact':np.array_equal(clean+noise,z['y']),'noise_std_empirical':float(np.std(z['y']-z['clean']))}
  observation_checks.append(check);obs[geom]=(z,m)

for cid in sorted(expected):
 dest=ROOT/'fits'/cid;state=read(dest/'state.json');assert state==records[cid];geom=cid.split('_sigma')[0];z,m=obs[geom]
 with np.load(dest/'predictions.npz',allow_pickle=False) as f:pred={k:f[k].copy() for k in f.files}
 cp=torch.load(dest/'checkpoint.pth',map_location='cpu',weights_only=True);sd=cp['model']
 tensors=list(sd.values());cpfinite=all(torch.isfinite(v).all().item() for v in tensors if torch.is_tensor(v));history=state['history']
 expected_B=torch.randn(256,2,generator=torch.Generator().manual_seed(0))*state['sigma'];Bmatch=torch.equal(sd['B'],expected_B)
 mm={}
 for name,x in pred.items():
  err=(x[:,4:-4,4:-4].astype(np.float64)-z['truth'][:,4:-4,4:-4].astype(np.float64))**2;mse=float(np.mean(err));psnr=-10*math.log10(mse)
  mm[name]={'mse':mse,'psnr':psnr};max_mse_error=max(max_mse_error,abs(mse-state['metrics'][name]['mse']));max_psnr_error=max(max_psnr_error,abs(psnr-state['metrics'][name]['psnr']))
 xy=z['xy'];obs_pred=pred['unclamped'][:,xy[:,1],xy[:,0]].T;ploss=float(np.mean((obs_pred.astype(np.float64)-z['y'].astype(np.float64))**2));max_prediction_obs_loss_error=max(max_prediction_obs_loss_error,abs(ploss-state['observation_loss']))
 optimizer_steps=[float(v['step']) for v in cp['optimizer']['state'].values() if 'step' in v]
 fit={'id':cid,'steps':state['steps'],'history_length':len(history),'history_finite':all(math.isfinite(v) for v in history),'checkpoint_steps':cp['steps'],'checkpoint_history_matches':cp['history']==history,'all_model_tensors_finite':cpfinite,'FF_gaussian_B_matches_seed0':Bmatch,'model_shapes':{k:list(v.shape) for k,v in sd.items()},'optimizer_steps_set':sorted(set(optimizer_steps)),'scheduler_last_epoch':cp['scheduler']['last_epoch'],'state_status':state['status'],'observation_sha_match':state['observation_sha256']==m['observation_sha256'],'prediction_sha_match':sha(dest/'predictions.npz')==state['predictions_sha256'],'prediction_shape_finite':all(x.shape==(3,64,64) and np.isfinite(x).all() for x in pred.values()),'clamped_matches':np.array_equal(pred['clamped'],np.clip(pred['unclamped'],0,1)),'metrics_recomputed_float64':mm,'observation_loss_from_saved_grid_prediction':ploss,'final_observation_loss_recorded':state['observation_loss'],'failures':state.get('failures',[]),'attempts':state['attempts'],'peak_allocated':state['peak_allocated_gpu_bytes'],'peak_reserved':state['peak_reserved_gpu_bytes']}
 fits.append(fit)
out['observation_checks']=observation_checks;out['fits']=fits;out['max_recompute_errors']={'truth_avgpool':max_crop_error,'MSE':max_mse_error,'PSNR_dB':max_psnr_error,'saved_grid_observation_loss':max_prediction_obs_loss_error}
byfit={x['id']:x for x in fits};comparison=[]
for im,sigma in itertools.product(P['cohort']['images'],P['model']['sigma']):
 values=[byfit[f'{im}_ph{ph}_sigma{sigma}']['metrics_recomputed_float64']['clamped']['psnr'] for ph in ['00','01','02']]
 comparison.append({'image':im,'sigma':sigma,'psnr_ph00_ph01_ph02':values,'delta01_minus00':values[1]-values[0],'delta02_minus00':values[2]-values[0],'delta02_minus01':values[2]-values[1],'counterexample_to_naive_odd_phase_advantage':values[2]>values[1]})
out['same_cost_phase_comparisons']=comparison
out['resource_checks']={'device':device,'recorded_gpu_phase_seconds':resource['gpu_phase_seconds'],'invocations':resource['invocations'],'all_complete_steps2000':all(x['steps']==x['checkpoint_steps']==x['history_length']==2000 for x in fits),'recorded_failure_count':sum(len(x['failures']) for x in fits),'fit_wall_sum':sum(a['wall_seconds'] for x in fits for a in x['attempts']),'peak_reserved_max':max(x['peak_reserved'] for x in fits),'memory_fraction_cap_bytes':device['total_memory']*device['memory_fraction'],'peak_below_recorded_cap':max(x['peak_reserved'] for x in fits)<device['total_memory']*.15,'phase_budget_ok':resource['gpu_phase_seconds']<=7200,'absolute_deadline_ok':all(r['ended_unix']<=frozen['deadline_unix'] for r in resource['invocations']),'invocation_wall_matches':all(abs(r['ended_unix']-r['started_unix']-r['full_wall_seconds'])<1e-4 for r in resource['invocations'])}
out['limits']=['Two previously seen development images and one seed; no confirmatory effect or generalization evidence.','Only horizontal phase changes in 2D point sampling; do not transfer 8-site stationary Gaussian covariance-map ranks.','AST reuses original Claude FF; no new model selector or learned architecture proposed.','Weights, history and endpoint records corroborate completion; saved finite-state checks do not independently replay all training gradients.','Device and memory cap are recorded provenance; audit does not independently establish exclusive GPU use or historical process count.','Seed1001 noise restarts identically across geometries/images: within duplicate frames draws are independent, but residual arrays across geometries are coupled.','Resource times are shared-device wall times; no GPU busy-time or exclusive speed claim.']
out['cuda_initialized_after']=torch.cuda.is_initialized();out['audit_wall_seconds']=time.monotonic()-began
(ROOT/'独立开发复核-agent.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['fits','observation_checks']},ensure_ascii=False,indent=2,default=lambda x:x.item()))
