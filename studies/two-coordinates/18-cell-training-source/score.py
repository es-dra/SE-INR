"""Frozen gates and paired, within-dataset image bootstrap; seeds are fixed."""
import argparse,gzip,hashlib,json
from itertools import product
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args();r=Path(a.run_root)
P=json.loads((r/'protocol.json').read_text());rows=[];paths=[]
for arm in ['native','nocell','constant','shuffle']:
 for seed in [1,2]:
  d=r/'evaluation'/f'{arm}_s{seed}';assert (d/'COMPLETE.json').exists(),str(d)
  path=d/'rows.jsonl.gz';paths.append(path)
  with gzip.open(path,'rt') as f:slot=[json.loads(line) for line in f]
  complete=json.loads((d/'COMPLETE.json').read_text());assert len(slot)==complete['rows'],str(d)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==complete['rows_sha256'],str(d)
  assert hashlib.sha256((r/'FROZEN.sha256').read_bytes()).hexdigest()==complete['frozen_sha256'],str(d)
  modes=['native','mean']+[f'marginal_{i}' for i in range(4)] if arm=='shuffle' else ['native']
  assert set(complete['modes'])==set(modes),str(d)
  keys=[]
  for x in slot:
   assert x['arm']==arm and x['seed']==seed and x['scale'] in P['scales'] and x['eval_mode'] in modes,str(d)
   assert x['kind'] in ('benchmark','recovery')
   keys.append((x['kind'],x['dataset'],x['image'],x['scale'],x['eval_mode'],x.get('lo')))
  assert len(keys)==len(set(keys)),f'Duplicate rows: {d}'
  expected=set()
  for ds,n in [('Set5',5),('Set14',14),('BSD100',100),('Urban100',100)]:
   ims=sorted({x['image'] for x in slot if x['kind']=='benchmark' and x['dataset']==ds});assert len(ims)==n,(d,ds)
   expected.update(('benchmark',ds,img,sc,mo,None) for img,sc,mo in product(ims,P['scales'],modes))
  ims=[f'{i:04d}' for i in range(869,901)]
  expected.update(('recovery','DIV2K-fresh',img,sc,mo,k/4) for img,sc,mo in product(ims,P['scales'],modes) for k in range(sc*4))
  assert set(keys)==expected,f'Incomplete/extra grid: {d}'
  rows.extend(slot)
B=[x for x in rows if x['kind']=='benchmark'];R=[x for x in rows if x['kind']=='recovery'];dslist=['Set5','Set14','BSD100','Urban100'];scales=P['scales'];nboot=P['bootstrap']['replicates'];rng=np.random.default_rng(P['bootstrap']['seed'])
index={(x['arm'],x['seed'],x['dataset'],x['image'],x['scale'],x['eval_mode']):x['y_psnr'] for x in B}
images={ds:sorted({x['image'] for x in B if x['dataset']==ds}) for ds in dslist};draws={ds:rng.integers(len(images[ds]),size=(nboot,len(images[ds]))) for ds in dslist}
assert [len(images[ds]) for ds in dslist]==[5,14,100,100]

def cube(arm,ds,mode='native'):
 modes=[f'marginal_{i}' for i in range(4)] if mode=='marginal' else [mode]
 return np.array([[[np.mean([index[(arm,seed,ds,img,s,mo)] for mo in modes]) for s in scales] for seed in [1,2]] for img in images[ds]])

def compare(arm,reference='native',mode='native',refmode='native'):
 # Positive means reference is better than arm.
 d={ds:cube(reference,ds,refmode)-cube(arm,ds,mode) for ds in dslist}
 v=float(np.mean([x.mean() for x in d.values()]));boot=np.mean([x.mean(axis=(1,2))[draws[ds]].mean(axis=1) for ds,x in d.items()],axis=0)
 out={'reference':reference,'arm':arm,'eval_mode':mode,'difference_db':v,'ci95':np.percentile(boot,[2.5,97.5]).tolist(),'ci97_5':np.percentile(boot,[1.25,98.75]).tolist(),'seed_means':[float(np.mean([x[:,i,:].mean() for x in d.values()])) for i in range(2)],'per_dataset_scale':[]}
 for ds,x in d.items():
  for i,s in enumerate(scales):
   perimage=x[:,:,i].mean(axis=1);bc=perimage[draws[ds]].mean(axis=1)
   out['per_dataset_scale'].append({'dataset':ds,'scale':s,'difference_db':float(perimage.mean()),'ci95':np.percentile(bc,[2.5,97.5]).tolist()})
 return out
main={arm:compare(arm) for arm in ['constant','shuffle']};equiv={arm:compare(arm,'nocell') for arm in ['constant','shuffle']};sensitivity={mode:compare('shuffle',mode=mode) for mode in ['mean','marginal']}
gates={}
for arm,x in main.items():
 gates[f'T1_{arm}']={'pass':bool(x['difference_db']>=.05 and x['ci97_5'][0]>0 and min(x['seed_means'])>0),'criterion':'native-minus-arm >=0.05 dB, Bonferroni-adjusted 97.5% paired-image CI lower>0, both fixed seeds positive','observed':x['difference_db'],'ci':x['ci97_5'],'seed_means':x['seed_means']}
for arm,x in equiv.items():
 gates[f'T2_{arm}_equivalent_nocell']={'pass':bool(x['ci95'][0]>-.03 and x['ci95'][1]<.03),'criterion':'95% CI wholly within +/-0.03 dB; two equivalence tests with Bonferroni alpha .025 each','observed':x['difference_db'],'ci':x['ci95']}
ridx={(x['arm'],x['seed'],x['image'],x['scale'],x['eval_mode'],x['lo']):(x['target'],x['error']) for x in R}
rimages=sorted({x['image'] for x in R});assert len(rimages)==32
rdraw=rng.integers(len(rimages),size=(nboot,len(rimages)));recovery=[]
for arm in ['nocell','constant','shuffle']:
 for s in scales:
  for lo in sorted({x['lo'] for x in R if x['scale']==s}):
   native=np.array([[ridx[('native',seed,img,s,'native',lo)] for seed in [1,2]] for img in rimages]);other=np.array([[ridx[(arm,seed,img,s,'native',lo)] for seed in [1,2]] for img in rimages])
   assert np.allclose(native[:,:,0],other[:,:,0],rtol=0,atol=0)
   delta=(other[:,:,1].sum(axis=0)-native[:,:,1].sum(axis=0))/native[:,:,0].sum(axis=0)
   nums=(other[:,:,1]-native[:,:,1])[rdraw].sum(axis=1);den=native[:,:,0][rdraw].sum(axis=1);bb=(nums/den).mean(axis=1)
   recovery.append({'arm':arm,'scale':s,'lo':lo,'hi':min(lo+.25,float(s)),'native_R':float((1-native[:,:,1].sum(axis=0)/native[:,:,0].sum(axis=0)).mean()),'arm_R':float((1-other[:,:,1].sum(axis=0)/other[:,:,0].sum(axis=0)).mean()),'native_minus_arm_R':float(delta.mean()),'seed_deltas':delta.tolist(),'ci95':np.percentile(bb,[2.5,97.5]).tolist()})
region=[]
for arm in ['nocell','constant','shuffle']:
 for s in scales:
  for label,test in [('low_u_lt_0.75',lambda x:x['lo']<.75),('transition_0.75_to_1',lambda x:.75<=x['lo']<1),('high_1_to_2',lambda x:1<=x['lo']<2),('tail_u_ge_2',lambda x:x['lo']>=2)]:
   selected=[x for x in recovery if x['arm']==arm and x['scale']==s and test(x)]
   if selected:region.append({'arm':arm,'scale':s,'region':label,'mean_bin_delta_R':float(np.mean([x['native_minus_arm_R'] for x in selected])),'max_abs_bin_delta_R':max(abs(x['native_minus_arm_R']) for x in selected),'interpretation':'descriptive equal-bin summary; all bins retained, no gate or selection'})
absolute=[]
for arm in ['native','nocell','constant','shuffle']:
 for mode in (['native','mean','marginal'] if arm=='shuffle' else ['native']):
  for ds in dslist:
   x=cube(arm,ds,mode)
   for i,s in enumerate(scales):absolute.append({'arm':arm,'eval_mode':mode,'dataset':ds,'scale':s,'psnr':float(x[:,:,i].mean()),'seed_means':x[:,:,i].mean(axis=0).tolist()})
out={'predictions':gates,'primary_contrasts':main,'equivalence':equiv,'shuffle_policy_sensitivity':sensitivity,'absolute_y_psnr':absolute,'recovery':recovery,'recovery_regions':region,'bootstrap':P['bootstrap'],'scope':'Image uncertainty conditional on two trained seeds; dataset and scale balanced; not population-of-seeds CI.','inputs':{str(x.relative_to(r)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}}
(r/'scores').mkdir(exist_ok=False);(r/'scores'/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(gates,indent=2))
