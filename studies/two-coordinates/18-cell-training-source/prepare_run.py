"""Deploy to a NEW server run root and freeze before any train/eval call."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--source',required=True);a=p.parse_args();r=Path(a.run_root).resolve();src=Path(a.source).resolve();here=Path(__file__).resolve().parent
assert not r.exists(),f'Run root already exists: {r}'
P=json.loads((here/'protocol.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path,value in P['expected_source_sha256'].items():assert sha(src/path)==value,path
for item in P['baseline_checkpoints'].values():assert sha(item['path'])==item['sha256'],item['path']
for old in ['claude-20261019-encoder-train-v5','claude-20261020-seed2-v8']:
 for path,value in P['expected_source_sha256'].items():
  original=Path('/workspace/SE-INR/runs')/old/path
  assert sha(original)==value,str(original)
r.mkdir(parents=True)
for p in here.iterdir():
 if p.suffix in ('.py','.sh','.json'):shutil.copy2(p,r/p.name)
for folder in ['src','entrypoints']:shutil.copytree(src/folder,r/'source'/folder,ignore=shutil.ignore_patterns('__pycache__'))
for name,target in [('models','src/models'),('datasets','src/datasets'),('utils.py','src/utils.py')]:os.symlink(target,r/'source'/name)
(r/'configs').mkdir();base=yaml.safe_load((src/'configs/train-div2k/train-liif-edsr-200ep.yaml').read_text())
for arm in ['constant','shuffle']:
 for seed in [1,2]:
  cfg=json.loads(json.dumps(base));cfg['train_dataset']['dataset']['args']['root_path']=P['data_root']+'/DIV2K_train_HR';cfg['val_dataset']['dataset']['args']['root_path']=P['data_root']+'/DIV2K_valid_HR';cfg['model']['name']='liif-cell-source';cfg['model']['args'].update(cell_mode=arm,cell_seed=180000+seed)
  (r/'configs'/f'{arm}_s{seed}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
# Freeze original training/validation and benchmark input identities before running.
inventory={}
for folder in ['DIV2K_train_HR','DIV2K_valid_HR','Set5/HR','Set14/HR','BSD100/HR','Urban100/HR']:
 paths=sorted((Path(P['data_root'])/folder).glob('*.png'));assert paths,folder
 inventory[folder]={str(x):sha(x) for x in paths}
(r/'data_manifest.json').write_text(json.dumps(inventory,indent=1)+'\n')
meta={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source':str(src),'git_head':subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),'checkpoint_sha256':P['baseline_checkpoints'],'predictions_frozen_before_training':True}
(r/'freeze_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
files=[x for x in sorted(r.rglob('*')) if x.is_file() and not x.is_symlink()]
(r/'FROZEN.sha256').write_text(''.join(f'{sha(x)}  {x.relative_to(r)}\n' for x in files))
print(json.dumps({'run_root':str(r),'frozen_sha256':sha(r/'FROZEN.sha256'),'files':len(files),'train_runs_started':0},indent=2))
