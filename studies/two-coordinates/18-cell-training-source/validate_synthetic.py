"""CPU-only synthetic checks. Does not read training data or trained checkpoints."""
import argparse,gzip,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();here=Path(__file__).resolve().parent
sys.path[:0]=[a.source,str(here)]
import torch
import models
from models.liif import LIIF
import cell_source
spec={'encoder_spec':{'name':'edsr-baseline','args':{'no_upsampling':True}},'imnet_spec':{'name':'mlp','args':{'out_dim':3,'hidden_list':[256]*4}}}
checks={}
for seed in [1,2]:
 torch.manual_seed(seed);native=LIIF(**spec);state=torch.get_rng_state().clone()
 for mode in ['constant','shuffle']:
  torch.manual_seed(seed);m=cell_source.CellSourceLIIF(cell_mode=mode,cell_seed=180000+seed,**spec)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in native.state_dict().items())
  assert torch.equal(state,torch.get_rng_state())
 checks[f'initial_weights_and_global_rng_seed{seed}']=True
original=LIIF.query_rgb
LIIF.query_rgb=lambda self,coord,cell:cell
with tempfile.TemporaryDirectory() as tmp:
 os.environ['CELL_SOURCE_AUDIT_PATH']=str(Path(tmp)/'audit.json')
 for mode in ['constant','shuffle']:
  m=cell_source.CellSourceLIIF(cell_mode=mode,cell_seed=180001,**spec);m.train();m.feat=torch.empty(16,1,48,48)
  native=torch.tensor([96/k for k in range(48,64)],dtype=torch.float32).view(16,1,1).repeat(1,4,2)/48
  coord=torch.zeros_like(native);st=torch.get_rng_state().clone()
  for _ in range(1000):
   z=m.query_rgb(coord,native)
   if mode=='shuffle':assert torch.equal(z[:,0,0].sort().values,native[:,0,0].sort().values);assert (z[:,0,0]!=native[:,0,0]).all()
   else:assert torch.allclose(z*48,torch.full_like(z,cell_source.CONSTANT_CELL),atol=1e-7,rtol=0)
  assert torch.equal(st,torch.get_rng_state());m.eval();report=json.loads((Path(tmp)/'audit.json').read_text());assert report['samples']==16000
  if mode=='shuffle':assert report['fixed_index_count']==0;assert report['exact_marginal_sum_error']==0;assert report['exact_marginal_square_error']==0
  for H,W in [(48,48),(100,87),(30,40)]:
   m.feat=torch.empty(16,1,H,W);c=m.fixed_cell(native,cell_source.CONSTANT_CELL);c[...,0]*=H;c[...,1]*=W
   assert torch.allclose(c,torch.full_like(c,cell_source.CONSTANT_CELL),atol=1e-7,rtol=0)
  checks[f'{mode}_assignment_rng_audit']=report
LIIF.query_rgb=original
with tempfile.TemporaryDirectory() as tmp:
 r=Path(tmp);(r/'FROZEN.sha256').write_text('synthetic-only');protocol=json.loads((here/'protocol.json').read_text());protocol['bootstrap']['replicates']=100;(r/'protocol.json').write_text(json.dumps(protocol))
 for arm in ['native','nocell','constant','shuffle']:
  for seed in [1,2]:
   out=r/'evaluation'/f'{arm}_s{seed}';out.mkdir(parents=True);modes=['native','mean']+[f'marginal_{i}' for i in range(4)] if arm=='shuffle' else ['native'];rows=[]
   for ds,n in [('Set5',5),('Set14',14),('BSD100',100),('Urban100',100)]:
    for img in range(n):
     for s in protocol['scales']:
      for mode in modes:rows.append(dict(kind='benchmark',arm=arm,seed=seed,dataset=ds,image=str(img),scale=s,eval_mode=mode,y_psnr=30+(0 if arm=='native' else -.1)+img*.001+seed*.01+s*.001))
   for img in range(869,901):
    for s in protocol['scales']:
     for mode in modes:
      for k in range(s*4):rows.append(dict(kind='recovery',arm=arm,seed=seed,dataset='DIV2K-fresh',image=f'{img:04d}',scale=s,eval_mode=mode,lo=k/4,hi=(k+1)/4,target=1.,error=.1 if arm=='native' else .11))
   with gzip.open(out/'rows.jsonl.gz','wt') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
   (out/'COMPLETE.json').write_text(json.dumps({'rows':len(rows),'modes':modes,'rows_sha256':hashlib.sha256((out/'rows.jsonl.gz').read_bytes()).hexdigest(),'frozen_sha256':hashlib.sha256((r/'FROZEN.sha256').read_bytes()).hexdigest()}))
 result=subprocess.run([sys.executable,str(here/'score.py'),'--run-root',str(r)],capture_output=True,text=True);assert result.returncode==0,result.stderr
 summary=json.loads((r/'scores/summary.json').read_text());assert all(x['pass'] for x in summary['predictions'].values());checks['scoring_pass_case']=summary['predictions']
 # Duplicate row must fail rather than silently replace the dictionary entry.
 out=r/'evaluation'/'native_s1';path=out/'rows.jsonl.gz'
 with gzip.open(path,'rt') as f:one=f.readline()
 with gzip.open(path,'at') as f:f.write(one)
 c=json.loads((out/'COMPLETE.json').read_text());c['rows']+=1;c['rows_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(out/'COMPLETE.json').write_text(json.dumps(c))
 bad=subprocess.run([sys.executable,str(here/'score.py'),'--run-root',str(r)],capture_output=True,text=True);assert bad.returncode!=0 and 'Duplicate rows' in bad.stderr;checks['duplicate_rejected']=True
Path(a.output).write_text(json.dumps({'status':'pass','scope':'CPU synthetic checks only; zero trained checkpoints or image data read, zero GPU inference/training','constant_cell':cell_source.CONSTANT_CELL,'validated_script_sha256':{name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ['cell_source.py','score.py','protocol.json','validate_synthetic.py']},'checks':checks},indent=2)+'\n')
print('Synthetic checks passed')
