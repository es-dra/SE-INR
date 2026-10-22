"""Use the unchanged, frozen trainer; register the cell-only subclass first."""
import argparse, hashlib, json, os, resource, runpy, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--run-root', required=True); p.add_argument('--arm', choices=['constant','shuffle'], required=True); p.add_argument('--seed', type=int, choices=[1,2], required=True); a=p.parse_args()
r=Path(a.run_root).resolve(); src=r/'source'; os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
assert len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==1, 'Exactly one visible GPU per arm'
sys.path[:0]=[str(src),str(src/'entrypoints'),str(r)]
import cell_source
out=r/'save'/f'{a.arm}_s{a.seed}'
if out.exists(): raise FileExistsError(f'Never overwrite/retry an existing arm: {out}')
config=r/'configs'/f'{a.arm}_s{a.seed}.yaml'
os.environ['CELL_SOURCE_AUDIT_PATH']=str(r/'logs'/f'cell_audit_{a.arm}_s{a.seed}.json')
import utils
original_set_save_path=utils.set_save_path
start=time.monotonic()
def set_save_path(path):
    original_log=original_set_save_path(path)
    def log(message, *args, **kwargs):
        original_log(message, *args, **kwargs)
        if str(message).startswith('epoch '):
            record={'epoch_message':str(message),'elapsed_seconds':time.monotonic()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            cmd=['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=uuid,name,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits']
            try:
                check=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
                record['gpu_csv']=check.stdout.strip();record['gpu_query_exit']=check.returncode
            except Exception as exc:
                record['gpu_query_error']=type(exc).__name__+': '+str(exc)
            with open(r/'logs'/f'resources_{a.arm}_s{a.seed}.jsonl','a') as f:f.write(json.dumps(record)+'\n')
    return log
utils.set_save_path=set_save_path
sys.argv=[str(src/'entrypoints/train.py'),'--config',str(config),'--name',out.name,'--saveFolder',str(r/'save'),'--seed',str(a.seed)]
t=time.time(); runpy.run_path(str(src/'entrypoints/train.py'),run_name='__main__')
ck=out/'epoch-last.pth'
import torch
c=torch.load(ck,map_location='cpu',weights_only=False); assert c['epoch']==200
(out/'COMPLETE.json').write_text(json.dumps({'epoch':200,'seconds':time.time()-t,'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()},indent=2)+'\n')
