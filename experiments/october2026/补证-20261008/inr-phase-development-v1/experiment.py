"""Bounded FF point-sampling development. Original top-level code never runs."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp, path)


def extract(source):
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ('FF', 'gauss_B')]
    assert len(nodes) == 2 and {n.name for n in nodes} == {'FF', 'gauss_B'}
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source)+'::AST-subset', 'exec')


def interrupted(signum, frame):
    raise RuntimeError('interrupted by signal '+str(signum))


def run(args):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    p = json.loads((HERE/'protocol.json').read_text())
    source = args.original_source or HERE/'original-e1ff.py'
    source_sha = sha(source)
    if source_sha != sha(HERE/'original-e1ff.py'):
        raise RuntimeError('Original Claude source differs from archived bytes')
    scope = dict(torch=torch, nn=nn, math=math)
    exec(extract(source), scope)
    FF, gauss_B = scope['FF'], scope['gauss_B']
    root = args.root
    data = args.data_root
    if (data/'DIV2K_valid_HR').is_dir(): data = data/'DIV2K_valid_HR'
    inputs = {im:dict(path=str(data/(im+'.png')), sha256=sha(data/(im+'.png'))) for im in p['cohort']['images']}
    identities = dict(protocol_sha256=sha(HERE/'protocol.json'), source_sha256=sha(Path(__file__)), original_source_sha256=source_sha, data=inputs)
    started = time.time()
    if root.exists():
        if not args.resume: raise RuntimeError('Existing root requires explicit --resume')
        frozen = json.loads((root/'frozen.json').read_text())
        if frozen['identities'] != identities: raise RuntimeError('Frozen source/data changed; use a new version/root')
        resource = json.loads((root/'resources.json').read_text())
        if args.deadline_unix is not None and args.deadline_unix > frozen['deadline_unix']:
            raise RuntimeError('Resume cannot extend deadline')
    else:
        root.mkdir(parents=True)
        frozen = dict(identities=identities, started_unix=started, deadline_unix=min(started+7200, args.deadline_unix or float('inf')))
        save_json(root/'frozen.json', frozen)
        resource = dict(gpu_phase_seconds=0., invocations=[])
        save_json(root/'resources.json', resource)
    if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable; no implicit CPU training fallback')
    os.nice(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0)))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.cuda.set_device(args.visible_index)
    torch.cuda.set_per_process_memory_fraction(.15, args.visible_index)
    device = torch.device('cuda', args.visible_index)
    phase_start = time.monotonic()
    old_phase_seconds = resource['gpu_phase_seconds']
    current = None
    completed = []
    def checkpoint():
        if time.time() >= frozen['deadline_unix'] or old_phase_seconds+time.monotonic()-phase_start >= 7200:
            raise TimeoutError('Frozen 2h absolute/cumulative CUDA-phase budget exhausted')
    def partial(state, model, opt, scheduler):
        torch.cuda.synchronize(device)
        dest = root/'fits'/state['id']
        dest.mkdir(parents=True, exist_ok=True)
        payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), steps=state['steps'], history=state['history'])
        tmp = dest/'.checkpoint.tmp'
        torch.save(payload, tmp); os.replace(tmp, dest/'checkpoint.pth')
        save_json(dest/'state.json', state)
    try:
        checkpoint()
        properties = torch.cuda.get_device_properties(device)
        save_json(root/'device.json', dict(name=properties.name, uuid=str(getattr(properties,'uuid','unknown')), total_memory=properties.total_memory, visible_index=args.visible_index, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), memory_fraction=.15))
        for im in p['cohort']['images']:
            checkpoint()
            with Image.open(data/(im+'.png')) as image:
                image=image.convert('RGB'); w,h=image.size
                if min(w,h)<128: raise RuntimeError('Native image smaller than128')
                box=((w-128)//2,(h-128)//2,(w-128)//2+128,(h-128)//2+128)
                native=np.asarray(image.crop(box),dtype=np.float32)/255.
            truth=F.avg_pool2d(torch.from_numpy(native).permute(2,0,1)[None],2)[0]
            for phases in p['cohort']['phase_sets']:
                geometry=im+'_ph'+''.join(map(str,phases)); obsdir=root/'observations';obsdir.mkdir(exist_ok=True)
                xy=[];values=[]
                for phase in phases:
                    yy,xx=torch.meshgrid(torch.arange(0,64,4),torch.arange(phase,64,4),indexing='ij')
                    xy.append(torch.stack([xx,yy],-1).reshape(-1,2))
                    values.append(truth[:,yy,xx].permute(1,2,0).reshape(-1,3))
                xy=torch.cat(xy);clean=torch.cat(values)
                y=clean+.01*torch.randn(clean.shape,generator=torch.Generator().manual_seed(1001))
                coords=(xy.float()+.5)/64
                obsfile=obsdir/(geometry+'.npz')
                if obsfile.exists():
                    with np.load(obsfile,allow_pickle=False) as old:
                        assert np.array_equal(old['y'],y.numpy()) and np.array_equal(old['coords'],coords.numpy()) and np.array_equal(old['truth'],truth.numpy())
                else: np.savez_compressed(obsfile,y=y.numpy(),clean=clean.numpy(),coords=coords.numpy(),xy=xy.numpy(),truth=truth.numpy())
                save_json(obsdir/(geometry+'.json'),dict(image=im,phases=phases,crop=box,source=inputs[im],observation_sha256=sha(obsfile),n_observations=len(y),seed=1001))
                v,y_gpu=coords.to(device),y.to(device)
                for sigma in p['model']['sigma']:
                    checkpoint()
                    cid=geometry+'_sigma'+str(sigma);dest=root/'fits'/cid
                    old_state=json.loads((dest/'state.json').read_text()) if (dest/'state.json').exists() else None
                    if old_state and old_state['status']=='complete': completed.append(old_state);continue
                    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
                    model=FF(gauss_B(sigma)).to(device)
                    opt=torch.optim.Adam(model.parameters(),lr=.001)
                    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2000,eta_min=.00005)
                    state=dict(id=cid,image=im,phases=phases,sigma=sigma,steps=0,history=[],status='in_progress',observation_sha256=sha(obsfile),attempts=old_state.get('attempts',[]) if old_state else [],failures=old_state.get('failures',[]) if old_state else [])
                    if old_state:
                        payload=torch.load(dest/'checkpoint.pth',map_location=device,weights_only=False)
                        model.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
                        state.update(steps=payload['steps'],history=payload['history'])
                        if old_state['observation_sha256']!=sha(obsfile):raise RuntimeError('Observations changed')
                    fit_started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
                    current=(state,model,opt,scheduler,fit_started)
                    model.train()
                    for step in range(state['steps'],2000):
                        checkpoint();opt.zero_grad(set_to_none=True)
                        loss=(model(v)-y_gpu).square().mean()
                        if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite observation loss')
                        loss.backward()
                        if any(not torch.isfinite(t.grad).all().item() for t in model.parameters() if t.grad is not None):raise RuntimeError('Nonfinite gradient')
                        opt.step();scheduler.step()
                        if any(not torch.isfinite(t).all().item() for t in model.parameters()):raise RuntimeError('Nonfinite weight')
                        state['steps']=step+1;state['history'].append(float(loss.item()))
                        if (step+1)%250==0: partial(state,model,opt,scheduler)
                    checkpoint();model.eval()
                    with torch.no_grad():
                        yy,xx=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
                        grid=torch.stack([(xx+.5)/64,(yy+.5)/64],-1).reshape(-1,2).to(device)
                        pred=torch.cat([model(c) for c in grid.split(1024)]).T.reshape(3,64,64).cpu()
                        observation_loss=float((model(v)-y_gpu).square().mean().item())
                    if not torch.isfinite(pred).all().item():raise RuntimeError('Nonfinite full prediction')
                    clamped=pred.clamp(0,1)
                    metrics={}
                    for name,estimate in [('unclamped',pred),('clamped',clamped)]:
                        mse=float((estimate[:,4:-4,4:-4]-truth[:,4:-4,4:-4]).square().mean().item())
                        metrics[name]=dict(mse=mse,psnr=-10*math.log10(max(mse,1e-30)))
                    dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/'predictions.npz',unclamped=pred.numpy(),clamped=clamped.numpy())
                    torch.cuda.synchronize(device)
                    state.update(status='complete',observation_loss=observation_loss,metrics=metrics,predictions_sha256=sha(dest/'predictions.npz'),peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(device))
                    state['attempts'].append(dict(wall_seconds=time.monotonic()-fit_started,status='complete'))
                    partial(state,model,opt,scheduler);completed.append(state);current=None
                    print('FIT_COMPLETE',cid,metrics,flush=True)
        save_json(root/'summary.json',dict(status='complete',n_fits=len(completed),expected_fits=12,development_only=True,fits=completed))
        (root/'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')
    except Exception as error:
        if current:
            state,model,opt,scheduler,fit_started=current
            state.update(status='budget_incomplete' if isinstance(error,TimeoutError) else 'failed',failure=dict(type=type(error).__name__,message=str(error)))
            state['failures'].append(state['failure'])
            state['attempts'].append(dict(wall_seconds=time.monotonic()-fit_started,status=state['status']))
            partial(state,model,opt,scheduler)
        save_json(root/'INCOMPLETE.json',dict(type=type(error).__name__,message=str(error),completed_fits=len(completed),development_only=True))
        raise
    finally:
        torch.cuda.synchronize(device)
        elapsed=time.monotonic()-phase_start
        resource['gpu_phase_seconds']=old_phase_seconds+elapsed
        resource['invocations'].append(dict(started_unix=started,ended_unix=time.time(),gpu_phase_wall_seconds=elapsed,full_wall_seconds=time.time()-started))
        save_json(root/'resources.json',resource)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--original-source',type=Path)
    parser.add_argument('--visible-index','--gpu-index-or-visible',type=int,default=0)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        extract(args.original_source or HERE/'original-e1ff.py')
        json.loads((HERE/'protocol.json').read_text())
        print('AST/protocol check passed; original top-level code not executed; no torch/CUDA loaded')
    else:
        if args.root is None or args.data_root is None:parser.error('--root and --data-root required')
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        try:
            run(args)
        except Exception as error:
            if args.root.exists() and not (args.root/'INCOMPLETE.json').exists():
                save_json(args.root/'INCOMPLETE.json',dict(type=type(error).__name__,message=str(error),stage='setup_or_checkpoint_failure',development_only=True))
            raise
