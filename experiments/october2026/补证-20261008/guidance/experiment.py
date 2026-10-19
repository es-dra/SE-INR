#!/usr/bin/env python3
"""Phase-separated, resumable guidance experiment. No execution on import."""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from core import (ARMS, Geometry, Operator, NumericalFailure, Measurement,
                  frame_params, sampling_points, spectrum, gp_mean,
                  cropped_mse, calibrated_scale, random_field, noise_like, sync)

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(path.read_text())


class BudgetExpired(RuntimeError):
    pass


class Budget:
    """One process, absolute 24-hour wall deadline, measured GPU phase time.

    Pauses consume the absolute wall budget. A resume never extends it.
    Resources ledger is atomic and preserved after incomplete runs.
    """
    def __init__(self, root, device, hours=24):
        self.path = root / 'resources.json'
        self.device = device
        self.state = read_json(self.path) if self.path.exists() else dict(
            started_unix=time.time(), gpu_phase_seconds=0., invocations=[], budget_hours=hours)
        self.deadline = self.state['started_unix'] + self.state['budget_hours'] * 3600
        self.start = time.monotonic()
        self.old_gpu = self.state['gpu_phase_seconds']
        self.last_save = 0.
        self.check()

    def save(self):
        active = time.monotonic() - self.start
        self.state['gpu_phase_seconds'] = self.old_gpu + (active if self.device.startswith('cuda') else 0.)
        self.state['last_updated_unix'] = time.time()
        atomic_json(self.path, self.state)
        self.last_save = time.monotonic()

    def check(self):
        active = time.monotonic() - self.start
        gpu = self.old_gpu + (active if self.device.startswith('cuda') else 0.)
        if time.time() > self.deadline or gpu > self.state['budget_hours'] * 3600:
            self.save()
            raise BudgetExpired('Frozen wall/GPU budget exhausted; incomplete cases remain incomplete')
        if time.monotonic() - self.last_save > 30:
            self.save()


@contextlib.contextmanager
def exclusive(root):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / '.process.lock', 'a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns this experiment output; only one GPU process is permitted')
        yield


def freeze(root, mini):
    record = dict(protocol_sha256=sha(HERE / 'protocol.json'),
                  core_sha256=sha(HERE / 'core.py'), experiment_sha256=sha(HERE / 'experiment.py'),
                  summary_sha256=sha(HERE / 'summarize.py'), torch_version=torch.__version__,
                  mode='development-miniature' if mini else 'formal',
                  geometry=dict(hr=32, scale=4, frames=16, subsamples=2, field=64, shave=2)
                  if mini else Geometry().__dict__)
    path = root / 'frozen.json'
    if path.exists() and read_json(path) != record:
        raise RuntimeError('Code/protocol/mode changed; existing task cannot be silently resumed')
    if not path.exists():
        atomic_json(path, record)
    return Geometry(**record['geometry'])


def case_id(image, noise):
    return f'{image}_n{noise:g}'.replace('.', 'p')


def image_tensor(path, g):
    # E1's central 2HR native crop followed by 2x pixel averaging.
    im = Image.open(path).convert('RGB')
    w, h = im.size
    if min(w, h) < 2 * g.hr:
        raise ValueError(f'Image too small for native crop: {path}')
    left, top = (w - 2*g.hr)//2, (h - 2*g.hr)//2
    data = np.asarray(im.crop((left, top, left+2*g.hr, top+2*g.hr)), dtype=np.float32) / 255.
    return F.avg_pool2d(torch.from_numpy(data).permute(2,0,1)[None], 2)[0]


def calibrate(args, root, g, budget):
    dest = root / 'calibration.npz'
    if dest.exists() and (root / 'calibration_sources.json').exists():
        print('calibration exists; resume skip', flush=True)
        return
    total = torch.zeros(g.field, g.field, dtype=torch.float64)
    sources = []
    for i in range(1, 33):
        budget.check()
        path = args.data_root / 'DIV2K_train_HR' / f'{i:04d}.png'
        x = image_tensor(path, g).double()
        x -= x.mean((1, 2), keepdim=True)
        z = torch.zeros(3, g.field, g.field, dtype=torch.float64)
        z[:, :g.hr, :g.hr] = x
        p = torch.fft.fft2(z).abs().square().mean(0)
        p[0,0] = 0
        p /= p.sum()
        total += p
        sources.append(dict(image=f'{i:04d}', path=str(path), sha256=sha(path)))
    total /= 32
    total *= g.field ** 2 / total.sum()
    atomic_npz(dest, unit_spectrum=total.numpy())
    atomic_json(root / 'calibration_sources.json', dict(sources=sources, spectrum_sha256=sha(dest)))


def prepare(args, root, g, budget):
    image, noise = args.image, args.noise
    if args.mini and image not in ('0801', '0802'):
        raise ValueError('Miniature development is restricted to 0801/0802')
    if image not in [f'{i:04d}' for i in range(801, 813)] + [f'{i:04d}' for i in range(813, 825)]:
        raise ValueError('Image outside frozen development/target IDs')
    cid = case_id(image, noise)
    obs = root / 'observations' / (cid + '.npz')
    truth = root / 'evaluation_truth' / (cid + '.npz')
    if obs.exists() and truth.exists() and (root / 'observations' / (cid+'.json')).exists():
        print('case exists; resume skip', flush=True)
        return
    path = args.data_root / 'DIV2K_valid_HR' / (image + '.png')
    x = image_tensor(path, g).to(device=args.device, dtype=torch.float64)
    d, a = frame_params(image, g)
    op = Operator(g, sampling_points(d, a, g, args.device, torch.float64))
    field = torch.zeros(3, g.field, g.field, device=args.device, dtype=torch.float64)
    field[:, :g.hr, :g.hr] = x
    with Measurement(op, args.device) as meter:
        budget.check()
        noiseless = op.forward(field)
        y = noiseless + noise * noise_like(noiseless, 1001)
    atomic_npz(obs, y=y.cpu().numpy(), displacements=d, angles=a,
               noise=np.asarray(noise), geometry_json=np.asarray(json.dumps(g.__dict__)))
    atomic_npz(truth, x=x.cpu().numpy())
    atomic_json(root / 'observations' / (cid+'.json'), dict(
        image=image, noise=noise, noise_seed=1001, matched_operator=True,
        source_image_sha256=sha(path), observation_sha256=sha(obs), truth_sha256=sha(truth),
        preparation=meter.record, selection_may_read_truth=False))


def observation_only(root, cid, g, device):
    """The only target-data reader used by select. Never loads evaluation_truth."""
    with np.load(root / 'observations' / (cid+'.npz'), allow_pickle=False) as f:
        if json.loads(str(f['geometry_json'])) != g.__dict__:
            raise ValueError('Observation geometry differs from frozen geometry')
        y = torch.as_tensor(f['y'], device=device, dtype=torch.float64)
        d, a, noise = f['displacements'].copy(), f['angles'].copy(), float(f['noise'])
    with np.load(root / 'calibration.npz', allow_pickle=False) as f:
        unit = torch.as_tensor(f['unit_spectrum'], device=device, dtype=y.dtype)
    return y, Operator(g, sampling_points(d, a, g, device, y.dtype)), unit, noise


def select(args, root, g, budget):
    cid = case_id(args.image, args.noise)
    path = root / 'selections' / (cid+'.json')
    state = read_json(path) if path.exists() else dict(case=cid, status='in_progress',
        scores={}, cv={}, reconstructions={}, failures=[], started_unix=time.time())
    if state['status'] == 'complete':
        print('selection complete; resume skip', flush=True)
        return
    if state['failures'] and not args.retry_failed:
        raise RuntimeError('Failure recorded; use --retry-failed to retain history and explicitly retry')
    y, op, unit, noise = observation_only(root, cid, g, args.device)
    if noise != args.noise:
        raise ValueError('CLI noise differs from observation')
    started_attempt = time.monotonic()
    state.setdefault('attempts', [])
    try:
        with Measurement(op, args.device) as common_meter:
            energy_fields = [random_field(unit, 6000+r, args.device, y.dtype) for r in range(4)]
            energy_y = [op.forward(f) for f in energy_fields]
            del energy_fields
            scale, scale_details = calibrated_scale(y, energy_y, noise)
        with Measurement(op, args.device) as risk_input_meter:
            fields = [random_field(unit, 7000+r, args.device, y.dtype) for r in range(4)]
            unit_y = [op.forward(f) for f in fields]
        state.setdefault('common_attempts', []).append(common_meter.record)
        state['common'] = dict(cost=common_meter.record, scale=scale, details=scale_details,
                               risk_input_cost=risk_input_meter.record,
                               reference_sha256=sha(root/'calibration.npz'))
        atomic_json(path, state)
        for arm in ARMS:
            if arm in state['scores']:
                continue
            costs, risks, numerics = [], [], []
            ps = spectrum(arm, scale, g, args.device, y.dtype)
            for r in range(4):
                with Measurement(op, args.device) as meter:
                    budget.check()
                    f = fields[r] * math.sqrt(scale)
                    yr = unit_y[r] * math.sqrt(scale) + noise * noise_like(unit_y[r], 8000+r)
                    xh, info = gp_mean(yr, op, ps, noise, budget_check=budget.check)
                    risks.append(cropped_mse(xh, f[:, :g.hr, :g.hr], g))
                costs.append(meter.record)
                numerics.append(info)
            state['scores'][arm] = dict(mean=float(np.mean(risks)), probe_risks=risks,
                                       cost=costs, numerics=numerics)
            atomic_json(path, state)
        for arm in ARMS:
            if arm in state['cv']:
                continue
            risks, costs, numerics = [], [], []
            for fold in range(4):
                validation = np.arange(g.frames)[np.arange(g.frames)%4 == fold]
                training = np.arange(g.frames)[np.arange(g.frames)%4 != fold]
                train_y = y[training]
                train_scale, _ = calibrated_scale(train_y, [v[training] for v in energy_y], noise)
                ps = spectrum(arm, train_scale, g, args.device, y.dtype)
                with Measurement(op, args.device) as meter:
                    budget.check()
                    xh, info = gp_mean(train_y, op, ps, noise, training, budget_check=budget.check)
                    field = torch.zeros(3,g.field,g.field,device=args.device,dtype=y.dtype)
                    field[:,:g.hr,:g.hr] = xh
                    risks.append(float(((op.forward(field, validation)-y[validation])**2).mean()))
                costs.append(meter.record)
                numerics.append(info)
            state['cv'][arm] = dict(mean=float(np.mean(risks)), fold_risks=risks,
                                   cost=costs, numerics=numerics)
            atomic_json(path, state)
        # Stable candidate order breaks exact ties. No HR is available here.
        chosen = min(ARMS, key=lambda a: state['scores'][a]['mean'])
        cv = min(ARMS, key=lambda a: state['cv'][a]['mean'])
        state['chosen'] = dict(analysis=chosen, cv=cv, default='power_2')
        for arm in dict.fromkeys(state['chosen'].values()):
            if arm in state['reconstructions']:
                continue
            with Measurement(op, args.device) as meter:
                xh, info = gp_mean(y, op, spectrum(arm,scale,g,args.device,y.dtype), noise,
                                   budget_check=budget.check)
            dest = root / 'reconstructions' / (cid+'_'+arm+'.npz')
            atomic_npz(dest, x=xh.cpu().numpy())
            state['reconstructions'][arm] = dict(path=str(dest.relative_to(root)), sha256=sha(dest),
                                                cost=meter.record, numerics=info)
            atomic_json(path,state)
        state.update(status='complete', completed_unix=time.time(),
                     observations_sha256=sha(root/'observations'/(cid+'.npz')),
                     frozen_sha256=sha(root/'frozen.json'))
        atomic_json(path,state)
    except Exception as e:
        state['status'] = 'budget_incomplete' if isinstance(e,BudgetExpired) else 'failed'
        state['failures'].append(dict(type=type(e).__name__, message=str(e), unix=time.time(),
                                      traceback=traceback.format_exc()))
        atomic_json(path,state)
        raise
    finally:
        sync(args.device)
        state['attempts'].append(dict(seconds=time.monotonic()-started_attempt,counts=op.counts(),status=state['status']))
        atomic_json(path,state)


def expected_cases(split):
    images = range(813,825) if split == 'target' else range(801,803)
    return [case_id(f'{i:04d}',n) for i in images for n in (.005,.01,.02)]


def freeze_selections(args,root,g,budget):
    if args.mini and args.split != 'development':
        raise ValueError('Miniature results cannot freeze target selection')
    frozen = {}
    for cid in expected_cases(args.split):
        budget.check()
        p=root/'selections'/(cid+'.json')
        if not p.exists() or read_json(p)['status'] != 'complete':
            raise RuntimeError(f'Cannot evaluate incomplete selection cohort: {cid}')
        state=read_json(p)
        if state['frozen_sha256'] != sha(root/'frozen.json'):
            raise RuntimeError(f'Selection belongs to a different freeze: {cid}')
        if state['common']['reference_sha256'] != sha(root/'calibration.npz'):
            raise RuntimeError(f'Calibration changed after selection: {cid}')
        if state['observations_sha256'] != sha(root/'observations'/(cid+'.npz')):
            raise RuntimeError(f'Observations changed after selection: {cid}')
        for arm,item in state['reconstructions'].items():
            if sha(root/item['path']) != item['sha256']:
                raise RuntimeError(f'Reconstruction altered before freeze: {cid}/{arm}')
        frozen[cid]=sha(p)
    dest=root/('selection_cohort_'+args.split+'.json')
    content=dict(split=args.split, cases=frozen, frozen_sha256=sha(root/'frozen.json'))
    if dest.exists() and read_json(dest) != content:
        raise RuntimeError('Selection cohort changed after freeze')
    atomic_json(dest,content)


def score(args,root,g,budget):
    cid=case_id(args.image,args.noise)
    split='target' if int(args.image)>=813 else 'development'
    cohort_path=root/('selection_cohort_'+split+'.json')
    if not cohort_path.exists():
        raise RuntimeError('Must freeze the entire declared cohort before any HR scoring')
    cohort=read_json(cohort_path)
    selection_path=root/'selections'/(cid+'.json')
    if cid not in cohort['cases'] or sha(selection_path)!=cohort['cases'][cid]:
        raise RuntimeError('Selection not in frozen cohort or altered')
    output=root/'evaluation'/(cid+'.json')
    state=read_json(output) if output.exists() else dict(case=cid,status='in_progress',candidates={},failures=[])
    if state['status']=='complete':
        print('evaluation complete; resume skip',flush=True)
        return
    if state['failures'] and not args.retry_failed:
        raise RuntimeError('Evaluation failure recorded; explicit --retry-failed required')
    selection=read_json(selection_path)
    y,op,unit,noise=observation_only(root,cid,g,args.device)
    truth_path=root/'evaluation_truth'/(cid+'.npz')
    observation_meta=read_json(root/'observations'/(cid+'.json'))
    if sha(truth_path)!=observation_meta['truth_sha256']:
        raise RuntimeError('Evaluation truth changed since matched observation preparation')
    with np.load(truth_path,allow_pickle=False) as f:
        truth=torch.as_tensor(f['x'],device=args.device,dtype=y.dtype)
    scale=selection['common']['scale']
    try:
        for arm in ARMS:
            if arm in state['candidates']:
                continue
            cache=selection['reconstructions'].get(arm)
            if cache:
                if sha(root/cache['path'])!=cache['sha256']:
                    raise RuntimeError('Frozen reconstruction checksum failed')
                with np.load(root/cache['path'],allow_pickle=False) as f:
                    xh=torch.as_tensor(f['x'],device=args.device,dtype=y.dtype)
                cost=dict(reused_frozen_output=True,seconds=0.)
                info=cache['numerics']
            else:
                with Measurement(op,args.device) as meter:
                    xh,info=gp_mean(y,op,spectrum(arm,scale,g,args.device,y.dtype),noise,
                                    budget_check=budget.check)
                cost=meter.record
            mse=cropped_mse(xh,truth,g)
            state['candidates'][arm]=dict(mse=mse,psnr=-10*math.log10(max(mse,1e-30)),cost=cost,numerics=info)
            atomic_json(output,state)
        oracle=min(ARMS,key=lambda a:state['candidates'][a]['mse'])
        chosen=selection['chosen']
        default_psnr=state['candidates'][chosen['default']]['psnr']
        oracle_psnr=state['candidates'][oracle]['psnr']
        state.update(status='complete', oracle=oracle, chosen=chosen,
                     selection_cohort_sha256=sha(cohort_path),truth_sha256=sha(truth_path),
                     metrics={method:dict(psnr=state['candidates'][arm]['psnr'],
                        regret_db=oracle_psnr-state['candidates'][arm]['psnr'],
                        improvement_over_default_db=state['candidates'][arm]['psnr']-default_psnr,
                        severe_degradation=state['candidates'][arm]['psnr']-default_psnr < -1.)
                        for method,arm in chosen.items()})
        atomic_json(output,state)
    except Exception as e:
        state['status']='budget_incomplete' if isinstance(e,BudgetExpired) else 'failed'
        state['failures'].append(dict(type=type(e).__name__,message=str(e),unix=time.time()))
        atomic_json(output,state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('calibrate','prepare','select','freeze-selections','score'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('/workspace/SE-INR/Data'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--image',default='0801')
    parser.add_argument('--noise',type=float,choices=(.005,.01,.02),default=.01)
    parser.add_argument('--split',choices=('development','target'),default='target')
    parser.add_argument('--mini',action='store_true',help='32-pixel geometry, development 0801/0802 only')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if args.mini and int(args.image)>=813:
        parser.error('Miniature mode may never access target cases')
    root=args.output.resolve()
    with exclusive(root):
        g=freeze(root,args.mini)
        budget=Budget(root,args.device)
        started=time.time()
        try:
            if args.device.startswith('cuda'):
                if not torch.cuda.is_available():
                    raise RuntimeError('Requested CUDA is unavailable')
                torch.cuda.set_device(args.device)
                # grid_sample backward is nondeterministic on CUDA; disclose, do not promise bitwise repeatability.
            torch.manual_seed(0)
            torch.set_num_threads(4)
            globals()[args.phase.replace('-','_')](args,root,g,budget)
        finally:
            sync(args.device) if torch.cuda.is_available() else None
            budget.state['invocations'].append(dict(phase=args.phase,image=args.image,noise=args.noise,
                                                   started_unix=started,ended_unix=time.time(),device=args.device))
            budget.save()


if __name__=='__main__':
    main()
