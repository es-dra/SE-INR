"""Equal-cell-step inference. Each checkpoint is an immutable, independently logged shard."""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
STAT_NAMES = ['J', 'D', 'A', 'V', 'E']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def spectral_stats(target, native, changed, scale, shave=4):
    """No model calls. Rectangular RGB Parseval sums include DC and Nyquist."""
    target, native, changed = [np.asarray(x, dtype=np.float64)[shave:-shave, shave:-shave]
                                for x in (target, native, changed)]
    assert target.shape == native.shape == changed.shape and target.shape[-1] == 3
    error, delta = target - native, changed - native
    spatial = np.array([np.mean(delta**2), np.mean((error-delta)**2)-np.mean(error**2),
                        np.mean(error*delta), np.mean(error**2), np.mean(target**2)])
    ft, fe, fd = [np.fft.fft2(x, axes=(0, 1)) for x in (target, error, delta)]
    n = target.shape[0] * target.shape[1]
    J = np.sum(np.abs(fd)**2, axis=2) / (3 * n*n)
    A = np.sum((fe.conj()*fd).real, axis=2) / (3 * n*n)
    V = np.sum(np.abs(fe)**2, axis=2) / (3 * n*n)
    E = np.sum(np.abs(ft)**2, axis=2) / (3 * n*n)
    spectrum = np.stack([J, J-2*A, A, V, E], axis=-1)
    radius = np.maximum(np.abs(np.fft.fftfreq(target.shape[0]))[:, None],
                        np.abs(np.fft.fftfreq(target.shape[1]))[None, :])
    positive = radius > 0
    rbin = np.minimum(np.floor(radius*32).astype(int), 16)
    ubin = np.minimum(np.floor(2*scale*radius*4).astype(int), 4*scale-1)
    def pooled(indices, count):
        return np.stack([np.bincount(indices[positive], weights=spectrum[..., k][positive],
                                    minlength=count) for k in range(len(STAT_NAMES))], axis=-1)
    rb, ub, dc = pooled(rbin, 17), pooled(ubin, 4*scale), spectrum[0, 0]
    total = spectrum.sum(axis=(0, 1))
    np.testing.assert_allclose(total, spatial, atol=2e-12, rtol=2e-9)
    np.testing.assert_allclose(rb.sum(axis=0)+dc, spatial, atol=2e-12, rtol=2e-9)
    np.testing.assert_allclose(ub.sum(axis=0)+dc, spatial, atol=2e-12, rtol=2e-9)
    return {'total': spatial.tolist(), 'dc': dc.tolist(), 'r_bins': rb.tolist(),
            'u_bins': ub.tolist(), 'parseval_max_abs': float(np.max(np.abs(total-spatial)))}


def condition(scale, h):
    multiplier = 1 + h*scale/2
    c0, c1 = 2/scale, 2/scale+h
    assert multiplier > 0 and c1 > 0
    return {'scale': scale, 'h': h, 'multiplier': multiplier,
            'decoder_cell_native': c0, 'decoder_cell_perturbed': c1,
            'native_cell_in_training_interval': .5 <= c0 <= 2,
            'perturbed_cell_in_training_interval': .5 <= c1 <= 2}


def worker(args, protocol, manifest):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT / 'src'))
    import models
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = ROOT / 'outputs' / f'{args.model}-seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    try:
        checkpoint = manifest['checkpoints'][f'{args.model}:seed{args.seed}']
        assert sha(checkpoint['path']) == checkpoint['sha256'], 'Checkpoint changed after freeze'
        ck = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
        assert ck['epoch'] == protocol['checkpoint_epoch']
        model = models.make(ck['model'], load_sd=True).cuda().eval()
        metadata = {'model': args.model, 'seed': args.seed,
                    'started_at_utc': datetime.now(timezone.utc).isoformat(),
                    'device_name': torch.cuda.get_device_name(0),
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'cuda_device_order': os.environ['CUDA_DEVICE_ORDER'],
                    'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                    'frozen_sha256': sha(ROOT / 'FROZEN.sha256'), 'checkpoint': checkpoint}
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        count, max_parseval, max_cell_error = 0, 0., 0.
        with torch.no_grad(), gzip.open(out / 'records.jsonl.gz', 'wt') as stream:
            for ds, images in manifest['images'].items():
                for image in images:
                    assert sha(image['path']) == image['sha256'], 'Image changed after freeze'
                    H, W = image['crop_hw']
                    with Image.open(image['path']) as raw:
                        target = np.asarray(raw.convert('RGB').crop((0, 0, W, H)), dtype=np.float32)/255
                    tensor = torch.from_numpy(target.transpose(2, 0, 1).copy())[None]
                    yy, xx = np.meshgrid((np.arange(H)+.5)/H, (np.arange(W)+.5)/W, indexing='ij')
                    coordinates = torch.from_numpy((np.stack([yy, xx], axis=-1).reshape(-1, 2)*2-1).astype(np.float32))
                    for scale in protocol['scales']:
                        lr = F.interpolate(tensor, size=(H//scale, W//scale), mode='bicubic',
                                           align_corners=False, antialias=True).clamp(0, 1)
                        model.gen_feat((lr*2-1).cuda())
                        assert tuple(model.feat.shape[-2:]) == (H//scale, W//scale)
                        def predict(h):
                            c = condition(scale, h)
                            external = torch.tensor([2/H, 2/W], dtype=torch.float32) * c['multiplier']
                            observed = external.numpy()*np.array([H//scale, W//scale]) - 2/scale
                            assert np.max(np.abs(observed-h)) < 2e-7, (scale, h, observed)
                            cell_error = float(np.max(np.abs(observed-h)))
                            chunks = []
                            for lo in range(0, len(coordinates), protocol['inference']['query_chunk']):
                                q = coordinates[lo:lo+protocol['inference']['query_chunk']][None].cuda()
                                cell = external.to(q.device).reshape(1, 1, 2).expand_as(q)
                                chunks.append(model.query_rgb(q, cell)[0].float().cpu().numpy())
                            p = np.concatenate(chunks, axis=0).reshape(H, W, 3)*.5+.5
                            assert np.isfinite(p).all()
                            return p, cell_error
                        native, cell_error = predict(0.)
                        max_cell_error = max(max_cell_error, cell_error)
                        for h in protocol['steps']:
                            changed, cell_error = predict(h)
                            record = {'dataset': ds, 'image': image['image'], 'model': args.model,
                                      'seed': args.seed, 'hr_hw': [H, W], **condition(scale, h),
                                      **spectral_stats(target, native, changed, scale,
                                                       protocol['preprocessing']['shave'])}
                            stream.write(json.dumps(record, separators=(',', ':'))+'\n')
                            max_parseval = max(max_parseval, record['parseval_max_abs'])
                            max_cell_error = max(max_cell_error, cell_error)
                            count += 1
                            del changed
                        stream.flush()
                        del native
                    print(json.dumps({'model': args.model, 'seed': args.seed, 'dataset': ds,
                                      'image': image['image'], 'records': count,
                                      'seconds': round(time.time()-start, 1)}), flush=True)
        expected = sum(x['count'] for x in protocol['datasets'].values()) * len(protocol['scales']) * len(protocol['steps'])
        assert count == expected
        complete = {'records': count, 'seconds': time.time()-start, 'training_runs': 0,
                    'parseval_max_abs': max_parseval, 'actual_decoder_step_max_error': max_cell_error,
                    'records_sha256': sha(out / 'records.jsonl.gz'),
                    'frozen_sha256': sha(ROOT / 'FROZEN.sha256')}
        (out / 'COMPLETE.json').write_text(json.dumps(complete, indent=2)+'\n')
        print(json.dumps(complete), flush=True)
    except Exception:
        (out / 'FAILED.json').write_text(json.dumps({'seconds': time.time()-start, 'error': traceback.format_exc()}, indent=2)+'\n')
        raise


def orchestrate(args, protocol):
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=False)
    devices = args.gpus.split(',')
    assert devices and len(devices) == len(set(devices))
    jobs = [(m, s) for m in protocol['models'] for s in protocol['seeds']]
    start = time.time()
    active, finished = {}, []
    receipt = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'frozen_sha256': sha(ROOT / 'FROZEN.sha256'), 'gpus': devices,
               'cuda_device_order': os.environ['CUDA_DEVICE_ORDER'], 'shard_timeout_hours': 12}
    (ROOT / 'RUN_STARTED.json').write_text(json.dumps(receipt, indent=2)+'\n')
    failure = None
    try:
        while jobs or active:
            for gpu in devices:
                if gpu not in active and jobs and failure is None:
                    model, seed = jobs.pop(0)
                    log = (ROOT / f'{model}-seed{seed}.log').open('x')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                    proc = subprocess.Popen([sys.executable, str(ROOT/'run.py'), '--model', model,
                                             '--seed', str(seed)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = (proc, time.time(), log, model, seed)
                    print(f'Start {model} seed {seed} on GPU {gpu}', flush=True)
            for gpu, (proc, ts, log, model, seed) in list(active.items()):
                code = proc.poll()
                if code is None and time.time()-ts > 12*3600:
                    proc.terminate()
                    failure = f'{model} seed {seed} exceeded 12-hour shard budget'
                if code is not None:
                    log.close()
                    del active[gpu]
                    finished.append({'model': model, 'seed': seed, 'gpu': gpu, 'returncode': code})
                    print(f'Finish {model} seed {seed}, exit={code}', flush=True)
                    if code:
                        failure = f'{model} seed {seed} failed with exit {code}'
            if failure:
                raise RuntimeError(failure)
            if active:
                time.sleep(10)
        (ROOT / 'INFERENCE_COMPLETE.json').write_text(json.dumps({'seconds': time.time()-start,
                    'shards': finished, 'frozen_sha256': receipt['frozen_sha256']}, indent=2)+'\n')
    except Exception:
        for proc, _, log, _, _ in active.values():
            proc.terminate()
            log.close()
        (ROOT / 'FAILED.json').write_text(json.dumps({'error': traceback.format_exc(),
                             'finished': finished, 'seconds': time.time()-start}, indent=2)+'\n')
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--orchestrate', action='store_true')
    args = ap.parse_args()
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    manifest = json.loads((ROOT / 'input_manifest.json').read_text())
    if args.orchestrate:
        orchestrate(args, protocol)
    else:
        assert args.model in protocol['models'] and args.seed in protocol['seeds']
        worker(args, protocol, manifest)


if __name__ == '__main__':
    main()
