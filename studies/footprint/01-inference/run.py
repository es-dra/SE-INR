#!/usr/bin/env python3
"""Footprint-extrapolation inference on existing checkpoints. No training.

Per image and checkpoint: symmetric scales (references, large scales, consistency
sources), asymmetric scale pairs under three cell policies, cross-scale consistency
and a downscaling condition. Each checkpoint is one shard that writes an immutable
records.jsonl.gz; --orchestrate runs the shards over the named GPUs.

  python3 run.py --orchestrate --gpus 0,1          # from a frozen run root
  python3 run.py --model liif --seed 1             # one shard (CUDA_VISIBLE_DEVICES set by the orchestrator)
  python3 run.py --smoke /tmp/footprint-smoke      # CPU self-test with random models and synthetic images
"""
import argparse
import gzip
import hashlib
import json
import math
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
MODCROP = 24
Y_COEFFS = (65.738 / 256, 129.057 / 256, 25.064 / 256)
U_EDGES_2D = (0.0, 0.5, 1.0, 1.5, 2.0)
U_BIN, U_MAX = 0.25, 4.0


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fnum(x):
    """Float formatting that keeps 1.5 as 1.5 and 2 as 2 in condition names."""
    return str(int(x)) if float(x).is_integer() else str(x)


class Engine:
    """Device-agnostic prediction and measurement helpers around a loaded model."""

    def __init__(self, protocol, device):
        import torch
        self.torch = torch
        self.device = device
        self.chunk = protocol['inference']['query_chunk']

    # ----- geometry -----
    def coords(self, H, W):
        t = self.torch
        ys = ((t.arange(H, dtype=t.float32) + .5) / H) * 2 - 1
        xs = ((t.arange(W, dtype=t.float32) + .5) / W) * 2 - 1
        grid = t.stack(t.meshgrid(ys, xs, indexing='ij'), dim=-1).reshape(-1, 2)
        return grid.to(self.device)

    def degrade(self, hr, size):
        F = self.torch.nn.functional
        return F.interpolate(hr, size=size, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)

    def predict(self, model, H, W, rel_cell):
        """Query the (H, W) grid with decoder cell rel_cell = (c_y, c_x). Returns HxWx3 float32 in image units, unclipped."""
        t = self.torch
        fh, fw = model.feat.shape[-2:]
        external = t.tensor([rel_cell[0] / fh, rel_cell[1] / fw], dtype=t.float32, device=self.device)
        coords = self.coords(H, W)
        out = []
        for lo in range(0, H * W, self.chunk):
            q = coords[lo:lo + self.chunk][None]
            cell = external.view(1, 1, 2).expand_as(q)
            out.append(model.query_rgb(q, cell)[0].float())
        pred = t.cat(out, dim=0).view(H, W, 3) * .5 + .5
        if not t.isfinite(pred).all():
            raise FloatingPointError('non-finite prediction')
        return pred

    # ----- metrics -----
    def psnr_y(self, pred, target, shave):
        t = self.torch
        w = t.tensor(Y_COEFFS, dtype=t.float64, device=self.device)
        p = (pred.clamp(0, 1).double() * w).sum(-1)[shave:-shave, shave:-shave]
        g = (target.double() * w).sum(-1)[shave:-shave, shave:-shave]
        return float(-10 * t.log10(((p - g) ** 2).mean()))

    def psnr_rgb(self, pred, target, shave):
        t = self.torch
        d = (pred.double() - target.double())[shave:-shave, shave:-shave]
        return float(-10 * t.log10((d ** 2).mean()))

    def spectra(self, pred, target, shave):
        """Energy spectra of error and target over the shaved window (float64 FFT); sums equal the spatial mean squares."""
        t = self.torch
        err = (target.double() - pred.double())[shave:-shave, shave:-shave]
        tgt = target.double()[shave:-shave, shave:-shave]
        n = err.shape[0] * err.shape[1]
        V = (t.fft.fft2(err, dim=(0, 1)).abs() ** 2).sum(-1) / (3 * n * n)
        E = (t.fft.fft2(tgt, dim=(0, 1)).abs() ** 2).sum(-1) / (3 * n * n)
        fy = t.fft.fftfreq(err.shape[0], device=self.device).abs()
        fx = t.fft.fftfreq(err.shape[1], device=self.device).abs()
        check = float(abs(V.sum() - (err ** 2).mean()))
        return V, E, fy, fx, check

    def u_profile(self, V, E, fy, fx, sy, sx):
        """1-D bins in u = max(2 s_y |f_y|, 2 s_x |f_x|): width U_BIN up to U_MAX plus overflow; DC separate."""
        t = self.torch
        u = t.maximum(2 * sy * fy[:, None], 2 * sx * fx[None, :])
        nb = int(round(U_MAX / U_BIN))
        idx = t.clamp((u / U_BIN).floor().long(), max=nb)
        dc = (fy[:, None] == 0) & (fx[None, :] == 0)
        idx = idx[~dc]
        v = t.zeros(nb + 1, dtype=t.float64, device=self.device).scatter_add_(0, idx, V[~dc])
        e = t.zeros(nb + 1, dtype=t.float64, device=self.device).scatter_add_(0, idx, E[~dc])
        return {'u_edges': [round(i * U_BIN, 4) for i in range(nb + 1)] + ['inf'],
                'V': v.tolist(), 'E': e.tolist(), 'dc_V': float(V[0, 0]), 'dc_E': float(E[0, 0])}

    def u_grid(self, V, E, fy, fx, sy, sx):
        """5x5 grid over (u_y, u_x) with edges U_EDGES_2D plus overflow; DC excluded."""
        t = self.torch
        edges = t.tensor(U_EDGES_2D[1:], dtype=t.float64, device=self.device)
        iy = t.bucketize(2 * sy * fy, edges, right=True)
        ix = t.bucketize(2 * sx * fx, edges, right=True)
        k = len(U_EDGES_2D)
        idx = (iy[:, None] * k + ix[None, :])
        dc = (fy[:, None] == 0) & (fx[None, :] == 0)
        idx = idx[~dc]
        v = t.zeros(k * k, dtype=t.float64, device=self.device).scatter_add_(0, idx, V[~dc])
        e = t.zeros(k * k, dtype=t.float64, device=self.device).scatter_add_(0, idx, E[~dc])
        return {'edges': list(U_EDGES_2D) + ['inf'], 'V': v.view(k, k).tolist(), 'E': e.view(k, k).tolist()}

    def r_profile(self, V, E, fy, fx, nb=8):
        """Bins in the Chebyshev radius r (cycles per output pixel), width 0.5/nb; DC excluded."""
        t = self.torch
        r = t.maximum(fy[:, None], fx[None, :])
        idx = t.clamp((r * 2 * nb).floor().long(), max=nb - 1)
        dc = (fy[:, None] == 0) & (fx[None, :] == 0)
        idx = idx[~dc]
        v = t.zeros(nb, dtype=t.float64, device=self.device).scatter_add_(0, idx, V[~dc])
        e = t.zeros(nb, dtype=t.float64, device=self.device).scatter_add_(0, idx, E[~dc])
        return {'r_edges': [round(i * 0.5 / nb, 5) for i in range(nb + 1)], 'V': v.tolist(), 'E': e.tolist()}

    @staticmethod
    def box2(x):
        H, W, C = x.shape
        return x.view(H // 2, 2, W // 2, 2, C).mean(dim=(1, 3))


def isize(x, s):
    v = x / s
    assert abs(v - round(v)) < 1e-9, (x, s)
    return int(round(v))


def process_image(engine, model, protocol, target, meta, shave=4):
    """Yield all records for one image and one loaded model."""
    H, W = target.shape[:2]
    hr = target.permute(2, 0, 1)[None]
    cond = protocol['conditions']
    base = {'dataset': meta['dataset'], 'image': meta['image'], 'hr_hw': [H, W]}
    checks = []

    sym_pred = {}
    cons_hi = {float(hi) for hi, lo in cond['consistency_pairs_hi_lo']}
    for s in cond['symmetric_scales']:
        s = float(s)
        lr = engine.degrade(hr, (isize(H, s), isize(W, s)))
        model.gen_feat(lr * 2 - 1)
        pred = engine.predict(model, H, W, (2 / s, 2 / s))
        V, E, fy, fx, chk = engine.spectra(pred, target, shave)
        checks.append(chk)
        yield {**base, 'kind': 'sym', 'scale': s, 'in_support': 1 <= s <= 4,
               'psnr_y': engine.psnr_y(pred, target, math.ceil(s)),
               'mse_rgb': float(((pred.double() - target.double())[shave:-shave, shave:-shave] ** 2).mean()),
               'u': engine.u_profile(V, E, fy, fx, s, s)}
        if s in cons_hi:
            sym_pred[s] = pred
            lo = s / 2
            h2, w2 = H // 2, W // 2
            pred_lo = engine.predict(model, h2, w2, (2 / lo, 2 / lo))
            hi_box = engine.box2(pred)
            t_box = engine.box2(target)
            t_bic = engine.degrade(hr, (h2, w2))[0].permute(1, 2, 0)
            yield {**base, 'kind': 'cons', 'hi': s, 'lo': lo,
                   'psnr_hibox_vs_lo': engine.psnr_rgb(hi_box, pred_lo, shave),
                   'psnr_lo_vs_tbox': engine.psnr_rgb(pred_lo, t_box, shave),
                   'psnr_lo_vs_tbic': engine.psnr_rgb(pred_lo, t_bic, shave),
                   'psnr_hibox_vs_tbox': engine.psnr_rgb(hi_box, t_box, shave),
                   'psnr_hibox_vs_tbic': engine.psnr_rgb(hi_box, t_bic, shave),
                   'psnr_tbic_vs_tbox': engine.psnr_rgb(t_bic, t_box, shave)}
            del pred_lo, hi_box
        del pred

    for sy, sx in cond['asymmetric_pairs_sy_sx']:
        sy, sx = float(sy), float(sx)
        lr = engine.degrade(hr, (isize(H, sy), isize(W, sx)))
        model.gen_feat(lr * 2 - 1)
        g = math.sqrt(sy * sx)
        policies = {'native': (2 / sy, 2 / sx), 'swap': (2 / sx, 2 / sy), 'iso': (2 / g, 2 / g)}
        for name, rel in policies.items():
            pred = engine.predict(model, H, W, rel)
            V, E, fy, fx, chk = engine.spectra(pred, target, shave)
            checks.append(chk)
            yield {**base, 'kind': 'asym', 'sy': sy, 'sx': sx, 'policy': name,
                   'decoder_cell': list(rel), 'in_support': max(sy, sx) <= 4,
                   'psnr_y': engine.psnr_y(pred, target, math.ceil(max(sy, sx))),
                   'mse_rgb': float(((pred.double() - target.double())[shave:-shave, shave:-shave] ** 2).mean()),
                   'u': engine.u_profile(V, E, fy, fx, sy, sx),
                   'grid': engine.u_grid(V, E, fy, fx, sy, sx)}
            del pred

    d = cond['downscale']
    side = int(meta.get('downscale_side') or d['side'])
    top, left = (H - side) // 2, (W - side) // 2
    crop = target[top:top + side, left:left + side]
    inp = crop.permute(2, 0, 1)[None]
    model.gen_feat(inp * 2 - 1)
    half = side // 2
    pred = engine.predict(model, half, half, (d['decoder_cell'], d['decoder_cell']))
    t_box = engine.box2(crop)
    t_bic = engine.degrade(inp, (half, half))[0].permute(1, 2, 0)
    nearest = crop[::2, ::2]
    V, E, fy, fx, chk = engine.spectra(pred, t_box, shave)
    checks.append(chk)
    yield {**base, 'kind': 'down', 'side': side, 'scale': d['output_scale'],
           'psnr_vs_box': engine.psnr_rgb(pred, t_box, shave), 'psnr_vs_bic': engine.psnr_rgb(pred, t_bic, shave),
           'nearest_vs_box': engine.psnr_rgb(nearest, t_box, shave), 'nearest_vs_bic': engine.psnr_rgb(nearest, t_bic, shave),
           'bic_vs_box': engine.psnr_rgb(t_bic, t_box, shave),
           'r': engine.r_profile(V, E, fy, fx)}
    yield {'kind': 'check', 'parseval_max_abs': max(checks)}


def expected_records(protocol):
    c = protocol['conditions']
    return len(c['symmetric_scales']) + len(c['consistency_pairs_hi_lo']) + 3 * len(c['asymmetric_pairs_sy_sx']) + 1


def worker(args, protocol, manifest):
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    import models  # noqa: E402
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
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
        assert ck['epoch'] == protocol['checkpoint_epoch'], ck['epoch']
        for k, v in checkpoint.get('override', {}).items():   # test-time ablation: model args changed, weights unchanged
            ck['model']['args'][k] = v
        model = models.make(ck['model'], load_sd=True).to(device).eval()
        engine = Engine(protocol, device)
        metadata = {'model': args.model, 'seed': args.seed, 'registry_name': ck['model']['name'],
                    'started_at_utc': datetime.now(timezone.utc).isoformat(), 'device': device,
                    'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'torch_version': torch.__version__, 'frozen_sha256': sha(ROOT / 'FROZEN.sha256'),
                    'checkpoint': checkpoint}
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        count, parseval = 0, 0.
        per_image = expected_records(protocol)
        with torch.no_grad(), gzip.open(out / 'records.jsonl.gz', 'wt') as stream:
            for ds, images in manifest['images'].items():
                for image in images:
                    assert sha(image['path']) == image['sha256'], 'Image changed after freeze'
                    H, W = image['crop_hw']
                    with Image.open(image['path']) as raw:
                        arr = np.asarray(raw.convert('RGB').crop((0, 0, W, H)), dtype=np.float32) / 255
                    target = torch.from_numpy(arr.copy()).to(device)
                    meta = {'dataset': ds, 'image': image['image'],
                            'downscale_side': manifest.get('downscale_side')}
                    n = 0
                    for record in process_image(engine, model, protocol, target, meta):
                        if record['kind'] == 'check':
                            parseval = max(parseval, record['parseval_max_abs'])
                            continue
                        record.update(model=args.model, seed=args.seed)
                        stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                        n += 1
                    assert n == per_image, (n, per_image)
                    count += n
                    stream.flush()
                    print(json.dumps({'model': args.model, 'seed': args.seed, 'dataset': ds, 'image': image['image'],
                                      'records': count, 'seconds': round(time.time() - start, 1)}), flush=True)
        expected = sum(len(v) for v in manifest['images'].values()) * per_image
        assert count == expected, (count, expected)
        complete = {'records': count, 'seconds': time.time() - start, 'training_runs': 0,
                    'parseval_max_abs': parseval, 'records_sha256': sha(out / 'records.jsonl.gz'),
                    'frozen_sha256': sha(ROOT / 'FROZEN.sha256')}
        (out / 'COMPLETE.json').write_text(json.dumps(complete, indent=2) + '\n')
        print(json.dumps(complete), flush=True)
    except Exception:
        (out / 'FAILED.json').write_text(json.dumps({'seconds': time.time() - start,
                                                     'error': traceback.format_exc()}, indent=2) + '\n')
        raise


def orchestrate(args, protocol, manifest):
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    (ROOT / 'outputs').mkdir(exist_ok=False)
    devices = args.gpus.split(',')
    assert devices and len(devices) == len(set(devices))
    jobs = [tuple(k.split(':seed')) for k in manifest['checkpoints']]
    hours = protocol['inference']['shard_timeout_hours']
    start = time.time()
    (ROOT / 'RUN_STARTED.json').write_text(json.dumps({
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'frozen_sha256': sha(ROOT / 'FROZEN.sha256'),
        'gpus': devices, 'jobs': jobs, 'shard_timeout_hours': hours}, indent=2) + '\n')
    active, finished, failure = {}, [], None
    while jobs or active:
        for gpu in devices:
            if gpu not in active and jobs and failure is None:
                model, seed = jobs.pop(0)
                log = (ROOT / f'{model}-seed{seed}.log').open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                proc = subprocess.Popen([sys.executable, str(ROOT / 'run.py'), '--model', model, '--seed', seed],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (proc, time.time(), log, model, seed)
                print(f'Start {model} seed {seed} on GPU {gpu}', flush=True)
        for gpu, (proc, ts, log, model, seed) in list(active.items()):
            code = proc.poll()
            if code is None and time.time() - ts > hours * 3600:
                proc.terminate()
                failure = f'{model} seed {seed} exceeded the {hours}-hour shard budget'
            if code is not None:
                log.close()
                del active[gpu]
                finished.append({'model': model, 'seed': seed, 'gpu': gpu, 'returncode': code})
                print(f'Finish {model} seed {seed}, exit={code}', flush=True)
                if code:
                    failure = f'{model} seed {seed} failed with exit {code}'
        if failure and not active:
            (ROOT / 'INFERENCE_FAILED.json').write_text(json.dumps({'failure': failure, 'finished': finished}, indent=2) + '\n')
            raise RuntimeError(failure)
        if active:
            time.sleep(10)
    (ROOT / 'INFERENCE_COMPLETE.json').write_text(json.dumps({
        'seconds': time.time() - start, 'finished': finished, 'frozen_sha256': sha(ROOT / 'FROZEN.sha256')}, indent=2) + '\n')


def smoke(args, protocol):
    """CPU self-test: random models of the three families, two synthetic images, then the full worker path."""
    import torch
    src = ROOT / 'src' if (ROOT / 'src').exists() else ROOT.parents[2] / 'src'
    sys.path.insert(0, str(src))
    import models  # noqa: E402
    root = Path(args.smoke).resolve()
    root.mkdir(parents=True, exist_ok=False)
    enc = {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}
    specs = {
        'liif': {'name': 'liif', 'args': {'encoder_spec': enc, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256] * 4}}}},
        'lte': {'name': 'lte', 'args': {'encoder_spec': enc, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256] * 3}}, 'hidden_dim': 256}},
        'sc-inr': {'name': 'sc_inr_signed_phiz', 'args': {'encoder_spec': enc, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256] * 3}},
                   'hidden_dim': 256, 'num_freqs': 128, 'num_angles': 8, 'freq_min': 0.1, 'freq_max': 2.0, 'omega_bound': 2.1,
                   'learn_phase': True, 'phase_kernel_size': 1, 'phase_bias': False, 'phase_zero_init': True}},
    }
    torch.manual_seed(0)
    (root / 'ckpt').mkdir()
    (root / 'images').mkdir()
    manifest = {'images': {}, 'checkpoints': {}, 'downscale_side': 48, 'smoke': True}
    for name, spec in specs.items():
        m = models.make(spec)
        spec = dict(spec, sd=m.state_dict())
        p = root / 'ckpt' / f'{name}.pth'
        torch.save({'model': spec, 'epoch': protocol['checkpoint_epoch']}, p)
        manifest['checkpoints'][f'{name}:seed1'] = {'path': str(p), 'sha256': sha(p)}
    rng = np.random.default_rng(0)
    for ds, hw in (('A', (96, 72)), ('B', (72, 96))):
        rows = []
        for i in range(2):
            yy, xx = np.mgrid[:hw[0], :hw[1]]
            img = (0.5 + 0.3 * (np.sin(yy / 5 + i) * np.cos(xx / 7))[..., None] + 0.1 * rng.standard_normal((*hw, 3))).clip(0, 1)
            img = (img * 255).astype(np.uint8)
            p = root / 'images' / f'{ds}{i}.png'
            Image.fromarray(img).save(p)
            rows.append({'path': str(p), 'image': f'{ds}{i}', 'sha256': sha(p), 'original_hw': list(hw), 'crop_hw': list(hw)})
        manifest['images'][ds] = rows
    (root / 'input_manifest.json').write_text(json.dumps(manifest, indent=1))
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=1))
    for name in ('run.py', 'score.py'):
        (root / name).write_bytes((ROOT / name).read_bytes())
    os.symlink(src, root / 'src')
    (root / 'FROZEN.sha256').write_text('smoke\n')
    for name in specs:
        code = subprocess.call([sys.executable, str(root / 'run.py'), '--model', name, '--seed', '1', '--device', 'cpu'], cwd=root)
        assert code == 0, name
    code = subprocess.call([sys.executable, str(root / 'score.py'), '--run-root', str(root)], cwd=root)
    assert code == 0
    print('smoke ok:', root)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--seed')
    ap.add_argument('--device', default=None)
    ap.add_argument('--orchestrate', action='store_true')
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--smoke', default=None)
    a = ap.parse_args()
    proto = json.loads((ROOT / 'protocol.json').read_text())
    if a.smoke:
        smoke(a, proto)
    else:
        man = json.loads((ROOT / 'input_manifest.json').read_text())
        if a.orchestrate:
            orchestrate(a, proto, man)
        else:
            worker(a, proto, man)
