#!/usr/bin/env python3
"""Freeze protocol, code and input provenance into a new run root before inference.

  python3 freeze.py --run-root /workspace/SE-INR/runs/claude-20261101-footprint-v1
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MODCROP = 24
FILES = ['protocol.json', 'run.py', 'score.py', 'freeze.py', 'launch.sh']


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', required=True)
    ap.add_argument('--source-root', default=str(HERE.parents[2] / 'src'))
    ap.add_argument('--data-root', default='/workspace/SE-INR/Data')
    ap.add_argument('--seeds', default='1,2,3')
    args = ap.parse_args()
    root = Path(args.run_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f'Refusing to freeze into a non-empty directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(HERE / name, root / name)
    shutil.copytree(args.source_root, root / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    protocol = json.loads((root / 'protocol.json').read_text())

    manifest = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'source_root': str(Path(args.source_root).resolve()),
                'data_root': args.data_root, 'downscale_side': protocol['conditions']['downscale']['side'],
                'images': {}, 'checkpoints': {}, 'pixels': 0}
    side = manifest['downscale_side']
    for ds, spec in protocol['datasets'].items():
        files = sorted(Path(args.data_root, spec['root']).glob('*.png'))
        if 'first' in spec:
            files = [f for f in files if spec['first'] <= int(f.stem) <= spec['last']]
        assert files, (ds, spec)
        rows = []
        for path in files:
            with Image.open(path) as im:
                w, h = im.size
            H, W = h // MODCROP * MODCROP, w // MODCROP * MODCROP
            assert min(H, W) >= side, (path, H, W)
            rows.append({'path': str(path), 'image': path.stem, 'sha256': sha(path), 'original_hw': [h, w], 'crop_hw': [H, W]})
            manifest['pixels'] += H * W
        manifest['images'][ds] = rows

    import torch
    seeds_present = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        paths = {m: Path(protocol['checkpoint_root'].format(seed=seed, model=m)) for m in protocol['models']}
        if all(p.exists() for p in paths.values()):
            seeds_present.append(seed)
            for m, p in paths.items():
                ck = torch.load(p, map_location='cpu', weights_only=True)
                assert ck['epoch'] == protocol['checkpoint_epoch'], (p, ck['epoch'])
                n = int(sum(v.numel() for v in ck['model']['sd'].values()))
                manifest['checkpoints'][f'{m}:seed{seed}'] = {'path': str(p), 'sha256': sha(p), 'registry_name': ck['model']['name'],
                                                              'epoch': ck['epoch'], 'parameters': n}
    assert seeds_present, 'no complete seed found'
    manifest['seeds'] = seeds_present
    (root / 'input_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')

    lines = []
    for name in FILES + ['input_manifest.json']:
        lines.append(f'{sha(root / name)}  {name}')
    for path in sorted((root / 'src').rglob('*.py')):
        lines.append(f'{sha(path)}  {path.relative_to(root)}')
    (root / 'FROZEN.sha256').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'run_root': str(root), 'seeds': seeds_present, 'images': {k: len(v) for k, v in manifest['images'].items()},
                      'checkpoints': list(manifest['checkpoints']), 'frozen_sha256': sha(root / 'FROZEN.sha256')}, indent=1))


if __name__ == '__main__':
    main()
