"""Freeze protocol, prediction, implementation and input provenance before inference."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sha(p):
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', required=True)
    args = ap.parse_args()
    root = Path(args.run_root).resolve()
    if (root / 'FROZEN.sha256').exists() or (root / 'outputs').exists():
        raise RuntimeError('Refusing to overwrite an existing freeze or run')
    src = Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    names = ['protocol.json', 'predictions.json', 'run.py', 'score.py', 'freeze.py', 'launch.sh', 'validate.py']
    for name in names:
        if (src / name).resolve() != (root / name).resolve():
            shutil.copy2(src / name, root / name)
    protocol = json.loads((root / 'protocol.json').read_text())
    shutil.copytree(protocol['source_root'], root / 'src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'images': {}, 'checkpoints': {}, 'full_image_pixels': 0}
    for ds, spec in protocol['datasets'].items():
        files = sorted(Path(spec['root']).glob('*.png'))
        if 'first' in spec:
            files = [p for p in files if spec['first'] <= int(p.stem) <= spec['last']]
        assert len(files) == spec['count'], (ds, len(files))
        manifest['images'][ds] = []
        for path in files:
            with Image.open(path) as image:
                w, h = image.size
            H, W = h // 12 * 12, w // 12 * 12
            assert min(H, W) > 8
            manifest['images'][ds].append({'path': str(path), 'image': path.stem,
                                           'sha256': sha(path), 'original_hw': [h, w], 'crop_hw': [H, W]})
            manifest['full_image_pixels'] += H * W
    for model in protocol['models']:
        for seed in protocol['seeds']:
            path = Path(protocol['checkpoint_template'].format(model=model, seed=seed))
            manifest['checkpoints'][f'{model}:seed{seed}'] = {'path': str(path), 'sha256': sha(path)}
    (root / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files = [root / n for n in names + ['input_manifest.json']]
    files.extend(sorted((root / 'src').rglob('*.py')))
    (root / 'FROZEN.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in files))
    print(json.dumps({'run_root': str(root), 'frozen_files': len(files),
                      'freeze_sha256': sha(root / 'FROZEN.sha256'),
                      'images': sum(len(x) for x in manifest['images'].values()),
                      'full_image_pixels': manifest['full_image_pixels']}, indent=2))


if __name__ == '__main__':
    main()
