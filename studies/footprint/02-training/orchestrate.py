#!/usr/bin/env python3
"""Train the stage-02 arms and benchmark each checkpoint, N jobs at a time over the named GPUs. Resumable.

  python3 orchestrate.py --run-root /workspace/SE-INR/runs/claude-20261102-footprint-train-v1 --gpus 1,2 \
      [--arms lte-200ep,sc-inr-200ep,...] [--seeds 1,2] [--dry-run]

Layout under run-root: code/ (frozen copy of the repository), save/<arm>-s<seed>/, logs/, bench.jsonl, DONE_<arm>-s<seed>.
Each job: entrypoints/train.py (200 epochs) -> tools/benchmark.py on epoch-last.pth. A finished job is skipped on rerun.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_ARMS = ['lte-200ep', 'sc-inr-200ep', 'liif-asym-200ep', 'lte-asym-200ep', 'sc-inr-nosinc-200ep',
                'sc-inr-exact-200ep', 'lte-linear-200ep']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', required=True)
    ap.add_argument('--gpus', required=True)
    ap.add_argument('--arms', default=','.join(DEFAULT_ARMS))
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--data-root', default='/workspace/SE-INR/Data')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = Path(args.run_root).resolve()
    code = root / 'code'
    if not code.exists():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO, code, symlinks=True,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'save', 'runs', 'Data', 'paper'))
        (root / 'FREEZE.json').write_text(json.dumps({
            'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip(),
            'arms': args.arms.split(','), 'seeds': args.seeds.split(',')}, indent=2) + '\n')
    data_link = code / 'Data'
    if not data_link.exists():
        os.symlink(args.data_root, data_link)
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'save').mkdir(exist_ok=True)
    jobs = [(a, s) for a in args.arms.split(',') for s in args.seeds.split(',')]
    jobs = [(a, s) for a, s in jobs if not (root / f'DONE_{a}-s{s}').exists()]
    gpus = args.gpus.split(',')
    print(f'{len(jobs)} jobs on GPUs {gpus}: {jobs}', flush=True)
    if args.dry_run:
        return
    env_base = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONUNBUFFERED='1')
    active = {}
    while jobs or active:
        for gpu in gpus:
            if gpu in active or not jobs:
                continue
            arm, seed = jobs.pop(0)
            name = f'{arm}-s{seed}'
            script = (f"set -e; cd {code}; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
                      f"python3 entrypoints/train.py --config configs/train-div2k/footprint/train-{arm}.yaml "
                      f"--name {name} --saveFolder {root}/save --device {gpu} --seed {seed}; "
                      f"CUDA_VISIBLE_DEVICES={gpu} python3 tools/benchmark.py --ckpt {root}/save/{name}/epoch-last.pth "
                      f"--name {name} --data-root {args.data_root} --datasets set5 set14 bsd100 urban100 div2k "
                      f"--scales 2 3 4 6 8 12 --out {root}/bench.jsonl; echo done > {root}/DONE_{name}")
            log = open(root / 'logs' / f'{name}.log', 'a')
            proc = subprocess.Popen(['bash', '-c', script], env=env_base, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (proc, name, log, time.time())
            print(f'{datetime.now().isoformat(timespec="seconds")} start {name} on GPU {gpu}', flush=True)
        for gpu, (proc, name, log, t0) in list(active.items()):
            code_ = proc.poll()
            if code_ is not None:
                log.close()
                del active[gpu]
                print(f'{datetime.now().isoformat(timespec="seconds")} finish {name} exit={code_} '
                      f'after {(time.time() - t0) / 3600:.1f} h', flush=True)
                if code_:
                    (root / f'FAILED_{name}').write_text(f'exit {code_}\n')
        if active:
            time.sleep(60)
    print('all jobs finished', flush=True)


if __name__ == '__main__':
    main()
