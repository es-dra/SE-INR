#!/usr/bin/env bash
set -euo pipefail
TASK=/workspace/SE-INR/codex-20261008-paper-continuation
CODE=$TASK/guidance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-675adc8d-a6ea-1c8b-f833-f8ae8a87cdaa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
run() {
  nice -n 10 python3 -u -c 'import torch,runpy;torch.cuda.set_per_process_memory_fraction(.15);runpy.run_path("/workspace/SE-INR/codex-20261008-paper-continuation/guidance/experiment.py",run_name="__main__")' "$@"
}
export PYTHONPATH=$CODE
ROOT=$TASK/guidance-development-full-v4
run calibrate --output "$ROOT"
run prepare --output "$ROOT" --image 0801 --noise .01
run select --output "$ROOT" --image 0801 --noise .01
date -u > "$ROOT/NUMERICAL_SMOKE_COMPLETE"
ROOT=$TASK/guidance-development-mini-v4
run calibrate --output "$ROOT" --mini
for image in 0801 0802; do
  for noise in .005 .01 .02; do
    run prepare --output "$ROOT" --mini --image "$image" --noise "$noise"
    run select --output "$ROOT" --mini --image "$image" --noise "$noise"
  done
done
run freeze-selections --output "$ROOT" --mini --split development
for image in 0801 0802; do
  for noise in .005 .01 .02; do
    run score --output "$ROOT" --mini --image "$image" --noise "$noise"
  done
done
python3 "$CODE/summarize.py" --output "$ROOT" --split development
date -u > "$ROOT/COMPLETE"
