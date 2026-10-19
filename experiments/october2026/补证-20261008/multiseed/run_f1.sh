#!/usr/bin/env bash
set -euo pipefail
TASK=/workspace/SE-INR/codex-20261008-paper-continuation
SOURCE=/workspace/SE-INR/claude-20261005-f1s
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-4fd8e39c-ecb7-159e-b1f9-930c3066676e
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TASK/g1/out" "$TASK/logs"
date -u > "$TASK/g1/STARTED"
sha256sum "$SOURCE/f1s_h6c.py" "$SOURCE/ana_g1.py" > "$TASK/g1/source.sha256"
for seed in 2 3; do
  mkdir -p "$TASK/g1/ckpt/seed$seed" "$TASK/g1/out/seed$seed"
  for model in lte sc-inr; do
    ln -sfn "/workspace/SE-INR/SC-ASISR/artifacts/checkpoints/seed$seed/$model" "$TASK/g1/ckpt/seed$seed/$model"
  done
  for model in lte-nocellphase sc-inr-nosinc; do
    ln -sfn "$SOURCE/ckpt/seed$seed/$model" "$TASK/g1/ckpt/seed$seed/$model"
  done
  for scale in 2 3 4 8; do
    timeout 1800 nice -n 10 python3 -u -c 'import torch,runpy;torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35);runpy.run_path("/workspace/SE-INR/claude-20261005-f1s/f1s_h6c.py",run_name="__main__")' --ckroot "$TASK/g1/ckpt/seed$seed" --models lte lte-nocellphase sc-inr sc-inr-nosinc --scale "$scale" --out "$TASK/g1/out/seed$seed/f1s_x$scale.json" > "$TASK/logs/g1_s${seed}_x${scale}.log" 2>&1
  done
done
cd "$TASK/g1"
python3 "$SOURCE/ana_g1.py" > "$TASK/logs/g1_analysis.log" 2>&1
date -u > "$TASK/g1/COMPLETE"
