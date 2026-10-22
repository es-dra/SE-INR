#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
GPU_IDS="${1:-0,1,2,3}"
sha256sum --check FROZEN.sha256
python3 validate.py
python3 run.py --orchestrate --gpus "$GPU_IDS"
sha256sum --check FROZEN.sha256
python3 score.py --run-root .
