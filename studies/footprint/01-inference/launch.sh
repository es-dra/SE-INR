#!/usr/bin/env bash
# Run from a frozen run root: bash launch.sh <comma-separated GPU ids>
set -euo pipefail
cd -- "$(dirname -- "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
GPU_IDS="${1:?usage: bash launch.sh <gpu ids, e.g. 1,2>}"
sha256sum --check --quiet FROZEN.sha256
python3 run.py --orchestrate --gpus "$GPU_IDS"
sha256sum --check --quiet FROZEN.sha256
python3 score.py --run-root .
