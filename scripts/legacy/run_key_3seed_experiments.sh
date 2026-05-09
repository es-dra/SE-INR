#!/usr/bin/env bash
set -euo pipefail

# Run remaining 3-seed experiments for the key paper models.
# This launcher starts one background pipeline per model/GPU. Each pipeline runs
# seed2 training -> seed2 benchmark eval -> seed3 training -> seed3 benchmark eval.
# It intentionally reuses the existing training/evaluation entry points.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/seeds results/seeds save-seeds/seed2 save-seeds/seed3

# Selected GPUs based on nvidia-smi snapshot at launch time:
#   GPU 3: RTX 4090, essentially idle, primary choice.
#   GPU 2: RTX 4090, low memory occupancy, secondary choice.
#   GPU 1: A800 80GB, high utilization but large free memory; selected over GPU 0
#          because GPU 0 had higher memory occupancy.
# If the machine state changes, edit these assignments before launching.

run_model_pipeline() {
  local gpu="$1"
  local model_slug="$2"
  local config="$3"
  local model_name="$4"
  local seed2="$5"
  local seed3="$6"

  local log_file="logs/seeds/${model_slug}_seed2_seed3_gpu${gpu}.nohup.log"
  echo "[LAUNCH] ${model_slug}: GPU=${gpu}, seeds=${seed2},${seed3}, log=${log_file}"

  nohup bash -c "
    set -euo pipefail
    cd '$ROOT_DIR'
    echo '[INFO] pipeline model=${model_slug} gpu=${gpu} seeds=${seed2},${seed3}'
    echo '[INFO] start seed2 train: ' \\$(date -Is)
    python train.py --config '$config' --name '$model_slug' --saveFolder './save-seeds/seed2' --device '$gpu' --seed '$seed2'
    echo '[INFO] start seed2 eval: ' \\$(date -Is)
    python eval_full.py --device '$gpu' --save_root './save-seeds/seed2' --models '$model_name' --output './results/seeds/benchmark_seed2.json' --skip_existing
    echo '[INFO] start seed3 train: ' \\$(date -Is)
    python train.py --config '$config' --name '$model_slug' --saveFolder './save-seeds/seed3' --device '$gpu' --seed '$seed3'
    echo '[INFO] start seed3 eval: ' \\$(date -Is)
    python eval_full.py --device '$gpu' --save_root './save-seeds/seed3' --models '$model_name' --output './results/seeds/benchmark_seed3.json' --skip_existing
    echo '[INFO] pipeline done: ' \\$(date -Is)
  " > "$log_file" 2>&1 &

  echo "$!" > "logs/seeds/${model_slug}_seed2_seed3_gpu${gpu}.pid"
  echo "[PID] ${model_slug}: $(cat "logs/seeds/${model_slug}_seed2_seed3_gpu${gpu}.pid")"
}

run_model_pipeline 3 liif configs/train-div2k/train-liif.yaml LIIF 2 3
run_model_pipeline 2 lte configs/train-div2k/train-lte.yaml LTE 2 3
run_model_pipeline 1 sc-inr-adaptive configs/train-div2k/train-sc-inr-adaptive.yaml SC-INR 2 3

printf '\n[INFO] Active seed pipeline PIDs:\n'
cat logs/seeds/*_seed2_seed3_gpu*.pid
