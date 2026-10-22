#!/bin/bash
# Called inside the launcher's single 24h deadline for the entire slot.
set -euo pipefail
R=$1; arm=$2; seed=$3; gpu=$4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
label=${arm}_s${seed}
python3 "$R/train_arm.py" --run-root "$R" --arm "$arm" --seed "$seed" > "$R/logs/train_${label}.log" 2>&1
python3 "$R/evaluate.py" --run-root "$R" --arm "$arm" --seed "$seed" > "$R/logs/eval_${label}.log" 2>&1
if [ "$arm" = constant ]; then baseline=native; else baseline=nocell; fi
python3 "$R/evaluate.py" --run-root "$R" --arm "$baseline" --seed "$seed" > "$R/logs/eval_${baseline}_s${seed}.log" 2>&1
