#!/bin/bash
MODEL_PATH=$1
MODEL_NAME=$2
BASE=/workspace/SE-INR/Equivariant-ASISR
cd $BASE

BENCHMARKS=(Set5 Set14 BSD100 Urban100)
SCALES=(2 3 4)

echo "=== MODEL: $MODEL_NAME ==="

for bench in "${BENCHMARKS[@]}"; do
  for scale in "${SCALES[@]}"; do
    config="configs/test/test-${bench,,}-${scale}.yaml"
    result=$(python test.py --config "$config" --model "$MODEL_PATH" --device cpu 2>&1 | grep "^result:" | awk '{print $2}')
    if [ -n "$result" ]; then
      echo "$MODEL_NAME | ${bench} x${scale} | PSNR = ${result} dB"
    else
      echo "$MODEL_NAME | ${bench} x${scale} | PSNR = ERR"
    fi
  done
done
