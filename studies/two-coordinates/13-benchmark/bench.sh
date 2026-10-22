#!/bin/bash
# usage: bench.sh <name> <ckpt> <gpu>
R=/workspace/SE-INR/runs/claude-20261022-benchmark-v13; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for f in cfg/*.yaml; do
  t=$(basename $f .yaml)
  v=$(cd /workspace/SE-INR/SE-INR-current && python3 $R/test_crop.py --config $R/$f --model $2 --device $3 2>/dev/null | grep -o "result: [0-9.]*" | awk '{print $2}')
  echo "$1 $t $v" >> results.txt
done
echo done > DONE_$1
