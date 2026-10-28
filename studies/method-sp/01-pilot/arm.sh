#!/bin/bash
# usage: arm.sh <arm>   train (GPU1) -> benchmark 16 cells -> R(u) probe
R=/workspace/SE-INR/runs/claude-20261029-sp-pilot-v1; cd $R/code
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python3 entrypoints/train.py --config ../train-$1.yaml --name $1 --saveFolder $R/save --device 1 --seed 1 > $R/logs/train_$1.log 2>&1
CK=$R/save/$1/epoch-last.pth
for f in $R/cfg/*.yaml; do
  t=$(basename $f .yaml)
  v=$(python3 entrypoints/test.py --config $f --model $CK --device 1 2>/dev/null | grep -o "result: [0-9.]*" | awk '{print $2}')
  echo "$1 $t $v" >> $R/results.txt
done
CUDA_VISIBLE_DEVICES=1 python3 tools/frequency_probe.py --ckpt $CK --data /workspace/SE-INR/Data/DIV2K_valid_HR --scales 2 3 4 --out $R/probe_$1.json > $R/logs/probe_$1.log 2>&1
echo done > $R/DONE_$1
if [ $(ls $R/DONE_* | wc -l) -eq 5 ] && [ ! -f $R/score.json ]; then python3 $R/score.py > $R/score.out 2>&1; fi
