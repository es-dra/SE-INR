#!/bin/bash
# usage: arm.sh <arm> <config> <gpu>   train seed 2 then evaluate; writes DONE_<arm>
R=/workspace/SE-INR/claude-20261020-seed2-v8; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python3 entrypoints/train.py --config $2 --name liif-$1-seed2 --saveFolder $R/save --device $3 --seed 2 > train_$1.log 2>&1
CUDA_VISIBLE_DEVICES=$3 python3 run_v3.py --out eval_$1 --src $R/src --ckpt save/liif-$1-seed2/epoch-last.pth --name liif > eval_$1.log 2>&1
echo done > DONE_$1
if ls DONE_edsr DONE_rdn DONE_tinydec DONE_nocell >/dev/null 2>&1 && [ ! -f score_v8.json ]; then python3 score_v8.py > score_v8.out 2>&1; fi
