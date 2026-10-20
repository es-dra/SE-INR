#!/bin/bash
R=/workspace/SE-INR/claude-20261021-gaussdeg-v12; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python3 entrypoints/train.py --config train-gauss-$1.yaml --name liif-gauss-$1 --saveFolder $R/save --device $2 --seed 1 > train_$1.log 2>&1
CUDA_VISIBLE_DEVICES=$2 python3 run_v3_gauss.py --out eval_$1 --src $R/src --ckpt save/liif-gauss-$1/epoch-last.pth --name liif > eval_$1.log 2>&1
echo done > DONE_$1
if [ -f DONE_edsr ] && [ -f DONE_rdn ] && [ -f DONE_nocell ] && [ ! -f score_v12.json ]; then python3 score_v12.py > score_v12.out 2>&1; fi
