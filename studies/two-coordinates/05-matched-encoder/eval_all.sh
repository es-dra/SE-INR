#!/bin/bash
cd /workspace/SE-INR/claude-20261019-encoder-train-v5
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
for a in edsr rdn tinydec nocell; do
  n=$a; [ $a = edsr ] && c=liif-edsr-200ep || c=liif-$a-200ep; [ $a = rdn ] && c=liif-rdn-200ep
  [ -d eval_$a ] || python3 run_v3.py --out eval_$a --src /workspace/SE-INR/claude-20261019-encoder-train-v5/src --ckpt save/$c/epoch-last.pth --name liif > eval_$a.log 2>&1
done
V7=/workspace/SE-INR/claude-20261020-swinir-v7
[ -d $V7/eval50_swinir ] || python3 run_v3.py --out $V7/eval50_swinir --src $V7/src --ckpt $V7/save/liif-swinir-200ep/epoch-50.pth --name liif > $V7/eval50_swinir.log 2>&1
echo done > eval_all.done
