#!/bin/bash
# Server-side: wait for epoch 200, then evaluate and score with the frozen v5 scorer (S1-S3 = T1-T3).
cd /workspace/SE-INR/claude-20261020-swinir-v7
until grep -q "epoch 200/200" save/liif-swinir-200ep/log.txt; do sleep 600; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python3 run_v3_crop.py --out eval200_swinir --src $PWD/src --ckpt save/liif-swinir-200ep/epoch-last.pth --name liif > eval200_swinir.log 2>&1
mkdir -p score200 && cd score200 && ln -sfn /workspace/SE-INR/claude-20261019-encoder-train-v5/eval_edsr eval_edsr && ln -sfn ../eval200_swinir eval_rdn
python3 /workspace/SE-INR/claude-20261019-encoder-train-v5/score_v5.py > score_epoch200.json 2>&1
echo done > ../FINISHED200
