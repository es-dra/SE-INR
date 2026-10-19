#!/bin/bash
# F1 seed补充：按原协议（configs/train-div2k 原配置，1000 epoch）只改 --seed 与保存目录。用户 2026-10-04T21:42Z 确认。
cd /workspace/SE-INR/SC-ASISR; export CUDA_DEVICE_ORDER=PCI_BUS_ID; export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
P=.venv/bin/python
run(){ nohup $P entrypoints/train.py --config configs/train-div2k/train-$1.yaml --seed $2 --device $3 --saveFolder /workspace/SE-INR/claude-20261005-f1s/ckpt/seed$2 > /workspace/SE-INR/claude-20261005-f1s/logs/$1_seed$2.log 2>&1 & }
run sc-inr-nosinc 2 0
run sc-inr-nosinc 3 2
run lte-nocellphase 2 3
run lte-nocellphase 3 0
# --- resume after reboot (2026-10-06T00:48:05Z) ---
# nohup .venv/bin/python entrypoints/train.py --config /workspace/SE-INR/claude-20261005-f1s/resume_cfg/train-sc-inr-nosinc-seed2.yaml --name sc-inr-nosinc --seed 2 --device 2 --saveFolder /workspace/SE-INR/claude-20261005-f1s/ckpt/seed2
# nohup .venv/bin/python entrypoints/train.py --config /workspace/SE-INR/claude-20261005-f1s/resume_cfg/train-lte-nocellphase-seed3.yaml --name lte-nocellphase --seed 3 --device 3 --saveFolder /workspace/SE-INR/claude-20261005-f1s/ckpt/seed3
