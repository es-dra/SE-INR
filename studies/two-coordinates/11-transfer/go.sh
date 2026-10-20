#!/bin/bash
cd /workspace/SE-INR/claude-20261020-transfer-v11
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
python3 transfer_v11.py > run.log 2>&1; echo done > DONE
