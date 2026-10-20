#!/bin/bash
cd /workspace/SE-INR/claude-20261020-swinir-v7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec python3 entrypoints/train.py --config train-liif-swinir-200ep.yaml --name liif-swinir-200ep --saveFolder /workspace/SE-INR/claude-20261020-swinir-v7/save --device 0 --seed 1
