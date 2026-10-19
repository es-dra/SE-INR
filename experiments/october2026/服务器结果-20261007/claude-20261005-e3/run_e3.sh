#!/bin/bash
cd /workspace/SE-INR/claude-20261005-e3; export CUDA_DEVICE_ORDER=PCI_BUS_ID
( CUDA_VISIBLE_DEVICES=0 python3 e3_h6c.py --scale 2 --out /workspace/SE-INR/claude-20261005-e3/out/e3_x2.json > out/e3_x2.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=2 python3 e3_h6c.py --scale 3 --out /workspace/SE-INR/claude-20261005-e3/out/e3_x3.json > out/e3_x3.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=3 python3 e3_h6c.py --scale 8 --out /workspace/SE-INR/claude-20261005-e3/out/e3_x8.json > out/e3_x8.log 2>&1 ) &
wait; echo done > out/ALLDONE
