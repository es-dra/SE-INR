#!/bin/bash
cd /workspace/SE-INR/claude-20261005-f1; export CUDA_DEVICE_ORDER=PCI_BUS_ID
( CUDA_VISIBLE_DEVICES=0 python3 f1_h6c.py --scale 2 --out $PWD/out/f1_x2.json > out/f1_x2.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=2 python3 f1_h6c.py --scale 3 --out $PWD/out/f1_x3.json > out/f1_x3.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=3 python3 f1_h6c.py --scale 4 --out $PWD/out/f1_x4.json > out/f1_x4.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=0 python3 f1_h6c.py --scale 8 --out $PWD/out/f1_x8.json > out/f1_x8.log 2>&1 ) &
wait; echo done > out/ALLDONE
