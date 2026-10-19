#!/bin/bash
cd /workspace/SE-INR/claude-20261004-e1; export CUDA_DEVICE_ORDER=PCI_BUS_ID
( export CUDA_VISIBLE_DEVICES=0; python3 e1ff.py --T 16 --skip 0 --n 4 --noise 0.0 --out out/ff_T16_n0.0_s0.json > out/ff_T16_n0.0_s0.log 2>&1 ) &
( export CUDA_VISIBLE_DEVICES=1; python3 e1ff.py --T 16 --skip 4 --n 4 --noise 0.005 --out out/ff_T16_n0.005_s4.json > out/ff_T16_n0.005_s4.log 2>&1 ) &
( export CUDA_VISIBLE_DEVICES=0; python3 e1ff.py --T 16 --skip 4 --n 4 --noise 0.02 --out out/ff_T16_n0.02_s4.json > out/ff_T16_n0.02_s4.log 2>&1 ) &
wait; echo done > out/RERUN_DONE
