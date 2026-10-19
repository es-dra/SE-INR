#!/bin/bash
# L1/L2 formal inference. GPU1 not used; GPU2/3 shared with seed2/3 training (small, batch=1); Thera on CPU at nice 19.
cd /workspace/SE-INR/claude-20261005-L12; export CUDA_DEVICE_ORDER=PCI_BUS_ID; mkdir -p out logs
( CUDA_VISIBLE_DEVICES=3 python3 l1.py --group inr --outdir out > logs/l1_inr.log 2>&1; CUDA_VISIBLE_DEVICES=3 python3 l2.py --outdir out > logs/l2.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=2 python3 l1.py --group gsasr --outdir out > logs/l1_gsasr.log 2>&1; CUDA_VISIBLE_DEVICES=2 python3 l1.py --group contsr --outdir out > logs/l1_contsr.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES= nice -n 19 /workspace/SE-INR/claude-20260919/thera/venv312/bin/python l1.py --group thera --outdir out > logs/l1_thera.log 2>&1 ) &
wait; date -u > out/ALLDONE
