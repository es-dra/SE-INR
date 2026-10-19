#!/bin/bash
cd /workspace/SE-INR/claude-20261005-d11; export CUDA_VISIBLE_DEVICES="" NTH=8
cat jobs.txt | xargs -P 32 -L 1 bash -c 'python3 d11gp.py --noise $0 --skip $1 --n 1 --out /workspace/SE-INR/claude-20261005-d11/out/d11_n$0_i$1.json > /workspace/SE-INR/claude-20261005-d11/out/d11_n$0_i$1.log 2>&1'
echo done > out/ALLDONE
