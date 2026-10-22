#!/bin/bash
cd /workspace/SE-INR/claude-20261021-gaussdeg-v12
python3 run_v3_gauss_cpu.py --out eval_rdn --src $PWD/src --ckpt save/liif-gauss-rdn/epoch-last.pth --name liif > eval_rdn.log 2>&1
python3 score_v12.py > score_v12.out 2>&1; echo done > SCORED
