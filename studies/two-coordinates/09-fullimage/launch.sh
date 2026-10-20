#!/bin/bash
R=/workspace/SE-INR/claude-20261020-fullimage-v9; cd $R; mkdir -p out logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
SRC=/workspace/SE-INR/runs/claude-20261019-cell-mismatch-v1/src; C=/workspace/SE-INR/SC-ASISR/artifacts/checkpoints; S=/workspace/SE-INR/SC-ASISR/artifacts
python3 run_full.py --bicubic --name bicubic --out out/bicubic.json > logs/bicubic.log 2>&1
for mo in liif lte sc-inr; do for sd in 1 2 3; do python3 run_full.py --src $SRC --ckpt $C/seed$sd/$mo/epoch-last.pth --name ${mo}_s$sd --out out/${mo}_s$sd.json > logs/${mo}_s$sd.log 2>&1; done; done
python3 run_full.py --src $S/source_snapshots/gaussiansr/prechange --ckpt $C/external/gaussiansr/epoch-last-39024d186928.pth --name gaussiansr --needs-scale --out out/gaussiansr.json > logs/gaussiansr.log 2>&1
python3 score_v9.py > score_v9.out 2>&1; echo done > DONE
