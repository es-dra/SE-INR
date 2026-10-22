#!/bin/bash
# Diagnose why no-cell cost differs between the DIV2K-crop probe (~0 at x3+) and benchmark PSNR (~-0.1 dB).
R=/workspace/SE-INR/runs/claude-20261022-benchmark-v13; cd $R; mkdir -p diag
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
V5=/workspace/SE-INR/runs/claude-20261019-encoder-train-v5/save
python3 - <<'P'
import yaml
c=yaml.safe_load(open('cfg/set14-4.yaml'))
for s in (4,6):
    c['test_dataset']['dataset']['args']['root_path']='/workspace/SE-INR/Data/DIV2K_valid_HR'
    c['test_dataset']['dataset']['args']['first_k']=None
    c['test_dataset']['wrapper']['args'].update(scale_min=s,scale_max=s); c['eval_type']=f'benchmark-{s}'
    yaml.safe_dump(c,open(f'diag/div2k-{s}.yaml','w'))
P
for a in edsr-200ep edsr-nocell-200ep; do for s in 4 6; do
 v=$(cd /workspace/SE-INR/SE-INR-current && python3 $R/test_crop.py --config $R/diag/div2k-$s.yaml --model $V5/liif-$a/epoch-last.pth --device 0 2>/dev/null | grep -o "result: [0-9.]*"); echo "benchmark-protocol DIV2K-valid100 $a x$s $v" >> diag/results.txt; done; done
cd /workspace/SE-INR/SE-INR-current
for a in edsr-200ep edsr-nocell-200ep; do
 python3 tools/frequency_probe.py --ckpt $V5/liif-$a/epoch-last.pth --data /workspace/SE-INR/Data/Set14/HR --crop 0 --scales 4 6 --out $R/diag/probe-set14-$a.json > /dev/null 2>&1
done
echo done > $R/diag/DONE
