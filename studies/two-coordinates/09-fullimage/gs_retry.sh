#!/bin/bash
# GaussianSR full-image rerun with cuDNN disabled (grid_sample on large images hits CUDNN_STATUS_NOT_SUPPORTED). Model, data and scoring unchanged.
R=/workspace/SE-INR/claude-20261020-fullimage-v9; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
S=/workspace/SE-INR/SC-ASISR/artifacts; C=$S/checkpoints
python3 -c "import torch,runpy,sys; torch.backends.cudnn.enabled=False; sys.argv=['run_full.py','--src','$S/source_snapshots/gaussiansr/prechange','--ckpt','$C/external/gaussiansr/epoch-last-39024d186928.pth','--name','gaussiansr','--needs-scale','--out','out/gaussiansr.json']; runpy.run_path('run_full.py', run_name='__main__')" > logs/gaussiansr_retry.log 2>&1
python3 score_v9.py > score_v9.out 2>&1; echo done > DONE2
