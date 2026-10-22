#!/bin/bash
R=/workspace/SE-INR/runs/claude-20261022-benchmark-v13; cd $R
V5=/workspace/SE-INR/runs/claude-20261019-encoder-train-v5/save; V8=/workspace/SE-INR/runs/claude-20261020-seed2-v8/save; V7=/workspace/SE-INR/runs/claude-20261020-swinir-v7/save
(./bench.sh edsr_s1 $V5/liif-edsr-200ep/epoch-last.pth 0; ./bench.sh edsr_s2 $V8/liif-edsr-seed2/epoch-last.pth 0; ./bench.sh swinir_s1 $V7/liif-swinir-200ep/epoch-last.pth 0) &
(./bench.sh rdn_s1 $V5/liif-rdn-200ep/epoch-last.pth 1; ./bench.sh rdn_s2 $V8/liif-rdn-seed2/epoch-last.pth 1) &
(./bench.sh tinydec_s1 $V5/liif-edsr-tinydec-200ep/epoch-last.pth 2; ./bench.sh tinydec_s2 $V8/liif-tinydec-seed2/epoch-last.pth 2) &
(./bench.sh nocell_s1 $V5/liif-edsr-nocell-200ep/epoch-last.pth 3; ./bench.sh nocell_s2 $V8/liif-nocell-seed2/epoch-last.pth 3) &
wait; echo all > DONE_ALL
