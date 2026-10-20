#!/bin/bash
cd /workspace/SE-INR/claude-20261020-seed2-v8
./arm.sh rdn train-liif-rdn-200ep.yaml 1 &
./arm.sh edsr train-liif-edsr-200ep.yaml 2 &
(./arm.sh tinydec train-liif-edsr-tinydec-200ep.yaml 3; ./arm.sh nocell train-liif-edsr-nocell-200ep.yaml 3) &
wait
