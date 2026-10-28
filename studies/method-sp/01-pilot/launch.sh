#!/bin/bash
cd /workspace/SE-INR/runs/claude-20261029-sp-pilot-v1
for a in base film uloss full wide; do ./arm.sh $a & done
wait
