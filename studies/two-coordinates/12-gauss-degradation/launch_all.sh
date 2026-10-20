#!/bin/bash
cd /workspace/SE-INR/claude-20261021-gaussdeg-v12
./arm.sh rdn 0 &
(./arm.sh edsr 1; ./arm.sh nocell 1) &
wait
