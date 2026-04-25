#!/bin/bash
# Benchmark evaluation loop for all 3 models on all benchmarks/scales

BASE=/workspace/SE-INR/Equivariant-ASISR
cd $BASE

MODELS=(
    "save/edsr-baseline-liif/epoch-best.pth:LIIF_baseline"
    "save/edsr-baseline-liif-EQ/epoch-best.pth:LIIF_EQ"
    "save/se-inr/epoch-best.pth:SE_INR"
)

BENCHMARKS=("Set5" "Set14" "BSD100" "Urban100")
SCALES=(2 3 4)

for model_path in "${MODELS[@]}"; do
    IFS=':' read -r model name <<< "$model_path"
    echo ""
    echo "========================================"
    echo "MODEL: $name ($model)"
    echo "========================================"

    for bench in "${BENCHMARKS[@]}"; do
        for scale in "${SCALES[@]}"; do
            echo ""
            HR_PATH="/workspace/SE-INR/Data/${bench}/HR"
            if [ ! -d "$HR_PATH" ]; then
                echo "  [SKIP] $bench/HR not found"
                continue
            fi

            CONFIG="configs/test/test-${bench,,}-${scale}.yaml"

            RESULT=$(python test.py \
                --config "$CONFIG" \
                --model "$model" \
                --device cpu 2>&1 | grep "result:" | awk '{print $2}')

            if [ -n "$RESULT" ]; then
                echo "  $bench x$scale => PSNR = $RESULT dB"
            else
                echo "  $bench x$scale => FAILED"
            fi
        done
    done
done
