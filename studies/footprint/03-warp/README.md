# Footprint stage 03: warping evaluation (inference only)

Protocols, footprint modes and written expectations: [`protocol.json`](protocol.json). Single script `warp.py`; CPU self-test: `python3 warp.py --smoke /tmp/warp-smoke`.

## Data (server)

1. LTEW warped sets (Google Drive links in the LTEW README, folders "isc"/"osc" for Set5W, Set14W, B100W, Urban100W, DIV2KW): place as `load/benchmark/<Set>/LR_warp_in_scale`, `.../LR_warp_out_of_scale` next to the existing `HR` folders. Each folder holds `*.png` inputs and matching `*.pth` matrices.
2. SRWarp validation set for the convention check: `https://cv.snu.ac.kr/dataset/srwarp/input_valid_v3.tar` and `gt_valid.tar`.

## Run

```bash
cd /workspace/SE-INR/SE-INR-current
CK="liif=/workspace/SE-INR/SC-ASISR/artifacts/checkpoints/seed1/liif/epoch-last.pth lte=.../seed1/lte/epoch-last.pth sc-inr=.../seed1/sc-inr/epoch-last.pth"
# convention check (bicubic-point should be a sane PSNR; if it is ~10 dB add --invert-matrix)
python3 studies/footprint/03-warp/warp.py --protocol ltew-warp --hr <srwarp gt_valid dir> --warp-dir <srwarp input_valid dir> --ckpt $CK --out runs/warp/check.jsonl
# LTEW splits
for S in Set5 Set14 B100 Urban100; do for T in in_scale out_of_scale; do
  python3 studies/footprint/03-warp/warp.py --protocol ltew-warp --hr load/benchmark/$S/HR --warp-dir load/benchmark/$S/LR_warp_$T --ckpt $CK --out runs/warp/ltew-$S-$T.jsonl
done; done
# LTEW asymmetric pairs (per dataset list in warp.py LTEW_PAIRS)
python3 studies/footprint/03-warp/warp.py --protocol ltew-asym --hr /workspace/SE-INR/Data/BSD100/HR --pairs 4.00,1.40 1.50,3.00 3.50,1.45 8.00,2.80 3.00,6.00 7.00,2.90 --ckpt $CK --out runs/warp/asym-b100.jsonl
# minification split
python3 studies/footprint/03-warp/warp.py --protocol minify --hr /workspace/SE-INR/Data/Set14/HR --strengths 0.3 0.6 0.9 --ss 8 --ckpt $CK --out runs/warp/minify-set14.jsonl
python3 studies/footprint/03-warp/warp.py --summarize runs/warp/minify-set14.jsonl
```

Repeat with seed 2 checkpoints and, once trained, add `exact=<path>`.
