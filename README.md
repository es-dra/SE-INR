# Two Coordinates of Arbitrary-Scale Super-Resolution

Code, registered study records and retrospective audits for the paper **"Recovery and Conditioning in Arbitrary-Scale Super-Resolution: A Two-Coordinate Analysis"** ([PDF](paper/main.pdf)).

**2026-10-23 revision:** the manuscript incorporates saved-output decomposition, residual localization and controls for the actual cell-input step. These analyses use existing predictions and sufficient statistics only: no training or model inference. Original protocols, failures and paper source are preserved. See the [evidence and limits](studies/two-coordinates/README.md).

**Question.** Where do recovery and conditioning responses occur in frequency, and what does a small cell-perturbation cost actually establish?

**Findings.**

1. *Recovery in u.* Profiles are similar across the tested decoder designs; strict cross-scale collapse fails some registered checks. Capacity effects occur mainly from u ≈ 0.75 upward. Matched families share an encoder architecture, not learned weights; GaussianSR is an external checkpoint with a different recipe.
2. *Cell response beyond a scalar filter.* Changes in recovery R and fitted filter strength w jointly account for about 41% of the doubled-cell ×2→×6 MSE-cost decrease. The non-scalar net term accounts for 54% on DIV2K and 59% on Set14. Most of that decline occurs in r ∈ [0.1875,0.375); images can contribute in opposite directions. Finer real-frequency fits reduce residual energy but leave a quadrature component.
3. *Intervention size and error alignment matter.* A fixed multiplier m gives per-axis decoder-input step h = 2(m−1)/s. Matched-step comparisons across 214 images do not support universal sensitivity loss. Output changes may correct existing errors; dB costs also depend on native MSE. Existing no-cell retraining loses roughly 0.1 dB across all benchmark scales.

Original registered stages retain protocols, scoring revisions and failed predictions. Stages 15–16 are explicitly retrospective diagnostics, not held-out prediction tests. Exact accounting and fitted projections do not identify a causal network mechanism or explain the training benefit of cell.

## Layout

| Path | Content |
| --- | --- |
| `src/models` | EDSR, RDN and SwinIR encoders; LIIF, LTE and SC-INR decoders |
| `src/datasets` | data loading; `gauss_wrapper.py` adds the Gaussian-blur degradation |
| `entrypoints/train.py`, `entrypoints/test.py` | training and benchmark evaluation |
| `configs/` | training and test configurations (`*-200ep.yaml`: matched-training arms of the paper) |
| `tools/frequency_probe.py` | reusable probe: band recovery R(u) and cell-sweep costs for any checkpoint |
| `studies/two-coordinates/` | frozen protocols, analysis code, scores and summaries of every study stage |
| `paper/` | LaTeX source, figures and PDF |

## Setup

```bash
pip install -r requirements.txt
ln -s /path/to/datasets Data   # expects Data/DIV2K_train_HR, Data/DIV2K_valid_HR, Data/Set14/HR, ...
```

`datasets`, `models` and `utils.py` at the repository root are links to the modules in `src/`, so scripts run from the root.

## Usage

Train a matched arm (one GPU):

```bash
python entrypoints/train.py --config configs/train-div2k/train-liif-edsr-200ep.yaml --name liif-edsr --device 0 --seed 1
```

Benchmark PSNR (e.g. Set14 ×4):

```bash
python entrypoints/test.py --config configs/test/test-set14-4.yaml --model save/liif-edsr/epoch-last.pth --device 0
```

Probe a trained model: recovery per u-band, and the cost of doubling or halving the cell:

```bash
python tools/frequency_probe.py --ckpt save/liif-edsr/epoch-last.pth --data Data/DIV2K_valid_HR \
    --scales 2 3 4 6 --cell-mult 2 0.5 --out probe.json
python tools/frequency_probe.py --summary probe.json
```

Use `--crop 0` for full images, `--degradation gauss` for the Gaussian-blur degradation, and `--needs-scale` for models whose `query_rgb` takes a scale argument (GaussianSR).

## Reproducing the paper

Registered folders retain their protocols, original scripts, scoring outputs and recorded hashes. Stages 15–16 contain analysis specifications, extracted sufficient statistics, source manifests, validation results and analysis scripts. Their uncertainty estimates condition on existing trained seeds. Historical scripts keep their server paths; tools/frequency_probe.py provides configurable paths for the original recovery and cell measurements. Trained weights and datasets are not included.

## Acknowledgements and licence

This code builds on LIIF, LTE, EDSR-PyTorch, BTC (RDN encoder) and SwinIR; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This repository is released under the BSD 3-Clause licence ([`LICENSE`](LICENSE)).
