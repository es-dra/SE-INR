# Recovery, Cell Response and Learning in Arbitrary-Scale Super-Resolution

> **Study status, 2026-10-23:** stages 17 and 18 are complete and independently verified. The manuscript reports all four stage-18 prediction outcomes. Original evaluation failures and the authorised recovery are retained in the evidence record.

Code and evidence for **"Shared Recovery, Sustained Cell Responses, and Training Gains in Arbitrary-Scale Super-Resolution"** ([source](paper/main.tex), [PDF](paper/main.pdf)).

The paper asks where additional learning capacity improves reconstruction, how to measure the decoder's response to cell, and why cell helps during training. The [study record](studies/two-coordinates/README.md) connects each claim to its protocol and results, including failed predictions.

**Findings.**

1. **Shared recovery profiles, localised capacity gains.** Recovery profiles are close across the tested decoder designs. Stronger encoders, larger MLPs and longer training affect recovery mainly from input-normalised frequency u ≈ 0.75 upward; low-frequency saturation persists with a second, Gaussian-blur degradation. This is a result for the tested models and recipes, not universal decoder independence. Strict cross-scale collapse fails its registered gates.
2. **Cell responses persist or increase at equal decoder-input steps.** From ×2 to ×6, response energy J has an endpoint ratio J6/J2 ≥ 1 in 11 of 12 dataset–step conditions, with point estimates 0.94–2.47. Individual paired-bootstrap 95% interval lower bounds span 0.81–1.57 and upper bounds 1.08–4.48; these are ranges of condition-specific bounds, not one pooled interval. Fixed-multiplier cost reductions reflect both the shrinking input step h = 2(m−1)/s and error alignment, rather than establishing weaker cell responses. This endpoint pattern is descriptive, not a newly registered success. The original registered gates and the failed stronger Set14 endpoint prediction remain in the prediction ledger and manuscript Table 2.
3. **Correctly paired scale information supplies the cell training gain.** Native cell exceeds constant-cell and shuffled-cell training by 0.100 dB [0.061, 0.149] and 0.097 dB [0.058, 0.145] (97.5% paired-image intervals), under the same 200-epoch, two-seed recipe. Both controls retain the two input dimensions and pass equivalence to no-cell within ±0.03 dB. All four frozen predictions pass. This supports correct pairing beyond extra input dimensions, without identifying a unique internal mechanism.

The retrospective decomposition remains a bounded supporting result: changes in recovery R and fitted filter strength w jointly explain about 41% of the fixed-multiplier ×2→×6 MSE-cost decrease. Net cost also depends on error alignment and non-scalar responses. Detailed projections and image rankings are in stages 15–16; they are not prospective tests or a unique causal explanation.

## Paper and evidence

- [Manuscript source](paper/main.tex) and [PDF](paper/main.pdf).
- [Complete prediction ledger](studies/two-coordinates/README.md), including original failures and scoring amendments.
- [Stage 17 frozen protocol](studies/two-coordinates/17-equal-step-response/protocol.json), [formal scores](studies/two-coordinates/17-equal-step-response/score/summary.json), [independent verification](studies/two-coordinates/17-equal-step-response/score/independent_validation.json) and [frequency summaries](studies/two-coordinates/17-equal-step-response/score/frequency_summary.json).
- [Stage 18 frozen protocol](studies/two-coordinates/18-cell-training-source/protocol.json), [predictions](studies/two-coordinates/18-cell-training-source/predictions.json) and [synthetic intervention checks](studies/two-coordinates/18-cell-training-source/synthetic_validation.json), [final scores](studies/two-coordinates/18-cell-training-source/scores/summary.json) and [independent verification](studies/two-coordinates/18-cell-training-source/scores/independent_validation.json).
- [Server run locations](studies/two-coordinates/RUNS.md) and [SHA256/path archive manifest](studies/two-coordinates/server-artifacts.json).

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
CUDA_DEVICE_ORDER=PCI_BUS_ID python entrypoints/train.py --config configs/train-div2k/train-liif-edsr-200ep.yaml --name liif-edsr --device 0 --seed 1
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

Registered folders retain protocols, predictions, original scripts, compact scores and hashes. Stages 17–18 froze their protocols, predictions, inputs and scoring scripts before their respective runs. Their integrated launchers execute measurement/training through scoring, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`; frozen historical paths identify the exact server inputs. Freeze-manifest SHA256 values are `a77f84a54fb09676a6e6703b345ff8ce30d42597bfc874f29561bff552f9fb06` (17) and `b601cf3d197cc81bfe0e6691f1c310457c8c936e49985e7e073062e4e06c1fb1` (18).

Image-bootstrap intervals resample images within each dataset and condition on the available trained seeds. Stages 17–18 use previously studied images/checkpoints for new, prospectively specified measurements; they do not claim a blind held-out-image discovery. Standard benchmark Y-PSNR and spectral R(u) use different, explicitly documented degradation and clipping protocols.

Expanded data, saved predictions, weights and intermediate versions stay on the server. The archive manifest records exact paths, byte counts and SHA256 values; Git history preserves removed tracked versions. Restore the documented inputs before rerunning historical scripts. Final figure rendering is reproducible with [the shared-figure renderer](paper/render_shared_figures.py) and [the equal-step plotter](studies/two-coordinates/17-equal-step-response/paper_figure.py). Figure presentation is separate from frozen scoring. Datasets and trained weights are not included.

## Public release

[Public-release instructions](docs/PUBLIC_RELEASE.md) create an independent repository from the current `main` snapshot with no inherited history or tags. The existing repository and its `archive-2026-10` tag remain unchanged.

## Acknowledgements and licence

This code builds on LIIF, LTE, EDSR-PyTorch, BTC (RDN encoder) and SwinIR; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This repository is released under the BSD 3-Clause licence ([`LICENSE`](LICENSE)).

Stage 18 evaluated eight checkpoint slots and verified all 50,328 records. Training finished once; initial evaluations failed during CUDA initialization. The authorised evaluation-only recovery reused unchanged checkpoints and frozen evaluator/scorer bytes. Original failure paths and final large-output hashes are retained in the [result manifest](studies/two-coordinates/18-cell-training-source/results-manifest.json). No training was repeated.
