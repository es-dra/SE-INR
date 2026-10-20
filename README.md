# Two Coordinates of Arbitrary-Scale Super-Resolution

Code and pre-registered study records for the paper
**"Recovery Lives in u, Conditioning Acts in r: Where Arbitrary-Scale Super-Resolution Gains Come From"** ([`paper/main.pdf`](paper/main.pdf)).

**Question.** Implicit-decoder ASISR models (LIIF and successors) keep adding decoder and cell-conditioning designs, yet gains shrink at large scales. Where in frequency can a decoder or the cell input still change the output?

**Findings.**
1. *Recovery lives in u.* Per-band recovery depends on frequency normalised by the LR Nyquist limit, u = 2·s·r. It is identical across four decoder designs on a shared encoder. Learned capacity (training time, encoder, decoder MLP width) raises it only above u ≈ 0.75.
2. *Conditioning acts in r.* Changing the cell filters the output by a low-pass in HR frequency r. Its cost is the recovered energy it filters away, so the cell's leverage vanishes as the scale grows. Removing the cell input costs 0.10 dB at ×2 and nothing measurable from ×3 on.

Every experiment fixed its protocol, predictions and scoring script before running; failed predictions are reported (Table 2 of the paper, and [`studies/two-coordinates`](studies/two-coordinates/README.md)).

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

Each folder in `studies/two-coordinates/` holds the protocol (`protocol*.json`), the exact scripts that ran, the scoring script, its output and `FROZEN.sha256` with hashes recorded before results were read. Those scripts keep the absolute server paths they ran with. To rerun on another machine, use `tools/frequency_probe.py`, which implements the same measurement with configurable paths. Checkpoints and datasets are not included.

## Acknowledgements and licence

This code builds on LIIF, LTE, EDSR-PyTorch, BTC (RDN encoder) and SwinIR; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The repository licence is to be added before public release.
