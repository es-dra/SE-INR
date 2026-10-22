# Recovery, Cell Response and Learning in Arbitrary-Scale Super-Resolution

> **Review handoff, 2026-10-23:** stage 17 is complete and independently verified. This source and PDF are the rebuilt review draft, with four body pages and references on page 5. Stage 18 remains in progress and is explicitly pending in the manuscript. At the user's request, the non-experiment deliverables are handed off now; Claude can continue from the running jobs without restarting them.

Code and evidence for **"Shared Recovery Profiles, Calibrated Cell Responses, and Scale-Conditioned Learning in Arbitrary-Scale Super-Resolution"** ([source](paper/main.tex), [PDF](paper/main.pdf)).

The paper asks where additional learning capacity improves reconstruction, how to measure the decoder's response to cell, and why cell helps during training. The [study record](studies/two-coordinates/README.md) connects each claim to its protocol and results, including failed predictions.

**Findings.**

1. **Shared recovery profiles, localised capacity gains.** Recovery profiles are close across the tested decoder designs. Stronger encoders, larger MLPs and longer training affect recovery mainly from input-normalised frequency u ≈ 0.75 upward; low-frequency saturation persists with a second, Gaussian-blur degradation. This is a result for the tested models and recipes, not universal decoder independence. Strict cross-scale collapse fails its registered gates.
2. **Equal decoder-input steps reveal substantial cell responses.** A fixed multiplier m gives per-axis step h = 2(m−1)/s. The prospective full-image experiment controls h on 246 images, three decoder families and two seeds. Eleven of 36 adjacent-scale comparisons show a material increase under the family-wide corrected test, including seven positive-step comparisons within the training cell interval. Response energy J can increase while net error cost D decreases. The stronger Set14 endpoint prediction fails at h = +0.25.
3. **Dimension-preserving training controls test the source of cell's benefit.** Native and no-cell checkpoints are compared with constant-cell and shuffled-cell training at 200 epochs and two seeds. The latter arms preserve input dimensions while removing the correct cell pairing. Frozen Y-PSNR and R(u) scoring is pending completion of these runs.

The retrospective decomposition remains a bounded supporting result: changes in recovery R and fitted filter strength w jointly explain about 41% of the fixed-multiplier ×2→×6 MSE-cost decrease. Net cost also depends on error alignment and non-scalar responses. Detailed projections and image rankings are in stages 15–16; they are not prospective tests or a unique causal explanation.

## Review and evidence entry points

- [Manuscript source](paper/main.tex) and [PDF](paper/main.pdf).
- [Complete prediction ledger](studies/two-coordinates/README.md), including original failures and scoring amendments.
- [Stage 17 frozen protocol](studies/two-coordinates/17-equal-step-response/protocol.json), [formal scores](studies/two-coordinates/17-equal-step-response/score/summary.json), [independent verification](studies/two-coordinates/17-equal-step-response/score/independent_validation.json) and [frequency summaries](studies/two-coordinates/17-equal-step-response/score/frequency_summary.json).
- [Stage 18 frozen protocol](studies/two-coordinates/18-cell-training-source/protocol.json), [predictions](studies/two-coordinates/18-cell-training-source/predictions.json) and [synthetic intervention checks](studies/two-coordinates/18-cell-training-source/synthetic_validation.json).
- [Server run locations](studies/two-coordinates/RUNS.md) and [SHA256/path archive manifest](studies/two-coordinates/server-artifacts.json).

## 接续给 Claude：只剩训练结果相关收尾

截至北京时间 2026-10-23 05:32，constant/shuffle 的 seed 1 均为 73/200，seed 2 分别为 100/200、101/200；四个训练主进程均实际存活。此处是交付时快照，接手时核实实时状态，不凭旧轮次判断停机。

- 服务器：`research-175`；正式源码 checkout：`/workspace/SE-INR/SE-INR-current`。运行目录：`/workspace/SE-INR/runs/codex-20261023-cell-training-source-v1`。原始 `SC-ASISR` 工作树不动。
- 正式流程已自动衔接训练→全部评估→冻结评分，成功标记为 `COMPLETE.txt`、结果为 `scores/summary.json`。四个 slot 退出码在 `statuses/`；失败记录保留，不为获得通过而重跑。训练主 PID 为 139536–139539，顺序调度器为 136928；PID 仅辅助定位，须结合命令行核验。
- 独立复核已自动接续：`/workspace/SE-INR/runs/codex-20261023-deploy-protocols/verify_after_training.py`（观察进程 PID 142794）。它只在正式流程完成后运行 `verify_completed.py`；结果为 `scores/independent_validation.json`，退出回执为 `INDEPENDENT_CHECK_EXIT.json`。如复核自身失败，保留日志并修复复核问题，不修改冻结实验或重跑训练。
- 结果齐全后，依据四项 T1/T2 实际判定更新摘要、第三贡献、第 4 节、表 1 的 constant/shuffle 两行、表 2 的 stage-18 两行和结论；报告 R(u) 与推理策略敏感性。等价检验失败不能写成证明不等价。正式主张还要条件于两个种子和当前配方。
- 同步小型结果及 SHA256/服务器路径到 `18-cell-training-source/`，更新本 README、研究记录和 RUNS；重新编译并保持 4 页正文＋第 5 页仅参考文献，再提交和同步三端。大型逐图结果及 checkpoint 留在服务器。

**当前自评：** 前两条贡献已有完整数据支撑，等步长控制是最明确的新增贡献；第三条目前只有冻结设计，不能把整稿称作已完成的投稿终稿。按 B 会分析稿方向重建，仍需 Claude 审阅新意、论点连贯性及训练结论。A 会层面的可迁移预测或设计收益尚未建立，本次不扩展实验范围。

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

## Acknowledgements and licence

This code builds on LIIF, LTE, EDSR-PyTorch, BTC (RDN encoder) and SwinIR; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This repository is released under the BSD 3-Clause licence ([`LICENSE`](LICENSE)).
