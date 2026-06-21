# Mechanism Suite 2026-05-12

本目录汇总 `SC-INR-EQ` 训练完成后的单 seed benchmark 后续机制诊断。
当前状态是 `diagnostic/exploratory`，不能直接升级为主论文 claim。

## 包含的检查

- 6.1 Cell response curve：
  `sampling_response/cell_response_curve.csv`。
- 6.2 Scale gain curve：
  `scale_gain_curve.csv`。
- 6.3 Sampling consistency metric：
  `sampling_consistency/consistency_summary.csv`。
- 6.4 Frequency/response visualization：
  `sampling_response/response_distribution_summary.csv`。
- 6.5 OOD cell extrapolation diagnostic：
  `cell_extrapolation_diagnostic.csv`。
- 6.6 Cell intervention experiment：
  `cell_intervention_summary.csv`。

说明：旧自动图已从本地工作区清理；本 legacy 目录保留 CSV、README 和 run config
用于审计。若需要图，应用当前细分脚本重新生成。

## 关键观察

- `SC-INR-EQ` 单 seed benchmark 完整覆盖 4 个 benchmark 数据集和
  9 个尺度，36/36 个结果为 finite。
- `SC-INR-EQ` 平均 PSNR：ID `31.1513`，OOD `22.7779`，
  ALL `25.5691`。
- 相对 `LTE-EQ`：ID `+0.0841 dB`，OOD `+0.0749 dB`，
  ALL `+0.0780 dB`。
- 相对 final `SC-INR`：ID `+0.0326 dB`，OOD `+0.0014 dB`，
  ALL `+0.0118 dB`；这更像 Rot-E exploratory integration 基本打平
  final SC-INR，而不是新的主结论。
- 6.3 consistency 负控仍然明显：`LTE-PhaseZ`、`LTE-NoCellPhase` 和
  `SC-INR-NoSinc` 的 consistency PSNR 都高于完整 `SC-INR`/`SC-INR-EQ`。
  因此 same-LR consistency 只能作为 diagnostic，不能单独证明 sinc
  observation correctness。
- 6.6 cell intervention 中，`LTE` 对错误 cell 更敏感；`LTE-NoCellPhase`
  和 `LTE-PhaseZ` 对 cell 不响应；`SC-INR`/`SC-INR-EQ` 有非零但较小的
  analytic cell response。

## 使用边界

允许表述：

- `SC-INR-EQ` 在单 seed benchmark 上相对 `LTE-EQ` 有小幅正增益。
- `SC-INR-EQ` 与 final `SC-INR` 单 seed 基本打平。
- 机制诊断显示 `SC-INR`/`SC-INR-EQ` 的 cell response 是有限的 analytic
  response，而不是 LTE 的 learned cell phase。
- same-LR consistency 需要和 fidelity、NoSinc 负控、cell-response 诊断一起解释。

禁止表述：

- `SC-INR-EQ` 已经成为主方法。
- `SC-INR-EQ` 证明 strict scale equivariance 或 whole-network scale
  equivariance。
- 6.3 consistency alone 证明 sampling correctness。
- sinc response 的收益已经被 6.1-6.6 单独因果证明。

## 生成命令

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python scripts/analysis/analyze_sampling_response.py \
  --mode both \
  --out artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/sampling_response \
  --device cuda:0 \
  --models SC-INR,SC-INR-NoSinc,SC-INR-EQ \
  --datasets bsd100,urban100 \
  --max_images 3 \
  --lr_scale 4 \
  --scales 2,3,4,6,8,12,16,24,30 \
  --cell_models LTE,LTE-NoCellPhase,LTE-PhaseZ,SC-INR,SC-INR-NoSinc,SC-INR-EQ \
  --cell_dataset urban100 \
  --cell_image img_004.png \
  --cell_lr_scale 4 \
  --cell_ref_scale 4 \
  --cell_scales 1,1.5,2,3,4,6,8,12,16,24,30 \
  --max_queries 2048 \
  --eval_bsize 50000

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python scripts/analysis/evaluate_seed1_aux_metrics.py \
  --models LTE,LTE-NoCellPhase,LTE-PhaseZ,SC-INR,SC-INR-NoSinc,SC-INR-EQ \
  --datasets bsd100,urban100 \
  --max_images 5 \
  --consistency_pairs '8,4;16,4;30,4' \
  --device cuda:0 \
  --eval_bsize 50000 \
  --out artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/sampling_consistency \
  --skip_quality \
  --skip_visuals

# 旧的总编排脚本 scripts/analysis/run_mechanism_suite.py 已从活跃代码面退役。
# 本目录只保留当时已经生成的 CSV/图和 run_config 供审计；如需复跑，
# 应优先使用当前仍保留的细分脚本重新生成相应诊断。
```

`SC-INR-EQ` benchmark 命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python entrypoints/eval_full.py \
  --device 0 \
  --save_root save \
  --models SC-INR-EQ \
  --output artifacts/raw_results/seed1/benchmark_sc_inr_eq.json
```
