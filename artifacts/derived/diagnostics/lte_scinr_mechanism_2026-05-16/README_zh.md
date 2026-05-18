# LTE vs SC-INR 机制诊断 2026-05-16

本目录把 `LTE -> SC-INR` 主脉络中的机制证据放到同一个协议下检查：

- 固定 x4 LR 和 query grid；
- 只改变 decoder 输入 cell multiplier：`1,2,4`；
- 同时记录 footprint oracle、输出 cell sensitivity、LTE `h_p(c)`、SC-INR `W(omega,c)`、
  以及 MLP 前 effective amplitude proxy。

## 覆盖

- models: `LTE,LTE-NoCellPhase,LTE-PhaseZ,SC-INR-NoPhi,SC-INR,SC-INR-NoSinc`
- datasets: `bsd100,urban100`
- max_images: `5`
- crop size: `64`，`0` 表示不用 crop。
- signal/oracle max queries: `256`

## 主要输出

- `mechanism_metrics.csv`：逐图、逐模型、逐 cell multiplier 指标。
- `mechanism_summary.csv`：按 dataset/model/cell multiplier 汇总。
- `mechanism_overall.csv`：按 model/cell multiplier 汇总。
- `gate_conclusion.json`：自动 gate 摘要。
- `figures/`：小型机制曲线图。

## Gate 结论

```json
{
  "supports_footprint_vs_nosinc": true,
  "supports_footprint_vs_lte": true,
  "has_lte_cell_phase_signal": true,
  "has_sc_inr_active_response_signal": true,
  "nosinc_active_response_cell_insensitive": true,
  "mean_metrics_m_gt_1": {
    "SC-INR_delta_tracking_rmse_y": 0.041131370700895786,
    "LTE_delta_tracking_rmse_y": 0.041980653814971444,
    "SC-INR-NoSinc_delta_tracking_rmse_y": 0.04353192122653127,
    "SC-INR_oracle_rmse_y": 0.025121342199976384,
    "LTE_oracle_rmse_y": 0.02623160401435992,
    "SC-INR-NoSinc_oracle_rmse_y": 0.02729273666769359,
    "LTE_cell_phase_delta_rms_vs_native": 0.5152796506881714,
    "SC-INR_active_response_delta_rms_vs_native": 0.3081212617456913,
    "SC-INR-NoSinc_active_response_delta_rms_vs_native": 0.0
  },
  "claim_boundary": "该 gate 只支持同一 LR/query 下的机制差异：LTE 有 learned h_p(c) cell-phase 信号，SC-INR 有非零 analytic response/effective-amplitude proxy，且 footprint oracle 上可与 LTE/NoSinc 比较。它不能证明 sinc 唯一因果或最终 RGB exact box integral。"
}
```

## 解释边界

- `cell_phase_*` 只对 LTE 的 learned `h_p(c)` 有直接语义；`LTE-NoCellPhase`
  应为无 cell phase，`LTE-PhaseZ` 的 feature phase 不随 cell 变化。
- `effective_pair_*` 是 SC-INR MLP 前 Fourier input 的 coefficient-pair magnitude proxy，
  不能解释为最终 RGB 频谱振幅。
- footprint oracle 是有限 HR 上的 piecewise-constant box proxy，不是真实连续场景积分。
- 如果本目录结果与 benchmark 冲突，应优先收缩 claim，而不是把诊断指标升级为主证据。
