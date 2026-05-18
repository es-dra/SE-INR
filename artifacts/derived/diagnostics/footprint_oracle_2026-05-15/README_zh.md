# Footprint Oracle Diagnostics 2026-05-15

本目录评估 decoder cell response 是否跟随真实 footprint 观测，而不是只比较模型输出之间的
same-LR consistency。

## 协议

- 固定 LR：HR 图像 bicubic downsample 到 x4 LR。
- 固定 query：x4 HR grid。
- 只改变 decoder 输入的 cell multiplier：`1,2,4`。
- oracle target：在同一 HR grid 上，对 HR 图像做 piecewise-constant box average。
- 模型：`LTE,LTE-NoCellPhase,SC-INR-NoPhi,SC-INR,SC-INR-NoSinc`。
- 数据：`bsd100,urban100` sorted 前 `10` 张。

## 指标

- `oracle_psnr_y` / `oracle_rmse_y`：模型输出 vs box-averaged oracle target。
- `delta_tracking_rmse_y`：模型 cell-change 与 oracle cell-change 的差异，越低越好。
- `cell_sensitivity_rmse_y`：模型相对 native cell 的输出变化幅度。
- `oracle_change_rmse_y`：真实 oracle 相对 native footprint 的变化幅度。
- `weak_oracle`：当 oracle 本身变化低于 `0.002` 时标记，不能用于强结论。

## 当前 gate 摘要

- gate_status: `supports_sc_inr_over_nosinc_on_at_least_one_primary_metric`
- best_vs_nosinc_delta_tracking: `-0.0028196469414979303`
- best_vs_nosinc_oracle_rmse: `-0.003749020172016424`

## 关键结果

跨 BSD100/Urban100 前 10 张平均：

| Model | Oracle RMSE m=2 | Oracle RMSE m=4 | Tracking RMSE m=2 | Tracking RMSE m=4 | Cell sensitivity m=2/m=4 |
| --- | ---: | ---: | ---: | ---: | --- |
| LTE | 0.02504 | 0.02309 | 0.03077 | 0.04834 | 0.01148 / 0.03299 |
| LTE-NoCellPhase | 0.02611 | 0.02653 | 0.03237 | 0.05134 | 0.00000 / 0.00000 |
| SC-INR-NoPhi | 0.02476 | 0.02057 | 0.03069 | 0.04708 | 0.01029 / 0.02795 |
| SC-INR | 0.02464 | 0.02148 | 0.03062 | 0.04745 | 0.01058 / 0.02955 |
| SC-INR-NoSinc | 0.02612 | 0.02750 | 0.03237 | 0.05134 | 0.00000 / 0.00000 |

解释：`SC-INR` 在两个主指标上优于 `SC-INR-NoSinc`，说明 analytic cell response
确实比 cell-insensitive 负控更跟随 footprint oracle。`SC-INR-NoPhi` 在该诊断上略强于
final `SC-INR`，因此本结果支持 footprint-response 机制，不支持“final variant 在所有辅助指标上最强”。

## 解释边界

- 这是 HR piecewise-constant box oracle，不是真实连续场景的精确积分。
- 该诊断可检验 cell response 是否跟随 footprint target，但不能单独证明 sinc 是唯一因果。
- same-LR consistency 仍然是 diagnostic-only；cell-insensitive 模型可能 self-consistency 很高。
- 若 `SC-INR-NoSinc` 或其他负控更好，应收缩 claim，而不是追加相邻指标寻找正结果。

## 文件

- `footprint_oracle_metrics.csv`：逐图、逐模型、逐 cell multiplier 指标。
- `footprint_oracle_summary.csv`：按 dataset/model/cell multiplier 汇总。
- `footprint_oracle_overall.csv`：按 model/cell multiplier 汇总。
- `figures/`：摘要图。
- `run_config.json`：运行参数。
