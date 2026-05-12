# Mechanism Suite 2026-05-12 摘要

本目录是 6.1-6.6 机制诊断和 `SC-INR-EQ` exploratory 分析的 active 摘要入口。
完整运行产物已移入：

`artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/`

## 可用结论

- `SC-INR-EQ` 只支持 exploratory 表述：它展示 SC-INR decoder contract 可以与
  Rot-E plumbing 结合，但不是主方法。
- 6.3 same-LR consistency 的负控结果说明：cell-insensitive / NoSinc 类模型可以获得
  很高 consistency，因此该指标不能单独证明 sampling correctness。
- 6.1-6.6 适合进入 mechanism discussion 或 appendix，不能替代 3-seed benchmark。

## 禁止用途

- 不得把 `SC-INR-EQ` 写成 main method。
- 不得写 strict scale equivariance 或 whole-network sampling correctness。
- 不得把 consistency 单指标当作 sinc response 的正证据。

## 完整产物

完整 CSV、图和 run config 见 legacy archive：

- `artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/README_zh.md`
- `artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/scale_gain_curve.csv`
- `artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/cell_extrapolation_diagnostic.csv`
- `artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/cell_intervention_summary.csv`
