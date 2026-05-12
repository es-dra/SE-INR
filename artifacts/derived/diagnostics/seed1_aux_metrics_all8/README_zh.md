# seed1 auxiliary metrics all8

本目录保存 seed1 辅助质量、一致性和频域指标的原始派生 CSV。它是辅助证据源，
不是论文表格的直接入口。

## 文件边界

- `quality_metrics.csv`、`consistency_metrics.csv`：逐图逐尺度明细，主要用于复查。
- `quality_summary.csv`、`consistency_summary.csv`：按模型/数据集/尺度汇总。
- `paper/`：从本目录派生出的论文可读表格、图和解读入口。
- `run_config.json`：运行配置和覆盖范围。

## 使用边界

论文和 claim ledger 应优先引用 `paper/` 子目录中的整理结果，或通过
`artifacts/derived/evidence/README_zh.md` 进入本目录。顶层 CSV 可以用于审计和再分析，
但不要直接把所有顶层结果视为 paper-ready evidence。

本目录主要覆盖旧 no-phase 主线 `SC-INR-NoPhi` 及相关 seed1 诊断模型；它不等同于最终
`SC-INR` 的完整多 seed 证据。
