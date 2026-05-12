# Artifacts 目录说明

`artifacts/` 保存所有训练、评估和派生产物。它不是源码目录。

## 分层

- `checkpoints/`：训练得到的 checkpoint、保存配置、训练日志。
- `raw_results/`：正式 raw JSON 结果，按 seed/protocol 存放。
- `derived/`：从 raw 结果派生出的 CSV、TEX、图和候选论文产物。
- `legacy/`：历史产物，保留用于追溯，不代表当前结论。
- `results/`：兼容旧 `results/...` 路径的 symlink tree。
- `logs/`：训练和后台任务日志，兼容旧 `logs/...` 路径。

## 正式 Raw 结果

- `raw_results/seed1/benchmark.json`
- `raw_results/seed1/benchmark_signed_phiz.json`
- `raw_results/seed2/benchmark.json`
- `raw_results/seed3/benchmark.json`

## 当前论文相关派生结果

- `derived/benchmarks/`：当前 canonical benchmark 总表、论文主表和 seed/context 表。
- `derived/evidence/README_zh.md`：辅助证据与机制分析入口索引。
- `derived/diagnostics/`：appendix / diagnostic 层级的辅助证据。
- `derived/paper_figures/qualitative_selected_seed1/`：当前保留的定性图入口。

命名注意：最终候选在论文中统一写作 `SC-INR`。seed1 历史 raw result
曾用 `SC-INR+PhiZ` 表示最终候选、用 raw key `SC-INR` 表示
`SC-INR-NoPhi`；seed2/3 的 `benchmark_sc_inr.json` 中 `SC-INR` 已是最终候选。
详见 `configs/registry/models.yaml` 和 `paper/model_taxonomy.md`。

## 引用规则

论文中优先引用 `raw_results/` 和 `derived/` 中有 experiment card 支撑的产物。
`legacy/` 默认不得引用，除非正文明确说明其诊断性质。
