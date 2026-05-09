# Artifacts 目录说明

`artifacts/` 保存所有训练、评估和派生产物。它不是源码目录。

## 分层

- `checkpoints/`：训练得到的 checkpoint、保存配置、训练日志。
- `raw_results/`：正式 raw JSON 结果，按 seed/protocol 存放。
- `derived/`：从 raw 结果派生出的 CSV、TEX、图和候选论文产物。
- `smoke/`：调试和 smoke 输出，不应作为论文证据引用。
- `legacy/`：历史产物，保留用于追溯，不代表当前结论。
- `results/`：兼容旧 `results/...` 路径的 symlink tree。
- `logs/`：训练和后台任务日志，兼容旧 `logs/...` 路径。

## 正式 Raw 结果

- `raw_results/seed1/benchmark.json`
- `raw_results/seed1/benchmark_signed_phiz.json`
- `raw_results/seed2/benchmark.json`
- `raw_results/seed3/benchmark.json`
- `raw_results/continuous/seed1/*.json`
- `raw_results/diagnostics/seed1/*.json`

## 当前论文相关派生结果

- `derived/analysis/benchmark_progress_2026-05-09/`
- `derived/analysis/seed1_aux_metrics_all8/`
- `derived/paper_candidates/qualitative_phiz_candidates/`

命名注意：最终候选 `SC-INR` 在当前 raw result 中仍使用历史 key
`SC-INR+PhiZ`；旧 raw key `SC-INR` 对应 `SC-INR-NoPhi`。详见
`paper/model_taxonomy.md`。

## 引用规则

论文中优先引用 `raw_results/` 和 `derived/` 中有 experiment card 支撑的产物。
`smoke/` 和 `legacy/` 默认不得引用，除非正文明确说明其诊断性质。
