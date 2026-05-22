# Artifacts 目录说明

`artifacts/` 保存所有训练、评估和派生产物。它不是源码目录。

## 分层

- `checkpoints/`：训练得到的 checkpoint、保存配置、训练日志。
- `raw_results/`：正式 raw JSON 结果，按 seed/protocol 存放。
- `derived/`：从 raw 结果派生出的 CSV、TEX、图和候选论文产物。
- `legacy/`：历史产物，保留用于追溯，不代表当前结论。
- `results/`：便捷入口，只链接 canonical benchmark JSON。
- `logs/`：训练和后台任务日志，兼容旧 `logs/...` 路径。

## 正式 Raw 结果

- `raw_results/seed1/benchmark.json`
- `raw_results/seed1/benchmark_signed_phiz.json`
- `raw_results/seed1/benchmark_sc_inr_nosinc.json`
- `raw_results/seed1/benchmark_sc_inr_eq.json`
- `raw_results/seed2/benchmark.json`
- `raw_results/seed2/benchmark_sc_inr.json`
- `raw_results/seed3/benchmark.json`
- `raw_results/seed3/benchmark_sc_inr.json`

## 当前论文相关派生结果

- `derived/benchmarks/`：当前 canonical benchmark 总表、canonical JSON、论文主表和 seed/context 表。
- `derived/evidence/README_zh.md`：辅助证据与机制分析入口索引。
- `derived/diagnostics/`：appendix / diagnostic 层级的辅助证据。
- `derived/paper_figures/qualitative_selected_seed1/`：当前保留的定性图入口。

命名注意：最终候选在论文中统一写作 `SC-INR`。seed1 历史 raw result
曾用 `SC-INR+PhiZ` 表示最终候选、用 raw key `SC-INR` 表示
`SC-INR-NoPhi`；seed2/3 的 `benchmark_sc_inr.json` 中 `SC-INR` 已是最终候选。
详见 `configs/registry/models.yaml` 和 `paper/model_taxonomy.md`。

## 引用规则

论文中优先引用 `derived/benchmarks/` 和 `derived/` 中有 experiment card 支撑的产物；
`raw_results/` 作为 provenance 保留，不直接作为论文表格入口。
`legacy/` 默认不得引用，除非正文明确说明其诊断性质。

## 源码管理边界

`artifacts/` 中只有少量 canonical 入口适合进入源码管理：

- `README_zh.md` 和各子目录索引；
- `raw_results/` 下正式 benchmark JSON；
- `derived/benchmarks/` 下论文主表和 canonical CSV/JSON；
- `derived/evidence/README_zh.md`；
- 经人工确认的少量 `derived/paper_figures/` 候选图。

大体量 checkpoint、训练日志、本地数据、scratch/smoke 输出、diagnostic 中间图和 legacy 大图默认只保留在本地，
不作为源码管理内容。需要把新的 artifact 晋级为论文证据时，先更新
`paper/ARTIFACTS_ALLOWED.md`，再考虑是否放行 `.gitignore`。
