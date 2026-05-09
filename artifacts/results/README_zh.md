# results 兼容路径说明

`results/` 是旧路径兼容层，真实目标是 `artifacts/results/`。这里大多数文件是
指向 `artifacts/raw_results/`、`artifacts/derived/`、`artifacts/smoke/` 或
`artifacts/legacy/` 的 symlink。

论文、报告和新脚本应优先使用 canonical 路径：

- 正式 raw JSON：`artifacts/raw_results/`
- 可信派生表图：`artifacts/derived/`
- 当前允许引用清单：`paper/ARTIFACTS_ALLOWED.md`
- claim 与证据边界：`paper/claims_evidence_matrix.md`

不要因为某个文件能从 `results/...` 打开就默认可引用。特别注意：

- `results/analysis/qualitative_phiz_smoke/` 是 smoke/debug 输出；
- `results/analysis/seed1_aux_metrics_smoke/` 是 smoke/debug 输出；
- `results/analysis/figures/` 和 `results/figures/` 属于历史兼容图；
- 旧 `results/analysis/benchmark_seed_mean_std.csv` 不应作为当前多 seed 表；
- `results/analysis/qualitative_phiz_candidates/` 只是候选图池，不是最终论文图。

