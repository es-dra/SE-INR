# results 便捷入口说明

`results/` 是根目录便捷入口，真实目标是 `artifacts/results/`。这里不再链接旧
raw JSON，只保留指向 canonical benchmark JSON 的 symlink，避免后续从历史 raw key
误读模型语义。

当前文件：

- `benchmark_all_models_canonical.json`：所有模型的 canonical benchmark；顶层模型名均为
  当前论文展示名，包含 `SC-INR-EQ` seed1 exploratory/context 结果。
- `benchmark_paper_main_3seed.json`：论文主模型 `LIIF`、`LTE`、`SC-INR` 的三 seed
  benchmark JSON。

canonical 源路径：

- 正式 raw JSON：`artifacts/raw_results/`
- 可信派生 benchmark：`artifacts/derived/benchmarks/`
- 当前允许引用清单：`paper/ARTIFACTS_ALLOWED.md`
- claim 与证据边界：`paper/claims_evidence_matrix.md`

使用边界：

- `results/benchmark.json`、`results/benchmark_seed1_with_signed_phiz.json`、
  `results/seeds/benchmark_seed*.json` 已删除；这些旧入口会暴露历史 raw key。
- 论文表格和 claim 仍应优先引用 `artifacts/derived/benchmarks/`，`results/` 只作短路径。
- raw JSON 保留在 `artifacts/raw_results/` 作为 provenance，不直接作为论文表格入口。
