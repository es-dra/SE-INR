# Canonical Benchmarks

本目录是当前论文 benchmark 的活跃入口。它不改动 raw JSON，而是把每一行
paper-facing 结果显式展开为 `source_file`、`raw_key`、`canonical_model`、
`model_status`、`seed`、`dataset`、`scale`、`checkpoint`、`protocol_id` 和
`raw_key_semantics`，避免后续再从文件名或 raw key 猜模型语义。

## 文件

- `benchmark_all_models_long.csv`：所有 benchmark 记录的 canonical long table。
- `benchmark_by_seed_model.csv`：每个 seed/model 的 ID/OOD/ALL 汇总。
- `paper_main_3seed.csv`：论文主表使用的 LIIF/LTE/SC-INR per-seed benchmark。
- `paper_main_3seed_summary.csv`：论文主表 mean/std 汇总。
- `paper_main_3seed_paired_delta.csv`：`SC-INR` 相对 LIIF/LTE 的 paired delta。
- `paper_context_seed1.csv`：seed1 context/diagnostic 模型，包括 NoPhi、NoSinc、
  EQ 和 LTE-side diagnostics。

## 命名 caveat

`artifacts/raw_results/seed1/benchmark_signed_phiz.json` 中的 raw key
`SC-INR` 表示 `SC-INR-NoPhi`，seed1 final candidate 是 raw key
`SC-INR+PhiZ`。`seed2/benchmark_sc_inr.json` 和
`seed3/benchmark_sc_inr.json` 中的 clean raw key `SC-INR` 才表示 final
candidate。论文表格、claim 和后续 agent 恢复应优先读取本目录，而不是直接读取
raw JSON key。
