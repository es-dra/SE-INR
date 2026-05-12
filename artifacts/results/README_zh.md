# results 兼容路径说明

`results/` 是旧路径兼容层，真实目标是 `artifacts/results/`。这里仅保留少量指向
`artifacts/raw_results/` 的 symlink，方便旧命令读取历史 raw JSON。

论文、报告和新脚本应优先使用 canonical 路径：

- 正式 raw JSON：`artifacts/raw_results/`
- 可信派生表图：`artifacts/derived/`
- 当前允许引用清单：`paper/ARTIFACTS_ALLOWED.md`
- claim 与证据边界：`paper/claims_evidence_matrix.md`

不要因为某个文件能从 `results/...` 打开就默认可引用。特别注意：

- 旧多 seed 表 symlink 已从兼容层删除；当前多 seed 表在
  `artifacts/derived/benchmarks/`。
- 旧 debug、analysis 和候选图池 symlink 已删除；当前定性图入口以
  `artifacts/derived/paper_figures/qualitative_selected_seed1/` 和
  `paper/ARTIFACTS_ALLOWED.md` 为准。
