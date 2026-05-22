# Derived Artifacts

这里保存从 raw results 派生出的分析表、图、论文候选材料。

## 主要目录

- `benchmarks/`：当前 benchmark 活跃入口，包含 all-model long table、论文主表和 seed1 context。
- `evidence/README_zh.md`：辅助证据索引，指向 diagnostics、qualitative 和 exploratory 摘要。
- `diagnostics/`：appendix / diagnostic 层级的派生证据，不是主 benchmark。
- `paper_figures/qualitative_selected_seed1/`：用户确认的 selected qualitative examples。

## 使用规则

派生产物必须能追溯到 `artifacts/raw_results/`、checkpoint 或生成脚本。
论文 benchmark 表只从 `benchmarks/` 进入；辅助证据先查 `evidence/README_zh.md`。
未确认自动候选、debug/smoke 输出和旧 planning table 不属于 active evidence。

## 源码管理边界

默认进入源码管理的是 `benchmarks/`、`evidence/README_zh.md`、必要 README/索引和少量人工确认的
paper figures。`diagnostics/` 中的大图、debug 图、自动候选池和可再生成中间结果默认只本地保留；
若后续需要作为论文证据，先在 `paper/ARTIFACTS_ALLOWED.md` 中声明用途、协议和 caveat。
