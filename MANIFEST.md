# 项目清单

本文档记录 `SC-ASISR` 的稳定结构约定。更详细的科学状态看
`docs/project/current_state.md`，论文证据边界看 `paper/claims_evidence_matrix.md`。

## 目录契约

| 路径 | 角色 | 说明 |
| --- | --- | --- |
| `src/` | 源码 | `src/models`、`src/datasets`、`src/utils.py` 是 canonical 源码位置 |
| `entrypoints/` | 正式 CLI | 训练、评估和少量历史诊断入口 |
| `configs/` | 配置与注册表 | 训练配置和 `configs/registry/` |
| `scripts/analysis/` | 分析脚本 | benchmark 聚合、诊断和机制分析 |
| `scripts/paper/` | 论文产物脚本 | 从已验证结果生成表图候选 |
| `scripts/viz/` | 可视化脚本 | 定性 crop 图导出 |
| `artifacts/checkpoints/` | checkpoint | 权重、训练配置、训练日志 |
| `artifacts/raw_results/` | 原始评估结果 | 正式 benchmark JSON |
| `artifacts/derived/benchmarks/` | 主 benchmark 表 | 论文主表 canonical 来源 |
| `artifacts/derived/diagnostics/` | 辅助诊断 | consistency、response、mechanism suite 摘要 |
| `artifacts/derived/evidence/` | 证据索引 | 辅助证据总入口 |
| `artifacts/derived/paper_figures/` | 论文候选图 | 只保留用户确认的图 |
| `artifacts/legacy/` | 历史留档 | 只作 provenance/audit |
| `artifacts/results/` | results 便捷层 | 只链接 canonical benchmark JSON，不链接旧 raw JSON |
| `experiments/` | 实验卡 | 冻结协议和实验结论 |
| `paper/` | 论文工作区 | claim ledger、artifact 白名单、草稿 |
| `docs/` | 项目文档 | 当前状态和历史 archive |
| `memory/` | 每日记忆 | 中文科研记录；本地保留，不进入源码管理 |

## 兼容路径

根目录只保留以下 symlink：

- `models` -> `src/models`
- `datasets` -> `src/datasets`
- `utils.py` -> `src/utils.py`
- `save` -> `artifacts/checkpoints/seed1`
- `save-seeds` -> `artifacts/checkpoints/seeds`
- `results` -> `artifacts/results`
- `logs` -> `artifacts/logs`
- `Data` -> `artifacts/data_local`

`results` 只保留清晰命名的 canonical benchmark JSON 链接。新脚本、新文档和论文正文应使用 canonical 路径。

## checkpoint 命名契约

checkpoint 目录以论文展示名为 canonical 名称。旧训练名不再作为 checkpoint symlink
保留，只能在 raw-result provenance 或历史配置说明中出现。

| 展示名 | canonical checkpoint |
| --- | --- |
| `LIIF` | `artifacts/checkpoints/seed1/liif` |
| `LTE` | `artifacts/checkpoints/seed1/lte` |
| `LIIF-EQ` | `artifacts/checkpoints/seed1/liif-eq` |
| `LTE-EQ` | `artifacts/checkpoints/seed1/lte-eq` |
| `LTE-NoCellPhase` | `artifacts/checkpoints/seed1/lte-nocellphase` |
| `LTE-PhaseZ` | `artifacts/checkpoints/seed1/lte-phasez` |
| `SC-INR-FixedOmega` | `artifacts/checkpoints/seed1/sc-inr-fixed-omega` |
| `SC-INR-NoPhi` | `artifacts/checkpoints/seed*/sc-inr-nophi` |
| `SC-INR-NoPhi-Signed` | `artifacts/checkpoints/seed1/sc-inr-nophi-signed` |
| `SC-INR` | `artifacts/checkpoints/seed*/sc-inr` |
| `SC-INR-EQ` | `artifacts/checkpoints/seed1/sc-inr-eq` |
| `SC-INR-NoSinc` | `artifacts/checkpoints/seed1/sc-inr-nosinc` |

命名映射以 `configs/registry/models.yaml` 和 `paper/model_taxonomy.md` 为准。
raw JSON 中的历史 key 不改写，只通过注册表解释。

## 证据状态

| Artifact | 状态 | 论文用途 | caveat |
| --- | --- | --- | --- |
| `artifacts/raw_results/seed1/benchmark.json` | formal raw | seed1 baseline | 不含 Signed/PhiZ |
| `artifacts/raw_results/seed1/benchmark_signed_phiz.json` | formal raw | seed1 Signed/PhiZ | 单 seed 新变体 |
| `artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json` | formal raw | seed1 NoSinc | 单 seed 消融 |
| `artifacts/raw_results/seed1/benchmark_sc_inr_eq.json` | formal raw | seed1 SC-INR-EQ | exploratory extension |
| `artifacts/raw_results/seed2/benchmark.json` | formal raw | core multi-seed | core models |
| `artifacts/raw_results/seed2/benchmark_sc_inr.json` | formal raw | final SC-INR seed2 | clean raw key |
| `artifacts/raw_results/seed3/benchmark.json` | formal raw | core multi-seed | core models |
| `artifacts/raw_results/seed3/benchmark_sc_inr.json` | formal raw | final SC-INR seed3 | clean raw key |
| `artifacts/derived/benchmarks/` | derived | 主 benchmark 表 | paper benchmark canonical 来源 |
| `artifacts/derived/benchmarks/benchmark_all_models_canonical.json` | derived | 全模型 benchmark JSON | 顶层模型名均为 canonical，含 `SC-INR-EQ` |
| `artifacts/derived/benchmarks/benchmark_paper_main_3seed.json` | derived | 主模型三 seed JSON | 只含 LIIF/LTE/SC-INR |
| `artifacts/derived/evidence/README_zh.md` | derived | 辅助证据索引 | 指向 diagnostics 与 selected figures |
| `artifacts/derived/diagnostics/` | diagnostic | 机制和 consistency 支撑 | 不替代主 benchmark |
| `artifacts/derived/paper_figures/qualitative_selected_seed1/` | candidate | 定性图 | 只支持 selected examples |
| `artifacts/legacy/` | legacy | audit/provenance | 不能直接作为当前证据 |

严格可引用范围以 `paper/ARTIFACTS_ALLOWED.md` 为准。

## 当前 claim 边界

支持：

- `SC-INR` 相对 LIIF/LTE 的小幅 OOD/ALL PSNR 正增益；
- decoder-side sampling consistency / scale-decoupled observation；
- 机制诊断可作为解释辅助。

不支持：

- strict scale equivariance；
- `SC-INR-EQ` 作为主方法；
- 单一 consistency 分数或单 seed 诊断证明机制因果。
