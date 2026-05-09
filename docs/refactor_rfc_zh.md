# 工作区重构 RFC

## 目标

把当前脚本堆叠型研究目录重构为长期可维护的论文级研究工程，支持当前
SC-INR / SC-INR+PhiZ 论文，也支持后续 Rot-E 结合、LIIF+、sinc 消融、多
seed 复现和机制诊断。

## 核心原则

1. 不删除 checkpoint、raw JSON、config、log。
2. canonical 路径与 compatibility 路径分离。
3. raw / derived / smoke / legacy 必须分层。
4. 模型展示名、registry name、config、checkpoint dir 通过统一 registry 映射。
5. 论文 claim 必须能追溯到 experiment card 和 raw artifact。
6. 先轻量 registry/manifest 化，不引入大型实验框架。

## 已实施结构

```text
src/                         # canonical source code
entrypoints/                  # train/eval CLI
scripts/analysis/             # analysis scripts
scripts/paper/                # paper artifact preparation
scripts/viz/                  # qualitative figures
scripts/legacy/               # historical launchers
artifacts/checkpoints/        # checkpoints/configs/log.txt
artifacts/raw_results/        # formal raw JSON
artifacts/derived/            # derived tables/figures
artifacts/smoke/              # debug/smoke outputs
artifacts/legacy/             # historical outputs
artifacts/results/            # old results/ compatibility tree
configs/registry/             # model/protocol registry
experiments/                  # experiment cards
paper/                        # claims/evidence and paper workspace
docs/                         # project docs and historical archive
```

## Compatibility Contract

Root-level legacy paths remain as symlinks during transition:

- `models`, `datasets`, `utils.py`
- `train.py`, `test.py`, `eval_full.py`, `eval_continuous.py`, `eval_fce.py`,
  `eval_phase_intervention.py`
- `save`, `save-seeds`, `results`, `logs`, `figs`

This keeps existing commands usable while new code migrates to canonical paths.

## Artifact Status Definitions

- `formal_raw`: raw metrics/checkpoints/logs that define an experiment result.
- `derived`: computed summaries, tables, or plots derived from formal raw files.
- `candidate`: paper candidate material requiring human/evidence audit.
- `smoke`: debug or smoke-test output; not citable.
- `legacy`: historical material kept for traceability.
- `deprecated`: superseded and not recommended for new work.

## Immediate Follow-up

1. Migrate hard-coded model lists in evaluation/analysis scripts to
   `configs/registry/models.yaml`.
2. Add experiment cards for:
   - seed1 core benchmark;
   - seed1 auxiliary consistency;
   - signed omega;
   - qualitative PhiZ;
   - future w/o-sinc ablation.
3. Promote only audited paper figures/tables into `paper/`.
4. Add lightweight validation commands for path existence, registry consistency,
   and result JSON completeness.

## Non-goals

- No wholesale rewrite into Hydra/Lightning.
- No checkpoint renaming.
- No deletion of historical outputs before paper evidence stabilizes.
- No strict package installation requirement yet; symlink compatibility remains.
