# Project Story

本文档是当前项目叙事入口。历史长版记录已归档到
`docs/archive/project_story_legacy_2026-05-09.md`，其中包含旧命名
`SC-INR-Adaptive` 和早期结果解读；引用论文事实时应以本文档、
`paper/claims_evidence_matrix.md` 和 `paper/ARTIFACTS_ALLOWED.md` 为准。

## 研究问题

本项目研究 arbitrary-scale image super-resolution 中的 decoder-side scale
sampling consistency。核心问题不是证明整个网络严格 scale equivariant，而是
约束 output cell/scale 进入 implicit decoder 的方式，减少 LTE 中
cell-conditioned learned phase 在 OOD scale 上的任意外推。

当前推荐表述：

- scale sampling consistency
- sampling-consistent Fourier implicit decoder
- scale-decoupled observation
- decoder-side sampling consistency

避免表述：

- strict scale equivariance
- scale-equivariant INR
- whole-network scale equivariance

## 当前方法

LTE 中 Fourier 频率和系数来自图像特征，但 phase 由 output cell 直接预测。
SC-INR 系列将内容 Fourier basis 与 observation footprint 分开：

- 内容项由 feature 预测：`coef(z)`、`omega(z)`、可选 `phi(z)`；
- 观测 footprint 由 cell 通过解析 sinc response 进入；
- 禁止 learned `phase(cell)` shortcut。

当前最终候选的论文展示名建议为 **SC-INR**，canonical checkpoint alias 为
`save/sc-inr` / `artifacts/checkpoints/seed1/sc-inr`。历史 raw/checkpoint key
仍为 `SC-INR+PhiZ` / `sc-inr-phiz`。它使用 signed omega 和
feature-conditioned phase：

```text
coef(z), omega(z), phi(z)  -> content Fourier basis
cell                       -> analytic sinc response
```

raw JSON、训练日志和历史物理 checkpoint 目录不改写；新训练、新评估和论文展示
使用 canonical alias。展示层命名由 `paper/model_taxonomy.md` 和
`configs/registry/models.yaml` 约束。

## 实验系列

当前模型分为三类。

**基础 ASISR baseline**

- `LIIF`
- `LTE`

**正交旋转等变 baseline**

- `LIIF-EQ`
- `LTE-EQ`

它们用于说明 Rot-E 方向是正交增强，不应作为 SC-INR decoder-side consistency
的互斥替代。

**机制诊断与消融**

- `LTE-NoCellPhase`：raw key `LTE-NoCell`，去掉 LTE 的 cell-conditioned phase。
- `LTE-PhaseZ`：raw key `LTE-FeaturePhase`，将 LTE phase 改为 feature-conditioned。
- `SC-INR-FixedOmega`：raw key `SC-INR-Fixed`，固定频率加解析 sinc。
- `SC-INR-NoPhi`：raw key `SC-INR` / `SC-INR-Adaptive`，自适应频率但无 phase。
- `SC-INR-NoPhi-Signed`：raw key `SC-INR-Signed`，signed omega 但无 phase。
- `SC-INR`：raw key `SC-INR+PhiZ`，signed omega 加 feature-conditioned phase。

不建议把 `LTE-NoCell` 简写为 `LTE-P`。`P` 容易被理解成 phase、parameter、
prior 或 positive，审稿人不容易一眼看出它到底去掉了什么。更清楚的展示名是
`LTE-NoCellPhase`；若图表空间很紧，可用 `LTE-NoCell`.

## 当前证据边界

截至 2026-05-09：

- `SC-INR-NoPhi` 有 3 seed 核心 benchmark，OOD PSNR 相对 LTE 小幅稳定提升。
- 最终候选 `SC-INR`（raw `SC-INR+PhiZ`）目前只有 seed1 benchmark 和两张用户
  确认的 qualitative 候选图。
- seed1 auxiliary metrics 支持 decoder-side same-LR cross-scale observation
  consistency，但尚未覆盖最终候选 `SC-INR`。
- 还缺少 `SC-INR` 的 auxiliary metrics、多 seed、w/o-sinc 消融和最终论文图。

因此论文写作应区分：

- 已较稳的 family-level 结论：scale-decoupled observation 改善 OOD scale
  robustness 和 same-LR consistency；
- 仍需验证的 final-candidate 结论：feature-conditioned phase 是否在多 seed
  与 consistency 指标上保持优势。

## 证据入口

- 模型命名：`paper/model_taxonomy.md`
- claim 边界：`paper/claims_evidence_matrix.md`
- 可引用产物白名单：`paper/ARTIFACTS_ALLOWED.md`
- 核心 benchmark：`artifacts/derived/analysis/benchmark_progress_2026-05-09/`
- seed1 auxiliary metrics：`artifacts/derived/analysis/seed1_aux_metrics_all8/`
- qualitative 候选池：`artifacts/derived/paper_candidates/qualitative_phiz_candidates/`

## 下一步

1. 跑最终候选 `SC-INR` 的 auxiliary metrics。
2. 设计并训练 `SC-INR w/o sinc` 或等价 sinc 消融。
3. 将用户确认的两张 qualitative 候选图升级为正式 figure draft。
4. 若 auxiliary metrics 不显示 consistency 退化，再推进最终候选的多 seed。
