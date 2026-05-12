# Project Story

本文档是项目叙事入口。短恢复入口优先读 `docs/project/current_state.md`。
旧长版叙事已经删除，避免 `SC-INR-Adaptive` 等历史命名继续污染当前主线。
引用论文事实时应以本文档、`paper/claims_evidence_matrix.md` 和
`paper/ARTIFACTS_ALLOWED.md` 为准。

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

当前最终候选的论文展示名为 **SC-INR**，canonical checkpoint alias 为
`save/sc-inr` / `artifacts/checkpoints/seed*/sc-inr`。历史 raw/checkpoint key
在 seed1 中仍为 `SC-INR+PhiZ` / `sc-inr-phiz`；seed2/3 新评估文件使用清理后的
raw key `SC-INR`。它使用 signed omega 和 feature-conditioned phase：

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
- `SC-INR`：seed1 raw key `SC-INR+PhiZ`，seed2/3 clean raw key `SC-INR`，
  signed omega 加 feature-conditioned phase。
- `SC-INR-NoSinc`：signed omega 加 feature-conditioned phase，但关闭 analytic
  sinc response，实际使用 `W=1`。

不建议把 `LTE-NoCell` 简写为 `LTE-P`。`P` 容易被理解成 phase、parameter、
prior 或 positive，审稿人不容易一眼看出它到底去掉了什么。更清楚的展示名是
`LTE-NoCellPhase`；若图表空间很紧，可用 `LTE-NoCell`.

## 当前证据边界

截至 2026-05-11：

- `SC-INR-NoPhi` 有 3 seed 核心 benchmark，OOD PSNR 相对 LTE 小幅稳定提升。
- 最终候选 `SC-INR` 已有 3 seed benchmark；主展示应比较它相对 LIIF/LTE 的差距。
  `SC-INR` vs `SC-INR-NoPhi` 只展示 seed1 结构增量。seed1 auxiliary metrics 和
  两张用户确认的 qualitative 候选图仍可作为辅助证据。
- `SC-INR-NoSinc` benchmark 与 auxiliary metrics 已完成。Full benchmark 中 NoSinc
  弱于完整 `SC-INR`，但 same-LR self-consistency 反而更高。
- response/omega diagnostics 显示：NoSinc 的高 self-consistency 来自 zero active
  cell response，不是更正确的 footprint observation。

因此论文写作应区分：

- 已较稳的 family-level 结论：scale-decoupled observation 改善 OOD scale
  robustness 和 same-LR consistency；
- 已更新的 final-candidate 结论：`SC-INR` 3-seed benchmark 已完成；论文主表
  以 LIIF/LTE 为参照，NoPhi 差异只放 seed1 context；
- 机制 caveat：same-LR consistency 不能单独证明 analytic sinc response。

## 证据入口

- 模型命名：`paper/model_taxonomy.md`
- claim 边界：`paper/claims_evidence_matrix.md`
- 可引用产物白名单：`paper/ARTIFACTS_ALLOWED.md`
- 短恢复入口：`docs/project/current_state.md`
- 长任务账本：`memory/task_ledger.md`
- canonical benchmark：`artifacts/derived/benchmarks/`
- 辅助证据索引：`artifacts/derived/evidence/README_zh.md`
- 核心 benchmark provenance：`artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/`
- final `SC-INR` 3-seed benchmark provenance：
  `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-11/`
- seed1 final SC-INR auxiliary：`artifacts/derived/diagnostics/sc_inr_final_aux_metrics_seed1/`
- NoSinc auxiliary：`artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/`
- response/omega diagnostics：`artifacts/derived/diagnostics/response_omega_diagnostics_2026-05-10/`
- selected qualitative examples：`artifacts/derived/paper_figures/qualitative_selected_seed1/`

## 下一步

1. 论文表格只从 `artifacts/derived/benchmarks/` 进入。
2. 对辅助证据做 figure/table selection gate，区分正文、appendix 和 diagnostic-only。
3. 设计更直接的 footprint correctness 指标；不要继续堆叠 same-LR
   self-consistency 作为 sinc 主证据。
