# Current State Capsule

本文档是短恢复入口。长技术细节见 `docs/project/SC-INR_Exp.md`，论文 claim
边界见 `paper/claims_evidence_matrix.md`，可引用产物见
`paper/ARTIFACTS_ALLOWED.md`。

## 当前研究对象

项目研究 arbitrary-scale image super-resolution 中的 decoder-side scale
sampling consistency。当前不宣称 whole-network strict scale equivariance。

推荐表述：

- scale sampling consistency
- sampling-consistent Fourier implicit decoder
- scale-decoupled observation
- decoder-side sampling consistency

禁止表述：

- strict scale equivariance
- scale-equivariant INR
- whole-network scale equivariance

## 当前主方法

论文展示名：`SC-INR`

实现/配置：

- registry name: `sc_inr_signed_phiz`
- train config: `configs/train-div2k/train-sc-inr.yaml`
- canonical checkpoint: `artifacts/checkpoints/seed{1,2,3}/sc-inr`
- seed1 历史 raw key：`SC-INR+PhiZ`；旧 checkpoint 目录名不再保留。

结构：

- `coef(z)`, `omega(z)`, `phi(z)` 由 feature 预测。
- `omega = omega_bound * tanh(raw_omega)`, `omega_bound=2.1`。
- `phi(z)` 是 feature-conditioned phase，不能接收 cell/scale。
- cell/scale 只通过 analytic sinc response 进入完整 `SC-INR`。
- `SC-INR-NoSinc` 仅关闭该 response，实际使用 `W=1`。

## 命名边界

- `SC-INR`: final candidate, signed omega + feature phase + sinc。
- `SC-INR-NoPhi`: previous no-phase mainline, 3-seed 稳证据主体。
- `SC-INR-NoPhi-Signed`: signed omega but no phase。
- `SC-INR-NoSinc`: signed omega + feature phase, no analytic sinc response。
- `LIIF-EQ` / `LTE-EQ`: Rot-E 正交 baseline，不是 SC-INR 消融。
- `SC-INR-EQ`: exploratory Rot-E integration，保留 SC-INR decoder contract；
  目前只作为单 seed extension/diagnostic，不是主方法。
- `LTE-NoCellPhase` / `LTE-PhaseZ`: LTE-side diagnostic。

## 当前证据

较稳证据：

- `SC-INR-NoPhi` 有 3-seed core benchmark，OOD 相对 LTE 小幅稳定提升。
- `SC-INR-NoPhi` seed1 auxiliary 支持 strong same-LR consistency vs LTE。

final `SC-INR` evidence：

- 3-seed benchmark 已补齐，活跃入口为：
  `artifacts/derived/benchmarks/paper_main_3seed_summary.csv` 和
  `artifacts/derived/benchmarks/paper_main_3seed_paired_delta.csv`。
- final `SC-INR`: ID `31.0738 ± 0.0582`, OOD `22.7578 ± 0.0316`,
  ALL `25.5298 ± 0.0403`。
- paired delta vs LIIF: ID `+0.0437 ± 0.0757`, OOD `+0.0328 ± 0.0373`,
  ALL `+0.0364 ± 0.0499`。
- paired delta vs LTE: ID `-0.0048 ± 0.0836`, OOD `+0.0504 ± 0.0417`,
  ALL `+0.0320 ± 0.0555`。
- `SC-INR` vs `SC-INR-NoPhi` 只展示 seed1 结构增量：OOD `+0.0295`,
  ALL `+0.0421`；入口为 `artifacts/derived/benchmarks/paper_context_seed1.csv`，
  不在主表展示 seed2/seed3 两者差异。
- auxiliary 仍是 seed1：strong same-LR consistency vs LTE，但不是 consistency 最强变体。
- qualitative: 用户确认 Urban100 img012 x8 和 img004 x8 selected examples，已在
  `artifacts/derived/paper_figures/qualitative_selected_seed1/` 用当前命名重新导出。

NoSinc evidence：

- full seed1 benchmark 已并入 `artifacts/derived/benchmarks/paper_context_seed1.csv`；
  `SC-INR-NoSinc` vs full `SC-INR`，ID `-0.1397`, OOD `-0.0467`,
  All `-0.0777`。
- auxiliary: NoSinc same-LR self-consistency 远高于 full `SC-INR`，但这是
  zero active cell response 的 caveat，不是更正确 observation 的证据。
- response/omega diagnostics: NoSinc active attenuation 和 cell-only RMSE 均为 0；
  full `SC-INR` 保留有限 analytic response。
- effective amplitude diagnostics:
  `artifacts/derived/diagnostics/effective_amplitude_2026-05-12/` 只作为
  feature-level mechanism sanity check。它统计 MLP 前 coefficient-pair magnitude
  proxy 与 `W(omega,c)` 的关系，不能解释为最终 RGB 频谱振幅、数据集级趋势或
  多 seed 证据。
- footprint oracle diagnostics:
  `artifacts/derived/diagnostics/footprint_oracle_2026-05-15/` 固定 x4 LR/query，
  只改变 cell multiplier，并用 HR piecewise-constant box average 作为 oracle proxy。
  `SC-INR` 在 multiplier 2/4 下的 oracle RMSE 为 `0.02464/0.02148`，低于
  `SC-INR-NoSinc` 的 `0.02612/0.02750`；delta-tracking RMSE 为
  `0.03062/0.04745`，低于 NoSinc 的 `0.03237/0.05134`。该诊断支持 analytic
  footprint response 区分于 cell-insensitive 负控，但不是真实连续图像 exact integral
  或 sinc 唯一因果证明。
- LTE vs SC-INR unified mechanism diagnostics:
  `artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/` 使用 BSD100/Urban100
  前 5 张、中心 crop 64、每图 256 query、cell multiplier 1/2/4。m>1 平均下，
  `SC-INR` delta-tracking RMSE `0.04113`，低于 `LTE` 的 `0.04198` 和
  `SC-INR-NoSinc` 的 `0.04353`；oracle RMSE `0.02512`，低于 `LTE` 的
  `0.02623` 和 NoSinc 的 `0.02729`。`LTE` 的 learned `h_p(c)` delta RMS 为
  `0.51528`，`SC-INR` active response delta RMS 为 `0.30812`，NoSinc 为 `0`。
  该结果直接补强 `LTE phase(cell) -> SC-INR W(omega,c)` 主脉络，但仍是
  crop-based seed1 diagnostic。

SC-INR-EQ exploratory evidence：

- seed1 full benchmark 已完成：
  `artifacts/raw_results/seed1/benchmark_sc_inr_eq.json`，并并入
  `artifacts/derived/benchmarks/paper_context_seed1.csv`。
- `SC-INR-EQ`: ID `31.1513`, OOD `22.7779`, ALL `25.5691`。
- single-seed delta vs `LTE-EQ`: ID `+0.0841`, OOD `+0.0749`,
  ALL `+0.0780`。
- single-seed delta vs final `SC-INR`: ID `+0.0326`, OOD `+0.0014`,
  ALL `+0.0118`；应解释为 exploratory Rot-E integration 基本打平
  final `SC-INR`，不能升级为主方法。
- 6.1-6.6 mechanism suite 的 active 摘要在：
  `artifacts/derived/diagnostics/mechanism_suite_2026-05-12/README_zh.md`；
  完整探索产物只保存在
  `artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/`。
- 6.3 consistency 继续暴露 caveat：`LTE-PhaseZ`、`LTE-NoCellPhase` 和
  `SC-INR-NoSinc` 的 same-LR consistency 都高于完整 `SC-INR`/`SC-INR-EQ`，
  所以该指标只能 diagnostic-only。

## 当前最重要缺口

1. paper table 收敛：benchmark 已有 canonical 入口
   `artifacts/derived/benchmarks/`；下一步只从该入口进入论文表格。
2. footprint correctness 已有 seed1 小样本 diagnostic gate；下一步若进入论文强证据，
   需要扩大覆盖或继续作为 appendix diagnostic。不要继续把 same-LR consistency 当作 sinc 主证据。
3. `SC-INR-EQ` 是 exploratory 分支：benchmark 和机制 suite 已补，但仍缺少更严格的
   旋转通道顺序、group action、cell-only response 和 analytic sinc 积分语义属性测试。
4. 项目健康：旧 overview、旧 planning table 和未确认自动 qualitative candidates 已不应作为活跃入口。

## 当前行动原则

- final `SC-INR` multi-seed 已完成；论文 claim 按 3-seed 中性结论更新。
- 所有新训练使用 canonical name `sc-inr`，按 seed 分目录保存。
- 不覆盖已有 seed1、benchmark、checkpoint 或用户改动。
- 长结果写入 artifact；对话只汇报关键数值、命令、风险。
- 新 claim 必须同步到 `paper/claims_evidence_matrix.md` 和
  `paper/ARTIFACTS_ALLOWED.md`。
- 新 benchmark 表格必须从 `artifacts/derived/benchmarks/` 读取，不直接解释 raw JSON key。
