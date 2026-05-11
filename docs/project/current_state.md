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
- canonical checkpoint aliases: `artifacts/checkpoints/seed{1,2,3}/sc-inr`
- historical physical checkpoint: `artifacts/checkpoints/seed1/sc-inr-phiz`

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
- `LTE-NoCellPhase` / `LTE-PhaseZ`: LTE-side diagnostic。

## 当前证据

较稳证据：

- `SC-INR-NoPhi` 有 3-seed core benchmark，OOD 相对 LTE 小幅稳定提升。
- `SC-INR-NoPhi` seed1 auxiliary 支持 strong same-LR consistency vs LTE。

final `SC-INR` evidence：

- 3-seed benchmark 已补齐：
  `artifacts/derived/analysis/benchmark_progress_2026-05-11/`。
- final `SC-INR`: ID `31.0738 ± 0.0582`, OOD `22.7578 ± 0.0316`,
  ALL `25.5298 ± 0.0403`。
- paired delta vs LIIF: ID `+0.0437 ± 0.0757`, OOD `+0.0328 ± 0.0373`,
  ALL `+0.0364 ± 0.0499`。
- paired delta vs LTE: ID `-0.0048 ± 0.0836`, OOD `+0.0504 ± 0.0417`,
  ALL `+0.0320 ± 0.0555`。
- `SC-INR` vs `SC-INR-NoPhi` 只展示 seed1 结构增量：OOD `+0.0295`,
  ALL `+0.0421`；不在主表展示 seed2/seed3 两者差异。
- auxiliary 仍是 seed1：strong same-LR consistency vs LTE，但不是 consistency 最强变体。
- qualitative: 用户确认 Urban100 img012 x8 和 img004 x8 selected examples，已在
  `artifacts/derived/paper_figures/qualitative_selected_seed1/` 用当前命名重新导出。

NoSinc evidence：

- full seed1 benchmark: `SC-INR-NoSinc` vs full `SC-INR`，ID `-0.1397`,
  OOD `-0.0467`, All `-0.0777`。
- auxiliary: NoSinc same-LR self-consistency 远高于 full `SC-INR`，但这是
  zero active cell response 的 caveat，不是更正确 observation 的证据。
- response/omega diagnostics: NoSinc active attenuation 和 cell-only RMSE 均为 0；
  full `SC-INR` 保留有限 analytic response。

## 当前最重要缺口

1. paper table 收敛：core 3-seed、final `SC-INR` 3-seed、NoSinc ablation、aux/diagnostic。
2. footprint correctness 指标仍缺；不要继续把 same-LR consistency 当作 sinc 主证据。
3. `SC-INR-EQ` 是 exploratory 分支：当前训练正常，但还缺少旋转通道顺序、cell-only response
   和 analytic sinc 积分语义的属性测试。

## 当前行动原则

- final `SC-INR` multi-seed 已完成；论文 claim 按 3-seed 中性结论更新。
- 所有新训练使用 canonical name `sc-inr`，按 seed 分目录保存。
- 不覆盖已有 seed1、benchmark、checkpoint 或用户改动。
- 长结果写入 artifact；对话只汇报关键数值、命令、风险。
- 新 claim 必须同步到 `paper/claims_evidence_matrix.md` 和
  `paper/ARTIFACTS_ALLOWED.md`。
