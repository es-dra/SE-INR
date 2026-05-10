# Model Taxonomy And Display Names

本文档约束论文表格、图例和正文中的模型展示名。工程目录、checkpoint 目录、
raw JSON key 不做物理重命名；展示层通过本表映射。checkpoint 使用 canonical
alias 指向历史物理目录，避免新实验继续扩散旧命名。

## 命名原则

1. 最终候选使用简洁主名 `SC-INR`。
2. 旧主模型和诊断模型应变成围绕最终模型的消融或机制对照。
3. raw key 与历史物理 checkpoint 目录保留可追溯；新入口使用 canonical alias。
4. 图表中优先使用短但明确的名字，避免只用单字母缩写。

## 推荐展示表

| 展示名 | raw/result key | canonical checkpoint | 角色 | 说明 |
| --- | --- | --- | --- | --- |
| `LIIF` | `LIIF` | `artifacts/checkpoints/seed1/liif` | baseline | 标准 LIIF。 |
| `LTE` | `LTE` | `artifacts/checkpoints/seed1/lte` | baseline | 标准 LTE，含 cell-conditioned phase。 |
| `LIIF-EQ` | `LIIF-EQ` | `artifacts/checkpoints/seed1/liif-eq` | orthogonal baseline | Rot-E 旋转等变版本。 |
| `LTE-EQ` | `LTE-EQ` | `artifacts/checkpoints/seed1/lte-eq` | orthogonal baseline | Rot-E 旋转等变版本。 |
| `LTE-NoCellPhase` | `LTE-NoCell` | `artifacts/checkpoints/seed1/lte-nocellphase` | diagnostic | 移除 LTE 的 learned `phase(cell)`。 |
| `LTE-PhaseZ` | `LTE-FeaturePhase` | `artifacts/checkpoints/seed1/lte-phasez` | diagnostic | LTE phase 改为 feature-conditioned `phi(z)`。 |
| `SC-INR-FixedOmega` | `SC-INR-Fixed` | `artifacts/checkpoints/seed1/sc-inr-fixed-omega` | ablation | 固定 frequency basis + analytic sinc。 |
| `SC-INR-NoPhi` | `SC-INR`, `SC-INR-Adaptive` | `artifacts/checkpoints/seed1/sc-inr-nophi` | ablation / previous main | feature-conditioned omega，无 phase；旧论文名为 SC-INR-Adaptive。 |
| `SC-INR-NoPhi-Signed` | `SC-INR-Signed`, `SC-INR-Adaptive-Signed` | `artifacts/checkpoints/seed1/sc-inr-nophi-signed` | ablation | signed omega，无 phase。 |
| `SC-INR` | `SC-INR+PhiZ` | `artifacts/checkpoints/seed1/sc-inr` | final candidate | signed omega + feature-conditioned phase。 |
| `SC-INR-NoSinc` | `SC-INR-NoSinc` | `artifacts/checkpoints/seed1/sc-inr-nosinc` | ablation | signed omega + feature-conditioned phase，但移除 analytic sinc response；seed1 benchmark 和 auxiliary metrics 已完成。 |

## 对用户命名方案的判断

将最终 `SC-INR-PhaseZ` 简化为 `SC-INR` 是合理的，因为论文主方法应该有一个
干净的名字；`PhaseZ` 更适合放在消融说明里，而不是长期挂在主方法名上。

但不建议把所有对照都强行改成 `SC-INR-*`：

- `LIIF`、`LTE` 是外部 baseline，不应变成 SC-INR 消融；
- `LIIF-EQ`、`LTE-EQ` 是正交的 Rot-E baseline，也不应伪装成 SC-INR 消融；
- `LTE-NoCellPhase` 和 `LTE-PhaseZ` 是 LTE-side 机制诊断，保留 LTE 前缀更诚实；
- 真正的 SC-INR 消融应集中在 `FixedOmega`、`NoPhi`、`NoPhi-Signed`、`w/o sinc`
  等结构轴上。

不推荐 `LTE-P` 这个名字。它太短且语义不唯一，不能直接表达“移除
cell-conditioned phase”。图表空间足够时用 `LTE-NoCellPhase`；空间不足时可用
`LTE-NoCell`。

## 当前证据注意事项

- 3 seed OOD 稳定性目前属于 `SC-INR-NoPhi`，不是最终候选 `SC-INR`。
- 最终候选 `SC-INR` 目前对应 raw key `SC-INR+PhiZ`，证据仍是 seed1 preliminary。
- 正文若提前采用最终命名 `SC-INR`，必须在实验表注或方法说明中写明：
  `SC-INR` corresponds to the feature-conditioned phase variant; `SC-INR-NoPhi`
  denotes the previous no-phase variant.
- `save/...` 是 compatibility symlink，等价于 `artifacts/checkpoints/seed1/...`。
  新实验优先写 canonical alias，如 `save/sc-inr`；历史物理目录只在 provenance
  或 raw-result 说明中出现。
- `SC-INR-NoSinc` 的 same-LR self-consistency 很高，不能被解读为 NoSinc 更好；
  它说明当前 consistency 指标会奖励 cell-independent decoder，需要结合 fidelity
  和 full benchmark 解释。
