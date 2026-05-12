# 模型命名与展示名

本文档约束论文表格、图例和正文中的模型展示名。`configs/registry/models.yaml`
是机器可读注册表；本文档负责解释命名理由和使用边界。

## 命名原则

1. 最终候选主方法使用 `SC-INR`。
2. 旧主线和结构变化写成消融或机制对照，不继续扩散旧工程名。
3. raw JSON key 和历史训练名只作为 provenance 保留。
4. checkpoint 真实目录使用 canonical 名称；历史训练名保留为 symlink。
5. 图表中优先使用清晰展示名，避免只用含义不明的短缩写。

## 推荐展示表

| 展示名 | raw/result key | canonical checkpoint | 角色 | 说明 |
| --- | --- | --- | --- | --- |
| `LIIF` | `LIIF` | `artifacts/checkpoints/seed1/liif` | baseline | 标准 LIIF |
| `LTE` | `LTE` | `artifacts/checkpoints/seed1/lte` | baseline | 标准 LTE，含 cell-conditioned phase |
| `LIIF-EQ` | `LIIF-EQ` | `artifacts/checkpoints/seed1/liif-eq` | 正交 baseline | Rot-E 旋转等变版本 |
| `LTE-EQ` | `LTE-EQ` | `artifacts/checkpoints/seed1/lte-eq` | 正交 baseline | Rot-E 旋转等变版本 |
| `LTE-NoCellPhase` | `LTE-NoCell` | `artifacts/checkpoints/seed1/lte-nocellphase` | diagnostic | 移除 LTE 的 learned `phase(cell)` |
| `LTE-PhaseZ` | `LTE-FeaturePhase` | `artifacts/checkpoints/seed1/lte-phasez` | diagnostic | LTE phase 改为 feature-conditioned `phi(z)` |
| `SC-INR-FixedOmega` | `SC-INR-Fixed` | `artifacts/checkpoints/seed1/sc-inr-fixed-omega` | ablation | 固定 frequency basis + analytic sinc |
| `SC-INR-NoPhi` | legacy `SC-INR` / `SC-INR-Adaptive` | `artifacts/checkpoints/seed*/sc-inr-nophi` | 旧主线/消融 | feature-conditioned omega，无 phase |
| `SC-INR-NoPhi-Signed` | `SC-INR-Signed` | `artifacts/checkpoints/seed1/sc-inr-nophi-signed` | ablation | signed omega，无 phase |
| `SC-INR` | seed1 `SC-INR+PhiZ`; seed2/3 clean `SC-INR` | `artifacts/checkpoints/seed*/sc-inr` | 最终候选 | signed omega + feature-conditioned phase |
| `SC-INR-EQ` | `SC-INR-EQ` | `artifacts/checkpoints/seed1/sc-inr-eq` | exploratory | Rot-E plumbing + SC-INR decoder contract，不是主方法 |
| `SC-INR-NoSinc` | `SC-INR-NoSinc` | `artifacts/checkpoints/seed1/sc-inr-nosinc` | ablation | 移除 analytic sinc response |

## 关键边界

- `SC-INR` 是最终候选，不再写作 `SC-INR+PhiZ`。
- `SC-INR-NoPhi` 才是旧 raw key `SC-INR` 在部分 seed1/core 结果中的含义。
- `SC-INR-EQ` 只能写成可结合 Rot-E 的探索性扩展，不能作为新主方法。
- `SC-INR-NoSinc` 的 same-LR consistency 很高，说明该指标会奖励 cell-insensitive
  decoder；不能据此说 NoSinc 更符合 observation modeling。

## 论文写法建议

正文可以写：

> We denote the final feature-phase variant as SC-INR. The previous no-phase
> variant is reported as SC-INR-NoPhi.

不要写：

> SC-INR is strictly scale-equivariant.

也不要把历史 raw key、训练目录名或旧日志名放进正文主表。它们只应出现在
provenance、artifact 白名单或复现说明里。
