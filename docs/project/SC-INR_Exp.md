# SC-INR 历史技术记录

> 本文档是历史技术记录，只用于追溯早期 SC-INR 系列实验和命名迁移。
> 当前恢复入口请优先读 `docs/project/current_state.md`，模型/文献底稿请读
> `docs/project/model_details_liif_lte_scinr.md`，论文 claim 边界请读
> `paper/claims_evidence_matrix.md`，可引用 artifact 请读
> `paper/ARTIFACTS_ALLOWED.md`。
>
> 本文中的旧命名、旧 seed1 叙事和阶段性判断不得绕过上述 canonical 入口直接用于论文。

本文档记录当前 SC-INR 系列的模型动机、实现细节、命名体系和已有实验事实。
它是项目内部技术记录，不是最终论文正文。论文可引用材料以
`paper/ARTIFACTS_ALLOWED.md` 和 `paper/claims_evidence_matrix.md` 为准。

## 1. 问题定义

Arbitrary-scale image super-resolution (ASISR) 的目标是在连续放大倍率下从
低分辨率图像生成高分辨率图像。当前项目关注的问题不是完整网络的严格
scale equivariance，而是 decoder-side scale sampling consistency：

> 输出像素的 cell/scale 应该作为观测 footprint 进入隐式解码器，而不是通过
> learned cell-conditioned phase 任意改变局部 Fourier 函数。

推荐论文表述：

- scale sampling consistency
- sampling-consistent Fourier implicit decoder
- scale-decoupled observation
- decoder-side sampling consistency

禁止过度表述：

- strict scale equivariance
- scale-equivariant INR
- whole-network scale equivariance

## 2. 统一局部隐式框架

给定低分辨率输入图像 $I_{\mathrm{LR}}$，encoder 输出特征图：

$$
M = E(I_{\mathrm{LR}}).
$$

对目标坐标 $x_q$，在特征网格中取邻近 anchor $v$，得到局部特征和相对坐标：

$$
z=M(v), \qquad \delta=x_q-v.
$$

局部隐式解码器输出：

$$
\hat y = D(z,\delta,c),
$$

其中 $c$ 是输出像素在连续坐标域中的 cell size / footprint。LIIF、LTE 和
SC-INR 的核心差别是 $D$ 的构造方式，以及 $c$ 如何进入 $D$。

| 展示名 | 高频表达 | cell/scale 进入方式 | 当前角色 |
| --- | --- | --- | --- |
| `LIIF` | MLP 隐式函数 | 直接拼接进 MLP | baseline |
| `LTE` | feature-conditioned Fourier basis | learned `phase(cell)` | baseline |
| `LTE-NoCellPhase` | LTE Fourier basis | 移除 `phase(cell)` | LTE-side diagnostic |
| `LTE-PhaseZ` | LTE Fourier basis | feature-conditioned `phi(z)` | LTE-side diagnostic |
| `SC-INR-NoPhi` | Fourier basis + analytic sinc | cell 只进入 sinc；无 phase | 3-seed core variant |
| `SC-INR-NoPhi-Signed` | signed omega + sinc | cell 只进入 sinc；无 phase | signed omega ablation |
| `SC-INR` | signed omega + `phi(z)` + sinc | cell 只进入 sinc；phase 由 feature 预测 | final candidate |
| `SC-INR-NoSinc` | signed omega + `phi(z)` | 移除 analytic sinc response | seed1 benchmark and auxiliary complete; exposes consistency-metric caveat |

## 3. LTE 的机制风险

LTE 使用图像特征预测 coefficient 和 frequency：

$$
A=A(z), \qquad \omega=\omega(z).
$$

但 LTE 同时使用 cell 预测 phase：

$$
\cos\{\pi [\omega(z)^\top\delta + h_p(c)]\},\qquad
\sin\{\pi [\omega(z)^\top\delta + h_p(c)]\}.
$$

代码对应 `src/models/lte.py`：

- `coef = Conv2d(feature)`；
- `freq = Conv2d(feature)`；
- `phase = Linear(cell)`；
- `phase(cell)` 直接加到 Fourier argument。

这带来一个研究风险：同一个局部内容 $z$ 在不同 cell 下不再只是同一个连续函数
的不同观测，而更像是不同函数：

$$
f_{z,c}(\delta).
$$

训练尺度内 `phase(cell)` 可能提高拟合能力，但在 x6/x8/x16/x30 等训练外尺度上，
这个自由 learned mapping 缺乏采样理论约束。

## 4. 从像素观测到 sinc response

如果像素不是点采样，而是 footprint 上的平均观测，则一维 Fourier 分量

$$
f(x)=A\cos(\pi\omega x+\phi)
$$

在 cell 宽度 $c$ 上的 box average 为：

$$
\frac{1}{c}\int_{x_0-c/2}^{x_0+c/2} A\cos(\pi\omega x+\phi)\,dx
=
A\cos(\pi\omega x_0+\phi)\operatorname{sinc}\left(\frac{\omega c}{2}\right),
$$

其中使用 normalized sinc：

$$
\operatorname{sinc}(t)=\frac{\sin(\pi t)}{\pi t}.
$$

二维矩形 footprint 下：

$$
W(\omega,c)=
\operatorname{sinc}\left(\frac{\omega_h c_h}{2}\right)
\operatorname{sinc}\left(\frac{\omega_w c_w}{2}\right).
$$

因此，cell 的更自然角色是频率分量的解析观测响应，而不是 learned phase shift。

## 5. 当前 SC-INR 架构

当前最终候选展示名为 `SC-INR`，canonical checkpoint 为
`artifacts/checkpoints/seed{1,2,3}/sc-inr`（seed1 兼容路径 `save/sc-inr`）。
seed1 历史 raw key 为 `SC-INR+PhiZ`；旧 checkpoint 目录名 `sc-inr-phiz`
不再保留。seed2/3 新评估文件使用清理后的 raw key `SC-INR`。实现位于
`src/models/sc_inr_adaptive.py`，registry name 为 `sc_inr_signed_phiz`。

### 5.1 内容项

`SC-INR` 从特征预测 Fourier 内容参数：

$$
A=A(z), \qquad \omega=\omega(z), \qquad \phi=\phi(z).
$$

最终候选采用 signed bounded omega：

$$
\omega(z)=\omega_{\max}\tanh(g_\omega(z)),
$$

当前配置 `omega_bound=2.1`。这修复了旧 softplus Cartesian omega 只能取非负
分量、方向落在第一象限的风险。

Feature phase 使用 `phase_conv(z)`，当前配置：

- `learn_phase: true`
- `phase_kernel_size: 1`
- `phase_bias: false`
- `phase_zero_init: true`

重要约束：$\phi(z)$ 只能由 feature 预测，不能接收 cell/scale。

### 5.2 观测项

cell 只通过解析 sinc response 进入：

$$
W(\omega,c)=
\operatorname{sinc}\left(\frac{\omega_h c_h}{2}\right)
\operatorname{sinc}\left(\frac{\omega_w c_w}{2}\right).
$$

最终输入 MLP 的局部 Fourier 表示可写为：

$$
A(z)\odot
\left[
\cos\left(\pi(\omega(z)^\top\delta+\phi(z))\right),
\sin\left(\pi(\omega(z)^\top\delta+\phi(z))\right)
\right]
\odot W(\omega(z),c).
$$

输出还保留 LTE/LIIF 系列中的 local ensemble 和 LR bilinear skip connection。

### 5.3 坐标单位

数据 wrapper 输出 normalized `[-1,1]` 坐标和 cell。查询时：

- `rel_coord = coord - q_coord` 后乘以 feature map 的空间尺寸；
- `rel_cell` 也乘以同一 feature map 空间尺寸；
- 因此 Fourier phase 和 sinc response 中的 $\delta$、$c$、$\omega$ 单位一致；
- 代码中第 0 个坐标维对应 height，第 1 个坐标维对应 width。

论文公式如果写 $x/y$，必须明确它们对应代码中的 H/W 顺序，避免把
$\omega_x c_x$ 与 $\omega_y c_y$ 写反。

## 6. 命名与消融体系

当前展示名以 `paper/model_taxonomy.md` 为准。

| 展示名 | raw/result key | canonical checkpoint | 说明 |
| --- | --- | --- | --- |
| `SC-INR` | seed1 `SC-INR+PhiZ`; seed2/3 clean `SC-INR` | `artifacts/checkpoints/seed1/sc-inr` | 最终候选：signed omega + feature phase + sinc |
| `SC-INR-NoPhi` | `SC-INR`, `SC-INR-Adaptive` | `artifacts/checkpoints/seed1/sc-inr-nophi` | 旧主线：adaptive omega + sinc，无 phase |
| `SC-INR-NoPhi-Signed` | `SC-INR-Signed` | `artifacts/checkpoints/seed1/sc-inr-nophi-signed` | signed omega，无 phase |
| `SC-INR-FixedOmega` | `SC-INR-Fixed` | `artifacts/checkpoints/seed1/sc-inr-fixed-omega` | fixed omega + sinc |
| `SC-INR-NoSinc` | `SC-INR-NoSinc` | `artifacts/checkpoints/seed1/sc-inr-nosinc` | signed omega + feature phase，但移除 sinc |
| `LTE-NoCellPhase` | `LTE-NoCell`, `LTE-NoC` | `artifacts/checkpoints/seed1/lte-nocellphase` | LTE 诊断：移除 cell phase |
| `LTE-PhaseZ` | `LTE-FeaturePhase` | `artifacts/checkpoints/seed1/lte-phasez` | LTE 诊断：phase 改为 feature-conditioned |

不建议将 `LIIF`、`LTE`、`LIIF-EQ`、`LTE-EQ` 改成 `SC-INR-*`。它们是 baseline
或 Rot-E 正交 baseline。也不建议使用 `LTE-P`，因为 `P` 的语义不唯一。

checkpoint 目录和旧 raw key 不物理重命名；展示层通过 registry 和分析脚本映射。

## 7. 当前实验事实

### 7.1 seed1 最终候选 benchmark

来源：`artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/seed1_signed_phiz_summary.csv`。

| Model | ID PSNR | OOD PSNR | All PSNR | Delta OOD vs LTE | Delta OOD vs SC-INR-NoPhi |
| --- | ---: | ---: | ---: | ---: | ---: |
| `LIIF` | 30.9992 | 22.7111 | 25.4738 | +0.0195 | -0.0359 |
| `LTE` | 31.0450 | 22.6917 | 25.4761 | +0.0000 | -0.0554 |
| `SC-INR-NoPhi` | 31.0515 | 22.7471 | 25.5152 | +0.0554 | +0.0000 |
| `SC-INR-NoPhi-Signed` | 31.0567 | 22.7418 | 25.5134 | +0.0501 | -0.0053 |
| `SC-INR` | 31.1187 | 22.7766 | 25.5573 | +0.0849 | +0.0295 |
| `LIIF-EQ` | 31.1205 | 22.7469 | 25.5381 | +0.0552 | -0.0002 |

解释边界：

- `SC-INR` 是当前 seed1 PSNR 最强候选；
- `SC-INR` 相比 `SC-INR-NoPhi` 的 OOD 增益为 `+0.0295 dB`；
- 这是 `SC-INR` vs `SC-INR-NoPhi` 的展示位置；seed2/seed3 不需要继续展示两者差异。

### 7.2 核心 3-seed 与最终候选结果

来源：`artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/multiseed_core_summary.csv`。

| Model | Split | Mean PSNR | Std | Delta Mean vs LTE |
| --- | --- | ---: | ---: | ---: |
| `LIIF` | ID | 31.0301 | 0.0274 | -0.0485 |
| `LIIF` | OOD | 22.7251 | 0.0124 | +0.0177 |
| `LTE` | ID | 31.0786 | 0.0296 | +0.0000 |
| `LTE` | OOD | 22.7074 | 0.0137 | +0.0000 |
| `SC-INR-NoPhi` | ID | 31.0746 | 0.0203 | -0.0040 |
| `SC-INR-NoPhi` | OOD | 22.7581 | 0.0105 | +0.0507 |

解释边界：

- 这张表用于 family-level decoder-side sampling consistency 证据。
- 最终候选 `SC-INR` 的 3-seed 主展示应使用
  `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-11/final_sc_inr_vs_liif_lte_paper.csv`，
  即比较 `SC-INR` 与 `LIIF`、`LTE`，不在主表展示 seed2/seed3 的
  `SC-INR` vs `SC-INR-NoPhi` 差异。

### 7.3 auxiliary consistency evidence

旧的 all-model auxiliary evidence 来源于
`artifacts/derived/diagnostics/seed1_aux_metrics_all8/`，主要用于支撑
`SC-INR-NoPhi` 的机制判断。

Same-LR cross-scale observation consistency 相对 `LTE`：

| Model | Dataset | Consistency PSNR | Delta vs LTE |
| --- | --- | ---: | ---: |
| `LTE` | BSD100 | 44.1914 | +0.0000 |
| `SC-INR-NoPhi` | BSD100 | 52.8053 | +8.6140 |
| `LTE` | Urban100 | 39.2983 | +0.0000 |
| `SC-INR-NoPhi` | Urban100 | 47.0694 | +7.7712 |

这支持 `SC-INR-NoPhi` 的 decoder-side sampling consistency 解释。但
`LTE-NoCellPhase` 和 `LTE-PhaseZ` 的 consistency 也很高，因为它们削弱了 cell
响应，不能单独据此宣称它们更好，必须结合重建 PSNR、SSIM 和纹理误差。

最终候选 `SC-INR` 的 seed1 auxiliary metrics 位于
`artifacts/derived/diagnostics/sc_inr_final_aux_metrics_seed1/`。协议覆盖
`LTE`、`SC-INR-NoPhi`、`SC-INR-NoPhi-Signed`、`SC-INR`，BSD100 / Urban100
sorted 前 10 张，x4/x8/x16/x30 quality 和 x8/x16/x30 -> x4 consistency。

`SC-INR` 相对 `LTE` 保持强 same-LR consistency 优势：

| Dataset | LTE | `SC-INR` | SC-INR - LTE |
| --- | ---: | ---: | ---: |
| BSD100 | 44.1914 | 52.1421 | +7.9508 |
| Urban100 | 39.2979 | 47.1128 | +7.8149 |

但 `SC-INR` 不是 consistency 最强变体：BSD100 上低于 `SC-INR-NoPhi` 和
`SC-INR-NoPhi-Signed`，Urban100 上接近 `SC-INR-NoPhi` 但低于
`SC-INR-NoPhi-Signed`。这说明 PhiZ 没有破坏相对 LTE 的强一致性优势，但
consistency 本身不能单独决定最终主模型。

### 7.4 qualitative evidence

用户已确认两张 selected examples 有可见改善：

- `artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img012_x8_selected_gt_bicubic_lte_nophi_scinr.png`
- `artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img004_x8_selected_gt_bicubic_lte_nophi_scinr.png`

它们可以支持 selected qualitative improvement，不支持 average visual quality claim。

### 7.5 SC-INR-NoSinc auxiliary metrics

来源：`artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/`。

协议覆盖 `LTE`、`SC-INR-NoPhi`、`SC-INR-NoPhi-Signed`、完整 `SC-INR` 和
`SC-INR-NoSinc`；数据为 BSD100 / Urban100 sorted 前 10 张；quality scales 为
x4、x8、x16、x30；consistency pairs 为 x8/x16/x30 -> x4。共享模型 summary
与 `sc_inr_final_aux_metrics_seed1` 完全一致，因此可直接比较。

OOD PSNR-Y（x8/x16/x30 平均）：

| Dataset | `SC-INR` | `SC-INR-NoSinc` | NoSinc - SC-INR |
| --- | ---: | ---: | ---: |
| BSD100 | 22.8571 | 22.8495 | -0.0076 |
| Urban100 | 20.5864 | 20.6072 | +0.0208 |
| All | 21.7218 | 21.7284 | +0.0066 |

Same-LR consistency PSNR-Y（x8/x16/x30 -> x4 平均）：

| Dataset | `SC-INR` | `SC-INR-NoSinc` | NoSinc - SC-INR |
| --- | ---: | ---: | ---: |
| BSD100 | 52.1421 | 69.0870 | +16.9448 |
| Urban100 | 47.1128 | 64.6241 | +17.5112 |
| All | 49.6275 | 66.8555 | +17.2280 |

解释边界：

- auxiliary consistency 不支持“移除 sinc 破坏 same-LR self-consistency”；结果恰好相反。
- 更合理的解释是，NoSinc 关闭 cell-dependent analytic response 后更接近
  cell-independent point decoder，当前 same-LR 指标会奖励这种自一致性。
- 因此该指标必须和 fidelity、texture/highpass、full benchmark、response/omega
  diagnostics 共同解释；不能把高 consistency 单独写成更正确的 sampling-consistent
  observation。
- full benchmark 仍提示 sinc 对最终候选 PSNR 有贡献：NoSinc 相对完整 `SC-INR`
  的 OOD/All 平均分别低 `0.0467/0.0777 dB`。

### 7.6 response/omega diagnostics

来源：`artifacts/derived/diagnostics/response_omega_diagnostics_2026-05-10/`。

该诊断用于解释 7.5 中的反直觉结果。协议分两部分：

- response/omega 分布：`SC-INR-NoPhi`、`SC-INR-NoPhi-Signed`、`SC-INR`、
  `SC-INR-NoSinc`，BSD100 / Urban100 sorted 前 5 张，x4/x8/x16/x30。
- cell-only 曲线：`LTE`、`SC-INR-NoPhi`、`SC-INR-NoPhi-Signed`、`SC-INR`、
  `SC-INR-NoSinc`，同一 x4 LR、同一 query coordinate，只改变 cell scale。

Active attenuation mean：

| Model | x4 | x8 | x16 | x30 |
| --- | ---: | ---: | ---: | ---: |
| `SC-INR-NoPhi` | 0.037106 | 0.009583 | 0.002415 | 0.000688 |
| `SC-INR-NoPhi-Signed` | 0.036853 | 0.009494 | 0.002392 | 0.000681 |
| `SC-INR` | 0.040495 | 0.010570 | 0.002672 | 0.000762 |
| `SC-INR-NoSinc` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Cell-only RMSE-Y vs x4 cell：

| Model | x8 | x16 | x30 | x8/x16/x30 avg |
| --- | ---: | ---: | ---: | ---: |
| `LTE` | 0.005267 | 0.008329 | 0.010233 | 0.007943 |
| `SC-INR-NoPhi` | 0.002862 | 0.003462 | 0.003598 | 0.003308 |
| `SC-INR-NoPhi-Signed` | 0.002662 | 0.003192 | 0.003310 | 0.003055 |
| `SC-INR` | 0.002854 | 0.003395 | 0.003515 | 0.003255 |
| `SC-INR-NoSinc` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

解释：

- `SC-INR-NoSinc` 的 active response 恒为 1，所以 active attenuation 和
  cell-only RMSE 都为 0；这解释了它在 same-LR self-consistency 上的异常高分。
- 完整 `SC-INR` 不是 cell-independent：它保留非零 analytic cell response，
  因此同一坐标下改变 cell 会产生有限输出变化。
- `SC-INR` 的 cell-only sensitivity 明显低于 LTE，符合其移除 learned
  cell-conditioned phase、改用解析 observation response 的设计目标。
- 该诊断只能支持机制解释，不能替代 full benchmark、multi-seed 或视觉质量证据。

## 8. 仍缺失的关键实验

在正式收敛论文主线前，至少还需要：

1. continuous-scale curve：展示 x4 以外尺度变化趋势，而不只给离散表。
2. footprint correctness/property tests：尤其需要能区分 point self-consistency 与
   footprint observation correctness 的诊断。
3. response/omega diagnostics：继续补充 cell response curve、omega distribution、
   sinc attenuation 的覆盖面。
4. 正式 qualitative figure：只晋级人工确认图，并记录筛选协议。

后续 Rot-E 结合和 LIIF+ 原型属于正交扩展，不应在当前证据不足时混入主结论。

## 9. 当前可写论文结论

可以写：

- We study decoder-side scale sampling consistency for arbitrary-scale image
  super-resolution.
- Replacing LTE's learned cell-conditioned phase with a scale-decoupled analytic
  observation response improves same-LR cross-scale consistency in the
  no-phase variant.
- The final-candidate feature-phase SC-INR shows positive OOD/ALL PSNR gains
  over LIIF and LTE in the 3-seed benchmark; its comparison to SC-INR-NoPhi is
  reported as seed1 context.

不能写：

- SC-INR is strictly scale-equivariant.
- Feature-conditioned phase is a stable multi-seed PSNR gain over SC-INR-NoPhi.
- The gain is caused only by sinc, based only on the seed1 NoSinc benchmark.
- The visual quality is generally better, based only on two selected crops.

## 10. 工作判断

当前最稳的论文主线是：

1. 用 `SC-INR-NoPhi` 的 3-seed OOD 和 consistency 结果支撑 scale-decoupled
   observation 的基本有效性；
2. 用最终候选 `SC-INR` 的 3-seed benchmark 说明它相对 LIIF/LTE 的 OOD/ALL
   增益；`SC-INR` vs `SC-INR-NoPhi` 只作为 seed1 结构增量背景；
3. 用 `SC-INR-NoSinc` benchmark/auxiliary 的分歧提醒读者：same-LR
   self-consistency 不是 sinc 机制的单独证明；再用必要的 footprint correctness、
   response/omega diagnostics 和正式 qualitative figure 补齐论文闭环。
