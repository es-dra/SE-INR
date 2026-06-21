# SC-INR 算例与证据链建议 2026-06-07

本文档用于回答“如何找到更适合 SC-INR 的算例，并在较强结果前提下形成有事实支撑的叙述逻辑”。它不是新 benchmark，也不升级论文 claim。

## 推荐主算例

推荐采用“两层算例”：

1. **解析算例**：一维或二维方向性正弦/条纹信号。
2. **真实图像算例**：Urban100 建筑立面中的方向性重复结构，例如 `img_012.png` x8 crop `(y=432, x=864, size=96)`。

原因：

- SC-INR 的 decoder 是 local Fourier-like decoder，方向性重复纹理比平滑区域更能对应 `omega(z)`、`delta` 和 `W(omega,c)` 的变量角色。
- pixel footprint 对不同方向频率的影响不是单纯 area，而是沿 `x/y` 方向的窗口宽度，因此建筑条纹/窗格这类有方向频率的 patch 更适合说明方法。
- 该真实图像算例已有较强局部结果：`SC-INR` crop PSNR-Y `23.2229`，`LTE` 为 `20.4458`，`SC-INR-NoPhi` 为 `19.6163`，`Bicubic` 为 `18.8876`。

## 解析逻辑

先从一维信号说明：

`f(x)=A cos(pi (omega x + phi))`

如果输出像素被视为点采样，则 cell 不应改变响应：

`f(x_q)=A cos(pi (omega x_q + phi))`

如果输出像素有宽度 `c`，它对应的是窗口平均：

`1/c int_{x_q-c/2}^{x_q+c/2} A cos(pi (omega x + phi)) dx`

结果是：

`A sinc(omega c / 2) cos(pi (omega x_q + phi))`

这说明：

- `z` 和 `delta` 决定局部连续函数及查询位置；
- `cell` 不提供内容，也不应该直接移动纹理相位；
- `cell` 描述输出像素的 footprint，并决定某个频率分量在该窗口平均后还能保留多少；
- 高频或大 footprint 对应更强衰减。

二维矩形 footprint 下，对 separable box average 可得到：

`W(omega,c)=sinc(omega_x c_x / 2) sinc(omega_y c_y / 2)`

这里应强调：这是 decoder 输入层面的 analytic response，不是最终 RGB 的严格物理频谱，也不是证明真实图像等于正弦信号。

## 对应实现

`src/models/sc_inr_adaptive.py` 中，SC-INR 的关键路径是：

- `omega(z)` 由 `omega_conv(self.feat)` 给出；
- `phi(z)` 可由 `phase_conv(self.feat)` 给出，但不接收 cell；
- `q_phase = sum(q_omega * rel_coord) + phi(z)`；
- `W = sinc(omega_x * c_x / 2) * sinc(omega_y * c_y / 2)`；
- Fourier basis 先乘 `W`，再与 `q_coef` 相乘送入 MLP。

对照 LTE：

- `src/models/lte.py` 中 `q_freq += self.phase(rel_cell)`；
- 即 cell 通过 learned `h_p(c)` 进入 phase path。

因此叙事应写成：

> LTE 允许 cell 直接改变 Fourier-like phase；SC-INR 将 cell 从内容相位中移出，只让它通过 analytic response 改变各局部频率分量在 footprint 下的可观测强度。

## 事实支撑

### 1. 主 benchmark 支撑 OOD 场景

`artifacts/derived/benchmarks/paper_main_3seed_paired_delta.csv`：

- `SC-INR` vs `LTE`：OOD `+0.0504 ± 0.0417` dB，ALL `+0.0320 ± 0.0555` dB。
- ID vs `LTE` 为 `-0.0048 ± 0.0836` dB，因此主 claim 应收缩为 OOD/尺度外推方向，而不是所有尺度全面更强。

### 2. 真实图像算例支撑局部强结果

`artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img012_x8_selected_gt_bicubic_lte_nophi_scinr.png`

局部 crop：

| Model | PSNR-Y |
| --- | ---: |
| Bicubic | 18.8876 |
| LTE | 20.4458 |
| SC-INR-NoPhi | 19.6163 |
| SC-INR | 23.2229 |

该例子是建筑立面方向性重复条纹，适合承接解析条纹算例。使用时必须标注为 selected qualitative example，不能写成平均视觉质量证明。

### 3. Footprint oracle 支撑 cell 语义

`artifacts/derived/diagnostics/footprint_oracle_2026-05-15/footprint_oracle_overall.csv`：

- cell multiplier `4` 时，`SC-INR` oracle RMSE `0.02148`，低于 `LTE` 的 `0.02309` 和 `SC-INR-NoSinc` 的 `0.02750`。
- cell multiplier `4` 时，`SC-INR` delta-tracking RMSE `0.04745`，低于 `LTE` 的 `0.04834` 和 `SC-INR-NoSinc` 的 `0.05134`。
- `SC-INR-NoSinc` 的 cell sensitivity 为 `0`，说明去掉 response 后会退化成对 cell 不响应的负控。

### 4. 统一机制诊断支撑路径差异

`artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/mechanism_overall.csv`：

- `LTE` 的 `cell_phase_delta_rms_vs_native` 在 cell multiplier `2/4` 下为 `0.25764/0.77292`，说明 learned cell-phase 路径确实随 cell 改变。
- `SC-INR` 的 `active_response_delta_rms_vs_native` 在 cell multiplier `2/4` 下为 `0.17606/0.44018`，说明 analytic response 路径确实随 cell 改变。
- `SC-INR-NoSinc` 的 active response delta 为 `0`，作为负控成立。

## 推荐叙述顺序

1. 任意尺度 SR 中，目标像素不是抽象面积标量，而是在连续坐标域里有一个 footprint。
2. 单个像素值不能反推出局部连续信号；局部连续表示来自 LR feature `z` 和相对坐标 `delta` 条件化。
3. 对局部 Fourier-like 分量，有限 footprint 的 box-average 会带来 sinc response。
4. 因此 cell 更合理的角色不是移动局部纹理相位，而是调制频率分量在该 footprint 下的可观测强度。
5. SC-INR 在实现中把 cell 放在 `W(omega,c)` 路径；LTE 把 cell 放入 learned phase。
6. 在 Urban100 建筑条纹 x8 case 中，SC-INR 保留方向性重复结构，局部 PSNR 明显高于 LTE。
7. 在 footprint oracle 和机制诊断中，SC-INR 的 cell-induced change 更接近 HR box-average proxy，而 NoSinc 对 cell 不响应，支撑该机制解释。

## 可直接使用的短表述

> 更适合 SC-INR 的算例不是单个像素或平滑区域，而是具有方向性频率的局部纹理，例如建筑立面的重复窗格/条纹。对这类局部 Fourier-like 信号，输出像素的 footprint 会决定不同方向频率分量被平均后保留多少；这正好对应 `W(omega,c)=sinc(omega_x c_x/2)sinc(omega_y c_y/2)`。在 `Urban100/img_012` x8 的建筑条纹 crop 中，SC-INR 的局部 PSNR-Y 为 `23.22`，明显高于 LTE 的 `20.45`。同时，在 footprint oracle 诊断中，SC-INR 比 LTE/NoSinc 更接近 HR box-average proxy，说明该优势不只是视觉现象，而与 cell response 的建模语义一致。

## 边界

- 该证据链支持“SC-INR 的 footprint-aware cell path 更合理，并在 OOD/方向纹理区域有优势”。
- 不能写成 sinc 是唯一合理形式。
- 不能写成最终 RGB 是 exact box integral。
- 不能写成 SC-INR 在所有区域或所有尺度都明显优于 LTE/LIIF。
- `img_012` 是 selected qualitative example；如进入论文正文，应配合候选池统计或 neutral/failure 补充，降低 cherry-pick 风险。
