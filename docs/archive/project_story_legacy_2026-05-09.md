# SC-INR 项目故事：从尺度外泛化问题到尺度采样一致性

## 0. 文档目的

本文档用于系统记录当前工作区中 SC-INR / SE-INR 方向的研究故事、方法论、模型架构、实验流程和阶段性结论。它不是论文终稿，而是一份面向后续论文写作、组内交流和代码维护的技术叙事文档。

当前代码库位于 `Equivariant-ASISR/`，已经从原始 Rot-E ASISR 工作区重构为一个以 arbitrary-scale image super-resolution（ASISR）和 scale sampling consistency 为核心的研究项目。项目保留原始 LIIF/LTE 与 Rot-E 的旋转等变基线，并围绕 LTE 中 cell-conditioned phase 的尺度外泛化问题构建了一条逐步递进的模型与消融链条。

需要首先明确一个重要边界：当前 SC-INR 更准确地说是“尺度采样一致性”或“采样尺度一致性”的 INR 设计，而不是已经严格证明的数学意义上的尺度等变网络。它试图将输出 cell / 采样面积对重建结果的影响限制在具有解析采样意义的 sinc response 中，从而减少 LTE 类模型在未见过尺度上的任意 cell-dependent extrapolation。

---

## 1. 背景：ASISR 中的尺度外泛化问题

### 1.1 任意尺度超分辨率与 INR

Arbitrary-scale image super-resolution 的目标是：给定一张低分辨率图像，在任意连续放大倍率下生成高分辨率输出。与固定倍率 SR 不同，ASISR 需要模型在训练未覆盖的尺度上也能稳定输出。

LIIF 是该方向的重要代表。它将图像超分辨率问题转化为局部隐式函数查询问题：

1. 使用 encoder 将 LR image 编码成 feature map。
2. 对目标 HR 网格上的每个坐标 query，找到其在 LR feature grid 中的局部邻域。
3. 将 local feature、relative coordinate、cell size 等输入 MLP。
4. 输出该连续坐标处的 RGB 值。

LTE 在 LIIF 基础上进一步引入局部 Fourier / texture estimation。它不再仅依赖普通 MLP 表达局部函数，而是预测局部频率和系数，通过 cos/sin Fourier feature 表示局部纹理。这使 LTE 在高频结构上通常更强。

### 1.2 训练尺度与测试尺度的不一致

在当前项目中，训练数据通过 DIV2K HR 在线下采样生成。训练尺度主要覆盖 x1 到 x4，而 benchmark 和 continuous evaluation 覆盖 x2/x3/x4 的 ID 区间，以及 x6/x8/x12/x16/x24/x30 或连续 x1 到 x30 的 OOD 区间。

因此，核心问题不是模型能否在训练分布内重建，而是：

> 当输出 cell size 或目标尺度远离训练范围时，隐式解码器是否仍然以合理、稳定、可解释的方式响应尺度变化？

### 1.3 LTE 的潜在问题：cell-conditioned phase

当前 LTE baseline 的关键机制是：输出 cell 会通过一个 learned linear layer 进入 Fourier phase。简化表示为：

$$
q_{freq} = \omega(z) \cdot \delta + h_p(c)
$$

其中：

- $z$：encoder 产生的局部 feature。
- $\delta$：目标坐标相对 LR feature grid 的相对坐标。
- $c$：输出 cell size，对应目标像素在归一化坐标系中的面积。
- $h_p(c)$：由 cell 预测的 learned phase offset。

这个设计在 ID 尺度内可能有效，因为模型可以学习到训练尺度范围内的 cell-to-phase mapping。但在 OOD 尺度下，cell 输入落到训练范围之外，learned phase 分支可能产生不可控 extrapolation。换言之，LTE 将“采样尺度变化”交给了一个自由学习的相位函数，而不是交给有明确采样意义的解析算子。

当前项目的核心研究问题由此产生：

> 能否在保留 LTE/LIIF 框架主体的前提下，限制尺度变量 c 的作用方式，使其更符合采样理论，从而提升任意尺度尤其 OOD 尺度下的稳定性？

---

## 2. 研究目标与方法论主线

### 2.1 最初目标与当前重新定位

项目早期目标是探索 scale-equivariant implicit neural representation，即希望构造一种能够体现尺度变换结构的 INR。这个目标与 Rot-E ASISR 中的 rotation equivariance 有一定启发关系，但二者本质不同。

旋转等变可以较自然地通过群卷积、方向通道、旋转表示等机制实现。而尺度变换在图像超分辨率中同时牵涉：

- 输入图像退化过程。
- 输出采样密度。
- 坐标尺度。
- 像素面积积分。
- 高频 aliasing / anti-aliasing。
- 训练尺度分布和测试尺度分布。

因此，简单将旋转等变框架类比到尺度等变是不严谨的。当前代码和实验证据更支持将方法重新定位为：

> Scale Sampling Consistency for Implicit ASISR。

也就是说，当前 SC-INR 的核心不是证明严格尺度等变，而是把尺度相关变量 c 从任意 learned phase 中移出，放入有解析意义的 sampling response 中。

### 2.2 方法论核心：函数与采样分离

当前 SC-INR 方法论可以概括为：

> 连续图像函数 F 应主要由图像内容 z 和坐标 $\delta$ 决定；输出采样 cell c 不应任意改变 F 的相位或参数，而应通过解析采样权重影响最终像素观测值。

这形成了两层结构：

1. Continuous latent function layer：
   $$
   F(z, \delta)
   $$
   用局部 feature 和相对坐标描述连续图像信号。

2. Sampling observation layer：
   $$
   y = \mathcal{S}(F, c)
   $$
   用输出 cell size 表示目标像素面积上的采样/积分效应。

在 Fourier 表示中，如果局部信号由若干频率分量组成，那么对像素 cell 进行面积平均会对应频域中的 sinc 衰减。因此，对于某个频率 $\omega_k = (\omega_k^x, \omega_k^y)$，输出 cell $c=(c_x,c_y)$ 对该频率的解析响应可以写作：

$$
W_k(c) = \mathrm{sinc}(\omega_k^x c_x / 2) \cdot \mathrm{sinc}(\omega_k^y c_y / 2)
$$

SC-INR 的方法核心就是：

- Fourier phase 不再由 cell 任意预测。
- cell 只通过 $W_k(c)$ 影响频率分量。
- $W_k(c)$ 在 OOD cell 上仍然有明确解析定义。

### 2.3 为什么不是简单去掉 cell？

一个自然消融是 LTE-NoCell：直接移除 LTE 中的 cell-conditioned phase，使 phase 固定为 0。这个设计可以避免 OOD cell extrapolation，但它也丢掉了采样尺度信息。

SC-INR 与 LTE-NoCell 的区别在于：

- LTE-NoCell：cell 完全不影响 Fourier basis。
- SC-INR：cell 不影响连续函数相位，但仍通过 sinc response 影响不同频率的观测强度。

因此，SC-INR 不是“忽略尺度”，而是“限制尺度作用的形式”。这是当前方法最重要的思想区别。

---

## 3. 当前模型谱系

当前工作区保留 8 个模型，用于完整展示研究过程、基线比较和因果消融。

### 3.1 Baselines

#### LIIF

标准 LIIF baseline。当前实现位于 `models/liif.py`，训练配置为 `configs/train-div2k/train-liif.yaml`，checkpoint 位于 `save/liif`。

它是最基础的 implicit decoder baseline，使用 local feature、relative coordinate 和 cell 作为 MLP 输入。

#### LTE

标准 LTE baseline。当前实现位于 `models/lte.py`，训练配置为 `configs/train-div2k/train-lte.yaml`，checkpoint 位于 `save/lte`。

LTE 通过 Fourier feature 表达局部纹理，并使用 cell-conditioned phase：

$$
q_{freq} = \omega(z)\delta + h_p(c)
$$

它是当前方法链条中最重要的被分析对象。

### 3.2 Rotation-equivariant baselines

#### LIIF-EQ

Rot-E ASISR 中的 LIIF rotation-equivariant 版本。当前实现拆分到 `models/liif_eq.py`，训练配置为 `configs/train-div2k/train-liif-eq.yaml`，checkpoint 位于 `save/liif-eq`。

项目重构中特别将 LIIF 与 LIIF-EQ 拆开，是因为不带后缀的 `liif` 应表示原始 LIIF 架构，而 `liif_eq` 表示旋转等变版本。这也与 LTE / LTE-EQ 的命名逻辑一致。

#### LTE-EQ

Rot-E ASISR 中的 LTE rotation-equivariant 版本。当前实现位于 `models/lte_eq.py`，训练配置为 `configs/train-div2k/train-lte-eq.yaml`，checkpoint 位于 `save/lte-eq`。

LIIF-EQ 和 LTE-EQ 是判断当前尺度采样一致性机制能否与旋转等变方法正交结合的重要参考。

### 3.3 Diagnostic ablations

#### LTE-NoCell

实现位于 `models/lte_noc.py`，配置为 `configs/train-div2k/train-lte-no-cell.yaml`，checkpoint 位于 `save/lte-no-cell`。

它移除 LTE 的 cell-conditioned phase，使 phase offset 固定为 0。该模型回答的问题是：

> LTE 的 OOD 行为是否主要来自 cell-conditioned phase？如果删除该分支，泛化是否改善？

实验表明，简单删除 cell phase 并不总是足够，尤其 ID 性能可能下降。这说明问题不只是“cell 有害”，还涉及表达力和训练 co-adaptation。

#### LTE-FeaturePhase

实现位于 `models/lte_phase_z.py`，配置为 `configs/train-div2k/train-lte-feature-phase.yaml`，checkpoint 位于 `save/lte-feature-phase`。

它将 phase 从 cell-conditioned 改为 feature-conditioned：

$$
\phi = h_p(z)
$$

该模型用于区分两个假设：

1. OOD 问题来自 phase 分支本身。
2. OOD 问题来自 phase 对 cell 的直接依赖。

如果 feature-conditioned phase 更稳定，说明“phase 可以存在，但不应由 cell 任意预测”。

### 3.4 Proposed family: SC-INR

#### SC-INR-Fixed

实现位于 `models/sc_inr_fixed.py`，配置为 `configs/train-div2k/train-sc-inr-fixed.yaml`，checkpoint 位于 `save/sc-inr-fixed`。

这是固定频率版本。它使用固定 log-polar frequency basis：

$$
\omega_k \in \Omega_{fixed}
$$

对每个 query 计算：

$$
F_k(\delta) = [\cos(\pi(\omega_k \cdot \delta + \phi_k(z))),\ \sin(\pi(\omega_k \cdot \delta + \phi_k(z)))]
$$

然后乘以解析 sinc sampling weights：

$$
\tilde{F}_k(\delta,c)=F_k(\delta)W_k(c)
$$

最后通过 MLP 输出 RGB。

该模型验证了“cell 只进入 sinc response”的理论设计，但固定频率限制了表达能力。

#### SC-INR-Adaptive

实现位于 `models/sc_inr_adaptive.py`，配置为 `configs/train-div2k/train-sc-inr-adaptive.yaml`，checkpoint 位于 `save/sc-inr-adaptive`。

这是当前主模型。它将固定频率替换为 feature-conditioned adaptive frequency：

$$
\omega_k = \omega_k(z)
$$

同时保持 $\phi=0$，并保留 sinc sampling response：

$$
W_k(z,c)=\mathrm{sinc}(\omega_k^x(z)c_x/2)\mathrm{sinc}(\omega_k^y(z)c_y/2)
$$

最终形式可以概括为：

$$
y = \mathrm{MLP}\left(A(z) \odot [\cos(\pi\omega(z)\delta)W(z,c),\ \sin(\pi\omega(z)\delta)W(z,c)]\right) + y_{bilinear}
$$

其中：

- $A(z)$：由 `coef` convolution 预测的 amplitude / modulation。
- $\omega(z)$：由 `omega_conv` 预测的局部频率。
- $W(z,c)$：解析 sinc sampling response。
- $y_{bilinear}$：上采样输入的 residual connection。

SC-INR-Adaptive 的直觉是：

- 相比 SC-INR-Fixed，它恢复了局部频率的自适应表达能力。
- 相比 LTE，它避免了 cell-conditioned arbitrary phase。
- 相比 LTE-NoCell，它仍然保留了采样尺度对频率响应的影响。

---

## 4. 数据与训练流程

### 4.1 数据生成

训练数据使用 DIV2K HR，通过 online downsampling 构造任意尺度 LR/HR pair。核心 wrapper 是 `sr-implicit-downsampled`。

每次训练采样大致流程为：

1. 从 DIV2K HR 中取一张图像。
2. 从训练尺度范围中随机采样 scale，通常为 x1 到 x4。
3. 随机裁剪 HR patch。
4. 将 HR patch bicubic downsample 到固定 LR patch size，例如 48x48。
5. 从 HR patch 中采样 query coordinates 和 RGB ground truth。
6. 为每个 query 构造 cell size。

这种方式保证训练时模型看到连续尺度，而不是只看到固定 x2/x3/x4。

### 4.2 训练设置

主要模型使用相近训练预算：

- Dataset：DIV2K train HR。
- Validation：DIV2K valid HR 子集。
- Optimizer：Adam。
- LR：1e-4。
- Epoch：1000。
- Batch size：16。
- LR patch size：48。
- sample_q：2304。
- scale range：主要 x1 到 x4。

当前 SC-INR-Adaptive 的结果不依赖显式 consistency loss。虽然训练脚本中存在 optional consistency 分支，但当前主配置未启用该项。因此论文中不应将当前结果归因于额外 consistency loss。

---

## 5. 评估协议

### 5.1 离散 benchmark

离散 benchmark 结果保存在 `results/benchmark.json`。

数据集：

- Set5
- Set14
- BSD100
- Urban100

尺度划分：

- ID：x2, x3, x4
- OOD：x6, x8, x12, x16, x24, x30

指标：benchmark-style PSNR，使用 Y channel 与 scale-dependent shave。

### 5.2 连续尺度评估

连续尺度结果保存在：

- `results/continuous/set5.json`
- `results/continuous/set14.json`
- `results/continuous/bsd100.json`
- `results/continuous/urban100.json`

连续尺度评估脚本为 `eval_continuous.py`。当前版本已经修复一个关键对齐问题：必须先 crop HR 到与 scale 对齐的 target size，再从 cropped HR bicubic downsample 生成 LR。否则 LR 与 GT 会错位，并造成系统性 PSNR 偏差。

连续尺度覆盖 x1.0 到 x30.0，步长 0.5。需要注意 x1.0 结果通常非常特殊，PSNR 可能极高且模型间差异异常大，因此论文统计中应单独报告，不建议直接纳入主平均。

### 5.3 辅助实验

当前包含两个辅助分析：

1. FCE（Function Consistency Error）：结果保存在 `results/fce.json`。它用于测量固定 feature z 时，改变 cell c 是否会改变 decoder 的内部函数表示。
2. LTE phase intervention：结果保存在 `results/phase_intervention.json`。它比较原始 LTE、测试时强行 phase=0 的 LTE、以及从头训练的 LTE-NoCell。

---

## 6. 关键实验结果概览

### 6.1 离散 benchmark 总体趋势

SC-INR-Adaptive 在离散 benchmark 上相对非 EQ LIIF/LTE 有稳定但幅度不大的提升。尤其在 OOD 尺度上，它相对 LTE 的平均提升更清晰。

从 `results/benchmark.json` 可见，SC-INR-Adaptive 在四个数据集 x6 到 x30 的多数点上优于 LTE。例如：

- Set5 x30：SC-INR-Adaptive 20.5664，高于 LTE 20.4745。
- Set14 x30：SC-INR-Adaptive 19.8121，高于 LTE 19.7481。
- BSD100 x30：SC-INR-Adaptive 20.6854，高于 LTE 20.6313。
- Urban100 x30：SC-INR-Adaptive 18.3089，高于 LTE 18.2440。

相对 SC-INR-Fixed，SC-INR-Adaptive 也整体更强。这说明 data-driven local frequency 对表达力是必要的，固定频率虽然理论干净，但性能不是最优。

### 6.2 与 EQ baseline 的关系

需要谨慎的是，SC-INR-Adaptive 并不能无条件宣称全面超过 LIIF-EQ / LTE-EQ。LIIF-EQ 在 ID 上仍然很强，部分 OOD 点也与 SC-INR-Adaptive 接近甚至更高。

因此当前最稳妥的结论是：

- SC-INR-Adaptive 相对非 EQ LIIF/LTE 有稳定增益。
- SC-INR-Adaptive 相对 SC-INR-Fixed、LTE-NoCell、LTE-FeaturePhase 展示了更好的综合平衡。
- 与 rotation-equivariant baselines 的关系更复杂，后续更有价值的方向是将 SC-INR-Adaptive 与 Rot-E encoder/decoder 正交结合，而不是直接把二者视为互斥竞争方案。

### 6.3 消融链条的解释

当前 8 模型体系不是最终论文中都作为“方法”出现，而是构成一条因果链：

1. LIIF / LTE：基础 ASISR baseline。
2. LIIF-EQ / LTE-EQ：旋转等变 baseline，说明项目来源和正交方向。
3. LTE-NoCell：验证去掉 cell-conditioned phase 的影响。
4. LTE-FeaturePhase：验证 phase 可以来自 z，而不一定来自 c。
5. SC-INR-Fixed：验证 sinc sampling response 的理论设计。
6. SC-INR-Adaptive：最终主模型，在 sampling-consistent 结构中恢复数据驱动频率表达能力。

这条链条能够支持一个比单纯 benchmark 提升更强的故事：

> OOD scale generalization 的关键不只是模型容量，而是尺度变量进入隐式函数的方式。将 cell 从 learned phase 中移出，并通过解析 sampling response 注入，可以改善泛化稳定性。

---

## 7. 方法的创新点与边界

### 7.1 创新点

当前方法的主要创新点可以表述为：

1. 提出 scale sampling consistency 视角，将连续函数建模和像素采样观测分离。
2. 指出 LTE 中 cell-conditioned phase 可能导致 OOD cell extrapolation。
3. 设计 sinc-based sampling response，使 cell 只以解析频率响应形式影响输出。
4. 从 fixed frequency 发展到 adaptive frequency，使方法兼具理论约束和表达能力。
5. 通过 LTE-NoCell、LTE-FeaturePhase、SC-INR-Fixed、SC-INR-Adaptive 构建了一条机制清晰的消融链。

### 7.2 边界与风险

当前不应过度宣称：

- 不应声称已经实现严格 scale equivariance。
- 不应声称对所有 LIIF/LTE/Rot-E baselines 全面大幅超越。
- 不应把小于 0.1 dB 的提升写成强显著结论，除非后续补充多 seed。
- 不应将当前结果归因于 consistency loss，因为当前主配置未启用该项。

当前最适合的论文定位是：

> A sampling-consistent Fourier implicit decoder for arbitrary-scale super-resolution, improving OOD scale robustness by constraining how output cell size modulates local Fourier responses.

---

## 8. 与 Rot-E 的正交结合潜力

项目基于 Rot-E ASISR 源码，因此一个自然后续方向是将 SC-INR-Adaptive 与 rotation-equivariant encoder/decoder 结合。

当前 LIIF-EQ / LTE-EQ 证明旋转等变方向可以提升某些 benchmark 表现，但它主要处理旋转对称性，不直接解决输出尺度采样一致性。SC-INR-Adaptive 处理的是 cell / scale 如何进入 implicit decoder。

因此两者理论上是正交的：

- Rot-E：约束 feature / decoder 对旋转变换的响应。
- SC-INR：约束 output cell 对 Fourier response 的调制方式。

后续可以设计：

- SC-INR-Adaptive + EDSR-EQ encoder。
- SC-INR-Adaptive + equivariant MLP readout。
- Rotation-equivariant frequency prediction $\omega(z)$。
- 同时报告 rotation consistency 与 scale OOD PSNR。

这可能成为比当前单独 SC-INR 更强的论文扩展方向。

---

## 9. 当前工作区结构

当前主要结构如下：

```text
Equivariant-ASISR/
├── configs/train-div2k/
│   ├── train-liif.yaml
│   ├── train-liif-eq.yaml
│   ├── train-lte.yaml
│   ├── train-lte-eq.yaml
│   ├── train-lte-no-cell.yaml
│   ├── train-lte-feature-phase.yaml
│   ├── train-sc-inr-fixed.yaml
│   └── train-sc-inr-adaptive.yaml
├── models/
│   ├── liif.py
│   ├── liif_eq.py
│   ├── lte.py
│   ├── lte_eq.py
│   ├── lte_noc.py
│   ├── lte_phase_z.py
│   ├── sc_inr_fixed.py
│   └── sc_inr_adaptive.py
├── save/
│   ├── liif/
│   ├── liif-eq/
│   ├── lte/
│   ├── lte-eq/
│   ├── lte-no-cell/
│   ├── lte-feature-phase/
│   ├── sc-inr-fixed/
│   └── sc-inr-adaptive/
├── results/
│   ├── benchmark.json
│   ├── continuous/
│   ├── figures/
│   ├── fce.json
│   ├── phase_intervention.json
│   └── index.json
├── train.py
├── test.py
├── eval_full.py
├── eval_continuous.py
├── eval_fce.py
└── eval_phase_intervention.py
```

该结构的优点是：

- 不带后缀的 LIIF/LTE 表示原始 baseline。
- `-eq` 表示 Rot-E rotation-equivariant 版本。
- `no-cell`、`feature-phase`、`fixed`、`adaptive` 都是机制语义，而不是历史阶段命名。
- 结果文件集中到 `results/`，避免污染根目录。

---

## 10. 后续论文写作建议

建议论文按以下逻辑组织：

### Introduction

强调 ASISR 的 OOD scale generalization 问题，指出训练尺度和测试尺度不一致时，cell-conditioned decoder 可能产生不稳定 extrapolation。

### Method

重点写函数与采样分离：

1. 回顾 LIIF/LTE local implicit decoding。
2. 分析 LTE cell-conditioned phase。
3. 提出 sampling-consistent Fourier response。
4. 推导 sinc weight 的含义。
5. 从 fixed frequency 过渡到 adaptive frequency。

### Architecture

详细描述 SC-INR-Adaptive：

- encoder。
- local ensemble。
- `coef` branch。
- `omega_conv` branch。
- Fourier feature construction。
- sinc sampling response。
- MLP readout。
- bilinear residual。

### Experiments

建议分层报告：

1. Baselines：LIIF、LTE、LIIF-EQ、LTE-EQ。
2. Diagnostic ablations：LTE-NoCell、LTE-FeaturePhase、SC-INR-Fixed。
3. Proposed：SC-INR-Adaptive。
4. ID vs OOD benchmark。
5. Continuous-scale curves。
6. Phase intervention 与 FCE。

### Limitations

必须主动写：

- 当前不是严格 scale equivariance。
- 提升幅度较小，需要多 seed。
- 与 EQ baselines 的关系是正交而非简单全面超越。
- continuous Urban100 等结果需要确保最终完整。

---

## 11. 最终阶段性结论

当前项目已经形成了一条清晰的研究故事：

1. ASISR 模型需要在训练尺度之外泛化。
2. LTE 的 cell-conditioned phase 提供了强表达力，但也可能引入 OOD cell extrapolation。
3. 简单移除 cell 不是最优，因为尺度采样信息仍然重要。
4. 将 cell 限制到 sinc sampling response 中，可以更合理地表达像素面积采样效应。
5. 固定频率版本验证了理论结构，但表达力有限。
6. 自适应频率版本 SC-INR-Adaptive 在保持 sampling consistency 的同时恢复表达能力，成为当前最有价值的主模型。
7. 当前结果支持其相对非 EQ LIIF/LTE 的 OOD 稳定性提升，但不支持过度声称严格尺度等变或全面大幅超越所有 EQ baselines。

一句话总结：

> SC-INR-Adaptive 的核心贡献不是“让 INR 严格尺度等变”，而是提出了一种更合理的尺度采样注入方式：让输出 cell 通过解析 sinc response 调制局部 Fourier 表示，从而改善任意尺度超分辨率中的 OOD scale robustness。
