# INCODE 对 SC-INR method 的支撑与实验判断 2026-06-20

## 节点约束

本节点回答两个问题：

1. INCODE 这篇论文能从 method 层面为 SC-INR 提供什么支撑。
2. 当前是否需要新增实验，如果需要，应优先做什么最小实验。

禁止事后升级项：

- 不把 INCODE 写成 SC-INR 的直接前作或直接 baseline。
- 不用 INCODE 证明 sinc response 的性能优势。
- 不把“phase/amplitude 的解释”写成严格物理频谱结论。
- 不建议现在训练 INCODE baseline 与 LIIF/LTE/SC-INR 比 PSNR。

subagent 状态：

- 未使用真实 subagent。当前运行时工具要求只有用户明确要求 subagent/并行 agent 时才能创建；本轮采用主流程内模拟独立审查。

## 1. INCODE 的 method 核心

INCODE 不是 arbitrary-scale SR 方法，而是 conditional INR。它的核心是把 SIREN 的固定正弦激活：

$$
y_l=\sin(\omega_0(W_l y_{l-1}+b_l))
$$

扩展为可由先验 embedding 条件化控制的形式：

$$
y_l=a\sin(b\omega_0(W_l y_{l-1}+b_l)+c)+d.
$$

其中 harmonizer network 从任务先验 embedding 中预测 \(a,b,c,d\)，composer network 仍然负责从坐标映射到信号值。

INCODE 对这几个参数给了比较直观的图像解释：

- \(a\)：amplitude，控制响应强度，可增强特征，也可能放大噪声。
- \(b\)：frequency scaling，控制更细或更粗的细节粒度。
- \(c\)：phase shift，横向移动正弦波，影响 feature alignment 和 spatial arrangement。
- \(d\)：vertical shift，改变基线，类似亮度偏移。

这里对 SC-INR 最有用的是第三点：phase 不是一个无害的普通条件变量，它会移动正弦模式在空间中的对齐方式。

## 2. INCODE 能支撑 SC-INR 的哪一部分

INCODE 不能直接支撑 SC-INR 的 sinc response 有效，但能支撑一个更基础的 method 观点：

> 在 Fourier/SIREN-like decoder 中，不同参数对应不同图像属性；条件变量进入哪条路径，决定它能改变图像中的什么东西。

对应到 SC-INR：

- phase 路径对应局部结构的空间对齐，也就是条纹、边缘、纹理周期在局部坐标里落在哪里。
- frequency 路径对应局部纹理的方向和粒度。
- amplitude/response 路径对应某种局部模式显示得多强、多清楚。
- offset/残差路径对应亮度或低频基线。

因此，SC-INR 的 method 叙事可以从“条件路径分工”展开，而不是从“改了 LTE 的振幅”展开。

更具体地说，INCODE 支持我们这样讲：

1. 如果某个条件变量进入 phase，它就有能力移动局部结构。
2. `cell` 表示输出像素 footprint，本身不包含图像内容，也不应该决定条纹在哪里。
3. 所以 `cell` 不适合作为 phase shift 的来源。
4. `cell` 更适合进入 response path，因为 footprint 改变的是当前像素对局部连续信号的观测强度。

这正好对应 SC-INR 的设计：

$$
\theta=\omega(z)^\top\delta+\phi(z),
$$

其中 \(\omega,\phi\) 由图像 feature \(z\) 给出；而 `cell` 只进入：

$$
W(\omega,c)=
\operatorname{sinc}\left(\frac{\omega_xc_x}{2}\right)
\operatorname{sinc}\left(\frac{\omega_yc_y}{2}\right).
$$

换句话说，SC-INR 不是反对 phase，而是反对由 `cell` 控制 phase。

## 3. 对 method 写法的直接改动建议

当前 method 叙事应增加一个“parameter semantics”段落，位置建议放在 LTE 与 SC-INR 对比之前。

建议逻辑：

1. 先说 Fourier/SIREN-like decoder 的几个图像属性：frequency 决定方向/粒度，phase 决定空间对齐，amplitude/response 决定可见强度。
2. 再说 arbitrary-scale SR 中有三类变量：feature、relative coordinate、cell。
3. feature 和 coordinate 可以定义局部内容结构；cell 只定义输出像素的 footprint。
4. 因此，尺度条件不应进入 phase path，而应进入 observation response path。

可直接使用的中文表述：

> INCODE 对正弦激活参数的讨论给了一个有用视角：phase shift 会改变特征的空间对齐，frequency scaling 会改变细节粒度，amplitude 会改变响应强度。放到 arbitrary-scale SR 中，`cell` 只是输出像素的 footprint，并不是图像内容先验。因此，若让 `cell` 进入 phase path，就等于允许尺度条件移动局部纹理结构；而 SC-INR 的做法是把 phase 保留给 feature-conditioned content，把 `cell` 限制在 footprint response 中，只控制该局部频率在当前输出窗口下的可见强度。

可直接使用的英文表述：

> The parameter semantics of sinusoidal INRs suggest that phase controls spatial alignment, frequency controls local granularity, and amplitude controls response strength. In arbitrary-scale SR, the cell specifies the finite footprint of an output pixel rather than image content. Therefore, allowing the cell to shift the phase gives the scale condition direct access to structural alignment. SC-INR instead reserves the phase for feature-conditioned local content and routes the cell only through an analytic footprint response.

## 4. 当前不建议做的实验

### 4.1 不建议训练 INCODE baseline

原因：

- INCODE 是 per-signal 或 task-conditioned INR，通常针对单个信号拟合或由任务先验 embedding 条件化；LIIF/LTE/SC-INR 是从 LR 图像泛化到任意 query 的 ASISR 模型。
- 直接把 INCODE 加成 baseline 会改变任务定义，容易变成“单图拟合能力 vs 泛化式 SR”的不公平比较。
- 这会引入新的训练协议、先验 encoder、迭代预算和超参选择，短期内不能回答 SC-INR 的核心问题。

因此，不应把 INCODE 当作现在必须补的实验 baseline。

### 4.2 不建议只做更多 selected visual case

selected case 可以帮助解释，但导师已经指出问题是“太抽象”和“支撑不够落地”。如果继续只找更好看的图，容易强化展示偏置，不能回答 method claim。

## 5. 当前最值得做的实验

当前需要的不是新 baseline，而是一个轻量的 mechanism diagnostic，目标是验证：

> 当只改变 `cell`、固定 LR input 和 query coordinate 时，LTE 的变化是否更像 phase/structure shift，而 SC-INR 的变化是否更像 response/visibility change。

这正好对应 INCODE 给出的参数语义：phase 改 spatial alignment，response 改强度。

### 实验 A：cell-only intervention，可视化结构位移 vs 可见性变化

设置：

- 固定同一张 LR 图、同一组 query coordinate、同一 crop。
- 分别给模型输入不同 cell multiplier，例如 `0.5, 1, 2, 4`。
- 模型：LTE、LTE-PhaseZ、SC-INR、SC-INR-NoSinc。
- 选择建筑窗格、栏杆、条纹等方向性纹理 crop。

看什么：

- LTE：如果 cell 进入 phase，改变 cell 可能导致条纹位置、暗带对齐、局部纹理边界发生偏移。
- LTE-PhaseZ：cell 不再进入 phase，应作为“去掉 cell phase”的 control。
- SC-INR：改变 cell 应主要改变高频纹理的可见强度，结构位置不应明显漂移。
- SC-INR-NoSinc：去掉 sinc 后，cell 对 decoder 输入的 response 应消失或显著减弱，是 response path 的 negative control。

建议输出：

- 原 crop 输出图。
- cell multiplier 的差分图。
- Y 通道行/列 profile，观察峰谷位置是否移动。
- 简单指标：
  - delta image RMSE；
  - 梯度方向一致性；
  - profile peak shift 或 cross-correlation peak shift；
  - high-frequency energy change。

这个实验不需要重新训练，使用已有 checkpoint 即可。

### 实验 B：phase-shift proxy，对 LTE 的 cell phase 做显式度量

设置：

- 直接读取 LTE 的 `phase(rel_cell)` 输出。
- 在 `cell multiplier = 0.5,1,2,4` 下计算 phase vector 的变化量。
- 对比训练范围内尺度和 OOD 尺度，例如 x4、x16、x30。

看什么：

- 如果 `cell` 改变会带来非小的 phase vector 变化，就能从实现层证明 LTE 的 cell path 确实有移动 Fourier basis 的能力。
- 这不证明它一定造成视觉错误，但能支撑 method critique：cell-conditioned phase 在机制上允许 scale 改变结构对齐。

建议输出：

- `||h_p(c*m)-h_p(c)||` 的曲线。
- 与输出差分强度的相关性。
- 训练尺度范围与 OOD 尺度范围对比。

这个实验同样不需要重新训练。

### 实验 C：effective response 曲线，重新生成并收敛为可引用 artifact

项目里已有 `scripts/analysis/analyze_effective_amplitude.py`，可计算 SC-INR decoder 输入层面的：

$$
A_{\mathrm{eff}}=A(z)W(\omega(z),c).
$$

建议重新生成一个小而干净的 artifact：

- 模型：SC-INR、SC-INR-NoSinc。
- 数据：BSD100/Urban100 各 3-5 张即可，先做 diagnostic。
- 尺度：x2、x4、x8、x16、x30。
- 输出：按 low/mid/high frequency bin 的 `|W|`、effective energy ratio。

看什么：

- SC-INR 中 high-frequency response 是否随 footprint 变化呈合理趋势。
- NoSinc 中 active response 是否恒为 1，作为 negative control。

注意：

- 这只能说明 decoder 输入层面的 response 行为，不能直接说最终 RGB 频谱符合 box-average。

## 6. 实验优先级

优先级最高：实验 A。

理由：

- 它直接回答导师最关心的“图像层面到底是什么意思”。
- 它把 INCODE 的 phase/response 语义落到实际图像变化上。
- 它不需要训练新模型，成本低。
- 结果可以直接进入周报或组会图。

第二优先级：实验 B。

理由：

- 它从机制上证明 LTE 的 `cell -> phase` 路径确实会随 cell 改变 basis phase。
- 但它比实验 A 更偏内部量，单独展示容易再次显得抽象。

第三优先级：实验 C。

理由：

- 它有助于支撑 SC-INR 的 response path，但属于 decoder 输入诊断。
- 需要配合 NoSinc 和可视化解释，不能单独作为强证据。

## 7. 最强反对意见

反对意见 1：

> INCODE 允许 prior-conditioned phase，为什么 SC-INR 不能让 cell 进 phase？

回答：

INCODE 的条件来自图像/任务先验 embedding，它可以携带内容信息；而 SC-INR 中的 `cell` 只表示输出像素窗口大小，不携带局部内容。内容先验控制 phase 是合理的，窗口大小控制 phase 则会把观测条件和结构对齐混在一起。

反对意见 2：

> SC-INR 也有 feature-conditioned phase，是否也可能移动结构？

回答：

会。SC-INR 不是禁止 phase，而是把 phase 的来源限制为图像内容 feature。也就是说，结构对齐应由内容条件决定，而不是由输出尺度决定。

反对意见 3：

> `W` 乘在 Fourier feature 上，后面还有 MLP，怎么能说它是可见性？

回答：

不能说它是最终 RGB 频谱振幅。更准确说，它是 MLP 前 Fourier-like latent feature 的 response gate。图像层面可以解释为局部模式的可见强度 proxy，但必须保留“decoder 输入层面”的边界。

反对意见 4：

> NoSinc 有很高 consistency，是否说明 sinc 不重要？

回答：

不能这样解释。NoSinc 去掉了 cell response，输出对 cell 更不敏感，自然可能获得高 same-LR consistency；但这不等于它在重建质量、纹理误差或 footprint observation 上更合理。必须同时看 quality、texture 和 response 诊断。

## 8. 结论

INCODE 最适合用来支撑 SC-INR 的“参数语义与条件路径分工”：

- phase 是空间对齐路径；
- frequency 是局部粒度/方向路径；
- amplitude/response 是模式强度路径；
- `cell` 作为 footprint 不应控制空间对齐，而应控制观测响应。

当前不需要训练 INCODE baseline。最需要做的是一个轻量 cell-only intervention：固定图像和坐标，只改变 cell，展示 LTE 的变化是否更像结构/相位偏移，SC-INR 的变化是否更像纹理可见性变化。这个实验能把 INCODE 的 method 启发、SC-INR 的公式设计和图像层解释连起来。
