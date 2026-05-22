# LIIF / LTE 文献笔记与 SC-INR 对照底稿

本文档是后续讨论 `LIIF`、`LTE` 和 `SC-INR` 细节的文献笔记底稿。它的目标不是复述代码，而是解释这些方法为什么这样组织变量、哪些解释来自原文或实现、哪些解释是本项目基于常规信号建模思想给出的理解。

参考文献：

- Chen et al., *Learning Continuous Image Representation with Local Implicit Image Function*, CVPR 2021.
- Lee and Jin, *Local Texture Estimator for Implicit Representation Function*, CVPR 2022.

当前边界：本轮已经结合本 repo 实现核对了关键路径，但由于 PDF 下载环境不稳定，原文逐句引用仍需后续继续补。下面会显式区分“文献/实现事实”和“本项目解释”。

## 1. 共同问题：ASISR 为什么需要坐标查询

LIIF 和 LTE 都服务 arbitrary-scale image super-resolution。给定 LR 图像，encoder 提取特征图，然后 decoder 在连续坐标域上查询任意 HR 像素值。

普通 INR 常写成：

$$
I(x)=f_\theta(x)
$$

也就是直接把全局坐标输入网络，让网络表示一个连续信号。

但 LIIF 不是为每张图单独训练一个全局 INR，而是训练一个可泛化到任意输入图像的局部隐式函数。它的典型形式是：

$$
s=f_\theta(z^*, x_q-v^*)
$$

其中：

- $x_q$ 是 query 坐标；
- $v^*$ 是邻近的 LR feature 坐标；
- $z^*$ 是该位置的局部图像特征；
- $x_q-v^*$ 是 query 相对局部 feature anchor 的坐标差；
- $s$ 是该 query 位置的预测 RGB。

## 2. LIIF：为什么输入坐标差，而不是全局坐标

### 2.1 文献/实现事实

LIIF 使用局部特征条件化隐式函数。decoder 不只看坐标，还看局部图像特征：

$$
s=f_\theta(z^*, x_q-v^*)
$$

本 repo 的 `models/liif.py` 也是这样：先在 feature map 上取 `q_feat` 和 `q_coord`，然后构造：

$$
\delta=x_q-v^*
$$

并把 scaled relative coordinate 拼入 MLP。

### 2.2 为什么是坐标差

这里和普通 INR 的差别很关键。

普通 INR 的目标通常是表示一个固定信号，所以输入全局坐标 $x$ 是合理的。网络可以学习：在这个具体图像中，坐标 $x$ 对应什么颜色。

LIIF 的目标不是记住一张图，而是学习一个“局部解码规则”。这个规则要在所有位置、所有图像上共享。因此它更自然地使用局部坐标：

$$
\delta=x_q-v^*
$$

这表示 query 点相对于当前局部 feature anchor 的偏移。

这样做的直观意义是：

- $z^*$ 负责告诉 decoder “这里是什么内容”，例如边缘、纹理、平坦区域；
- $\delta$ 负责告诉 decoder “我在这个局部内容附近的哪个位置采样”；
- 同一个 MLP 可以复用于不同空间位置，因为它看到的是局部坐标，而不是图像全局坐标。

所以坐标差不是为了解决非整数倍率本身，而是为了把连续图像表示局部化。它把“空间绝对位置”转成“局部 feature 条件下的相对查询位置”。这和卷积网络中的平移共享思想一致：局部模式可以在不同位置复用。

### 2.3 local ensemble 的作用

如果只选一个最近邻 feature anchor，query 跨过 feature cell 边界时，条件特征会突然变化。LIIF 用 local ensemble 从周围多个 feature anchor 分别预测，再用面积权重融合。

因此更完整的理解是：

$$
I(x_q)=\sum_i w_i f_\theta(z_i, x_q-v_i)
$$

其中 $i$ 遍历邻近 anchor。这个机制让输出对 feature anchor 的切换更平滑。

### 2.4 feature unfolding

LIIF 还使用 feature unfolding，让一个 anchor 的特征包含邻域信息。常见实现是把 $3\times3$ 邻域 feature 展开后拼接。

这个设计解决的是局部上下文不足，不改变坐标差的基本含义。

### 2.5 cell decoding

LIIF 也使用 cell。cell 表示输出像素在连续坐标域中的 footprint size。当前实现中，如果 `cell_decode=True`，MLP 输入是：

$$
\mathrm{MLP\ input}=[z^*,\delta,c]
$$

所以不能说 LIIF 完全没有尺度信息。更准确地说：LIIF 把 cell 作为普通条件变量交给 MLP，自由学习 cell 的影响；它没有给 cell 设计显式的采样响应结构。

## 3. LIIF 的非整数倍率训练网格怎么理解

### 3.1 本 repo 的训练流程

当前训练 wrapper 是 `sr-implicit-downsampled`。训练时随机采样：

$$
s\sim U(1,4)
$$

如果 `inp_size=48`，则固定 LR patch 大小：

$$
W_{LR}=48
$$

HR patch 大小取：

$$
W_{HR}=\mathrm{round}(sW_{LR})
$$

然后从原始 HR 图上裁出大小为 $W_{HR}\times W_{HR}$ 的 HR patch，再 resize/downsample 到 $48\times48$ 的 LR patch。

之后，HR query 坐标来自这个 HR patch 的像素中心网格：

$$
x_q\in \mathrm{make\_coord}(W_{HR},W_{HR})
$$

cell 为：

$$
c_x=\frac{2}{H_{HR}},\qquad c_y=\frac{2}{W_{HR}}
$$

这里坐标域是 $[-1,1]$，所以 cell 是归一化坐标下一个 HR 像素的宽高。

### 3.2 非整数倍率时网格怎么算

非整数倍率不会要求 LR 网格和 HR 网格整数对齐。流程是：

1. 随机采样连续倍率 $s$；
2. 用 $\mathrm{round}(sW_{LR})$ 得到一个整数 HR patch size；
3. 在这个 HR patch 上生成离散 HR 像素中心坐标；
4. 将 HR patch resize 到固定 LR patch；
5. 模型输入 LR patch，输出在 HR 坐标上的采样值。

因此，非整数倍率的本质是 HR 网格大小可以是任意整数，而 LR 网格固定或由 floor/round 得到。query 坐标是连续归一化坐标，不要求每个 HR 像素中心都和 LR feature grid 整齐对齐。

这正是 LIIF 类方法的优势：模型学的是从局部 feature 到连续坐标查询的函数，而不是只能输出某个固定整数倍率的网格。

## 4. LTE 到底做了什么

### 4.1 它不是简单坐标编码

LTE 可以被粗略看成给 LIIF 加了 Fourier-like 表示，但它不只是普通 positional encoding。

普通坐标编码通常是对坐标 $x$ 使用固定频率：

$$
\gamma(x)=[\sin(\omega_1x),\cos(\omega_1x),\ldots]
$$

LTE 的重点是：频率和系数不是固定的，而是由图像局部 feature 预测出来的。也就是说，它试图估计局部纹理参数。

### 4.2 振幅和频率是谁的

在本 repo 实现中，LTE 先提取 feature map：

$$
M=E(I_{LR})
$$

然后用卷积分支预测：

$$
A(z)=h_A(z),\qquad \omega(z)=h_\omega(z)
$$

这里的“振幅/频率”不是 LR 图像原始像素的物理频谱，也不是最终 RGB 输出的严格频谱。更准确的说法是：它们是局部 feature-conditioned Fourier feature 的参数，服务于 decoder 的中间表示。

也就是说：

- 它们由 LR feature 预测；
- 它们描述的是 decoder 用来重建 HR query 的局部纹理基；
- 经过 MLP 混合后，不能再直接等同为最终 RGB 图像的 Fourier 振幅和频率。

### 4.3 为什么可以用卷积预测振幅和频率

实现事实：LTE 使用卷积从 feature map 预测 coefficient 和 frequency map。

常规解释是：卷积有空间共享和局部性假设，适合从图像 feature 中估计局部纹理参数。图像中的边缘、条纹、重复结构等局部模式在不同位置反复出现，所以用共享卷积分支预测局部参数是自然的。

这不是说卷积能“真实测量出物理频率”，而是说它学习一个从局部 feature 到 Fourier-like decoder 参数的映射：

$$
z\mapsto (A,\omega)
$$

这个思路和许多 implicit representation / neural field 方法中的“由条件特征调制局部函数参数”是一致的。

### 4.4 三角函数代表什么

LTE 使用：

$$
\cos(\pi\theta),\qquad \sin(\pi\theta)
$$

其中：

$$
\theta=\omega(z)^\top\delta+\phi
$$

三角函数表示 Fourier-like 局部基函数。它让 decoder 能更容易表达周期性、高频和纹理结构。相比直接把 $\delta$ 拼进 MLP，Fourier basis 对高频变化更友好。

但要注意：LTE 最后仍然把这些 Fourier feature 送入 MLP，因此它不是一个纯解析 Fourier 重建器，而是一个使用 Fourier-like 中间表示的神经 decoder。

### 4.5 相位为什么由尺度/cell 预测

代码事实是：LTE 用 cell 预测 additive phase：

$$
\theta=\omega(z)^\top\delta+h_p(c)
$$

其中：

$$
h_p(c)=\mathrm{Linear}(c)
$$

如果从 LTE 的设计动机去理解，cell 提供了输出像素大小信息。模型需要知道当前 query 对应的输出采样尺度，因此 LTE 让 cell 影响 Fourier feature 的相位项。

但从本项目的 sampling observation 视角看，这里有一个可质疑点：如果 cell 表示 pixel footprint，那么它更自然地对应“观测窗口大小”，即影响某个频率分量在窗口平均后保留多少强度，而不是任意移动纹理相位。

所以本项目不会说 LTE “一定错”，而是说：

$$
c\rightarrow h_p(c)
$$

给了模型一个 learned phase shortcut。它在训练尺度内可能有效，但在 OOD scale 下缺少解析约束。

## 5. 从 LTE 到 SC-INR：变量职责重分配

通过上面的分析，SC-INR 的倾向可以表述为：

1. feature 决定局部连续函数本身；
2. relative coordinate 决定在该局部函数上的查询位置；
3. cell 只描述输出像素 footprint，不应该直接改变局部纹理相位。

因此，SC-INR 将 LTE 的：

$$
c\rightarrow h_p(c)
$$

改成：

$$
(c,\omega(z))\rightarrow W(\omega(z),c)
$$

其中：

$$
W(\omega,c)=\operatorname{sinc}\left(\frac{\omega_xc_x}{2}\right)
\operatorname{sinc}\left(\frac{\omega_yc_y}{2}\right)
$$

局部 Fourier phase 由 feature 和 relative coordinate 决定：

$$
\theta=\omega(z)^\top\delta+\phi(z)
$$

cell 不进入 $\theta$，只进入 $W$。

## 6. SC-INR 按本项目倾向的解释

SC-INR 的核心解释是：把局部 Fourier 表示拆成“内容函数”和“观测响应”。

内容函数由 feature 决定：

$$
A(z),\omega(z),\phi(z)
$$

当前输出 cell 下的可观测强度由解析 response 决定：

$$
A_{\mathrm{eff}}(z,c)=A(z)W(\omega(z),c)
$$

这样，cell 的作用从“改变相位”变成“调制有效振幅”：

| 模型 | cell 的作用 | 本项目解释 |
| --- | --- | --- |
| LIIF | 直接拼入 MLP | cell 是普通条件变量。 |
| LTE | learned phase condition | cell 可以直接移动局部纹理相位。 |
| SC-INR | analytic response | cell 是 pixel footprint，调制频率分量的观测强度。 |

这也是为什么 SC-INR 的 claim 应该写成 decoder-side sampling consistency，而不是 whole-network scale equivariance。

## 7. 与本 repo 实现的对应

这一节只用于把文献理解和代码对上，不作为主叙事。

### 7.1 LIIF

本 repo 中：

$$
\mathrm{MLP\ input}=[q_{feat},\delta,c]
$$

默认 `feat_unfold=True`、`cell_decode=True`，使用 local ensemble。

### 7.2 LTE

本 repo 中：

$$
q_{coef}=h_A(z),\qquad q_{freq}=h_\omega(z)
$$

$$
\theta=q_{freq}^\top\delta+h_p(c)
$$

$$
\mathrm{MLP\ input}=q_{coef}\odot[\cos(\pi\theta),\sin(\pi\theta)]
$$

LTE/SC-INR 默认还加 bilinear LR residual；LIIF 当前实现没有这个 `upinput` 残差。

### 7.3 SC-INR

本 repo 最终候选为 `sc_inr_signed_phiz`：

$$
\omega(z)=\omega_{bound}\tanh(g_\omega(z)),\qquad \omega_{bound}=2.1
$$

$$
\theta=\omega(z)^\top\delta+\phi(z)
$$

$$
W=\operatorname{sinc}\left(\frac{\omega_xc_x}{2}\right)
\operatorname{sinc}\left(\frac{\omega_yc_y}{2}\right)
$$

$$
\mathrm{MLP\ input}=q_{coef}\odot W\odot[\cos(\pi\theta),\sin(\pi\theta)]
$$

其中 `torch.sinc(x)` 的定义是：

$$
\operatorname{sinc}(x)=\frac{\sin(\pi x)}{\pi x}
$$

所以写论文公式时必须小心归一化，避免多一个或少一个 $\pi$。

## 8. 仍需继续核查的问题

1. LIIF 原文如何准确表述 cell decoding，以及是否把 cell 解释为 pixel size / shape。
2. LTE 原文如何解释 cell-conditioned phase，它是称为 phase、scale-aware term，还是 texture query correction。
3. LTE 原文是否明确说明 coefficient/frequency 是 local texture 的参数，还是只从实现上体现。
4. SC-INR 的 sinc response 如果写成 box footprint 推导，公式中的坐标尺度、$\pi$ 和 `torch.sinc` 归一化必须逐项核对。
5. local ensemble 和 bilinear residual 会影响最终输出，不能把任何一个模型解释成单个局部 Fourier 分量直接输出 RGB。

## 9. 当前写作边界

可以写：

- LIIF 将图像表示为由局部特征条件化的连续隐式函数；
- 坐标差 $x_q-v^*$ 表示 query 在局部 feature anchor 坐标系中的位置；
- LTE 用 feature-conditioned Fourier-like 参数增强局部纹理表达；
- LTE 的 cell-conditioned phase 给 cell 一个直接改变 Fourier phase 的自由路径；
- SC-INR 将 cell 解释为 output footprint，并让它通过 analytic response 调制 Fourier feature。

不能写：

- LIIF 不使用 cell；
- LTE 的 coefficient/frequency 是最终 RGB 频谱的严格物理量；
- LTE 原文承认 `phase(cell)` 是 shortcut；
- SC-INR 证明了 exact box integral；
- SC-INR 是严格尺度等变网络；
- sinc 是唯一正确 response。
