# LIIF、LTE 与 SC-INR 方法脉络

本文档整理 `LIIF -> LTE -> SC-INR` 的方法链条。目标是说明三类 decoder
如何组织图像特征、查询坐标和输出尺度信息，而不是复述训练脚本或实验记录。

参考文献：

- Chen et al., *Learning Continuous Image Representation with Local Implicit Image Function*, CVPR 2021.
- Lee and Jin, *Local Texture Estimator for Implicit Representation Function*, CVPR 2022.

## 1. 基本问题

Arbitrary-scale image super-resolution 需要从一张低分辨率图像生成任意目标分辨率下的像素值。
这类方法通常先用 encoder 得到低分辨率特征图：

$$
M = E(I_{\mathrm{LR}}),
$$

再对目标坐标 $x_q$ 查询颜色值。关键问题是 decoder 如何使用：

- 局部图像特征 $z$；
- 查询位置与局部特征位置之间的相对坐标 $\delta$；
- 输出像素的 cell/footprint $c$。

普通 INR 常被写成：

$$
I(x)=f_\theta(x),
$$

即用一个网络表示单个连续信号。LIIF 类方法的目标不同：它不是为每张图单独拟合一个
全局坐标函数，而是学习一个可在不同图像、不同位置共享的局部解码规则。

## 2. LIIF

LIIF 的核心是 local implicit image function。给定 query 坐标 $x_q$，在特征图中找到邻近
anchor $v^*$ 及其特征 $z^*$，再用共享 MLP 预测该位置的 RGB：

$$
s=f_\theta(z^*, x_q-v^*).
$$

这里的坐标输入不是单独的全局坐标，而是相对坐标：

$$
\delta=x_q-v^*.
$$

这样做的含义很直接：

- $z^*$ 提供局部内容，例如边缘、纹理或平坦区域；
- $\delta$ 表示 query 在该局部 anchor 坐标系中的位置；
- 同一个 decoder 可以在不同空间位置复用，因为它看到的是局部坐标和局部特征。

因此，LIIF 与普通 INR 的主要差别不是“是否能处理非整数倍率”，而是从单图的全局坐标函数，
转向由 LR feature 条件化的局部共享函数。

更完整地，LIIF 通常还包含三个补充机制：

- local ensemble：从多个邻近 anchor 分别预测，再按面积权重融合，降低 anchor 切换带来的不连续；
- feature unfolding：把邻域特征并入当前 anchor，增加局部上下文；
- cell decoding：把输出像素大小 $c$ 作为 MLP 的输入条件。

所以不能把 LIIF 描述成“完全不使用尺度信息”。更准确的说法是：LIIF 可以把 cell 作为普通条件变量交给 MLP，
但没有为 cell 规定显式的频率响应或采样结构。

## 3. LTE

LTE 延续 LIIF 的局部隐式查询框架，但认为直接把相对坐标交给 MLP 对局部高频纹理并不充分。
因此，LTE 在 decoder 侧引入 local texture estimator，用图像特征预测局部 Fourier-like 表示。

在形式上，LTE 从局部特征 $z$ 预测 coefficient 和 frequency：

$$
A(z)=h_A(z),\qquad \omega(z)=h_\omega(z).
$$

这些量应理解为 decoder 中间特征的局部纹理参数，而不是最终 RGB 图像的严格物理频谱。
它们由 LR feature 预测，用来构造更适合表达纹理和高频变化的 basis。

对相对坐标 $\delta$，LTE 构造三角基：

$$
B(z,\delta,c)
=
\left[
\cos\left(\pi(\omega(z)^\top\delta+h_p(c))\right),
\sin\left(\pi(\omega(z)^\top\delta+h_p(c))\right)
\right].
$$

然后用 coefficient 调制该 basis，并送入 MLP：

$$
\hat y
=
g_\theta\left(A(z)\odot B(z,\delta,c)\right).
$$

LTE 的关键改动有两点：

1. 频率和 coefficient 来自局部 feature，而不是固定 positional encoding；
2. cell 通过 learned phase term $h_p(c)$ 进入 Fourier-like 表示。

第一点增强了局部纹理表达能力。第二点让输出像素尺度可以影响 decoder，但它是一个自由学习的 phase
路径。若从 pixel footprint 的角度看，cell 表示输出像素覆盖范围；它更自然地对应采样窗口对不同频率分量的观测响应。
因此，$h_p(c)$ 可以被视为一种有效但缺少解析约束的尺度条件路径：它允许 cell 改变局部 Fourier feature 的相位。

这里需要保持边界：LTE 的设计并不能简单概括为“错误”。它提供了强于 LIIF 的局部纹理建模方式；
需要关注的是 cell 进入 phase 路径后，在尺度外推时可能缺少结构约束。

## 4. SC-INR

SC-INR 保留 LTE 的局部 Fourier-like decoder 思路，但重新划分变量角色：

- 局部特征 $z$ 决定连续局部函数本身；
- 相对坐标 $\delta$ 决定在该局部函数上的查询位置；
- cell $c$ 表示输出像素 footprint，用于描述当前像素如何观测该局部函数。

因此，SC-INR 将 LTE 中的

$$
c\rightarrow h_p(c)
$$

改为

$$
(\omega(z),c)\rightarrow W(\omega(z),c).
$$

最终候选版本使用 feature-conditioned phase：

$$
A(z),\omega(z),\phi(z),
$$

并把局部 phase 写成：

$$
\theta=\omega(z)^\top\delta+\phi(z).
$$

cell 不进入 $\theta$。它只通过解析 response 调制 Fourier feature 的有效输入强度：

$$
W(\omega,c)
=
\operatorname{sinc}\left(\frac{\omega_x c_x}{2}\right)
\operatorname{sinc}\left(\frac{\omega_y c_y}{2}\right).
$$

于是 decoder 输入可以理解为：

$$
A_{\mathrm{eff}}(z,c)
=
A(z)\odot W(\omega(z),c),
$$

$$
\hat y
=
g_\theta\left(
A_{\mathrm{eff}}(z,c)
\odot
\left[
\cos(\pi\theta),\sin(\pi\theta)
\right]
\right).
$$

这个改动的重点不是简单增加一个函数，而是把 cell 从 learned phase condition
转移到 sampling response condition。也就是说，尺度信息不再直接移动局部纹理相位，
而是根据频率和 footprint 大小调制该频率分量在当前输出像素下的可观测强度。

这种解释应限定在 decoder 输入层面。由于后续还有 MLP、local ensemble 和残差路径，
不能把 $A(z)W(\omega,c)$ 直接写成最终 RGB 图像的严格频谱振幅。

## 5. 方法关系

三者可以按下面的链条理解：

| 方法 | 局部内容 | 坐标作用 | cell 作用 |
| --- | --- | --- | --- |
| LIIF | $z$ 条件化 MLP | 相对坐标 $\delta$ 输入 MLP | 作为普通条件变量 |
| LTE | $z$ 预测 coefficient / frequency | $\omega(z)^\top\delta$ 构成 Fourier-like basis | learned phase term $h_p(c)$ |
| SC-INR | $z$ 预测 coefficient / frequency / phase | $\omega(z)^\top\delta+\phi(z)$ 构成局部 phase | analytic footprint response $W(\omega,c)$ |

因此，SC-INR 的位置是对 LTE decoder 侧 cell 使用方式的约束化改造：

- 继承局部 Fourier-like 表达；
- 保留 feature-conditioned 内容参数；
- 去掉 cell-conditioned phase；
- 用解析 footprint response 表示 cell 对频率分量的观测影响。

## 6. 写作边界

可以写：

- LIIF 将图像表示为由局部特征条件化的连续隐式函数；
- LTE 用 feature-conditioned Fourier-like 参数增强局部纹理表达；
- LTE 的 cell-conditioned phase 为输出尺度提供了 learned phase 路径；
- SC-INR 将 cell 解释为 output footprint，并通过 analytic response 调制 decoder 输入特征；
- SC-INR 更准确的表述是 decoder-side sampling consistency 或 scale-decoupled observation。

不要写：

- LIIF 完全不使用 cell；
- LTE 的 coefficient/frequency 是最终 RGB 的严格物理频谱；
- LTE 原文承认 $h_p(c)$ 是 shortcut；
- $A(z)W(\omega,c)$ 是最终 RGB 图像的严格频谱振幅；
- SC-INR 证明了 exact box integral、严格尺度等变或 sinc 的唯一因果性。
