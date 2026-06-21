# SC-INR Method 叙事完整版 2026-06-07

本文档是一版可进入论文草稿的中文 Method 叙事。目标是讲清楚 SC-INR 的建模动机、变量角色、公式来源、实现对应和边界。它不是实验结果汇总，也不把诊断证据升级为强结论。

## 1. 问题设定：任意尺度 SR 中 decoder 在预测什么

Arbitrary-scale image super-resolution 的目标是：给定低分辨率图像 \(I_{\mathrm{LR}}\)，在任意目标分辨率下预测输出像素值。LIIF 类方法通常先用 encoder 得到低分辨率特征图：

$$
M = E(I_{\mathrm{LR}}),
$$

然后把目标图像中的每个输出像素转化为连续坐标查询。对于一个 query coordinate \(x_q\)，decoder 从特征图中取局部 feature，并预测该坐标对应的颜色：

$$
\hat y(x_q)=D(z,\delta,c).
$$

这里三个变量承担不同角色：

- \(z\)：由 LR 图像邻域编码得到的局部 feature，提供边缘、纹理、平滑区域等内容依据；
- \(\delta=x_q-v\)：query 坐标相对局部 feature anchor \(v\) 的位置，表示在局部坐标系中的查询点；
- \(c\)：目标输出像素在连续坐标域中的 cell size，也就是 pixel footprint 的尺寸。

这里需要特别区分：单个输出像素值本身不是局部连续信号的信息来源。一个像素值只是一个离散 RGB/Y 观测，仅凭它无法判断附近是边缘、周期纹理还是平坦区域。LIIF 类方法能够进行连续坐标外推，不是因为单个像素包含了连续信号，而是因为 encoder feature 从 LR 图像邻域中提取了上下文，decoder 学习了由 feature 条件化的局部连续解码规则。

因此，局部连续性主要来自：

1. LR feature 随空间位置的连续/平滑变化；
2. decoder 对相对坐标 \(\delta\) 的共享局部解码；
3. 训练中不同尺度、不同 query 坐标提供的监督。

而不是来自单个像素值本身。

## 2. 什么是 pixel footprint

在连续坐标表示中，一个目标像素不能只被理解成数学上的无限小点。给定目标分辨率 \(H\times W\)，每个输出像素在归一化坐标域 \([-1,1]^2\) 中对应一个有限小区域，其尺寸通常写为：

$$
c=\left(\frac{2}{H},\frac{2}{W}\right).
$$

这个有限区域就是该输出像素的 footprint。它描述的是：这个像素值对应连续图像域中的多大空间支持范围。

这也是为什么使用 footprint 而不是 area。area 只是一个标量，例如 \(c_xc_y\)，只能表示面积大小；footprint 保留了窗口在不同方向上的宽度，例如 \(c_x\) 和 \(c_y\)。对于方向性纹理或 Fourier 分量，横向窗口宽度和纵向窗口宽度会产生不同响应，因此只知道 area 不够。

简言之：

> pixel footprint 是目标输出像素在连续坐标域中的采样窗口。它不提供图像内容，但决定该像素值应被解释为点采样，还是一个有限窗口内的平均观测。

在 ASISR 中，当目标分辨率变化时，query 坐标和 footprint 都会变化。目标分辨率越高，单个像素 footprint 越小；目标分辨率越低，单个像素 footprint 越大。非整数倍率也不需要建立 LR 像素和 HR 像素的一一整数对应，只需要为目标分辨率生成对应的 continuous query grid 和 cell size。

实现时，`cell` 通常先在归一化目标坐标域中构造，再和相对坐标一样换算到局部 feature-grid 单位。例如代码中会做 `rel_cell[:, :, 0] *= feat.shape[-2]`、`rel_cell[:, :, 1] *= feat.shape[-1]`。因此下文公式中的 \(c\) 可理解为已经与 \(\delta\)、\(\omega\) 处在同一局部坐标单位下的 footprint 尺寸。

## 3. LIIF 与 LTE：已有 decoder 如何使用 cell

LIIF 的核心是 local implicit image function。它用局部 feature 和相对坐标预测颜色：

$$
\hat y=f_\theta(z,\delta,c),
$$

其中 cell 可以作为普通条件变量输入 MLP。LIIF 的优势是把全局坐标函数转化为可共享的局部解码规则；但它没有显式规定 cell 应如何影响局部频率或采样响应。

LTE 在 LIIF 的局部隐式查询框架上引入 local Fourier-like decoder。它从局部 feature 预测 coefficient 和 frequency：

$$
A(z)=h_A(z),\qquad \omega(z)=h_\omega(z).
$$

然后用相对坐标构造三角基。LTE 的关键设计是 cell 通过 learned phase term 进入 Fourier-like phase：

$$
B_{\mathrm{LTE}}(z,\delta,c)
=
\left[
\cos\left(\pi(\omega(z)^\top \delta+h_p(c))\right),
\sin\left(\pi(\omega(z)^\top \delta+h_p(c))\right)
\right].
$$

再送入 MLP：

$$
\hat y
=
g_\theta\left(A(z)\odot B_{\mathrm{LTE}}(z,\delta,c)\right).
$$

LTE 的这个设计有较强表达能力，尤其能增强局部纹理建模。但从 footprint 语义看，cell 直接进入 phase 会带来一个问题：输出像素的大小可以直接移动局部纹理相位。也就是说，不同输出尺度下，decoder 看到的可能不再是同一个 feature-conditioned 局部函数在不同 footprint 下的观测，而是一个被 cell 改写相位的局部函数。

这并不意味着 LTE 是错误的。更准确地说，LTE 给了 cell 一条自由学习的 phase path；这条路径在训练尺度附近可能有效，但在尺度外推时缺少明确的采样结构约束。

## 4. SC-INR 的核心建模假设

SC-INR 保留 LTE 的 local Fourier-like 表示能力，但重新划分变量角色：

- \(z\) 决定局部内容参数；
- \(\delta\) 决定在局部函数上的查询位置；
- \(c\) 只描述输出像素 footprint，即当前像素值对应多大的采样窗口。

因此，SC-INR 不让 cell 直接进入 phase。它把局部 phase 写成：

$$
\theta(z,\delta)=\omega(z)^\top\delta+\phi(z),
$$

其中

$$
A(z),\quad \omega(z),\quad \phi(z)
$$

都由 LR feature 预测。这里的 \(\phi(z)\) 是 feature-conditioned content phase，不接收 cell 或 scale。

cell 的作用被限制在 analytic response：

$$
(\omega(z),c)\rightarrow W(\omega(z),c).
$$

因此，SC-INR 的核心建模假设是：

> 输出尺度不应该直接移动局部纹理相位；它应该通过 pixel footprint 改变不同频率分量在该输出像素窗口下的可观测强度。

## 5. 为什么是 sinc response

为了说明 \(W(\omega,c)\) 的形式，可以先考虑一维局部 Fourier 分量：

$$
f(x)=A\cos(\pi(\omega x+\phi)).
$$

如果输出像素被视为点采样，那么在 \(x_q\) 处的输出就是：

$$
f(x_q)=A\cos(\pi(\omega x_q+\phi)).
$$

但如果输出像素对应宽度为 \(c\) 的 footprint，那么它更自然地对应窗口平均：

$$
\bar f(x_q,c)
=
\frac{1}{c}
\int_{x_q-c/2}^{x_q+c/2}
A\cos(\pi(\omega x+\phi))\,dx.
$$

积分后得到：

$$
\bar f(x_q,c)
=
A\operatorname{sinc}\left(\frac{\omega c}{2}\right)
\cos(\pi(\omega x_q+\phi)).
$$

这里

$$
\operatorname{sinc}(t)=\frac{\sin(\pi t)}{\pi t}.
$$

这个结果说明：footprint 不需要改变相位 \(\omega x_q+\phi\)，而是改变该频率分量经过窗口平均后的响应。当 footprint 变大时，高频分量更容易被平均掉；当 footprint 很小时，响应接近点采样。

二维矩形 footprint 下，如果沿 \(x/y\) 两个方向做 separable box average，则得到：

$$
W(\omega,c)
=
\operatorname{sinc}\left(\frac{\omega_x c_x}{2}\right)
\operatorname{sinc}\left(\frac{\omega_y c_y}{2}\right).
$$

这里 \(c=(c_x,c_y)\) 是 footprint 尺寸，\(\omega=(\omega_x,\omega_y)\) 是局部频率。它们共同决定一个输出像素窗口内包含多少频率振荡，以及该频率分量被平均后还能保留多少。

需要注意：这里的 sinc 是 signed response，不是永远为正的 attenuation coefficient；在直观解释中可以把其绝对值理解为响应幅度，但符号本身也是 box filter 对 Fourier 分量的解析响应的一部分。这个 sinc response 是 decoder 输入层面的采样响应启发。真实图像不严格等于单个 Fourier 分量，SC-INR 后面还有 MLP、local ensemble 和 residual upsampling，因此不能把最终 RGB 输出写成 exact box integral。

## 6. SC-INR decoder 形式

给定局部 feature \(z\)，SC-INR 预测：

$$
A(z),\qquad \omega(z),\qquad \phi(z).
$$

当前 final candidate 使用 signed bounded frequency parameterization：

$$
\omega(z)=\omega_{\max}\tanh(g_\omega(z)).
$$

局部 phase 为：

$$
\theta(z,\delta)=\omega(z)^\top\delta+\phi(z).
$$

cell 不进入 \(\theta\)。Fourier-like basis 为：

$$
B_{\mathrm{SC}}(z,\delta)
=
\left[
\cos(\pi\theta(z,\delta)),
\sin(\pi\theta(z,\delta))
\right].
$$

然后用 analytic response 调制 basis：

$$
W(\omega(z),c)
=
\operatorname{sinc}\left(\frac{\omega_x(z)c_x}{2}\right)
\operatorname{sinc}\left(\frac{\omega_y(z)c_y}{2}\right).
$$

MLP 前的 decoder 输入可概括写成：

$$
u
=
A(z)\odot \widetilde{W}(\omega(z),c)\odot B_{\mathrm{SC}}(z,\delta).
$$

其中 \(\widetilde{W}\) 表示把每个 frequency 的 response 同时作用到对应的 cos/sin pair 上。实现中先计算 \(W\in\mathbb{R}^{K}\)，分别乘到 \(K\) 个 cosine 和 \(K\) 个 sine basis，再与长度为 \(2K\) 的 coefficient 向量逐元素相乘。

最终输出为：

$$
\hat y
=
g_\theta(u).
$$

也可以把

$$
A_{\mathrm{eff}}(z,c)=A(z)\odot W(\omega(z),c)
$$

理解为 MLP 前 Fourier feature 的 effective amplitude proxy。但这个 proxy 只对应 decoder 中间输入，不是最终 RGB 图像的严格物理频谱振幅。

## 7. 和 LTE 的关键差别

LTE 和 SC-INR 都使用 local Fourier-like decoder，但 cell 的路径不同：

| 方法 | 内容参数 | phase | cell 路径 |
| --- | --- | --- | --- |
| LTE | \(A(z),\omega(z)\) | \(\omega(z)^\top\delta+h_p(c)\) | learned phase shift |
| SC-INR | \(A(z),\omega(z),\phi(z)\) | \(\omega(z)^\top\delta+\phi(z)\) | analytic response \(W(\omega,c)\) |

因此，SC-INR 的改动不是简单增加一个 sinc 函数，而是把 cell 从 content phase path 中移出，转移到 observation response path 中。

可以概括为：

> LTE lets the output cell move local texture phase; SC-INR lets the output footprint modulate how strongly each local frequency component is observed.

中文表述为：

> LTE 让尺度条件改变局部纹理相位；SC-INR 将尺度解释为输出像素 footprint，并用 analytic response 描述该 footprint 对局部频率分量的平均效应。

## 8. 训练与推理中的非整数倍率

LIIF 类方法处理非整数倍率的关键，是把输出图像看成连续坐标查询问题，而不是建立 LR/HR 像素的一一整数对应。

给定任意目标分辨率 \(H\times W\)，生成对应的 query grid：

$$
x_q\in[-1,1]^2,
$$

并为每个 query 设置：

$$
c=\left(\frac{2}{H},\frac{2}{W}\right).
$$

decoder 对每个 query 预测颜色即可。倍率是否为整数并不改变这个查询形式；它只改变 query 坐标分布和 cell size。

训练时随机采样不同下采样倍率，本质上是让模型见到不同分辨率、不同 cell size 和不同 query 坐标下的监督。模型学习的是共享局部解码规则，而不是为某个固定整数 scale 学一个专用映射。

SC-INR 在这个协议下进一步约束 cell 的作用：不同目标分辨率带来的 cell 变化，不再作为 learned phase shift，而是作为 footprint response 进入 decoder。

## 9. 更合适的解释算例

解释 SC-INR 时，不建议以单个像素或平滑区域作为主算例。单个像素不能反映局部连续信号，平滑区域也难以体现 frequency response 的必要性。

更合适的算例是具有方向性重复结构的局部纹理，例如建筑立面、窗格、栅格或斜向条纹。原因是：

- 这类结构可以自然理解为局部 Fourier-like 分量；
- \(\omega_x,\omega_y\) 有明确方向含义；
- footprint 在不同方向的宽度会影响对应频率分量的平均响应；
- OOD 大倍率下，错误的 phase/response 处理更容易表现为条纹错位、纹理漂移或过度平滑。

现有定性例子中，`Urban100/img_012.png` x8 crop `(y=432,x=864,size=96)` 是一个合适 case。该区域是建筑立面的方向性重复条纹：

| Model | Crop PSNR-Y |
| --- | ---: |
| Bicubic | 18.8876 |
| LTE | 20.4458 |
| SC-INR-NoPhi | 19.6163 |
| SC-INR | 23.2229 |

这个例子可以作为 selected qualitative case study，说明 SC-INR 在方向性重复结构上能恢复更清晰的局部模式。但它必须和完整 benchmark、footprint oracle 以及 NoSinc 负控一起解释，不能单独作为平均视觉质量证明。

## 10. 证据链如何支撑 Method 叙事

Method 叙事可以对应以下证据：

1. **主 benchmark**：final `SC-INR` 在三 seed 下相对 LTE 有 OOD `+0.0504 ± 0.0417` dB、ALL `+0.0320 ± 0.0555` dB 的正增益；ID 相对 LTE 为 `-0.0048 ± 0.0836` dB，因此主张应集中在 OOD/尺度外推方向。
2. **真实图像 case**：Urban100 建筑条纹 x8 selected crop 中，`SC-INR` PSNR-Y `23.2229`，高于 LTE `20.4458`。
3. **Footprint oracle**：在固定 LR/query、只改变 cell 的诊断中，cell multiplier `4` 时 `SC-INR` oracle RMSE `0.02148`，低于 LTE `0.02309` 和 NoSinc `0.02750`；tracking RMSE `0.04745`，低于 LTE `0.04834` 和 NoSinc `0.05134`。
4. **机制 gate**：LTE 的 learned cell-phase delta 非零，`SC-INR` 的 analytic active response delta 非零，NoSinc 为零，支持 “cell phase path” 与 “footprint response path” 的机制差异。

这些证据共同支持较稳的表述：

> SC-INR 将输出尺度建模为 pixel footprint，并在 Fourier-like decoder 中通过 analytic response 调制局部频率分量。该设计在当前证据下支持一种更有采样语义约束的 decoder，并在 OOD scale 和 selected 方向性重复纹理区域表现出更好的重建与 footprint tracking 行为。

## 11. 必须保留的边界

可以写：

- SC-INR 是 decoder-side sampling consistency 的一种实现；
- cell 表示输出像素 footprint，而不是图像内容来源；
- feature 条件化局部连续表示，cell 描述该表示被输出像素窗口观测时的采样响应；
- sinc response 来自 Fourier 分量的 box-average；
- SC-INR 相对 LTE/NoSinc 在当前诊断中更符合 footprint observation proxy。

不要写：

- 单个像素可以反映局部连续信号；
- SC-INR 证明了真实连续图像的 exact box integral；
- \(A(z)W(\omega,c)\) 是最终 RGB 的严格频谱振幅；
- sinc 是唯一因果；
- SC-INR 是 whole-network strict scale equivariant；
- selected qualitative examples 证明平均视觉质量整体更好；
- final `SC-INR` 在所有辅助指标上都强于 `SC-INR-NoPhi`。

## 12. 可直接放入论文的 Method 段落草稿

下面是一版更接近论文正文的连续叙事：

> In local implicit ASISR, an output pixel is queried by a continuous coordinate, but the pixel itself should not be confused with the source of continuous image information. A single RGB value does not determine the local signal around it. The local continuous representation is instead conditioned on the LR feature extracted from a spatial neighborhood, while the relative coordinate specifies where this representation is queried. The output cell specifies another aspect of the query: the finite footprint occupied by the target pixel in the continuous coordinate domain.
>
> This distinction is important for Fourier implicit decoders. LTE predicts local coefficients and frequencies from image features, but injects the output cell through a learned phase term. As a result, changing the target footprint can directly shift the local Fourier phase. We argue that the cell should instead describe the observation footprint of a fixed feature-conditioned local function. For a Fourier component, averaging over a box footprint does not shift its phase; it multiplies its response by a sinc term. This motivates replacing the learned cell-conditioned phase path with an analytic frequency-footprint response.
>
> SC-INR predicts \(A(z)\), \(\omega(z)\), and an optional content phase \(\phi(z)\) from the local feature. The local phase is \(\omega(z)^\top\delta+\phi(z)\), independent of the output cell. The cell enters only through \(W(\omega,c)=\sinc(\omega_x c_x/2)\sinc(\omega_y c_y/2)\), which modulates the Fourier feature before the MLP decoder. This design separates content, query position, and output footprint: features define the local function, coordinates query it, and the cell controls the footprint-dependent observation response.

## 13. 中文精简版

如果需要在组会中讲，可以压缩为：

> SC-INR 的核心不是简单给 LTE 加一个 sinc 函数，而是重新定义 cell 的角色。局部连续信息不是来自单个输出像素，而是来自 LR feature 条件化出的局部函数；相对坐标决定查询位置；cell 只表示目标像素在连续坐标域中的 footprint。对于局部 Fourier 分量，有限 footprint 的平均观测会自然产生 sinc response。因此，SC-INR 不让 cell 直接移动纹理相位，而是用 \(W(\omega,c)\) 调制该频率分量在当前 footprint 下的可观测强度。这使得 decoder 在尺度外推时有更明确的采样语义。
