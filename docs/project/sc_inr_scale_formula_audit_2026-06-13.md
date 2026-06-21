# SC-INR 定性图公平性与尺度公式叙事审查 2026-06-13

本文档回答两个问题：

1. `Urban100/img_016.png x30 crop=(672,384,96)` 图中 LIIF/LTE 输出结构明显扭曲，这个对比是否公平、问题在哪里。
2. SC-INR 的核心公式如何脱离“改 LTE 振幅”这类局部表述，从尺度变化和像素观测语义出发讲清楚。

## 1. 这张图从哪里来

该图来自：

`artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/figures/external_advantage_context_zoom/00_external_top_positive_Urban100_img_016_x30_crop21_context_zoom.png`

对应候选池记录在：

`artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_region_pool.csv`

记录内容：

| 字段 | 数值 |
| --- | ---: |
| dataset/image | Urban100 / img_016.png |
| scale | x30 |
| crop | y=672, x=384, size=96 |
| split | OOD |
| texture/edge/highpass bin | high / high / high |
| LIIF PSNR-Y | 10.3019 |
| LTE PSNR-Y | 10.9557 |
| SC-INR PSNR-Y | 18.9134 |
| min external delta | +7.9578 dB |

该目录的 README 明确说明：候选图只从 OOD 非平坦结构 crop 中选择，排序指标为
`min(SC-INR-LIIF, SC-INR-LTE)`。因此它不是随机样例，也不是平均视觉质量证据，而是
selected external-baseline advantage region。

## 2. 对比是否公平

从脚本和配置看，目前没有发现“给 LIIF/LTE 使用了不同输入或不同 crop”的公平性错误。

支持这一点的证据：

- 生成脚本是 `scripts/analysis/evaluate_advantage_regions.py`。
- 三个模型都使用同一张 HR 图像，通过 `make_lr_hr(img_path, scale)` 得到同一个 x30 LR 输入和同一个 HR GT。
- 预测时三者使用同一目标分辨率、同一 continuous query grid、同一 cell size。
- crop 坐标由 GT 网格确定，三者都在同一 `y=672, x=384, size=96` 区域计算局部 PSNR。
- 本地图只展示 `GT/Bicubic/LIIF/LTE/SC-INR`，其中 LIIF/LTE/SC-INR 都来自 seed1 checkpoint。
- 三者训练协议一致：DIV2K、`sr-implicit-downsampled` wrapper、`inp_size=48`、`scale_max=4`、`sample_q=2304`。x30 对三者都是 OOD。

需要保留的公平性边界：

- 这是 seed1 局部 crop 诊断，不是三 seed 定性结论。
- 这是按 SC-INR 同时优于 LIIF/LTE 排序后选出的 top positive case，本身带有展示选择偏置。
- local candidate pool 只覆盖 BSD100/Urban100 的前 30 张、每图每尺度最多 32 个 crop。
- 该图尚未迁移到 `paper_figures/` 并登记为正式论文图。
- “LIIF/LTE 特征扭曲”这个说法不够准确；图里直接观察到的是输出纹理/结构方向扭曲，不是内部 feature map。

因此结论应写成：

> 该图在同一推理协议内部是公平的，但它是专门筛出的 OOD 强优势案例。它可以用来解释极端尺度外推下的结构保持问题，不能单独作为 SC-INR 平均视觉优势或全局优越性的证据。

## 3. 为什么 LIIF/LTE 会出现明显扭曲

这张图最关键的背景是 x30。

`img_016.png` 的 HR 尺寸是 `1024x1024`。在 x30 评估中，脚本会裁到可整除区域，LR 约为
`34x34`，目标 HR 为 `1020x1020`。一个 `96x96` 的 HR crop 只覆盖：

$$
96/30 \approx 3.2
$$

个 LR 像素宽度。

也就是说，这个局部区域的重建几乎是在 3 个多 LR 像素的观测上外推一个高频、强方向性的建筑立面纹理。此时问题不是普通意义上的“补一点纹理”，而是：

- LR 观测非常稀疏；
- 建筑立面有规则重复条纹；
- 下采样后局部 phase/方向存在强歧义；
- 训练只见过 scale `1-4`，x30 的 cell 远小于训练范围。

训练中，feature-grid 单位下的 cell 大致是：

$$
c_{\mathrm{rel}} \approx \frac{2}{s}.
$$

训练尺度 `s in [1,4]` 时，`c_rel` 大约在 `[0.5, 2]`；x30 时：

$$
c_{\mathrm{rel}} \approx \frac{2}{30} \approx 0.067.
$$

这说明 x30 不只是“倍率更大”，而是把 decoder 推到了一个小 footprint 极限。

在这个极限下，一个合理的尺度模型应满足：

1. 输出像素 footprint 很小，应接近点采样；
2. 改变 cell 不应随意改变局部纹理的相位或方向；
3. 局部条纹的方向、间距和相位主要应由 LR feature 与坐标决定。

LIIF 把 cell 作为普通 MLP 输入，缺少这个极限约束。LTE 把 cell 通过 learned phase term 注入 Fourier-like phase：

$$
\theta_{\mathrm{LTE}}=\omega(z)^\top\delta+h_p(c).
$$

当 `c` 落到训练范围外时，`h_p(c)` 的外推没有保证满足“small footprint -> point sampling”的边界条件，也没有保证不移动局部相位。因此在极端 OOD 尺度上，LIIF/LTE 都可能出现纹理漂移、条纹弯曲或方向不稳。

这不是证明 LTE “错误”，而是说明：在这个 selected x30 case 中，LTE 的自由 cell-phase path 和 LIIF 的自由 cell conditioning 都更容易暴露尺度外推风险。

## 4. SC-INR 在这张图中更好的合理解释

SC-INR 的局部 phase 是：

$$
\theta_{\mathrm{SC}}=\omega(z)^\top\delta+\phi(z),
$$

其中 `omega` 和 `phi` 都来自 LR feature，不直接接收 cell。cell 只进入：

$$
W(\omega,c)=
\operatorname{sinc}\left(\frac{\omega_x c_x}{2}\right)
\operatorname{sinc}\left(\frac{\omega_y c_y}{2}\right).
$$

因此它满足两个尺度边界：

- 当 `c -> 0` 时，`W(omega,c) -> 1`，局部 Fourier 分量接近点采样，cell 不会移动相位；
- 当 `c` 变大时，高频分量会按 footprint response 被削弱，表示较大输出像素对高频纹理的平均观测。

对这张 x30 图，`c` 很小，所以重点不是 sinc 大幅压制高频，而是：

> SC-INR 在 small-footprint OOD 极限下保持了“cell 不改相位”的边界条件，让条纹位置和方向主要由 feature-conditioned local function 与坐标决定。

这比“SC-INR 给 LTE 的振幅注入换成 sinc”更接近本质。

图像上可以这样解释：

- LIIF/LTE 输出中黑色条纹有明显弯曲、错位和方向漂移；
- SC-INR 输出虽然仍有模糊和细节损失，但保留了更接近 GT 的建筑肋条主方向；
- 该现象与 x30 小 footprint 外推有关：尺度条件不应把局部纹理相位推走。

更进一步地说，SC-INR 不是因为 x30 下没有歧义而保持得更好，而是因为它把歧义约束在更合理的图像层面。

对建筑立面的条纹，可以把局部图像理解成一个连续的“条纹场”。这个条纹场有几类图像属性：

- 方向：黑色竖肋/斜肋朝哪个方向延伸；
- 间距：相邻条纹之间隔多远；
- 位置/相位：某一条黑线具体落在哪里；
- 可见性/对比度：这组条纹在当前尺度下看起来有多深、多清楚。

在 Fourier-like 表示里，这几类属性大致对应：

| 图像属性 | 公式角色 | 直观含义 |
| --- | --- | --- |
| 条纹方向 | \(\omega(z)\) 的方向 | 等相位线的朝向，决定黑线往哪边走 |
| 条纹间距 | \(\|\omega(z)\|\) | 频率越高，条纹越密 |
| 条纹位置 | \(\phi(z)\) 与 \(\omega(z)^\top\delta\) | 决定暗/亮带在局部坐标中落在哪里 |
| 条纹可见性 | \(A(z)W(\omega,c)\) | 决定这组条纹显示得强还是弱 |

图像中的黑色条纹不是由 \(W\) 决定“在哪里”的。黑线的位置来自 phase 的等值线，例如
\(\omega(z)^\top\delta+\phi(z)\) 接近某些暗带相位的位置。也就是说，方向、间距和位置属于几何/结构属性；可见性和对比度属于观测属性。

SC-INR 的关键是把这两层分开：

- \(z,\delta,\omega,\phi\) 决定条纹场的几何结构；
- \(c\) 只通过 \(W(\omega,c)\) 决定这个条纹场在当前 pixel footprint 下被看见多少。

因此，当尺度从训练范围外推到 x30 时，SC-INR 允许条纹变淡、变模糊或细节不足，但不允许 cell 这条路径直接把条纹整体平移、扭弯或改变方向。换句话说，SC-INR 不是消除所有不确定性，而是把不确定性从“结构位置是否乱动”转移到“该结构在当前尺度下显不显著”。

这也是为什么该图中 SC-INR 看起来更能保持建筑肋条方向：它的尺度变量没有权限直接改写条纹场的几何相位。LIIF/LTE 的 cell path 更自由，尤其 LTE 的 \(h_p(c)\) 直接进入 phase；在 x30 这种未见过的小 footprint 区域，cell 外推一旦产生不合适的 phase shift，就会表现为黑线位置漂移、局部弯曲或与邻近 anchor 的预测不一致。

需要保留边界：SC-INR 并不是凭空恢复 LR 中完全不存在的细节。它能保持，是因为 LR feature 仍然能从建筑立面的低分辨率观测中提取到方向性结构线索，而 SC-INR 的 decoder 先验要求这些线索形成一个更稳定的局部条纹场。如果 LR 输入本身已经把方向混叠成错误模式，SC-INR 也可能稳定地恢复错误结构。

## 4.1 为什么图上 LIIF 和 LTE 反而更像

从方法族上看，LTE 和 SC-INR 都是 Fourier-like decoder，确实比 LIIF 更接近。但“方法族相近”不等于“极端 OOD 下的视觉失败模式相近”。这张 x30 图里，LIIF 和 LTE 更像，主要说明它们落入了相似的失败盆地，而不是说明 LIIF 与 LTE 的核心形式更接近。

本地补充了一个轻量 pairwise 检查：

`artifacts/derived/diagnostics/sc_inr_formula_audit_2026-06-13/urban100_img016_x30_crop21_pairwise_output_similarity.csv`

在裁边后的 Y 通道上：

| Pair | RMSE-Y | Correlation |
| --- | ---: | ---: |
| LIIF vs LTE | 0.0671 | 0.9575 |
| GT vs SC-INR | 0.1133 | 0.9075 |
| GT vs LIIF | 0.3054 | -0.1546 |
| GT vs LTE | 0.2832 | -0.2171 |
| LTE vs SC-INR | 0.2318 | -0.1547 |

这说明视觉观察是成立的：LIIF 和 LTE 的输出确实非常相似。但它们相似的是同一种错误结构，而不是相似地接近 GT。

为什么会这样？有几个原因：

1. 三个模型共享大部分底层条件：同一个 EDSR encoder、同一个 LR 输入、同一个 query/cell、同一个 local ensemble 框架、同一个 x1-x4 训练协议。x30 时，LR 只提供很粗的方向和颜色线索，很多输出由这些共享条件支配。
2. LIIF 和 LTE 在这个 case 中都没有形成稳定的正确条纹场。LIIF 的 MLP cell conditioning 和 LTE 的 learned phase path 虽然形式不同，但在小 footprint OOD 区域都缺少明确边界约束，因此都可能退回到由低频平滑、局部 anchor 融合和错误相位共同产生的相似模糊条纹。
3. LTE 的 Fourier-like 结构并不保证它一定比 LIIF 更接近 SC-INR。LTE 的频率/phase 是自由学习的 decoder latent；当 cell phase 在 x30 外推不稳时，Fourier 表达能力反而可能表达出错误的弯曲或错位纹理。
4. SC-INR 与 LTE 的差异不是“是否有 Fourier basis”，而是“尺度变量有没有权限改写 phase”。在这张图里，这个差异比 Fourier family 相似性更主导视觉结果。

因此，更准确的解释是：

> LTE 和 SC-INR 在结构设计上更接近；但在 x30 selected crop 的视觉结果中，LIIF 和 LTE 更接近，是因为它们共享了同一类 OOD 尺度失败模式。SC-INR 则因为把 cell 从 phase path 中拿出来，使输出偏离了这种失败模式，并更接近 GT 的条纹场。

这也提示论文表述要小心：不能说“LTE 因为 Fourier-like 所以应该天然更像 SC-INR”。Fourier-like 只是表达形式，尺度外推时真正关键的是 cell 如何作用于局部结构。

## 5. 从尺度层面重新解释 SC-INR 公式

SC-INR 不应首先被讲成“对 LTE 的一个改动”。更好的起点是：

> arbitrary-scale SR 中，改变目标尺度不仅改变 query 坐标数量，也改变每个输出像素在连续图像域中的观测 footprint。

一个目标像素不是无限小点。它对应连续图像域中的一个小窗口。模型预测的像素值可以理解为：在当前尺度下，这个小窗口对局部连续信号的观测。

于是变量角色应分开：

| 变量 | 作用 | 图像含义 |
| --- | --- | --- |
| `z` | 定义局部内容 | 附近是边缘、条纹、窗格还是平坦区域 |
| `omega(z)` | 局部方向/频率 | 条纹朝哪个方向、间距多大 |
| `phi(z)` | 局部内容相位 | 条纹在局部坐标系中对齐到哪里 |
| `delta` | 查询位置 | 当前输出像素落在局部模式的什么位置 |
| `c` | 输出像素 footprint | 当前像素以多大窗口观测这个局部模式 |
| `W(omega,c)` | footprint response | 这个窗口还能看见多少该频率分量 |

最终形式：

$$
\hat y
=
g_\theta\left(
A(z)\odot \widetilde{W}(\omega(z),c)
\odot
\left[
\cos\pi(\omega(z)^\top\delta+\phi(z)),
\sin\pi(\omega(z)^\top\delta+\phi(z))
\right]
\right).
$$

这里的核心不是“振幅注入”，而是把任意尺度 SR 写成一个局部观测模型：

> feature 定义局部连续内容，coordinate 决定查询位置，cell 决定当前尺度下的观测算子。

## 6. 振幅在图像中到底是什么意思

这里的 `A(z)` 和 `A(z)W(omega,c)` 不是最终 RGB 图像的严格频谱振幅。更准确地说，它们是 MLP 前 Fourier-like latent feature 的分量强度。

在图像直觉上，可以这样理解：

- `omega` 决定一种局部模式的方向和间距，例如建筑立面中的黑白竖条；
- `phi` 决定这组竖条在局部坐标里的对齐位置；
- `A` 表示这个局部模式在当前 feature 中有多强，也就是它对输出颜色/亮度的潜在贡献；
- `W` 表示当前输出像素 footprint 对这个模式的观测响应；
- `A W` 表示这个模式在当前尺度下还“可见”多少。

因此，对建筑条纹来说，effective amplitude 更接近“条纹对比度/可见性”的中间表示：

- 如果 footprint 覆盖了多个明暗周期，高频条纹会被平均，`W` 变小，条纹对比度应降低；
- 如果 footprint 很小，接近点采样，`W -> 1`，条纹不应因为尺度条件本身而错位；
- 改变尺度可以改变纹理清晰度和可见性，但不应随意改变纹理的相位、方向和几何位置。

这就是 SC-INR 公式中“振幅”的图像含义。

为了避免停留在公式层面，可以把它翻译成更图像化的一句话：

> \(W(\omega,c)\) 控制的不是“这条纹理在哪里”，而是“在当前像素窗口下，这类方向性纹理还应该显得多明显”。

例如同一块建筑立面，在不同输出尺度下应该满足：

- 几何结构应连续：同一根竖肋不应因为输出倍率改变而突然弯曲或横向跳动；
- 观测清晰度可变化：倍率低、footprint 大时，密集条纹被平均，看起来更淡或更糊；
- 倍率高、footprint 小时，模型更接近点采样，已有的方向性结构可以更清楚地展开；
- 不确定时，合理的失败应更像“缺少细节/对比度不足”，而不是“条纹方向被改写”。

因此，SC-INR 的图像层面目标不是单纯增强高频，而是让尺度变化主要影响纹理的可见程度，而不是破坏局部几何组织。对建筑、窗格、栅格这类重复结构，这种区分很重要：人眼首先感知的是线条是否沿着同一个方向连续延伸，其次才是每条线有多锐利。

## 7. 为什么是 sinc，而不是一个随便的函数

考虑局部 Fourier 分量：

$$
f(x,y)=A\cos\pi(\omega_x x+\omega_y y+\phi).
$$

如果输出像素是一个 `c_x by c_y` 的矩形 footprint，那么它观测的是窗口平均：

$$
\frac{1}{c_x c_y}
\int_{-c_x/2}^{c_x/2}
\int_{-c_y/2}^{c_y/2}
f(x_q+u,y_q+v)\,du\,dv.
$$

对 Fourier 分量积分后得到：

$$
A
\operatorname{sinc}\left(\frac{\omega_x c_x}{2}\right)
\operatorname{sinc}\left(\frac{\omega_y c_y}{2}\right)
\cos\pi(\omega_x x_q+\omega_y y_q+\phi).
$$

这说明有限 footprint 的平均观测不会移动相位，而是给该频率分量乘上一个由频率和窗口尺寸共同决定的 response。

所以 sinc 的来源不是 LTE，而是矩形像素 footprint 对 Fourier 分量的解析响应。它提供了两个关键约束：

- `W(omega,0)=1`：小 footprint 极限接近点采样；
- 高频或大 footprint 下 response 变弱：较大像素不应凭空看见被平均掉的细节。

这也是为什么不能只用 area。`area=c_x c_y` 只知道窗口面积，不知道窗口在 x/y 两个方向上的宽度；方向性条纹需要 `omega_x c_x` 和 `omega_y c_y` 分别参与响应。

## 8. 和 LTE 的关系应如何讲

LTE 应放在后面作为参照，而不是作为 SC-INR 故事的起点。

推荐叙事顺序：

1. 任意尺度 SR 的目标像素有 footprint；
2. 尺度变化改变的是观测窗口，而不是图像内容本身；
3. 局部连续内容由 LR feature 和坐标条件化得到；
4. 对 Fourier-like 局部分量，footprint observation 自然对应 sinc response；
5. SC-INR 将 cell 放入 observation response path；
6. 回头看 LTE，它把 cell 放入 learned phase path，因此在尺度外推时缺少上述边界条件。

这样讲，SC-INR 的贡献是：

> 一种面向 arbitrary-scale SR 的局部尺度观测模型，而不是对 LTE 的小修小补。

## 9. 推荐在组会/论文中的表述

短版：

> SC-INR 的核心不是简单修改 LTE 的 cell 注入位置，而是把任意尺度 SR 中的输出像素重新解释为一个 finite-footprint observation。LR feature 定义局部连续内容，relative coordinate 决定查询位置，cell 决定当前尺度下这个像素以多大的窗口观测局部内容。对局部 Fourier-like 分量，矩形 footprint 的平均观测给出 sinc response，因此 cell 应调制该频率分量的可见强度，而不是直接移动局部纹理相位。

更适合 method 的版本：

> In arbitrary-scale SR, changing the target scale changes not only the query coordinates but also the observation footprint of each output pixel. We therefore separate the local content function from the scale-dependent observation. The LR feature predicts the local frequency, coefficient, and content phase; the relative coordinate queries this feature-conditioned local function; the cell specifies the finite footprint under which the function is observed. For a local Fourier component, box-footprint observation yields a sinc response. This motivates modulating the Fourier feature by `W(omega,c)` while keeping the phase independent of the cell.

## 10. 当前证据边界与下一步

这张 x30 图可以保留为一个很强的 selected case，但使用时必须配合候选池统计：

- x30 local pool `n=1560`；
- win external rate 为 `0.4167`；
- median min external delta 为 `-0.0188 dB`；
- q75 min external delta 为 `+0.0657 dB`。

也就是说，x30 下确实存在强优势区域，但不是所有局部 crop 都赢 LIIF/LTE。

建议后续动作：

1. 正式图不要只放 top positive；至少配一个 neutral 或 failure 作为边界，或者在 caption 中明确 selected advantage region。
2. 如果要把“结构方向保持”作为论文说法，需要建立稳定的方向/结构指标，不能只靠单图肉眼观察。
3. 重新修订 method 开头：从 scale/footprint observation 出发，LTE 只作为已有 Fourier implicit decoder 的对照。
4. 修复旧 diagnostics 的 artifact 依赖，避免文档引用已删除结果。

## 11. 汇总陈述

围绕这张图，最稳的陈述不是“LTE 错了，SC-INR 对了”，而是：

> 在 x30 这种远超训练尺度的场景下，输出像素的 footprint 进入了极小尺度区域。此时 LR 输入只能提供非常有限的局部线索，建筑立面这类重复条纹会出现严重的方向和相位歧义。LIIF、LTE、SC-INR 的对比是公平的，因为它们共享同一 LR 输入、query/cell、crop 和训练协议；但这张图是 selected advantage region，只能说明一个强案例。
>
> 图中 LIIF 和 LTE 的输出更像，并不说明它们的方法形式更接近，而是说明它们在这个极端 OOD case 中落入了相似的失败模式。LIIF 将 cell 作为普通 MLP 条件，LTE 将 cell 作为 learned phase term，两者都没有显式约束“尺度变化不能改写局部结构相位”。因此在 x30 小 footprint 外推时，它们都可能产生相似的条纹漂移和结构弯曲。
>
> SC-INR 的不同点不是简单加入一个振幅函数，而是把局部结构和尺度观测分开：feature、relative coordinate、omega 和 phi 决定局部条纹场的方向、间距和位置；cell 只通过 `W(omega,c)` 决定这个条纹场在当前像素 footprint 下有多明显。换句话说，尺度可以改变纹理的可见性和对比度，但不应该直接移动条纹的位置或方向。
>
> 因此，SC-INR 在该图中更好地保持建筑条纹，并不是因为它凭空恢复了所有高频细节，而是因为它减少了 cell path 在训练范围外直接扰动结构相位的自由度。它把不确定性从“结构会不会乱动”收缩到“结构在当前尺度下显不显著”。这正是 SC-INR 可以从尺度/footprint observation 角度讲清楚的核心。

这段陈述的边界是：如果 LR 输入本身已经把方向混叠错了，SC-INR 也可能稳定地恢复错误结构；该 selected case 不能代替全量 benchmark 或候选池级结构指标。
