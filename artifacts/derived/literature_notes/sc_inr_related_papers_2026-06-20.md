# SC-INR 相关文献阅读笔记 2026-06-20

本文档只记录两篇论文中对 SC-INR 周报和 method 叙事有用的部分，不作为完整文献综述。

## 1. PR-INR: From Coarse to Continuous

论文对象是 motion-robust anisotropic MRI reconstruction，不是图像超分。它的价值不在于可以直接作为 SC-INR 的 baseline，而在于它提供了一个比较清楚的 INR 建模方式：把一个复杂重建问题拆成几个角色明确的子问题。

PR-INR 把 MRI 重建拆成三步：

- MAD：先做 motion-aware diffusion，得到全局结构较稳定的粗重建。
- IDR：再用 implicit detail restoration 做残差细节修复，重点处理高频细节和边界。
- VCR：最后用 3D continuous representation 处理 anisotropic slice 下的体数据连续性。

对 SC-INR 有用的 insight：

- 它没有把 INR 写成“一个连续函数就能解决所有问题”，而是明确规定每个模块能改什么、不能乱改什么。
- diffusion 负责粗结构，但可能 hallucinate；IDR 只在残差/高频区域修正，避免让强模型自由改写可靠结构。
- 这能支持 SC-INR 的叙事：`cell` 不应该成为可以随意改变纹理位置和结构相位的条件，而应该被限制为输出像素 footprint 的观测响应。

可用于周报的一句话：

> 这篇论文给我的启发是，INR 的关键不只是连续表示能力，而是要把连续表示放在一个角色清楚的重建流程里：哪些变量负责结构，哪些变量负责细节，哪些变量只负责观测约束。

边界：

- 医学 MRI 有 k-space/data consistency，SC-INR 是自然图像 arbitrary-scale SR，两者任务不同。
- 只能借鉴“角色拆分”和“限制隐式模块自由度”的建模思路，不能说 PR-INR 直接证明 SC-INR 的 sinc response 有效。

## 2. INCODE: Implicit Neural Conditioning With Prior Knowledge Embeddings

论文对象是 conditional INR。它把 SIREN 的正弦激活扩展为可由先验 embedding 条件化控制的形式，核心是由 harmonizer network 预测正弦激活参数。

它对几个参数给出了很直观的图像解释：

- amplitude：控制响应强度，影响特征增强或噪声抑制。
- frequency scaling：控制更细或更粗的细节粒度。
- phase shift：横向移动正弦波，影响特征对齐和空间排列。
- vertical shift：改变整体基线，类似亮度偏移。

对 SC-INR 有用的 insight：

- 这篇论文可以帮助解释“相位”和“振幅/响应”在图像层面的区别。
- phase 更接近结构位置和空间对齐；如果让尺度条件直接改 phase，就等于允许输出尺度移动局部纹理结构。
- amplitude/response 更接近某种模式在当前观测下显得多强、多清楚。
- 因此，SC-INR 不应被讲成“在 LTE 上改了振幅”，而应讲成：把 cell 从结构相位路径中拿出来，只让它控制 footprint 下的可见强度。

可用于周报的一句话：

> INCODE 对正弦参数的解释提醒我，phase 对应的是空间对齐，amplitude/response 对应的是模式强弱；因此 SC-INR 中最关键的不是多加一个 sinc，而是避免让 cell 直接移动纹理位置。

边界：

- INCODE 本身允许 prior-conditioned phase，因为它的条件来自任务先验或图像 embedding。
- SC-INR 不是反对 phase，而是反对由 `cell` 这种输出窗口大小去控制 phase。phase 应由图像内容特征决定，cell 应只描述观测窗口。

## 3. 对本周周报的组织建议

本周周报可以不写长公式，重点写两件事：

1. 从 PR-INR 得到的启发：INR 不是把所有信息都塞进一个 MLP，而是要区分变量角色，限制每条条件路径能改变的图像属性。
2. 从 INCODE 得到的启发：正弦表示里的 phase、frequency、amplitude 在图像上分别对应空间位置/对齐、纹理粒度、可见强度；这可以把 SC-INR 的解释从公式层落到图像层。

对应到 SC-INR，可以这样落地：

- `z` 和 `delta` 决定局部结构，比如条纹方向、间距和位置。
- `cell` 表示输出像素在连续坐标域里的窗口大小。
- 大窗口看到的是窗口内平均后的结果，所以它应该降低高频纹理的可见强度，而不是移动条纹位置。
- 因此，SC-INR 的核心表述应是“结构由内容特征决定，尺度只改变该结构在当前像素 footprint 下的观测强度”。

## 4. 不能过度宣称的点

- 两篇论文不能作为 SC-INR 性能优势的直接证据，只能作为 method 叙事和建模动机的支撑。
- 周报里不要写“这些论文证明了 SC-INR 的设计正确”，更稳的说法是“这些论文帮助我把 SC-INR 的变量角色讲清楚”。
- 不要继续把 SC-INR 说成“修改 LTE 的振幅注入方式”。导师已经指出这个说法太像小修小补，应改成“从输出像素 footprint 出发，对尺度条件的作用范围进行约束”。
