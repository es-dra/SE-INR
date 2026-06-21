# INCODE 启发下的 SC-INR method 叙事与机制实验结果 2026-06-20

## 1. 本轮回答的问题

本轮不是做新的 benchmark，也不是把 INCODE 加成 baseline。问题是：

> INCODE 对正弦参数的解释，能否帮助我们把 SC-INR 的 method 从“改 LTE 振幅”提升到“条件路径分工”的层面；如果能，现有 SC-INR 是否有对应的机制证据。

本轮新增三类无训练诊断：

- `cell_phase_proxy`：检查是否存在显式 `cell -> phase` 路径。
- `effective_amplitude`：检查 SC-INR 的 analytic response 是否实际调制 decoder 输入强度。
- `cell_intervention`：固定 LR 输入和 query，只改变 cell，观察图像输出变化是否接近 HR box-average oracle。

输出目录：

`artifacts/derived/diagnostics/incode_method_support_2026-06-20/`

## 2. INCODE 对 method 的真正价值

INCODE 的重点不是 arbitrary-scale SR，而是 conditional INR。它把正弦激活写成：

$$
y=a\sin(b\omega_0 x+c)+d.
$$

它对参数的解释很适合支撑 SC-INR 的方法叙事：

- `frequency scaling` 控制细节粒度；
- `phase shift` 控制空间对齐；
- `amplitude` 控制响应强度；
- `offset` 控制基线。

这给 SC-INR 一个更清楚的切入点：

> 在 Fourier/SIREN-like decoder 中，条件变量进入哪条路径，决定它能改变图像的哪类属性。

因此，SC-INR 不应被讲成“把 LTE 的振幅换成 sinc”。更准确的说法是：

> SC-INR 把结构相位和尺度观测分开。局部结构由 feature-conditioned frequency/phase 和 query coordinate 决定；cell 只表示输出像素 footprint，因此只应进入 observation response，而不应直接移动 phase。

## 3. 代码机制证据：LTE 有 cell-conditioned phase，SC-INR 没有

LTE 的实现中：

```python
q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
```

也就是 `cell` 经过一个 linear layer 后直接加到 Fourier-like phase 上。

SC-INR 的实现中：

- `omega` 来自 feature；
- 可选 `phi` 来自 feature；
- `cell` 只进入：

$$
W(\omega,c)=
\operatorname{sinc}(\omega_xc_x/2)\operatorname{sinc}(\omega_yc_y/2).
$$

这与 INCODE 的参数语义正好对应：

- feature-conditioned phase 可以移动结构，因为 feature 有内容信息；
- cell-conditioned phase 不稳，因为 cell 只是窗口大小，不包含内容。

## 4. 实验 1：cell -> phase proxy

实验路径：

`artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_phase_proxy/`

设置：

- 模型：LTE、LTE-PhaseZ、SC-INR、SC-INR-NoSinc。
- 参考尺度：x4。
- 观察尺度：x2、x3、x4、x6、x8、x12、x16、x24、x30。
- 只计算内部 phase path，不做图像推理。

关键结果：

| 模型 | x30 相对 x4 的 mean phase delta | 是否有 cell phase |
| --- | ---: | ---: |
| LTE | 0.1747 | 有 |
| LTE-PhaseZ | 0 | 无 |
| SC-INR | 0 | 无 |
| SC-INR-NoSinc | 0 | 无 |

cell multiplier 从 1 变到 4 时：

| 模型 | mean phase delta | max phase delta |
| --- | ---: | ---: |
| LTE | 0.6049 | 2.0050 |
| LTE-PhaseZ | 0 | 0 |
| SC-INR | 0 | 0 |
| SC-INR-NoSinc | 0 | 0 |

解释：

- 这个实验直接证明 LTE 的 `cell` 具有移动 Fourier-like phase 的机制能力。
- 它不证明 LTE 一定出错，但说明 LTE 的尺度条件确实能进入空间对齐路径。
- SC-INR 和 LTE-PhaseZ 没有这条路径，说明它们把结构相位交给 feature，而不是 cell。

## 5. 实验 2：effective amplitude / response

实验路径：

`artifacts/derived/diagnostics/incode_method_support_2026-06-20/effective_amplitude/`

设置：

- 模型：SC-INR、SC-INR-NoSinc。
- 数据：BSD100/Urban100 各 3 张。
- 尺度：x2、x4、x8、x16、x30。
- 指标：按低/中/高频分组统计 `active_response_abs_mean` 和 `effective_energy_ratio`。

高频组关键结果：

| 模型 | x2 | x4 | x8 | x16 | x30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SC-INR active \|W\| | 0.6356 | 0.8912 | 0.9715 | 0.9928 | 0.9979 |
| SC-INR energy ratio | 0.3739 | 0.7695 | 0.9357 | 0.9835 | 0.9953 |
| SC-INR-NoSinc active \|W\| | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

解释：

- SC-INR 的 response 随 footprint 变化：低输出尺度/大 footprint 下，高频响应被明显压低；高输出尺度/小 footprint 下，response 接近 1。
- SC-INR-NoSinc 的 active response 恒为 1，说明它没有 footprint response。
- 这支持 SC-INR 的 method 语义：cell 不是移动结构，而是在 decoder 输入层控制局部频率分量的观测强度。

边界：

- 这里的 `effective amplitude` 是 MLP 前 latent feature 的 response proxy，不是最终 RGB 的严格频谱振幅。

## 6. 实验 3：cell-only intervention 图像诊断

实验路径：

`artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_intervention/`

设置：

- 固定 x4 LR 输入；
- 固定 query grid；
- 只把 cell multiplier 从 1 改成 4；
- 模型：LIIF、LTE、LTE-PhaseZ、SC-INR、SC-INR-NoSinc；
- HR box-average oracle 作为参考变化；
- 候选来自 Urban100/BSD100 小样本高纹理 crop。

top selected case：

`urban100/img_002.png, crop x=192, y=192, size=96`

| 模型 | tracking RMSE ↓ | delta-oracle corr ↑ | cell sensitivity |
| --- | ---: | ---: | ---: |
| LIIF | 0.0440 | 0.7849 | 0.0526 |
| LTE | 0.0455 | 0.7708 | 0.0602 |
| LTE-PhaseZ | 0.0709 | NaN | 0 |
| SC-INR | 0.0423 | 0.8045 | 0.0605 |
| SC-INR-NoSinc | 0.0709 | NaN | 0 |

小样本 18 个候选的均值：

| 模型 | tracking RMSE ↓ | oracle RMSE ↓ | cell sensitivity |
| --- | ---: | ---: | ---: |
| LIIF | 0.0703 | 0.0243 | 0.0379 |
| LTE | 0.0723 | 0.0299 | 0.0440 |
| LTE-PhaseZ | 0.0793 | 0.0361 | 0 |
| SC-INR | 0.0713 | 0.0279 | 0.0409 |
| SC-INR-NoSinc | 0.0793 | 0.0368 | 0 |

解释：

- SC-INR 对 cell 有非零响应，NoSinc 和 LTE-PhaseZ 基本无响应。
- 在 selected examples 中，SC-INR 的 cell-induced change 比 LTE/NoSinc 更接近 HR box-average oracle。
- 但这个小样本里 SC-INR 没有稳定超过 LIIF；因此不能把该实验写成“SC-INR 全面优于所有 baseline”。

最稳的图像层解释是：

> 这组实验显示，SC-INR 的 cell path 确实在改变图像观测，而不是完全失效；同时 NoSinc 和 LTE-PhaseZ 说明，如果去掉 footprint response 或去掉 cell 路径，输出几乎不随 cell 改变。LTE 则有 cell-conditioned phase，说明它的尺度条件可以进入空间对齐路径。SC-INR 的差异在于：它让 cell 改变纹理可见性，而不是直接改变纹理相位。

## 7. 对周报和 method 的推荐写法

可以写：

> 本周结合 INCODE 重新梳理了 SC-INR 的 method 叙事。INCODE 对正弦参数的解释说明，phase shift 对应空间对齐，amplitude/response 对应响应强度。放到 arbitrary-scale SR 中，cell 只是输出像素 footprint，不是图像内容。因此，cell 不应直接进入 phase path，否则尺度条件就有能力移动局部纹理位置。SC-INR 的设计是把 phase 留给 feature-conditioned content，把 cell 放到 analytic footprint response 中。

再接机制实验：

> 为了验证这个解释，我做了三组无训练诊断。第一，LTE 的 cell-phase proxy 显示，x30 相对 x4 会产生明显 phase 变化，而 LTE-PhaseZ/SC-INR/NoSinc 没有 cell-conditioned phase。第二，SC-INR 的 effective response 显示，高频分量在大 footprint 下被压低，在小 footprint 下趋近 1，而 NoSinc 恒为 1。第三，cell-only intervention 显示，固定 LR 和 query、只改变 cell 时，SC-INR 有非零响应，并在 selected examples 上比 LTE/NoSinc 更接近 HR box-average oracle。

必须加边界：

> 这些实验是机制诊断，不是新的 benchmark。它们支持 SC-INR 的 cell path 语义，但不能单独证明 SC-INR 全局视觉质量更好，也不能说明最终 RGB 是严格 box integral。

## 8. 专家审查

可靠结论：

- INCODE 可以作为 SC-INR method 的参数语义支撑。
- LTE 存在显式 `cell -> phase` 路径；SC-INR 没有。
- SC-INR 的 sinc response 在 decoder 输入层按 footprint 调制有效强度；NoSinc 无该响应。
- cell-intervention 小样本支持 SC-INR 的 cell path 更接近 footprint observation，而不是完全无响应。

证据不足：

- cell-intervention 是小样本 selected diagnostic，不是全量结论。
- SC-INR 没有在该小实验中稳定超过 LIIF，不能写成全面视觉优势。
- 仍不能把 `A(z)W(omega,c)` 说成最终 RGB 的严格物理振幅。

下一步建议：

- 周报中使用 top selected figure 和 phase/response 两张曲线即可，不需要继续扩展成大实验。
- 如果进入论文，需要把 cell-intervention 扩展成更规范的候选池或降级为 mechanism figure。
