# LIIF vs 普通 INR 机制图

本目录保存 `LIIF` 类局部隐式表示与典型坐标 `INR` 的机制对比图。

图的核心含义：

- 典型坐标 `INR`：用绝对坐标查询单个连续信号，形式为 $\hat{s}=f_\theta(x)$。
- `LIIF` 类方法：先由 `LR` 图像提取特征图，再用局部特征 `z^*=M(v^*)` 和相对坐标
  `\delta=x_q-v^*` 查询共享 decoder，形式为
  $\hat{s}=f_\theta(z^*, x_q-v^* [,c])$。

使用边界：

- 该图只说明 `LIIF` 类方法相对普通坐标 `INR` 的变量组织差异。
- 该图不涉及 `LTE`/`SC-INR` 的 Fourier、phase 或 sinc response。
- 不能用该图宣称普通 `INR` 不能超分、`LIIF` 不使用 cell，或 `LIIF` 具备严格尺度一致性。

生成命令：

```bash
python scripts/viz/draw_liif_vs_inr_mechanism.py
```

输出：

- `liif_vs_inr_mechanism.png`
- `liif_vs_inr_mechanism.pdf`
- `liif_vs_inr_mechanism.svg`
