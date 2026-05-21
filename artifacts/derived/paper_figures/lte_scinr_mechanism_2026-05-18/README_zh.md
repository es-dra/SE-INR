# LTE vs SC-INR 机制对比图

本目录保存论文/周报可用的机制对比图，突出 cell 在两种 decoder 中的不同角色：

- LTE：`cell -> learned phase h_p(c)`，cell 直接改变 Fourier phase。
- SC-INR：`cell + omega -> analytic response W(omega,c)`，cell 调制 MLP 前 Fourier feature 的 effective amplitude。

图是概念机制图，不展示 benchmark 数值，也不表示最终 RGB 是 exact box integral。
论文正文或周报引用时，应配合 `artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/`
中的 seed1 机制诊断结果，避免把结构图本身当作实验结论。

生成命令：

```bash
python scripts/viz/draw_lte_scinr_mechanism.py
```

输出：

- `lte_scinr_mechanism_comparison.png`
- `lte_scinr_mechanism_comparison.pdf`
- `lte_scinr_mechanism_comparison.svg`
