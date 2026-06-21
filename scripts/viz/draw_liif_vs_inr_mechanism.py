#!/usr/bin/env python3
"""绘制普通坐标 INR 与 LIIF 类局部隐式表示的核心差异图。

图只表达两类表示的变量组织差异：

- 典型坐标 INR：用绝对坐标查询单个连续信号，形式为 I_theta(x)=f_theta(x)。
- LIIF 类方法：先从 LR 图像提取特征图，再用局部特征 z* 和相对坐标 delta
  查询共享 decoder，形式为 s=f_theta(z*, x_q-v* [, c])。

该图不讨论 LTE/SC-INR 的 Fourier、phase 或 sinc response，避免把后续 decoder
机制和 LIIF 的基础局部条件化思想混在一起。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]


COLORS = {
    "bg": "#F7F8FA",
    "panel": "#FFFFFF",
    "text": "#20242A",
    "muted": "#5D6673",
    "line": "#8A94A6",
    "neutral": "#2F3A4A",
    "inr": "#6E5AA8",
    "inr_soft": "#F0ECFA",
    "liif": "#167A73",
    "liif_soft": "#E7F4F2",
    "feature": "#DCE8F7",
    "feature_edge": "#4D7EA8",
    "anchor": "#E08A36",
    "query": "#C94848",
}


def add_panel(ax, title: str, subtitle: str, color: str, soft: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel = FancyBboxPatch(
        (0.012, 0.012),
        0.976,
        0.976,
        boxstyle="round,pad=0.016,rounding_size=0.035",
        linewidth=1.1,
        edgecolor="#D9DEE8",
        facecolor=COLORS["panel"],
        zorder=0,
    )
    ax.add_patch(panel)
    header = FancyBboxPatch(
        (0.055, 0.835),
        0.89,
        0.115,
        boxstyle="round,pad=0.014,rounding_size=0.032",
        linewidth=1.2,
        edgecolor=color,
        facecolor=soft,
        zorder=1,
    )
    ax.add_patch(header)
    ax.text(
        0.5,
        0.905,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=color,
        zorder=2,
    )
    ax.text(
        0.5,
        0.865,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
        zorder=2,
    )


def add_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str,
    ec: str = "#ADB5C2",
    color: str = COLORS["text"],
    fontsize: float = 11,
    weight: str = "normal",
    radius: float = 0.03,
    lw: float = 1.2,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=1.22,
        zorder=3,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    lw: float = 1.5,
    rad: float = 0.0,
    style: str = "-|>",
    zorder: int = 1,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_inr_signal(ax, x0: float, y0: float, w: float, h: float) -> None:
    """画一个连续信号坐标域，用来表示普通 INR 查询绝对坐标。"""
    rect = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#C6CBD6",
        facecolor="#FBFBFD",
        zorder=1,
    )
    ax.add_patch(rect)

    xs = np.linspace(x0 + 0.02, x0 + w - 0.02, 80)
    for k, y in enumerate(np.linspace(y0 + 0.035, y0 + h - 0.035, 5)):
        vals = y + 0.012 * np.sin(np.linspace(0, 2.2 * np.pi, xs.size) + k * 0.8)
        ax.plot(xs, vals, color="#B8B0D8", lw=1.1, zorder=2)
    for k, x in enumerate(np.linspace(x0 + 0.04, x0 + w - 0.04, 5)):
        ys = np.linspace(y0 + 0.025, y0 + h - 0.025, 80)
        vals = x + 0.010 * np.sin(np.linspace(0, 2.0 * np.pi, ys.size) + k * 0.5)
        ax.plot(vals, ys, color="#D7D2EB", lw=1.0, zorder=2)

    q = (x0 + 0.67 * w, y0 + 0.58 * h)
    ax.add_patch(Circle(q, 0.018, facecolor=COLORS["query"], edgecolor="white", linewidth=1.0, zorder=4))
    ax.text(q[0] + 0.025, q[1] + 0.018, "$x$", fontsize=12, color=COLORS["query"], fontweight="bold")

    ax.text(
        x0 + w / 2,
        y0 - 0.035,
        "absolute coordinate in one signal",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
    )


def draw_feature_grid(ax, x0: float, y0: float, w: float, h: float) -> tuple[float, float]:
    """画一个 LR feature map，并返回 anchor 中心坐标。"""
    cols, rows = 5, 4
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            shade = 0.88 - 0.025 * ((r + c) % 3)
            color = (shade * 0.86, shade * 0.93, min(1.0, shade + 0.08))
            ax.add_patch(
                Rectangle(
                    (x0 + c * cw, y0 + r * ch),
                    cw,
                    ch,
                    linewidth=0.8,
                    edgecolor=COLORS["feature_edge"],
                    facecolor=color,
                    zorder=1,
                )
            )

    anchor_c, anchor_r = 2, 1
    ax.add_patch(
        Rectangle(
            (x0 + anchor_c * cw, y0 + anchor_r * ch),
            cw,
            ch,
            linewidth=2.3,
            edgecolor=COLORS["anchor"],
            facecolor="none",
            zorder=3,
        )
    )
    anchor = (x0 + (anchor_c + 0.5) * cw, y0 + (anchor_r + 0.5) * ch)
    query = (anchor[0] + 0.085, anchor[1] + 0.055)
    ax.add_patch(Circle(anchor, 0.015, facecolor=COLORS["anchor"], edgecolor="white", linewidth=1.0, zorder=4))
    ax.add_patch(Circle(query, 0.015, facecolor=COLORS["query"], edgecolor="white", linewidth=1.0, zorder=4))
    add_arrow(ax, anchor, query, color=COLORS["neutral"], lw=1.2, style="->", zorder=4)
    ax.text(anchor[0] - 0.056, anchor[1] - 0.045, "$v^*$", fontsize=11.5, color=COLORS["anchor"], fontweight="bold")
    ax.text(query[0] + 0.015, query[1] + 0.012, "$x_q$", fontsize=11.5, color=COLORS["query"], fontweight="bold")
    ax.text(
        (anchor[0] + query[0]) / 2 + 0.005,
        (anchor[1] + query[1]) / 2 + 0.028,
        "$\\delta=x_q-v^*$",
        fontsize=10.5,
        color=COLORS["neutral"],
    )
    ax.text(
        x0 + w / 2,
        y0 - 0.035,
        "local anchor coordinate system",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
    )
    return anchor


def draw_panel_inr(ax) -> None:
    add_panel(
        ax,
        "Typical coordinate INR",
        "query a global coordinate function for one signal",
        COLORS["inr"],
        COLORS["inr_soft"],
    )

    draw_inr_signal(ax, 0.10, 0.56, 0.28, 0.19)
    add_box(
        ax,
        (0.09, 0.38),
        (0.30, 0.105),
        "coordinate\n$x=(u,v)$",
        fc=COLORS["inr_soft"],
        ec=COLORS["inr"],
        color=COLORS["inr"],
        fontsize=12,
        weight="bold",
    )
    add_box(
        ax,
        (0.49, 0.47),
        (0.22, 0.16),
        "MLP\n$f_\\theta$",
        fc="#FFFFFF",
        ec=COLORS["neutral"],
        fontsize=14,
        weight="bold",
    )
    add_box(
        ax,
        (0.79, 0.47),
        (0.13, 0.16),
        "value\n$I_\\theta(x)$",
        fc="#FFFFFF",
        ec=COLORS["inr"],
        color=COLORS["inr"],
        fontsize=12,
        weight="bold",
    )
    add_arrow(ax, (0.39, 0.432), (0.49, 0.52), color=COLORS["inr"], lw=2.0)
    add_arrow(ax, (0.71, 0.55), (0.79, 0.55), color=COLORS["inr"], lw=2.0)

    add_box(
        ax,
        (0.18, 0.17),
        (0.64, 0.105),
        "$\\hat{s}=f_\\theta(x)$",
        fc=COLORS["inr_soft"],
        ec=COLORS["inr"],
        color=COLORS["inr"],
        fontsize=16,
        weight="bold",
    )
    ax.text(
        0.50,
        0.105,
        "The coordinate itself identifies where to sample the continuous signal.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )


def draw_panel_liif(ax) -> None:
    add_panel(
        ax,
        "LIIF-style local implicit decoding",
        "query a shared local decoder conditioned on LR features",
        COLORS["liif"],
        COLORS["liif_soft"],
    )

    add_box(
        ax,
        (0.07, 0.62),
        (0.17, 0.12),
        "LR image\n$I_{LR}$",
        fc="#FFFFFF",
        ec=COLORS["neutral"],
        fontsize=12,
    )
    add_box(
        ax,
        (0.31, 0.62),
        (0.18, 0.12),
        "encoder\n$E$",
        fc="#FFFFFF",
        ec=COLORS["neutral"],
        fontsize=12,
        weight="bold",
    )
    add_arrow(ax, (0.24, 0.68), (0.31, 0.68), color=COLORS["neutral"], lw=1.7)
    add_arrow(ax, (0.49, 0.68), (0.57, 0.68), color=COLORS["neutral"], lw=1.7)

    anchor = draw_feature_grid(ax, 0.57, 0.58, 0.34, 0.17)
    ax.text(0.74, 0.79, "feature map $M=E(I_{LR})$", ha="center", fontsize=10.5, color=COLORS["muted"])

    add_box(
        ax,
        (0.09, 0.38),
        (0.19, 0.10),
        "local feature\n$z^*=M(v^*)$",
        fc=COLORS["feature"],
        ec=COLORS["feature_edge"],
        fontsize=10.8,
        weight="bold",
    )
    add_box(
        ax,
        (0.36, 0.38),
        (0.20, 0.10),
        "relative coord\n$\\delta=x_q-v^*$",
        fc=COLORS["liif_soft"],
        ec=COLORS["liif"],
        color=COLORS["liif"],
        fontsize=10.8,
        weight="bold",
    )
    add_box(
        ax,
        (0.64, 0.35),
        (0.22, 0.15),
        "shared MLP\n$f_\\theta$",
        fc="#FFFFFF",
        ec=COLORS["neutral"],
        fontsize=13,
        weight="bold",
    )
    add_box(
        ax,
        (0.69, 0.17),
        (0.13, 0.09),
        "RGB\n$\\hat{s}$",
        fc="#FFFFFF",
        ec=COLORS["liif"],
        color=COLORS["liif"],
        fontsize=12,
        weight="bold",
    )
    add_arrow(ax, (anchor[0], anchor[1] - 0.02), (0.19, 0.48), color=COLORS["feature_edge"], lw=1.8, rad=0.25)
    add_arrow(ax, (0.56, 0.43), (0.64, 0.43), color=COLORS["liif"], lw=2.0)
    add_arrow(ax, (0.28, 0.43), (0.64, 0.43), color=COLORS["feature_edge"], lw=2.0, rad=-0.08)
    add_arrow(ax, (0.75, 0.35), (0.75, 0.26), color=COLORS["liif"], lw=2.0)

    add_box(
        ax,
        (0.15, 0.055),
        (0.70, 0.105),
        "$\\hat{s}=f_\\theta(z^*,\\,x_q-v^*\\,[,c])$",
        fc=COLORS["liif_soft"],
        ec=COLORS["liif"],
        color=COLORS["liif"],
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.50,
        0.305,
        "The same decoder is reused across images and positions; local features define the local signal.",
        ha="center",
        va="center",
        fontsize=10.3,
        color=COLORS["muted"],
    )


def draw(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.8, 8.3), facecolor=COLORS["bg"])
    gs = fig.add_gridspec(1, 2, left=0.035, right=0.965, top=0.88, bottom=0.065, wspace=0.055)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

    fig.text(
        0.5,
        0.955,
        "Global coordinate INR vs local feature-conditioned LIIF decoding",
        ha="center",
        va="center",
        fontsize=18,
        color=COLORS["text"],
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.915,
        "LIIF changes the query from an absolute coordinate-only function to a local, image-conditioned shared decoder.",
        ha="center",
        va="center",
        fontsize=12.3,
        color=COLORS["muted"],
    )

    draw_panel_inr(axes[0])
    draw_panel_liif(axes[1])

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"liif_vs_inr_mechanism.{ext}", dpi=240)
    plt.close(fig)

    readme = """# LIIF vs 普通 INR 机制图

本目录保存 `LIIF` 类局部隐式表示与典型坐标 `INR` 的机制对比图。

图的核心含义：

- 典型坐标 `INR`：用绝对坐标查询单个连续信号，形式为 $\\hat{s}=f_\\theta(x)$。
- `LIIF` 类方法：先由 `LR` 图像提取特征图，再用局部特征 `z^*=M(v^*)` 和相对坐标
  `\\delta=x_q-v^*` 查询共享 decoder，形式为
  $\\hat{s}=f_\\theta(z^*, x_q-v^* [,c])$。

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
"""
    (out_dir / "README_zh.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw normal coordinate INR vs LIIF-style mechanism figure.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "derived" / "paper_figures" / "liif_vs_inr_mechanism_2026-05-24",
    )
    args = parser.parse_args()
    draw(args.out)
    print(f"wrote LIIF vs INR mechanism figure to {args.out}")


if __name__ == "__main__":
    main()
