#!/usr/bin/env python3
"""绘制 SC-INR analytic footprint response 的机制说明图。

这张图服务周报/论文方法解释，重点回答三个问题：

- 为什么需要 response：cell 表示输出像素 footprint，而不是图像内容。
- 为什么是 sinc：Fourier 分量在 box footprint 上平均会产生 sinc response。
- 和 LTE 的区别：LTE 让 cell 改变 phase；SC-INR 让 cell 调制可观测强度。

该图是概念图，不展示实验数值，也不表示最终 RGB 是 exact box integral。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "derived" / "paper_figures" / "footprint_response_mechanism_2026-05-31"


COLORS = {
    "bg": "#F7F8FA",
    "panel": "#FFFFFF",
    "text": "#20242A",
    "muted": "#5D6673",
    "line": "#8A94A6",
    "lte": "#D9793D",
    "lte_soft": "#FFF1E8",
    "sc": "#187C83",
    "sc_soft": "#E7F4F3",
    "blue": "#4C72B0",
    "blue_soft": "#EAF0FA",
    "signal": "#2F3A4A",
    "query": "#C94848",
    "footprint": "#6E5AA8",
    "response": "#167A73",
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
        fontsize=14.5,
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
        fontsize=9.7,
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
    fontsize: float = 10.5,
    weight: str = "normal",
    radius: float = 0.026,
    lw: float = 1.15,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
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
        linespacing=1.2,
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
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_lte_panel(ax) -> None:
    add_panel(
        ax,
        "LTE: cell as phase control",
        "scale can move local texture phase",
        COLORS["lte"],
        COLORS["lte_soft"],
    )

    add_box(ax, (0.08, 0.68), (0.24, 0.075), "feature\n$z$", fc="#ECEFF4", fontsize=11)
    add_box(ax, (0.39, 0.70), (0.22, 0.065), "$\\omega(z), A(z)$", fc="#FFFFFF", fontsize=10.5)
    add_box(
        ax,
        (0.08, 0.49),
        (0.24, 0.075),
        "output cell\n$c$",
        fc=COLORS["lte_soft"],
        ec=COLORS["lte"],
        color=COLORS["lte"],
        fontsize=11,
        weight="bold",
    )
    add_box(
        ax,
        (0.39, 0.50),
        (0.22, 0.065),
        "learned\n$h_p(c)$",
        fc=COLORS["lte_soft"],
        ec=COLORS["lte"],
        color=COLORS["lte"],
        fontsize=10.5,
        weight="bold",
    )
    add_box(
        ax,
        (0.67, 0.58),
        (0.25, 0.14),
        "$\\theta_{LTE}$\n$=\\omega(z)^T\\delta+h_p(c)$",
        fc="#FFFFFF",
        ec=COLORS["lte"],
        color=COLORS["text"],
        fontsize=10.2,
    )
    add_arrow(ax, (0.32, 0.715), (0.39, 0.733))
    add_arrow(ax, (0.32, 0.527), (0.39, 0.532), color=COLORS["lte"], lw=2.0)
    add_arrow(ax, (0.61, 0.532), (0.67, 0.625), color=COLORS["lte"], lw=2.0)
    add_arrow(ax, (0.61, 0.733), (0.67, 0.665))

    xs = np.linspace(0.08, 0.92, 220)
    base = 0.30
    amp = 0.07
    phase_a = 0.0
    phase_b = 0.85
    ya = base + amp * np.sin(2 * np.pi * 3.1 * (xs - 0.08) + phase_a)
    yb = base - 0.10 + amp * np.sin(2 * np.pi * 3.1 * (xs - 0.08) + phase_b)
    ax.plot(xs, ya, color=COLORS["signal"], lw=2.0)
    ax.plot(xs, yb, color=COLORS["lte"], lw=2.0)
    ax.text(0.10, 0.375, "same local feature", fontsize=9.5, color=COLORS["signal"])
    ax.text(0.10, 0.175, "different cell shifts phase", fontsize=9.5, color=COLORS["lte"], fontweight="bold")
    ax.text(
        0.50,
        0.075,
        "Insight: cell can become a texture-position shortcut.",
        ha="center",
        fontsize=10.5,
        color=COLORS["lte"],
        fontweight="bold",
    )


def draw_box_average_panel(ax) -> None:
    add_panel(
        ax,
        "Why sinc?",
        "a pixel observes a footprint, not only a point",
        COLORS["blue"],
        COLORS["blue_soft"],
    )

    xs = np.linspace(0.07, 0.93, 400)
    y = 0.53 + 0.13 * np.sin(2 * np.pi * 4.4 * (xs - 0.07) + 0.25)
    ax.plot(xs, y, color=COLORS["signal"], lw=2.2)
    ax.axhline(0.53, color="#C7CEDA", lw=1.0, zorder=0)

    center = 0.50
    small_w = 0.11
    large_w = 0.31
    ax.add_patch(
        Rectangle(
            (center - small_w / 2, 0.38),
            small_w,
            0.30,
            facecolor="#C6B8E8",
            edgecolor=COLORS["footprint"],
            alpha=0.34,
            linewidth=1.4,
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (center - large_w / 2, 0.36),
            large_w,
            0.34,
            facecolor="#D8D0F0",
            edgecolor=COLORS["footprint"],
            alpha=0.20,
            linewidth=1.4,
            zorder=1,
        )
    )
    ax.plot([center, center], [0.34, 0.73], color=COLORS["query"], lw=1.5, ls="--")
    ax.text(center + 0.015, 0.715, "$x_0$", fontsize=11, color=COLORS["query"], fontweight="bold")
    ax.text(0.26, 0.335, "small $c$", fontsize=10, color=COLORS["footprint"], fontweight="bold")
    ax.text(0.60, 0.335, "large $c$", fontsize=10, color=COLORS["footprint"], fontweight="bold")

    add_box(
        ax,
        (0.10, 0.14),
        (0.80, 0.105),
        "$\\frac{1}{c}\\int_{x_0-c/2}^{x_0+c/2} A\\cos(\\pi\\omega x+\\phi)dx$\n"
        "$= A\\cos(\\pi\\omega x_0+\\phi)\\,\\mathrm{sinc}(\\omega c/2)$",
        fc="#FFFFFF",
        ec=COLORS["blue"],
        color=COLORS["text"],
        fontsize=9.6,
        radius=0.020,
    )
    ax.text(
        0.50,
        0.075,
        "Insight: footprint averaging has a frequency response.",
        ha="center",
        fontsize=10.5,
        color=COLORS["blue"],
        fontweight="bold",
    )


def draw_sc_panel(ax) -> None:
    add_panel(
        ax,
        "SC-INR: cell as observation response",
        "scale changes visibility, not phase",
        COLORS["sc"],
        COLORS["sc_soft"],
    )

    add_box(ax, (0.07, 0.68), (0.23, 0.075), "feature\n$z$", fc="#ECEFF4", fontsize=11)
    add_box(ax, (0.37, 0.71), (0.21, 0.060), "$A(z)$", fc="#FFFFFF", fontsize=10.5)
    add_box(ax, (0.37, 0.62), (0.21, 0.060), "$\\omega(z),\\phi(z)$", fc="#FFFFFF", fontsize=10.5)
    add_box(
        ax,
        (0.07, 0.48),
        (0.23, 0.075),
        "output cell\n$c$",
        fc=COLORS["sc_soft"],
        ec=COLORS["sc"],
        color=COLORS["sc"],
        fontsize=11,
        weight="bold",
    )
    add_box(
        ax,
        (0.63, 0.62),
        (0.30, 0.115),
        "$W(\\omega,c)$\n$=\\mathrm{sinc}(\\omega_xc_x/2)\\mathrm{sinc}(\\omega_yc_y/2)$",
        fc=COLORS["sc_soft"],
        ec=COLORS["sc"],
        color=COLORS["sc"],
        fontsize=9.4,
        weight="bold",
    )
    add_arrow(ax, (0.30, 0.715), (0.37, 0.738))
    add_arrow(ax, (0.30, 0.715), (0.37, 0.650))
    add_arrow(ax, (0.58, 0.650), (0.63, 0.678), color=COLORS["sc"], lw=2.0)
    add_arrow(ax, (0.30, 0.517), (0.63, 0.650), color=COLORS["sc"], lw=2.0, rad=-0.10)

    xs = np.linspace(0.08, 0.92, 240)
    y_native = 0.36 + 0.08 * np.sin(2 * np.pi * 3.2 * (xs - 0.08) + 0.2)
    y_large = 0.21 + 0.035 * np.sin(2 * np.pi * 3.2 * (xs - 0.08) + 0.2)
    ax.plot(xs, y_native, color=COLORS["signal"], lw=2.0)
    ax.plot(xs, y_large, color=COLORS["sc"], lw=2.0)
    ax.text(0.10, 0.445, "same phase", fontsize=9.5, color=COLORS["signal"])
    ax.text(0.10, 0.265, "larger footprint attenuates response", fontsize=9.5, color=COLORS["sc"], fontweight="bold")

    ax.text(
        0.50,
        0.075,
        "Insight: cell modulates what the pixel can observe.",
        ha="center",
        fontsize=10.5,
        color=COLORS["sc"],
        fontweight="bold",
    )


def make_figure(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), dpi=180)
    fig.patch.set_facecolor(COLORS["bg"])

    draw_lte_panel(axes[0])
    draw_box_average_panel(axes[1])
    draw_sc_panel(axes[2])

    fig.suptitle(
        "From phase shortcut to footprint response",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "SC-INR changes the role of cell: not a texture phase controller, but an analytic observation response for the output footprint.",
        ha="center",
        va="center",
        fontsize=11.2,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.90, bottom=0.08, wspace=0.045)

    stem = out_dir / "footprint_response_mechanism"
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    text = """# SC-INR footprint response 机制图

本目录保存用于周报/论文方法解释的机制图，核心说明：

- `LTE` 中 cell 通过 learned `h_p(c)` 进入 Fourier-like phase，可能让尺度直接移动局部纹理相位。
- `SC-INR` 中 cell 被解释为输出像素 footprint，只通过 analytic response `W(omega,c)` 调制频率分量的可观测强度。
- `sinc` 形式来自 Fourier 分量在 box footprint 上做平均时的频率响应。

图中的一句话 insight 是：

> LTE lets scale move texture phase; SC-INR lets scale change what the pixel can observe.

## 文件

- `footprint_response_mechanism.png`
- `footprint_response_mechanism.pdf`
- `footprint_response_mechanism.svg`

## 生成命令

```bash
python scripts/viz/draw_footprint_response_mechanism.py
```

## 使用边界

该图是概念机制图，不展示 benchmark 数值，也不能替代 NoSinc、footprint oracle 或 multi-seed benchmark。这里的 `W(omega,c)` 应理解为 Fourier decoder 输入层面的 observation response；由于模型后面仍有 MLP、local ensemble 和 residual path，不能把最终 RGB 输出写成严格 exact box integral。
"""
    (out_dir / "README_zh.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw SC-INR footprint response mechanism figure.")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    make_figure(out_dir)
    write_readme(out_dir)
    print(f"Wrote footprint response mechanism figure to {out_dir}")


if __name__ == "__main__":
    main()
