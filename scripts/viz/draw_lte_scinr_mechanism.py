#!/usr/bin/env python3
"""绘制 LTE 与 SC-INR 的机制对比图。

输出是论文/周报可用的矢量图，重点展示 cell 在两种 decoder 中进入计算的位置：

- LTE: cell -> learned phase h_p(c)
- SC-INR: cell + omega -> analytic footprint response W(omega, c)

该图是概念机制图，不展示 benchmark 数值，也不暗示最终 RGB 是 exact box integral。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]


COLORS = {
    "bg": "#F7F8FA",
    "panel": "#FFFFFF",
    "shared": "#ECEFF4",
    "text": "#20242A",
    "muted": "#5D6673",
    "lte": "#D9793D",
    "lte_soft": "#FFF1E8",
    "sc": "#187C83",
    "sc_soft": "#E7F4F3",
    "neutral": "#2F3A4A",
    "line": "#8A94A6",
}


def add_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str,
    ec: str = "#ADB5C2",
    color: str = COLORS["text"],
    fontsize: int = 11,
    weight: str = "normal",
    radius: float = 0.04,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.2,
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
        linespacing=1.25,
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
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arrow)


def draw_panel_lte(ax) -> None:
    add_box(
        ax,
        (0.06, 0.82),
        (0.88, 0.10),
        "LTE: cell-conditioned phase",
        fc=COLORS["lte_soft"],
        ec=COLORS["lte"],
        color=COLORS["lte"],
        fontsize=14,
        weight="bold",
    )

    add_box(ax, (0.08, 0.64), (0.22, 0.09), "local feature\n$z$", fc=COLORS["shared"], fontsize=12)
    add_box(ax, (0.40, 0.66), (0.20, 0.08), "$A(z)$", fc="#FFFFFF", fontsize=12)
    add_box(ax, (0.40, 0.55), (0.20, 0.08), "$\\omega(z)$", fc="#FFFFFF", fontsize=12)
    add_box(ax, (0.08, 0.42), (0.22, 0.09), "relative coord\n$\\delta$", fc=COLORS["shared"], fontsize=12)

    add_box(
        ax,
        (0.08, 0.22),
        (0.22, 0.10),
        "output cell\n$c$",
        fc=COLORS["lte_soft"],
        ec=COLORS["lte"],
        color=COLORS["lte"],
        fontsize=12,
        weight="bold",
    )
    add_box(
        ax,
        (0.40, 0.24),
        (0.22, 0.09),
        "learned phase\n$h_p(c)$",
        fc=COLORS["lte_soft"],
        ec=COLORS["lte"],
        color=COLORS["lte"],
        fontsize=12,
        weight="bold",
    )

    add_box(
        ax,
        (0.67, 0.42),
        (0.25, 0.19),
        "$\\gamma_{LTE}$\n$=A(z)\\odot[\\cos\\theta,\\sin\\theta]$\n$\\theta=\\pi(\\omega(z)^T\\delta+h_p(c))$",
        fc="#FFFFFF",
        ec=COLORS["neutral"],
        fontsize=10,
    )
    add_box(ax, (0.70, 0.20), (0.19, 0.08), "MLP\nRGB", fc="#FFFFFF", fontsize=11)

    add_arrow(ax, (0.30, 0.685), (0.40, 0.70))
    add_arrow(ax, (0.30, 0.685), (0.40, 0.59))
    add_arrow(ax, (0.60, 0.70), (0.67, 0.55))
    add_arrow(ax, (0.60, 0.59), (0.67, 0.52))
    add_arrow(ax, (0.30, 0.465), (0.67, 0.49))
    add_arrow(ax, (0.30, 0.27), (0.40, 0.285), color=COLORS["lte"], lw=2.0)
    add_arrow(ax, (0.62, 0.285), (0.67, 0.46), color=COLORS["lte"], lw=2.0)
    add_arrow(ax, (0.795, 0.42), (0.795, 0.28), color=COLORS["neutral"])

    ax.text(
        0.50,
        0.11,
        "cell changes Fourier phase",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["lte"],
        fontweight="bold",
    )


def draw_panel_sc(ax) -> None:
    add_box(
        ax,
        (0.06, 0.82),
        (0.88, 0.10),
        "SC-INR: analytic footprint response",
        fc=COLORS["sc_soft"],
        ec=COLORS["sc"],
        color=COLORS["sc"],
        fontsize=14,
        weight="bold",
    )

    add_box(ax, (0.08, 0.64), (0.22, 0.09), "local feature\n$z$", fc=COLORS["shared"], fontsize=12)
    add_box(ax, (0.38, 0.69), (0.18, 0.07), "$A(z)$", fc="#FFFFFF", fontsize=12)
    add_box(ax, (0.38, 0.58), (0.18, 0.07), "$\\omega(z)$", fc="#FFFFFF", fontsize=12)
    add_box(ax, (0.38, 0.47), (0.18, 0.07), "$\\phi(z)$", fc="#FFFFFF", fontsize=12)
    add_box(ax, (0.08, 0.42), (0.22, 0.09), "relative coord\n$\\delta$", fc=COLORS["shared"], fontsize=12)

    add_box(
        ax,
        (0.08, 0.22),
        (0.22, 0.10),
        "output cell\n$c$",
        fc=COLORS["sc_soft"],
        ec=COLORS["sc"],
        color=COLORS["sc"],
        fontsize=12,
        weight="bold",
    )
    add_box(
        ax,
        (0.61, 0.62),
        (0.31, 0.12),
        "$W(\\omega,c)$\n$=\\mathrm{sinc}(\\omega_hc_h/2)\\,\\mathrm{sinc}(\\omega_wc_w/2)$",
        fc=COLORS["sc_soft"],
        ec=COLORS["sc"],
        color=COLORS["sc"],
        fontsize=9.6,
        weight="bold",
    )
    add_box(
        ax,
        (0.62, 0.43),
        (0.29, 0.10),
        "$A_{eff}(z,c)=A(z)\\odot W(\\omega(z),c)$",
        fc="#FFFFFF",
        ec=COLORS["sc"],
        color=COLORS["sc"],
        fontsize=10.5,
        weight="bold",
    )
    add_box(
        ax,
        (0.62, 0.24),
        (0.29, 0.13),
        "$\\gamma_{SC}$\n$=A_{eff}\\odot[\\cos\\psi,\\sin\\psi]$\n$\\psi=\\pi(\\omega(z)^T\\delta+\\phi(z))$",
        fc="#FFFFFF",
        ec=COLORS["neutral"],
        fontsize=9.5,
    )
    add_box(ax, (0.67, 0.07), (0.19, 0.08), "MLP\nRGB", fc="#FFFFFF", fontsize=11)

    add_arrow(ax, (0.30, 0.685), (0.38, 0.725))
    add_arrow(ax, (0.30, 0.685), (0.38, 0.615))
    add_arrow(ax, (0.30, 0.685), (0.38, 0.505))
    add_arrow(ax, (0.56, 0.615), (0.61, 0.675), color=COLORS["sc"], lw=2.0)
    add_arrow(ax, (0.30, 0.27), (0.61, 0.655), color=COLORS["sc"], lw=2.0, rad=-0.12)
    add_arrow(ax, (0.56, 0.725), (0.62, 0.49))
    add_arrow(ax, (0.765, 0.62), (0.765, 0.53), color=COLORS["sc"], lw=2.0)
    add_arrow(ax, (0.56, 0.615), (0.62, 0.31))
    add_arrow(ax, (0.56, 0.505), (0.62, 0.30))
    add_arrow(ax, (0.30, 0.465), (0.62, 0.285))
    add_arrow(ax, (0.765, 0.43), (0.765, 0.37), color=COLORS["sc"], lw=2.0)
    add_arrow(ax, (0.765, 0.24), (0.765, 0.15), color=COLORS["neutral"])

    ax.text(
        0.50,
        0.01,
        "cell changes observable Fourier amplitude",
        ha="center",
        va="bottom",
        fontsize=12,
        color=COLORS["sc"],
        fontweight="bold",
    )


def draw(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.5, 8.6), facecolor=COLORS["bg"])
    gs = fig.add_gridspec(1, 2, left=0.035, right=0.965, top=0.90, bottom=0.08, wspace=0.055)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        panel = FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.015,rounding_size=0.035",
            linewidth=1.1,
            edgecolor="#D9DEE8",
            facecolor=COLORS["panel"],
            zorder=0,
        )
        ax.add_patch(panel)

    fig.text(
        0.5,
        0.965,
        "Role of output cell in local Fourier implicit decoding",
        ha="center",
        va="center",
        fontsize=18,
        color=COLORS["text"],
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "LTE uses cell as a learned phase condition; SC-INR uses cell as an analytic sampling-footprint response.",
        ha="center",
        va="center",
        fontsize=12.5,
        color=COLORS["muted"],
    )

    draw_panel_lte(axes[0])
    draw_panel_sc(axes[1])

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"lte_scinr_mechanism_comparison.{ext}", dpi=240)
    plt.close(fig)

    readme = """# LTE vs SC-INR 机制对比图

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
"""
    (out_dir / "README_zh.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw LTE vs SC-INR mechanism comparison figure.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "derived" / "paper_figures" / "lte_scinr_mechanism_2026-05-18",
    )
    args = parser.parse_args()
    draw(args.out)
    print(f"wrote mechanism figure to {args.out}")


if __name__ == "__main__":
    main()
