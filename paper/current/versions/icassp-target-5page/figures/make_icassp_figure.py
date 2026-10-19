"""Compact plotting of the same frozen seed1 SC ring-energy results.

No evaluation or model inference. Source provenance is unchanged from
l1-scale-recovery.sources.json; all datasets remain in the manuscript table.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p = Path(__file__).resolve().parent
d = json.loads((p / "l1-scale-recovery.sources.json").read_text())
plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(4.25, 3.35), sharey=True)
for j, ds in enumerate(["BSD100", "Set14"]):
    for i, typ in enumerate(["nu", "u"]):
        ax = axes[i, j]
        edges = d[typ + "_bin_edges"]
        centres = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
        for s, colour in zip([2, 3, 4], ["#2463a5", "#c15e14", "#248666"]):
            ax.plot(centres, d["datasets"][ds][typ][str(s)], lw=1.3,
                    marker="o", ms=2.3, color=colour, label=f"x{s}")
        ax.grid(alpha=.18)
        ax.set_ylim(-.04, 1.04)
        ax.set_xlabel("Frequency, $\\nu$" if typ == "nu" else "LR-Nyquist units, $u$")
        if j == 0:
            ax.set_ylabel("Recovery $R$")
        if i == 0:
            ax.set_title(ds)
        if typ == "u":
            ax.axvline(1, color="gray", ls=":", lw=.8)
axes[0, 0].legend(ncol=3, fontsize=9, frameon=False, loc="lower left",
                  handlelength=.65, handletextpad=.25, columnspacing=.6)
fig.tight_layout(pad=.35)
fig.savefig(p / "icassp-recovery.pdf", bbox_inches="tight")
fig.savefig(p / "icassp-recovery.png", dpi=180, bbox_inches="tight")
