"""Regenerate manuscript figures from saved, independently checked summaries."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[2] / "paper" / "figures"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN = "#2b6f9c", "#c56334", "#39846b"
inputs = []

def read(rel):
    p = ROOT / rel
    inputs.append({"path": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return list(csv.DictReader(p.open()))

def f(row, key):
    return float(row[key])

def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / (name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

pairs = read("localization-v2/pairs.csv")
images = read("localization-v2/image_ranking.csv")
projections = read("projection-summary-v3/scales.csv")
bands = read("localization-v2/bands.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35),
                         gridspec_kw={"width_ratios": [1, 1.06, 1]})
fig.subplots_adjust(left=.07, right=.99, top=.88, bottom=.21, wspace=.54)
sel = [r for r in bands if r["model"] == "mean_of_models" and f(r,"mu")==2]
delta = []
for b in range(16):
    a = next(r for r in sel if int(r["scale"])==2 and int(r["bin"])==b)
    z = next(r for r in sel if int(r["scale"])==6 and int(r["bin"])==b)
    delta.append((f(a,"H")-f(z,"H"))*1e6)
axes[0].bar((np.arange(16)+.5)/32, delta, width=.027,
            color=[ORANGE if 6<=b<12 else BLUE for b in range(16)])
axes[0].set(xlim=(0,.5), ylim=(0,40), xlabel="Output frequency r",
            ylabel=r"$H_2-H_6$ ($10^{-6}$ MSE)", title="(a) Frequency allocation")
axes[0].text(.98,.92,"Middle band: 70.9%",ha="right",
             transform=axes[0].transAxes,fontsize=7)
rows = [r for r in images if r["group"]=="all" and r["start"]=="2"
        and r["end"]=="6" and f(r,"mu")==2]
rows.sort(key=lambda r:-f(r,"signed_share_of_delta_H"))
assert len(rows)==14
y = np.arange(14)
vals = [100*f(r,"signed_share_of_delta_H") for r in rows]
axes[1].barh(y,vals,color=[BLUE if v>=0 else ORANGE for v in vals],height=.72)
axes[1].set_yticks(y,[r["image"] for r in rows],fontsize=6.5)
axes[1].invert_yaxis()
axes[1].axvline(0,color="#666666",lw=.5)
axes[1].set(xlabel="Share of net H decrease (%)",title="(b) All image contributions",
            xlim=(-22,33))
levels = ["pooled","A","B","C","D"]
for scale,offset,color,marker in [(2,-.11,ORANGE,"o"),(6,.11,BLUE,"s")]:
    vals = [f(next(r for r in projections if r["scope"]=="r_lt_0.5"
                   and f(r,"mu")==2 and int(r["scale"])==scale
                   and r["level"]==level),"epsilon_energy")*1e6 for level in levels]
    axes[2].scatter(np.arange(5)+offset,vals,color=color,marker=marker,s=20,
                    label=fr"$\times{scale}$",zorder=3)
axes[2].set_xticks(range(5),levels)
axes[2].set(ylabel=r"Residual energy ($10^{-6}$ MSE)",
            title="(c) Finer real projections",ylim=(0,330),xlabel="Fitted projection (see text)")
axes[2].legend(frameon=False,loc="upper right")
for ax in axes: ax.grid(axis="y",alpha=.16,zorder=0)
save(fig,"fig2_residual")

scales = read("step-controls-v1/scales.csv")
contrasts = read("step-controls-v1/primary_contrasts.csv")
fig,axes=plt.subplots(1,3,figsize=(7.25,2.05))
fig.subplots_adjust(left=.07,right=.99,top=.86,bottom=.23,wspace=.38)
for ds,color,marker in [("Set14",BLUE,"o"),("BSD100",GREEN,"s"),("Urban100",ORANGE,"^")]:
    rows=sorted([r for r in scales if r["dataset"]==ds and f(r,"mu")==2
                 and r["model"]=="mean_of_models"],
                key=lambda r:int(r["scale"]))
    assert len(rows)==4
    for ax,key in zip(axes[:2],["J","J_per_h2"]):
        ax.plot([int(r["scale"]) for r in rows],
                [f(r,key)/f(rows[0],key) for r in rows],
                color=color,marker=marker,ms=3.5,lw=1,label=ds)
    c=next(r for r in contrasts if r["dataset"]==ds and r["model"]=="mean_of_models"
           and r["family"].startswith("matched_positive")
           and r["start_scale"]=="2" and r["end_scale"]=="6")
    i=["Set14","BSD100","Urban100"].index(ds)
    v=f(c,"J_ratio_end_over_start")
    lo=f(c,"J_ratio_end_over_start_ci_low");hi=f(c,"J_ratio_end_over_start_ci_high")
    axes[2].errorbar(i,v,yerr=[[v-lo],[hi-v]],color=color,fmt=marker,ms=4,capsize=3)
for ax in axes[:2]:
    ax.set_xticks([2,3,4,6],[r"$\times2$",r"$\times3$",r"$\times4$",r"$\times6$"])
    ax.set(xlabel="SR scale",ylabel=r"Ratio to $\times2$")
    ax.axhline(1,color="#999999",lw=.6,ls=":")
axes[0].set(title="(a) Fixed multiplier: m = 2",ylim=(0,1.15))
axes[0].legend(frameon=False,loc="upper right")
axes[1].set(title=r"(b) Normalised by $h^2$",ylim=(0,1.8))
axes[2].axhline(1,color="#999999",lw=.6,ls=":")
axes[2].set_xticks(range(3),["Set14","BSD100","Urban100"])
axes[2].set(title=r"(c) Matched step: $h\approx1/3$",
            ylabel=r"$J_6/J_2$ (paired 95% CI)",ylim=(0,4.3))
for ax in axes: ax.grid(axis="y",alpha=.16)
save(fig,"fig3_step_controls")
(ROOT/"paper-figure-inputs.json").write_text(json.dumps(inputs,indent=2)+"\n")
