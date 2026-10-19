import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
d=json.loads((p/"l1-scale-recovery.sources.json").read_text())
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
fig,axs=plt.subplots(2,3,figsize=(8.1,4.3),sharey=True)
colors=["#2563eb","#d97706","#059669"]
for j,ds in enumerate(["Set14","BSD100","Urban100"]):
 for i,typ in enumerate(["nu","u"]):
  edges=d[typ+"_bin_edges"];x=[(a+b)/2 for a,b in zip(edges,edges[1:])]
  ax=axs[i,j]
  for s,c in zip([2,3,4],colors):ax.plot(x,d["datasets"][ds][typ][str(s)],color=c,marker="o",markersize=3,linewidth=1.5,label=f"x{s}")
  ax.grid(alpha=.2);ax.set_ylim(-.16,1.04);ax.set_xlabel("Absolute frequency (cycles/HR pixel)" if typ=="nu" else "Frequency / LR Nyquist, u")
  if typ=="u":ax.axvline(1,color="gray",linestyle=":",linewidth=1)
  if i==0:ax.set_title(ds)
  if j==0:ax.set_ylabel("Pooled recovery R")
axs[0,0].legend(frameon=False,ncol=3,fontsize=8,loc="lower left")
fig.tight_layout(pad=.7)
fig.savefig(p/"l1-scale-recovery.pdf",bbox_inches="tight")
fig.savefig(p/"l1-scale-recovery.png",dpi=180,bbox_inches="tight")
