"""Visualize frozen explanatory results; no analysis or experimental changes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent
d=json.loads((p/'results.json').read_text())
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.2]})
fig.subplots_adjust(left=.075,right=.98,bottom=.17,top=.85,wspace=.34)
colors={'Set14':'#CB635B','BSD100':'#496780','Urban100':'#168579'}
for ds,color in colors.items():
    rr=[r for r in d['groups'] if r['dataset']==ds]
    axes[0].scatter([r['D']['original_energy_pool'] for r in rr],
        [r['D']['fixed_cross_scale_energy_weights'] for r in rr],c=color,s=45,label=ds,zorder=3)
axes[0].plot([0,.07],[0,.07],ls=':',color='grey',lw=1)
axes[0].axhline(.05,color='grey',ls='--',lw=.8)
axes[0].axvline(.05,color='grey',ls='--',lw=.8)
axes[0].set_xlim(0,.07);axes[0].set_ylim(0,.07)
axes[0].set_xlabel('Original pooled D');axes[0].set_ylabel('Fixed-weight D')
axes[0].set_title('(a) Reweighting diagnostic (27 groups)',fontsize=13)
axes[0].legend(frameon=False,fontsize=10,loc='upper left')
ax=axes[1]
groups=[next(r for r in d['groups'] if r['dataset']==ds and r['model']=='sc-inr' and r['seed']==1) for ds in colors]
vals=[[t['D_u'] for t in r['per_image']] for r in groups]
ax.boxplot(vals,positions=[0,1,2],widths=.42,showfliers=False,medianprops={'color':'#1C2D3C'},whiskerprops={'color':'grey'},boxprops={'color':'grey'})
for i,(r,color) in enumerate(zip(groups,colors.values())):
    v=np.asarray(vals[i]);jitter=np.linspace(-.12,.12,len(v))
    ax.scatter(i+jitter,v,s=18,color=color,alpha=.65,zorder=3)
ax.scatter([0,1,2],[r['D']['original_energy_pool'] for r in groups],marker='D',s=75,c='black',label='Pooled',zorder=5)
ax.scatter([0,1,2],[r['D']['equal_image'] for r in groups],marker='s',s=60,c='#D89422',label='Equal-weight curve',zorder=4)
ax.axhline(.05,color='grey',ls='--',lw=.8)
ax.set_xticks([0,1,2],list(colors));ax.set_ylim(0,.23)
ax.set_ylabel('Cross-scale variation D');ax.set_title('(b) SC-INR, seed1',fontsize=13)
ax.legend(frameon=False,fontsize=10,loc='upper right')
for ax in axes:ax.grid(alpha=.12)
fig.savefig(p/'residuals.png',dpi=170)
