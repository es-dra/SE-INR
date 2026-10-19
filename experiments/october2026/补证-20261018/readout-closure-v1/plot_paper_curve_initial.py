"""Single-column rendering of unchanged frozen risk curves."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,time,resource,hashlib,signal
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
signal.alarm(60);start=time.monotonic();ru=resource.getrusage(resource.RUSAGE_SELF)
root=Path(__file__).resolve().parent;out=root/'figures';assert not (out/'risk-curves-paper.pdf').exists()
curves=json.loads((root/'curves.json').read_text());summary=json.loads((root/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.8,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.size':2.2,'ytick.major.size':2.2,'xtick.major.pad':1.5,'ytick.major.pad':1.5})
fig,axs=plt.subplots(2,2,figsize=(3.4,2.25)); colors={'zero':'#426a8c','h3':'#c56542'}
for row,model in enumerate(['lte','sc-inr']):
 for col,scale in enumerate([2,4]):
  ax=axs[row,col]
  for ref in ['zero','h3']:
   rr=[r for r in curves if r['model']==model and r['scale']==scale and r['ref']==ref and r['band']=='rgb']
   ax.plot([r['coefficient'] for r in rr],[r['mse_change_vs_native_percent'] for r in rr],color=colors[ref],lw=1.05,label=ref)
   s=next(s for s in summary['groups'] if s['metric']=='raw_shave4' and s['model']==model and s['scale']==scale and s['ref']==ref and s['band']=='rgb')['global_oracle']
   ax.plot(s['coefficient'],s['mse_change_vs_native_percent'],'o',ms=2.7,color=colors[ref],markeredgecolor='white',markeredgewidth=.3,clip_on=False)
  ax.axhline(0,color='#999999',lw=.5,ls='--');ax.grid(alpha=.17,lw=.4);ax.set_xlim(0,1);ax.set_xticks([0,.5,1],['0','.5','1']);ax.yaxis.set_major_locator(MaxNLocator(3));ax.set_title(('LTE' if model=='lte' else 'SC-INR')+f' ×{scale}',pad=2)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.58,1.02),ncol=2,frameon=False,handlelength=1.8,columnspacing=1.5,borderpad=0)
fig.subplots_adjust(left=.17,right=.97,bottom=.20,top=.82,hspace=.66,wspace=.48)
fig.text(.58,.035,r'Correction coefficient $\lambda$',ha='center',fontsize=7)
fig.text(.018,.51,'MSE change vs native (%)',va='center',rotation=90,fontsize=7)
fig.savefig(out/'risk-curves-paper.pdf');fig.savefig(out/'risk-curves-paper.png',dpi=220)
r1=resource.getrusage(resource.RUSAGE_SELF)
(out/'paper-curve-render-check.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'curves.json',root/'summary.json']},size_inches=[3.4,2.25],font_pt_range=[6.5,7],wall_seconds=time.monotonic()-start,cpu_user_seconds=r1.ru_utime-ru.ru_utime,cpu_sys_seconds=r1.ru_stime-ru.ru_stime,nice=os.getpriority(os.PRIO_PROCESS,0),gpu=0),indent=2)+'\n')
