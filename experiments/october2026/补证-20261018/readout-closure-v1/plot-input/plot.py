"""Render only frozen analysis results and protocol-selected existing examples."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import json,time,resource,hashlib,signal
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=False)
signal.alarm(120); ts=time.time(); mono=time.monotonic(); ru=resource.getrusage(resource.RUSAGE_SELF)
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
write(OUT/'plot-budget.json',dict(start_unix=ts,deadline_unix=ts+120,budget_seconds=120,cpu_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),gpu=0))
curves=json.loads((ROOT/'curves.json').read_text()); summary=json.loads((ROOT/'summary.json').read_text()); ex=json.loads((ROOT/'selected_examples.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors={'zero':'#426a8c','h3':'#c56542'}; names={'lte':'LTE','sc-inr':'SC-INR'}
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight'); fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(7,3.8))
for row,model in enumerate(['lte','sc-inr']):
 for col,scale in enumerate([2,4]):
  ax=axes[row,col]
  for ref in ['zero','h3']:
   rr=[r for r in curves if r['model']==model and r['scale']==scale and r['ref']==ref and r['band']=='rgb']
   ax.plot([r['coefficient'] for r in rr],[r['mse_change_vs_native_percent'] for r in rr],label=ref,color=colors[ref],lw=1.8)
   s=next(s for s in summary['groups'] if s['metric']=='raw_shave4' and s['model']==model and s['scale']==scale and s['ref']==ref and s['band']=='rgb')['global_oracle']
   ax.plot(s['coefficient'],s['mse_change_vs_native_percent'],'o',ms=4,color=colors[ref],markeredgecolor='white',markeredgewidth=.5)
  ax.axhline(0,color='#999999',lw=.7,ls='--');ax.grid(alpha=.18);ax.set_xlim(0,1);ax.set_title(f'{names[model]}  ×{scale}');ax.set_xlabel(r'Correction coefficient $\lambda$');ax.set_ylabel('MSE change vs native (%)')
axes[0,0].legend(title='Reference',frameon=False)
fig.subplots_adjust(hspace=.6,wspace=.38,bottom=.16);fig.text(.5,.015,'Raw RGB; shave 4. Per-panel y scales. Dots: exact HR-informed group optimum; native = 1.',ha='center',fontsize=7)
save(fig,'risk-curves')
fig,axes=plt.subplots(2,4,figsize=(12,5.0))
for row,ref in enumerate(['zero','h3']):
 for col,(model,scale) in enumerate([('lte',2),('lte',4),('sc-inr',2),('sc-inr',4)]):
  ax=axes[row,col]; rr=[r for r in curves if r['model']==model and r['scale']==scale and r['ref']==ref and r['band']=='rgb']; x=[r['coefficient'] for r in rr]
  for key,label,color in [('original_negative_rescued','Harmed → benefited','#188b78'),('original_positive_becomes_negative','Benefited → harmed','#bd5942'),('all_worse_than_native','Worse than native','#647386')]:ax.plot(x,[r[key] for r in rr],label=label,color=color,lw=1.4)
  ax.set_title(f'{names[model]} ×{scale} | {ref}');ax.set_xlim(0,1);ax.set_ylim(-1,65);ax.set_yticks([0,16,32,48,64]);ax.grid(alpha=.18);ax.set_xlabel(r'$\lambda$');ax.set_ylabel('Records (of 64)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.03));fig.subplots_adjust(hspace=.5,wspace=.35,bottom=.22);fig.text(.5,.005,'32 images × 2 seeds per panel; records are dependent. Strict sign threshold: 1e−8 MSE. Raw RGB, shave 4.',ha='center',fontsize=8)
save(fig,'rescue-tradeoff-curves')
loaded=[]
for r in ex:
 if r['status']!='selected':continue
 z=np.load(ROOT/'examples'/f"{r['image']}_x{r['scale']}__{r['model']}_seed{r['seed']}.npz");t=np.load(ROOT/'examples'/f"{r['image']}.npy").astype('float64')[4:-4,4:-4];p0=z['zero'].astype('float64')[4:-4,4:-4];pn=z['h2' if r['scale']==2 else 'h4'].astype('float64')[4:-4,4:-4]
 delta=np.mean((pn-t)**2-(p0-t)**2,axis=2);loaded.append((r,t,p0,pn,delta))
vmax=max(float(np.max(np.abs(v[4]))) for v in loaded)
for kind in ['median','tail']:
 ar=[v for v in loaded if v[0]['selection']==kind];fig,axes=plt.subplots(len(ar),4,figsize=(8.1,2.35*len(ar)+.55),squeeze=False)
 for row,(r,t,p0,pn,d) in enumerate(ar):
  for col,p in enumerate([t,p0,pn]):axes[row,col].imshow(np.clip(p,0,1),interpolation='nearest');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
  im=axes[row,3].imshow(d,cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest');axes[row,3].set_xticks([]);axes[row,3].set_yticks([])
  axes[row,0].set_ylabel(f"{names[r['model']]} {r['image']:04d}\n×{r['scale']}, seed {r['seed']}\n{r['label']}, {r['db']:+.4f} dB",fontsize=8)
  if row==0:
   for col,title in enumerate(['Ground truth','Zero-cell reference','Native cell','Squared-error change']):axes[row,col].set_title(title,fontsize=9)
 fig.subplots_adjust(left=.14,right=.9,hspace=.18,wspace=.06,bottom=.12,top=.94)
 cbax=fig.add_axes([.92,.22,.018,.56]);cb=fig.colorbar(im,cax=cbax);fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((0,0));cb.formatter=fmt;cb.update_ticks();cb.set_label('RGB-mean squared error (native − reference)',fontsize=8)
 fig.text(.52,.025,'Unamplified RGB display; raw RGB scores. Red: native worse. Shared map range across both panels.',ha='center',fontsize=7)
 save(fig,'examples-'+kind)
write(OUT/'render-metadata.json',dict(input_hashes={p.name:sha(p) for p in [ROOT/'plot.py',ROOT/'curves.json',ROOT/'summary.json',ROOT/'selected_examples.json']},selected_examples=ex,shared_signed_error_vmax=vmax,rgb_display_clipped=True,rgb_score='raw_shave4',note='LTE direction class missing among eligible RGB zero-reference examples; not replaced. No image or contrast selection after viewing.'))
r1=resource.getrusage(resource.RUSAGE_SELF);write(OUT/'plot-cost.json',dict(wall_seconds=time.monotonic()-mono,cpu_user_seconds=r1.ru_utime-ru.ru_utime,cpu_sys_seconds=r1.ru_stime-ru.ru_stime,peak_rss_platform_units=r1.ru_maxrss,nice=os.getpriority(os.PRIO_PROCESS,0),gpu=0,inference=0))
print(json.dumps({'status':'complete','wall_seconds':time.monotonic()-mono,'error_map_vmax':vmax}))
