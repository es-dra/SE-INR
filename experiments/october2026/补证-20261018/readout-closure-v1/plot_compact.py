"""Compact layout only; original plots and all analysis remain unchanged."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,time,signal,resource
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
signal.alarm(120); ts=time.monotonic();r0=resource.getrusage(resource.RUSAGE_SELF);root=Path(__file__).resolve().parent;out=root/'figures'
assert not (out/'examples-median-compact.pdf').exists()
sel=json.loads((root/'selected_examples.json').read_text());metadata=json.loads((out/'render-metadata.json').read_text());vmax=metadata['shared_signed_error_vmax']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(3,4,figsize=(8,5.5));checks=[]
for row,r in enumerate([r for r in sel if r.get('status')=='selected' and r['selection']=='median']):
 with np.load(root/'examples'/f"{r['image']}_x{r['scale']}__{r['model']}_seed{r['seed']}.npz") as z:
  t=np.load(root/'examples'/f"{r['image']}.npy").astype('float64')[4:-4,4:-4];p0=z['zero'].astype('float64')[4:-4,4:-4];pn=z['h2' if r['scale']==2 else 'h4'].astype('float64')[4:-4,4:-4]
 e0=np.mean((t-p0)**2);en=np.mean((t-pn)**2);db=float(10*np.log10(e0/en));assert abs(db-r['db'])<1e-12
 checks.append(dict(image=r['image'],model=r['model'],scale=r['scale'],seed=r['seed'],original_db=r['db'],pixel_recomputed_db=db,crop=[4,124,4,124]))
 for col,p in enumerate([t,p0,pn]):ax[row,col].imshow(np.clip(p,0,1),interpolation='nearest')
 delta=np.mean((pn-t)**2-(p0-t)**2,axis=2);im=ax[row,3].imshow(delta,cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
 for col in range(4):ax[row,col].set_xticks([]);ax[row,col].set_yticks([])
 name='LTE' if r['model']=='lte' else 'SC-INR'
 ax[row,0].set_ylabel(f"{name} {r['image']:04d}, ×{r['scale']}\nseed {r['seed']}, {r['label']}\n{db:+.4f} dB",fontsize=8)
 if row==0:
  for col,title in enumerate(['Ground truth','Zero-cell reference','Native cell','Squared-error change']):ax[row,col].set_title(title,pad=4)
fig.subplots_adjust(left=.12,right=.9,bottom=.025,top=.95,hspace=.07,wspace=.045)
cax=fig.add_axes([.92,.1,.017,.79]);cb=fig.colorbar(im,cax=cax);fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((0,0));cb.formatter=fmt;cb.update_ticks();cb.ax.tick_params(labelsize=7);cb.set_label('RGB-mean squared error (native − reference)',fontsize=8,labelpad=5)
fig.savefig(out/'examples-median-compact.pdf',bbox_inches='tight',pad_inches=.025);fig.savefig(out/'examples-median-compact.png',bbox_inches='tight',pad_inches=.025,dpi=180)
r1=resource.getrusage(resource.RUSAGE_SELF)
(out/'compact-render-check.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pixel_checks=checks,shared_error_map_vmax=vmax,layout_only=True,wall_seconds=time.monotonic()-ts,cpu_user_seconds=r1.ru_utime-r0.ru_utime,cpu_sys_seconds=r1.ru_stime-r0.ru_stime,nice=os.getpriority(os.PRIO_PROCESS,0),gpu=0),indent=2)+'\n')
