"""Plot saved confirmation results only; no inference or training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True)
args=ap.parse_args();root=Path(args.root);out=Path(args.out);out.mkdir()
tic=time.monotonic();rows=json.loads((root/'rows.json').read_text())
complete=json.loads((root/'COMPLETE.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(root/'rows.json')==complete['rows_sha256']
focal=sorted([r for r in rows if r['focal'] and r['roi']=='center' and r['band']==2],key=lambda r:r['family'])
sham=sorted([r for r in rows if r['template']=='control_low' and r['roi']=='center' and r['band']==0],key=lambda r:r['family'])
assert len(focal)==len(sham)==16
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
panels=[(focal,'native','uniform','Middle band: normal decoder'),(focal,'fixed_native','fixed_uniform','Middle band: fixed gates'),(sham,'native','uniform','Low-frequency control: RGB')]
for ax,(rr,a,b,title) in zip(axes,panels):
    y0=np.array([r['g_'+a] for r in rr])*1e6;y1=np.array([r['g_'+b] for r in rr])*1e6
    offsets=np.linspace(-.10,.10,len(rr))
    for dx,v0,v1 in zip(offsets,y0,y1):
        ax.plot([dx,1+dx],[v0,v1],color='#aab3bc',alpha=.48,lw=.8,zorder=1)
    ax.scatter(offsets,y0,color='#49677e',s=24,zorder=3)
    ax.scatter(1+offsets,y1,color='#18877b',s=24,zorder=3)
    ax.axhline(0,color='#b44d4d',ls='--',lw=1)
    ax.set_xticks([0,1],['Native allocation','Uniform allocation'])
    ax.tick_params(axis='x',labelsize=9)
    ax.set_title(title,fontsize=11)
    ax.set_xlim(-.3,1.3);ax.grid(axis='y',alpha=.15)
    ax.text(.03,.96,f'Positive gain: {sum(y0>0.01)}/16 → {sum(y1>0.01)}/16',transform=ax.transAxes,va='top',fontsize=10)
axes[0].set_ylabel('MSE improvement vs zero cell (×10⁻⁶)')
fig.suptitle('SC-INR ×4 · same feature-step norm · existing controlled samples',fontsize=12)
fig.savefig(out/'allocation-confirmation.png',dpi=180)
fig.savefig(out/'allocation-confirmation.pdf')
plt.close(fig)
assert time.monotonic()-tic<30
(out/'provenance.json').write_text(json.dumps({'source_sha256':sha(__file__),'rows_sha256':sha(root/'rows.json'),'figure_sha256':{p.name:sha(p) for p in out.iterdir()},'wall_seconds':time.monotonic()-tic,'scope':'Saved results only;16 dependent focal conditions and16 low controls. Center ROI; first two panels middle band, third RGB. Feature-step RMS matched to native across branches/queries/features; no RGB norm matching claim. No training/inference/GPU.'},indent=2)+'\n')
print(json.dumps({'wall_seconds':time.monotonic()-tic,'output':str(out)}))
