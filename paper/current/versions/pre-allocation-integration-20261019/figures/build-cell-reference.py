"""All frozen natural-image cases; no selection of favourable examples."""
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('rows');ap.add_argument('output');args=ap.parse_args()
path=Path(args.rows);rows=json.loads(path.read_text());out=Path(args.output)
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,4,figsize=(7.1,2.05))
for ax,(model,scale) in zip(axs,[(m,s) for m in ['lte','sc-inr'] for s in [2,4]]):
    for seed,color,marker in [(1,'#27748c','o'),(2,'#c56b3e','^')]:
        ar=[r for r in rows if r['model']==model and r['seed']==seed and r['scale']==scale and r['band']=='rgb']
        pairs={}
        for r in ar:pairs.setdefault(r['image'],{})[r['ref']]=r
        assert len(pairs)==32
        x=[p['zero']['db'] for p in pairs.values()];y=[p['h3']['db'] for p in pairs.values()]
        ax.scatter(x,y,s=15,color=color,marker=marker,alpha=.75,label=f'Seed {seed}',linewidths=0)
    ax.axhline(0,color='.45',lw=.65,ls='--');ax.axvline(0,color='.45',lw=.65,ls='--')
    ax.set_title(('LTE' if model=='lte' else 'SC-INR')+f' $\\times${scale}',fontsize=9)
    ax.set_xlabel('Native vs zero (dB)',fontsize=7)
    ax.tick_params(labelsize=6);ax.grid(alpha=.12)
axs[0].set_ylabel('Native vs nonzero $h=3$ (dB)',fontsize=7)
axs[-1].legend(fontsize=6,loc='best',frameon=False)
fig.tight_layout(pad=.55,w_pad=.7)
fig.savefig(out.with_suffix('.pdf'));fig.savefig(out.with_suffix('.png'),dpi=200)
out.with_suffix('.sources.json').write_text(json.dumps({'rows':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':256,'images':32,'selection':'all RGB rows, zero and h3 paired; per model/scale, both seeds','png_sha256':hashlib.sha256(out.with_suffix('.png').read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(out.with_suffix('.pdf').read_bytes()).hexdigest()},indent=2)+'\n')
