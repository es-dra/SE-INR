"""Plot complete fixed research outputs; no fitting, selection, or new inference."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--stage-a',required=True);ap.add_argument('--stage-b',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
A=Path(args.stage_a);B=Path(args.stage_b);out=Path(args.out);out.mkdir(exist_ok=True)
s=json.loads(A.read_text());r=json.loads((B/'rows.json').read_text())
plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
colors=['#48758b','#cb8152'];models=['lte','sc-inr'];labels=['LTE','SC-INR']
fig,ax=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw={'width_ratios':[1,1.25]})
x=np.arange(2);over=[];wrong=[]
for m in models:
 a=s['groups'][m+'/pooled_bands_descriptive']['failure_types'];over.append(a['overshoot']['rows']);wrong.append(a['wrong_direction']['rows'])
ax[0].bar(x,over,color=colors[0],width=.52,label='Overshoot');ax[0].bar(x,wrong,bottom=over,color=colors[1],width=.52,label='Wrong direction')
for i in range(2):
 ax[0].text(i,over[i]/2,str(over[i]),ha='center',va='center',color='white',fontweight='bold')
 ax[0].text(i,over[i]+wrong[i]/2 if wrong[i]>=3 else over[i]+wrong[i]+.7,str(wrong[i]),ha='center',va='center',color='white' if wrong[i]>=3 else '#333',fontweight='bold')
ax[0].set_xticks(x,labels);ax[0].set_ylabel('Harmful band outcomes');ax[0].set_ylim(0,33);ax[0].legend(frameon=False,loc='upper left',fontsize=9)
for i,m in enumerate(models):
 a=s['groups'][m+'/xpooled/fullRGB'];xx=[.25,.5,.75,1.];yy=[100*(a['lambda_results'][str(v)]['mean_mse']/a['native_mse_mean']-1) for v in xx]
 ax[1].plot(xx,yy,'o-',color=colors[i],label=labels[i],lw=2)
 ax[1].annotate(f'{yy[1]:.2f}%',(.5,yy[1]),xytext=(7,7),textcoords='offset points',color=colors[i])
ax[1].axhline(0,color='#777',lw=.8);ax[1].set_xticks([.25,.5,.75,1.]);ax[1].set_xlabel('Fraction of correction retained');ax[1].set_ylabel('RGB MSE increase vs. native (%)');ax[1].legend(frameon=False)
fig.tight_layout();fig.savefig(out/'failure-modes.png');plt.close(fig)
templates=['mid_axis','mid_cross','high_axis','high_cross'];titles=['Mid band / axis','Mid band / mixed','High band / axis','High band / mixed']
fig,axes=plt.subplots(2,4,figsize=(10,4.6),sharex=True,sharey='row')
for mi,m in enumerate(models):
 for ti,tmp in enumerate(templates):
  band=2 if tmp.startswith('mid') else 3;ax=axes[mi,ti];curves=[]
  for phase in range(4):
   for shift in range(4):
    rr=sorted([z for z in r if z['model']==m and z['roi']=='center' and z['template']==tmp and z['band']==band and z['phase']==phase and z['shift']==shift],key=lambda z:z['eta'])
    yy=[z['gain']*1e6 for z in rr];curves.append(yy);ax.plot([0,.5,1],yy,color=colors[mi],alpha=.27,lw=.8)
  ax.plot([0,.5,1],np.mean(curves,axis=0),color='#242d36',marker='o',lw=2,label='Mean of fixed grid')
  ax.axhline(0,color='#777',lw=.8,ls='--');ax.set_xticks([0,.5,1]);ax.grid(axis='y',alpha=.15)
  if mi==0:ax.set_title(titles[ti])
  if ti==0:ax.set_ylabel(labels[mi]+'\nGain ($10^{-6}$ MSE)')
  if mi==1:ax.set_xlabel(r'Path $\eta$')
fig.tight_layout();fig.savefig(out/'controlled-alias-phase.png');fig.savefig(out/'icassp-controlled.pdf');plt.close(fig)
manifest={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'stage_a_summary_sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'stage_b_rows_sha256':hashlib.sha256((B/'rows.json').read_bytes()).hexdigest(),'scope':'all32 calibration images; full controlled primary center ROI, all16 phase/shift families per template. No sampling CI or independent-replicate claim.','outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.suffix in ['.png','.pdf']}}
(out/'cause-figure-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
