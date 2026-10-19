"""Render audited complete G2 cohort; no tuning or outcome exclusions."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=OUT.parent.parent/'补证-20261008/guidance-formal-v4')
ROOT=parser.parse_args().root
audit=json.loads((ROOT/'独立图像复核.json').read_text())
assert audit['status']=='pass' and audit['n_cases']==36
summary=json.loads((ROOT/'summary_target.json').read_text())
assert summary['status']=='complete'
rows=[json.loads(p.read_text()) for p in sorted((ROOT/'evaluation').glob('*.json'))]
assert len(rows)==36 and all(r['status']=='complete' for r in rows)
images=sorted({r['case'].split('_')[0] for r in rows})
means={m:[np.mean([r['metrics'][m]['improvement_over_default_db'] for r in rows if r['case'].startswith(i+'_')]) for i in images] for m in ('analysis','cv')}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(7,2.7),gridspec_kw={'width_ratios':[2,1]},constrained_layout=True)
x=np.arange(12)
for m,shift,color,label in [('analysis',-.17,'#176B87','Calibration-risk choice'),('cv',.17,'#B65C45','Four-fold CV')]:
    ax[0].bar(x+shift,means[m],.32,color=color,label=label)
ax[0].axhline(0,color='.4',lw=.7)
ax[0].set_xticks(x,[i[-2:] for i in images]);ax[0].set(xlabel='Target image (DIV2K 08xx)',ylabel='PSNR gain over default (dB)',title='(a) All 12 images; three noises averaged')
ax[0].legend(frameon=False,fontsize=8)
methods=['default','cv','analysis'];cost=[summary['methods'][m]['successful_selection_and_final_seconds']/60 for m in methods]
ax[1].bar(range(3),cost,color=['#808080','#B65C45','#176B87'])
ax[1].set_xticks(range(3),['Default','CV','Analysis']);ax[1].set(yscale='log',ylim=(1,300),ylabel='Synchronized wall minutes',title='(b) Costs over all 36 cases')
for i,v in enumerate(cost):ax[1].text(i,v*1.15,f'{v:.1f}',ha='center',fontsize=8)
for a in ax:a.grid(axis='y',color='.92',lw=.6);a.set_axisbelow(True)
fig.savefig(OUT/'guidance-results.pdf');fig.savefig(OUT/'guidance-results.png',dpi=200)
(OUT/'guidance-results.sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'summary_target.json',ROOT/'独立图像复核.json',*sorted((ROOT/'evaluation').glob('*.json'))]},indent=2)+'\n')
