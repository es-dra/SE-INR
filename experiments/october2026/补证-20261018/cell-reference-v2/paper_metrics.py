"""Descriptive manuscript extraction; no selection, fitting, or inference."""
import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]);rows=json.loads((p/'rows.json').read_text());summary=json.loads((p/'summary.json').read_text())
table=[];cross=[]
for model in ['lte','sc-inr']:
 for scale in [2,4]:
  for ref in ['zero','h3']:
   ar=[r for r in rows if r['model']==model and r['scale']==scale and r['ref']==ref]
   d={'model':model,'scale':scale,'ref':ref}
   for band in ['rgb','bands']:
    rr=[r for r in ar if (r['band']!='rgb' if band=='bands' else r['band']=='rgb')]
    neg=[r for r in rr if r['gain'] < -1e-8]
    d[band]={'records':len(rr),'negative':len(neg),'overshoot':sum(r['label']=='overshoot' for r in neg),'direction':sum(r['label']=='direction' for r in neg),'material_harm_001db':sum(r['db']<-.01 for r in rr),'material_harm_005db':sum(r['db']<-.05 for r in rr),'min_db':min(r['db'] for r in rr),'max_db':max(r['db'] for r in rr),'half_rescued':sum(r['half_gain']>1e-8 for r in neg),'direction_harm_share':sum(-r['gain'] for r in neg if r['label']=='direction')/sum(-r['gain'] for r in neg) if neg else None}
   d['aggregate']=next(a for a in summary['aggregate'] if a['model']==model and a['scale']==scale and a['ref']==ref)
   table.append(d)
 for band in ['rgb','bands']:
  rr=[r for r in summary['cross_reference'] if r['model']==model and (r['band']!='rgb' if band=='bands' else r['band']=='rgb')]
  cross.append(dict(model=model,band=band,counts={key:sum(r['counts'][key] for r in rr) for key in ['both','zero_only','h3_only','neither']},same_label_both=sum(r['same_label_both_harmful'] for r in rr)))
result={'rows_sha256':hashlib.sha256((p/'rows.json').read_bytes()).hexdigest(),'table':table,'cross_reference_pooled_counts':cross,'note':'Counts combine seeds; primary mean and interval separate scale. Pooled cross-reference counts are descriptive repeated image-scale-seed records.'}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
for d in table:
 print(d['model'],d['scale'],d['ref'],'RGB',d['rgb'],'BANDS',d['bands'],'MEAN',d['aggregate'])
print('CROSS',cross)
