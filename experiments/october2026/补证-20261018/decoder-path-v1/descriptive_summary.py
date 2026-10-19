"""Descriptive reaggregation of completed decoder-path rows; no model execution.

Read-only frozen experiment root; output must be a new file outside that root.
Dependent conditions remain explicit. Counterfactual gate differences are not
feature-independent error attribution or a unique physical cause estimate.
"""
import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

ANCHORS=('ref','mid','native')
SIGNS=('positive','negative','boundary')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sign(x,t):return 'positive' if x>t else 'negative' if x< -t else 'boundary'
def stats(xs):
    return dict(n=len(xs),sum=math.fsum(xs),mean=math.fsum(xs)/len(xs) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
def transition(rr,anchor,quantity,stable):
    prefix='direction_' if quantity=='direction' else 'gain_sign_'
    sensitive='numerical_sensitive_'+quantity
    ss=[r for r in rr if not stable or not r[sensitive]]
    return dict(n=len(ss),excluded_sensitive=len(rr)-len(ss),counts={a+'->'+b:sum(r[prefix+anchor]==a and r[prefix+'full']==b for r in ss) for a in SIGNS for b in SIGNS})
def concordance(rr,quantity,stable):
    prefix='direction_' if quantity=='direction' else 'gain_sign_'
    sensitive='numerical_sensitive_'+quantity
    ss=[r for r in rr if not stable or not r[sensitive]]
    out={}
    for outcome in (*SIGNS,'all'):
        sub=[r for r in ss if outcome=='all' or r[prefix+'full']==outcome]
        consistent={s:sum(all(r[prefix+a]==s for a in ANCHORS) for r in sub) for s in SIGNS}
        out[outcome]=dict(n=len(sub),all_anchor_positive=consistent['positive'],all_anchor_negative=consistent['negative'],all_anchor_boundary=consistent['boundary'],mixed=len(sub)-sum(consistent.values()))
    return out
def harm_effect(rr,anchor,stable):
    ss=[r for r in rr if r['g_full']<-1e-8 and (not stable or not r['numerical_sensitive_gain'])]
    result=dict(n=len(ss),total_full_harm=math.fsum(-r['g_full'] for r in ss),note='Positive full-minus-anchor gain means gate changes improve the gain relative to this frozen-anchor counterfactual; full result can remain harmful. Summed errors across dependent conditions are descriptive, not population burden.',by_gain_change={})
    for category in ('improved','worsened','near_equal'):
        selected=[r for r in ss if ('improved' if r['g_full']-r['g_'+anchor]>1e-8 else 'worsened' if r['g_full']-r['g_'+anchor]<-1e-8 else 'near_equal')==category]
        result['by_gain_change'][category]=dict(n=len(selected),full_harm_sum=math.fsum(-r['g_full'] for r in selected),anchor_harm_sum=math.fsum(max(-r['g_'+anchor],0) for r in selected),full_minus_anchor_gain=stats([r['g_full']-r['g_'+anchor] for r in selected]),anchor_gain_sign_counts={s:sum(r['gain_sign_'+anchor]==s for r in selected) for s in SIGNS})
    return result
def group(rr):
    return dict(n=len(rr),unique_observations=len({r['observation'] for r in rr}),sensitive_direction=sum(r['numerical_sensitive_direction'] for r in rr),sensitive_gain=sum(r['numerical_sensitive_gain'] for r in rr),transitions={a:{q:{'all':transition(rr,a,q,False),'stable':transition(rr,a,q,True)} for q in ('direction','gain')} for a in ANCHORS},full_harm={a:{'all':harm_effect(rr,a,False),'stable':harm_effect(rr,a,True)} for a in ANCHORS},anchor_concordance={q:{'all':concordance(rr,q,False),'stable':concordance(rr,q,True)} for q in ('direction','gain')},gain_statistics={a:stats([r['g_'+a] for r in rr]) for a in (*ANCHORS,'full')})
def pair_summary(pp):
    return dict(n=len(pp),numerical_sensitive_direction=sum(p['numerical_sensitive_direction'] for p in pp),old32_positive_to_negative=sum(p['old32_positive_to_negative'] for p in pp),full64_positive_to_negative=sum(p['full_positive_to_negative'] for p in pp),anchors={a:dict(positive_to_negative=sum(p[a+'_positive_to_negative'] for p in pp),eta1_direction_counts={s:sum(sign(p['a1_'+a],1e-10)==s for p in pp) for s in SIGNS},eta0_direction_counts={s:sum(sign(p['a0_'+a],1e-10)==s for p in pp) for s in SIGNS},delta_alignment=stats([p['delta_a_'+a] for p in pp])) for a in ANCHORS},delta_full_alignment=stats([p['delta_a_full'] for p in pp]),delta_reference_alignment=stats([p['delta_a_ref'] for p in pp]),delta_residual_alignment=stats([p['delta_a_residual'] for p in pp]),max_alignment_decomposition_error=max([abs(p['delta_a_full']-p['delta_a_ref']-p['delta_a_residual']) for p in pp] or [0]))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    tic=time.monotonic();root=Path(args.root).resolve();out=Path(args.out).resolve()
    assert not out.exists(),'Do not overwrite an analysis result'
    assert root not in out.parents,'Do not write into frozen experiment root'
    complete=json.loads((root/'COMPLETE.json').read_text())
    assert sha(root/'rows.json')==complete['rows_sha256']
    assert sha(root/'summary.json')==complete['summary_sha256']
    rows=json.loads((root/'rows.json').read_text());pairs=json.loads((root/'eta_pairs.json').read_text())
    rowkeys=[(r['id'],r['model'],r['roi'],r['band']) for r in rows]
    assert len(rowkeys)==len(set(rowkeys))
    pairkeys=[(p['family'],p['model'],p['roi'],p['band']) for p in pairs]
    assert len(pairkeys)==len(set(pairkeys))
    for r in rows:
        for a in (*ANCHORS,'full'):
            assert r['direction_'+a]==sign(r['a_'+a],1e-10)
            assert r['gain_sign_'+a]==sign(r['g_'+a],1e-8)
            assert abs(r['g_'+a]-2*r['a_'+a]+r['b_'+a])<1e-12
        assert abs(r['g_full']-r['g_ref']-r['residual_gain_cross']+r['residual_gain_penalty'])<1e-12
    groups={}
    for model in sorted({r['model'] for r in rows}):
        for roi in ('center','full'):
            for band in range(4):
                rr=[r for r in rows if (r['model'],r['roi'],r['band'])==(model,roi,band)]
                groups[f'{model}/{roi}/b{band}']=group(rr)
    mid={}
    for model in sorted({r['model'] for r in rows}):
        pp=[p for p in pairs if p['model']==model and p['roi']=='center' and p['band']==2 and p['template'].startswith('mid_')]
        oldflips=[p for p in pp if p['old32_positive_to_negative']]
        mid[model]=dict(all_fixed_mid_pairs=pair_summary(pp),old32_flip_subset=pair_summary(oldflips),old32_flip_subset_stable=pair_summary([p for p in oldflips if not p['numerical_sensitive_direction']]),full_pair_rows=pp)
    result=dict(scope='Complete existing finite synthetic cohort. Every row retained; overlapping repeated conditions are not independent samples. Fixed-anchor interventions depend on decoder weights and anchor. No standalone feature-error claim.',input_root=str(root),input_sha256={n:sha(root/n) for n in ['rows.json','eta_pairs.json','summary.json','COMPLETE.json']},analysis_source_sha256=sha(__file__),rows=len(rows),groups=groups,mid_eta_direction_change=mid,wall_seconds=time.monotonic()-tic,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)
    assert time.monotonic()-tic<30,'Descriptive analysis exceeded fixed30s budget'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),sha256=sha(out),rows=len(rows),seconds=result['wall_seconds'])))
if __name__=='__main__':main()
