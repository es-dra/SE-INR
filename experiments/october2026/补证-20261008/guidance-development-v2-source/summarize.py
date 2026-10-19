"""CPU-only, frozen-cohort statistics. Partial completion gives no positive claim."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(image_values):
    values=np.asarray(image_values,dtype=np.float64)
    rng=np.random.default_rng(20261008)
    means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),
                ci95=[float(x) for x in np.percentile(means,[2.5,97.5])],
                unit='image; noise settings averaged within image',bootstrap=10000)


def successful_policy_cost(selection,method):
    # Shared final fit is counted once for EACH deploying policy.
    chosen=selection['chosen'][method]
    total=selection['common']['cost']['seconds']
    total+=selection['reconstructions'][chosen]['cost']['seconds']
    if method=='analysis':
        total+=selection['common']['risk_input_cost']['seconds']
        total+=sum(cost['seconds'] for arm in selection['scores'].values() for cost in arm['cost'])
    elif method=='cv':
        total+=sum(cost['seconds'] for arm in selection['cv'].values() for cost in arm['cost'])
    return total


def summarize(root,split='target'):
    cohort_path=root/('selection_cohort_'+split+'.json')
    if not cohort_path.exists():
        return dict(status='incomplete',positive_claims=False,
                    reason='No complete frozen selection cohort; do not summarize selected subsets')
    frozen_path=root/'frozen.json'
    if not frozen_path.exists() or json.loads(frozen_path.read_text()).get('summary_sha256')!=sha(Path(__file__).resolve()):
        raise RuntimeError('Statistics source changed after experiment freeze')
    cohort=json.loads(cohort_path.read_text())
    complete=[]; missing=[]; failed=[]; tampered=[]
    selections={}
    for cid,expected_hash in cohort['cases'].items():
        selection_path=root/'selections'/(cid+'.json')
        if not selection_path.exists() or sha(selection_path)!=expected_hash:
            tampered.append(cid);continue
        selections[cid]=json.loads(selection_path.read_text())
        ep=root/'evaluation'/(cid+'.json')
        if not ep.exists():
            missing.append(cid);continue
        evaluation=json.loads(ep.read_text())
        if evaluation['status']!='complete':
            failed.append(dict(case=cid,status=evaluation['status'],failures=evaluation.get('failures',[])))
            continue
        if evaluation['selection_cohort_sha256']!=sha(cohort_path):
            tampered.append(cid);continue
        complete.append(evaluation)
    if missing or failed or tampered:
        return dict(status='incomplete',positive_claims=False,completed=len(complete),expected=len(cohort['cases']),
                    missing=missing,failed=failed,tampered=tampered,
                    reason='Frozen cohort remains incomplete; no outcome-dependent subset statistics')
    image_ids=sorted({r['case'].split('_')[0] for r in complete})
    def image_means(fn):
        return [np.mean([fn(r) for r in complete if r['case'].startswith(image+'_')]) for image in image_ids]
    result=dict(status='complete',cohort_sha256=sha(cohort_path),n_images=len(image_ids),n_cases=len(complete),methods={})
    for method in ('analysis','cv','default'):
        gains=[r['metrics'][method]['improvement_over_default_db'] for r in complete]
        regrets=[r['metrics'][method]['regret_db'] for r in complete]
        result['methods'][method]=dict(
            gain_over_default=mean_ci(image_means(lambda r:r['metrics'][method]['improvement_over_default_db'])),
            regret_db=dict(mean=float(np.mean(regrets)),median=float(np.median(regrets)),p90=float(np.percentile(regrets,90))),
            severe_degradation_rate=float(np.mean([r['metrics'][method]['severe_degradation'] for r in complete])),
            worst_gain_db=float(min(gains)),
            successful_selection_and_final_seconds=sum(successful_policy_cost(selections[r['case']],method) for r in complete))
    comparison=mean_ci(image_means(lambda r:r['metrics']['analysis']['psnr']-r['metrics']['cv']['psnr']))
    result['analysis_minus_cv']=comparison
    result['by_noise']={}
    for noise in (.005,.01,.02):
        suffix=('_n'+f'{noise:g}').replace('.','p')
        rows=[r for r in complete if r['case'].endswith(suffix)]
        result['by_noise'][str(noise)]={method:mean_ci([r['metrics'][method]['improvement_over_default_db'] for r in rows])
                                      for method in ('analysis','cv')}
    analysis=result['methods']['analysis'];cv=result['methods']['cv']
    recorded_failures=sum(len(s.get('failures',[])) for s in selections.values())
    result['recorded_selection_failure_attempts']=recorded_failures
    result['claims']=dict(
        improves_default=analysis['gain_over_default']['ci95'][0]>0 and analysis['severe_degradation_rate']<=.05,
        better_than_cv_at_no_greater_cost=(comparison['ci95'][0]>0
            and analysis['severe_degradation_rate']<=cv['severe_degradation_rate']
            and analysis['successful_selection_and_final_seconds']<=cv['successful_selection_and_final_seconds']
            and recorded_failures==0))
    result['cost_caveat']='Policy timings count shared final fits once for each policy; calibration overhead and actual attempts/failed work are separately disclosed. No claim of GPU busy-time measurement.'
    if (root/'resources.json').exists():
        result['actual_resources']=json.loads((root/'resources.json').read_text())
        result['calibration_invocation_wall_seconds']=sum(v['ended_unix']-v['started_unix']
            for v in result['actual_resources']['invocations'] if v['phase']=='calibrate')
    result['actual_attempts']={cid:s.get('attempts',[]) for cid,s in selections.items()}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=('development','target'),default='target')
    args=parser.parse_args()
    result=summarize(args.output,args.split)
    dest=args.output/('summary_'+args.split+'.json')
    fd,tmp=tempfile.mkstemp(prefix='.'+dest.name,dir=dest.parent)
    with os.fdopen(fd,'w') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,dest)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
