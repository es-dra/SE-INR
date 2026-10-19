"""Independent CPU audit of frozen artifacts; never calls the GP/selection code."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
cohort = read(ROOT / 'selection_cohort_target.json')
expected = {f'{i:04d}_n{n}' for i in range(813, 825) for n in ('0p005', '0p01', '0p02')}
assert set(cohort['cases']) == expected
frozen = read(ROOT / 'frozen.json')
for name, key in [('core.py','core_sha256'),('experiment.py','experiment_sha256'),
                  ('protocol.json','protocol_sha256'),('summarize.py','summary_sha256')]:
    assert sha(ROOT.parent/'guidance'/name) == frozen[key]
assert (ROOT/'COMPLETE').is_file()
arms = read(ROOT.parent/'guidance/protocol.json')['candidate_arms']
rows = []; records = []; pixel_errors = []; formula_errors = []
for cid in sorted(expected):
    sp = ROOT/'selections'/f'{cid}.json'; ep = ROOT/'evaluation'/f'{cid}.json'
    s, e = read(sp), read(ep)
    assert sha(sp) == cohort['cases'][cid]
    assert s['frozen_sha256'] == sha(ROOT/'frozen.json')
    assert s['common']['reference_sha256'] == sha(ROOT/'calibration.npz')
    assert s['observations_sha256'] == sha(ROOT/'observations'/f'{cid}.npz')
    assert e['selection_cohort_sha256'] == sha(ROOT/'selection_cohort_target.json')
    assert s['status'] == e['status'] == 'complete' and not s['failures'] and not e['failures']
    assert set(s['scores']) == set(s['cv']) == set(e['candidates']) == set(arms)
    assert s['chosen']['analysis'] == min(arms, key=lambda a:s['scores'][a]['mean'])
    assert s['chosen']['cv'] == min(arms, key=lambda a:s['cv'][a]['mean'])
    assert s['chosen']['default'] == 'power_2'
    truth_path=ROOT/'evaluation_truth'/f'{cid}.npz'
    assert sha(truth_path) == e['truth_sha256'] == read(ROOT/'observations'/f'{cid}.json')['truth_sha256']
    with np.load(truth_path,allow_pickle=False) as f: truth=f['x'].astype(np.float64)
    cached={}
    for arm, item in s['reconstructions'].items():
        assert sha(ROOT/item['path']) == item['sha256']
        with np.load(ROOT/item['path'],allow_pickle=False) as f: pred=f['x'].astype(np.float64)
        mse=float(np.mean((pred[:,8:-8,8:-8]-truth[:,8:-8,8:-8])**2))
        cached[arm]=-10*np.log10(mse)
        pixel_errors.append(abs(mse-e['candidates'][arm]['mse']))
        assert abs(cached[arm]-e['candidates'][arm]['psnr']) < 1e-10
    psnr={a:float(-10*np.log10(e['candidates'][a]['mse'])) for a in arms}
    for arm in arms:
        formula_errors.append(abs(psnr[arm]-e['candidates'][arm]['psnr']))
        records.append(e['candidates'][arm]['numerics'])
    for section in ('scores','cv'):
        for arm in arms: records.extend(s[section][arm]['numerics'])
    oracle=max(arms,key=lambda a:psnr[a]); assert oracle==e['oracle']
    gains={m:cached[a]-cached['power_2'] for m,a in s['chosen'].items()}
    regrets={m:psnr[oracle]-cached[a] for m,a in s['chosen'].items()}
    for m in gains:
        assert abs(gains[m]-e['metrics'][m]['improvement_over_default_db'])<1e-10
        assert abs(regrets[m]-e['metrics'][m]['regret_db'])<1e-10
    rows.append(dict(case=cid,chosen=s['chosen'],oracle=oracle,gain=gains,regret=regrets))
assert all(n['converged'] and n['true_relative_residual']<1e-5 for n in records)
resources=read(ROOT/'resources.json')
score_start=min(v['started_unix'] for v in resources['invocations'] if v['phase']=='score')
last_selection=max(read(ROOT/'selections'/f'{cid}.json')['completed_unix'] for cid in expected)
assert score_start>last_selection
def ci(values):
    x=np.asarray(values);rng=np.random.default_rng(20261008)
    b=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()),ci95=np.percentile(b,[2.5,97.5]).tolist())
images=sorted({r['case'].split('_')[0] for r in rows})
stats={m:ci([np.mean([r['gain'][m] for r in rows if r['case'].startswith(i+'_')]) for i in images])
       for m in ('analysis','cv','default')}
stats['analysis_minus_cv']=ci([np.mean([r['gain']['analysis']-r['gain']['cv'] for r in rows if r['case'].startswith(i+'_')]) for i in images])
summary=read(ROOT/'summary_target.json')
for m in ('analysis','cv','default'):
    assert abs(stats[m]['mean']-summary['methods'][m]['gain_over_default']['mean'])<1e-10
    assert np.max(np.abs(np.asarray(stats[m]['ci95'])-summary['methods'][m]['gain_over_default']['ci95']))<1e-10
assert np.max(np.abs(np.asarray(stats['analysis_minus_cv']['ci95'])-summary['analysis_minus_cv']['ci95']))<1e-10
out=dict(status='pass',n_images=12,n_cases=36,cohort_sha256=sha(ROOT/'selection_cohort_target.json'),
    primary_pixel_mse_max_absolute_difference=max(pixel_errors),cached_predictions_checked=len(pixel_errors),
    all_candidate_mse_to_psnr_max_difference=max(formula_errors),numerical_records_checked=len(records),
    max_true_relative_residual=max(n['true_relative_residual'] for n in records),max_iterations=max(n['iterations'] for n in records),
    selection_before_first_scoring_seconds=score_start-last_selection,statistics_from_cached_predictions=stats,
    negative_gain_cases=[r for r in rows if r['gain']['analysis']<0],
    worst_analysis_oracle_regret=max(rows,key=lambda r:r['regret']['analysis']),
    scope='All cached chosen predictions independently scored from RGB pixels; uncached candidate predictions were not retained, so their PSNR/oracle audited from saved MSE and records, without independent GP rerun.',
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'frozen.json',ROOT/'summary_target.json',ROOT/'resources.json',ROOT/'selection_cohort_target.json']})
(ROOT/'独立图像复核.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('sources','scope')},ensure_ascii=False,indent=2))
