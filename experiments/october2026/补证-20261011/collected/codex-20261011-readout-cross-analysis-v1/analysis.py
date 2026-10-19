"""Frozen crossed scoring of existing predictions. No model imports or inference."""
import argparse
import hashlib
import json
import os
import resource
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base-source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--launch-record', required=True)
    a = ap.parse_args()
    launch = json.loads(Path(a.launch_record).read_text())
    limit = P['resource']['wall_and_cumulative_budget_seconds']
    assert 0 < launch['absolute_deadline_epoch'] - launch['start_epoch'] <= limit
    start = time.monotonic()
    out = Path(a.out)
    assert not out.exists(), 'no resume'
    out.mkdir(parents=True)
    run, base = Path(a.run), Path(a.base_source)
    status = 'failed'

    def guard():
        if time.time() > launch['absolute_deadline_epoch'] or time.monotonic() - start > limit:
            raise TimeoutError('fixed 120s deadline exceeded')

    try:
        guard()
        assert sha(base / 'experiment.py') == P['base_source_sha256']
        assert sha(base / 'protocol.json') == P['base_protocol_sha256']
        frozen_base = json.loads((run / 'frozen.json').read_text())
        assert frozen_base['protocol_sha256'] == P['base_protocol_sha256']
        assert frozen_base['source_sha256']['experiment.py'] == P['base_source_sha256']
        freeze = json.loads((run / 'prediction_freeze.json').read_text())
        assert freeze['cohort_sha256'] == sha(run / 'prediction_cohort.json')
        cohort = json.loads((run / 'prediction_cohort.json').read_text())
        assert freeze['files'] == len(cohort)
        assert len(cohort) == 152, '72 synthetic x2 models +2 natural x4 models must all be frozen'
        for filename, expected in cohort.items():
            guard()
            assert sha(run / filename) == expected
        manifest = json.loads((run / 'input_manifest.json').read_text())
        for filename, expected in manifest.items():
            guard()
            assert sha(run / filename) == expected
        cases = [c for c in json.loads((run / 'cases.json').read_text()) if c['kind'] == 'synthetic']
        C = P['cohort']
        expected = {(n, k, tuple(d), pi) for n in C['input_sizes'] for k in C['frequencies']
                    for d in C['directions'] for pi in range(len(C['phases']))}
        actual = {(c['n'], c['k'], tuple(c['direction']), c['phase_index']) for c in cases}
        assert actual == expected and len(cases) == C['synthetic_inputs']
        assert len({c['id'] for c in cases}) == len(cases)
        for c in cases:
            assert abs(c['phase'] - C['phases'][c['phase_index']]) < 1e-14
            for model in C['models']:
                assert 'predictions/' + c['id'] + '__' + model + '.npz' in cohort
        write(out / 'frozen.json', {'source_sha256': sha(HERE / 'analysis.py'),
              'protocol_sha256': sha(HERE / 'protocol.json'), 'base_prediction_cohort_sha256': freeze['cohort_sha256'],
              'launch': launch, 'budget_seconds': limit, 'nice': os.getpriority(os.PRIO_PROCESS, 0),
              'input_manifest_sha256': sha(run / 'input_manifest.json'), 'base_source': str(base),
              'base_run': str(run), 'inference_runs': 0, 'training_runs': 0})
        # Import after the recorded clock; external launch should cap BLAS/OpenMP to four threads.
        import numpy as np
        v = (np.arange(C['hr_reference'], dtype=float) + .5) / C['hr_reference']
        v = v[C['query_crop'][0]:C['query_crop'][1]]
        q = np.stack(np.meshgrid(v, v, indexing='ij'), -1).reshape(-1, 2)
        mse = lambda x, y: float(np.mean((x - y) ** 2))
        rows, targets = [], []
        for c in cases:
            guard()
            ky, kx = np.array(c['direction'], dtype=float) * c['k']
            cosine = np.cos(2 * np.pi * (ky * q[:, 0] + kx * q[:, 1]) + c['phase'])
            point_truth = np.load(run / 'truth' / (c['id'] + '.npy'))
            assert np.max(abs(point_truth - (.5 + .25 * cosine))) <= 1e-12
            for model in C['models']:
                with np.load(run / 'predictions' / (c['id'] + '__' + model + '.npz')) as z:
                    point = z['point'].astype('float64')
                    assert point.shape == (len(q), 3) and np.isfinite(point).all()
                    for target_name, factors in C['targets'].items():
                        wh = np.array(factors, dtype=float) / C['hr_reference']
                        T = float(np.sinc(ky * wh[0]) * np.sinc(kx * wh[1]))
                        target = (.5 + .25 * T * cosine)[:, None]
                        S = float(np.mean((target - .5) ** 2))
                        assert S > 1e-8
                        point_mse = mse(point, target)
                        block = []
                        for hint in C['hints']:
                            pred = z['native_' + hint].astype('float64')
                            assert pred.shape == point.shape and np.isfinite(pred).all()
                            err = mse(pred, target)
                            d, e = pred - point, point - target
                            energy = float(np.mean(d * d)) / S
                            alignment = 2 * float(np.mean(e * d)) / S
                            gain = (point_mse - err) / S
                            row = dict(case=c['id'], model=model, n=c['n'], k=c['k'],
                                       direction=c['direction'], phase_index=c['phase_index'],
                                       band=1 if c['k'] < 8 else 2 if c['k'] < 16 else 3,
                                       target=target_name, hint=hint, matched=target_name == hint,
                                       signal_energy=S, T=T, raw_mse=err, point_raw_mse=point_mse,
                                       gain=gain, change_energy=energy, alignment=alignment,
                                       identity_residual=gain + energy + alignment)
                            rows.append(row)
                            block.append(row)
                        best = min(r['raw_mse'] for r in block)
                        matched = next(r for r in block if r['matched'])
                        ties = [r['hint'] for r in block if r['raw_mse'] <= best + P['mse_tie_tolerance']]
                        tv = dict(case=c['id'], model=model, n=c['n'], k=c['k'], direction=c['direction'],
                                  phase_index=c['phase_index'], target=target_name,
                                  matched_best=target_name in ties, best_hints=ties,
                                  matched_raw_regret=matched['raw_mse'] - best)
                        if target_name in ['vertical', 'horizontal']:
                            tv['equal_area_raw_mse_contrast'] = next(r['raw_mse'] for r in block if r['hint'] == 'vertical') - next(r['raw_mse'] for r in block if r['hint'] == 'horizontal')
                        targets.append(tv)
        assert len(rows) == C['expected_rows']
        assert len(targets) == C['synthetic_inputs'] * len(C['models']) * len(C['targets'])
        guard()
        write(out / '2304rows.json', rows)
        groups = []
        for model in C['models']:
            for n in C['input_sizes']:
                for pi in range(2):
                    subset = [r for r in rows if r['model'] == model and r['n'] == n and r['phase_index'] == pi]
                    ts = [r for r in targets if r['model'] == model and r['n'] == n and r['phase_index'] == pi]
                    g = dict(model=model, n=n, phase_index=pi, phase_role='calibration_description' if pi == 0 else 'held_phase_description',
                             rows=len(subset), targets=len(ts), matched_best_count=sum(t['matched_best'] for t in ts),
                             matched_best_fraction=float(np.mean([t['matched_best'] for t in ts])),
                             matched_raw_regret_mean=float(np.mean([t['matched_raw_regret'] for t in ts])),
                             matched_raw_regret_max=max(t['matched_raw_regret'] for t in ts), signs=[], equal_area=[])
                    for matched_only in [True, False]:
                        for band in [1, 2, 3]:
                            rs = [r for r in subset if r['band'] == band and (not matched_only or r['matched'])]
                            pos = sum(r['gain'] > P['sign_margin'] for r in rs)
                            neg = sum(r['gain'] < -P['sign_margin'] for r in rs)
                            g['signs'].append(dict(matched_only=matched_only, band=band, count=len(rs), positive=pos,
                                                  negative=neg, inactive=len(rs) - pos - neg, both_signs=bool(pos and neg)))
                    for target in ['vertical', 'horizontal']:
                        for direction in C['directions']:
                            et = [t for t in ts if t['target'] == target and t['direction'] == direction]
                            values = [t['equal_area_raw_mse_contrast'] for t in et]
                            g['equal_area'].append(dict(target=target, direction=direction, count=len(values),
                                                       raw_mse_contrast_mean=float(np.mean(values)),
                                                       raw_mse_contrast_min=min(values), raw_mse_contrast_max=max(values)))
                    groups.append(g)
        directional_pairs = []
        for t in targets:
            if t['direction'] != [1, 0] or t['target'] not in ['vertical', 'horizontal']:
                continue
            counterpart = next(u for u in targets if u['model'] == t['model'] and u['n'] == t['n'] and u['k'] == t['k'] and u['phase_index'] == t['phase_index'] and u['target'] == t['target'] and u['direction'] == [0, 1])
            directional_pairs.append(dict(model=t['model'], n=t['n'], k=t['k'], phase_index=t['phase_index'],
                                          target=t['target'], contrast_y=t['equal_area_raw_mse_contrast'],
                                          contrast_x=counterpart['equal_area_raw_mse_contrast'],
                                          directional_difference=t['equal_area_raw_mse_contrast'] - counterpart['equal_area_raw_mse_contrast']))
        write(out / 'summary.json', dict(rows=len(rows), groups=groups, target_scores=targets,
              equal_area_directional_pairs=directional_pairs,
              max_identity_residual=max(abs(r['identity_residual']) for r in rows),
              scope=P['scope'], fitted_coefficients=0, inference_runs=0, training_runs=0,
              interpretation='Target-aware scoring oracle and unsupported-query diagnostic; no mechanism or deployable selector claim.'))
        guard()
        (out / 'COMPLETE').write_text('complete; scoring saved frozen predictions only\n')
        status = 'complete'
    except Exception:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        r = resource.getrusage(resource.RUSAGE_SELF)
        write(out / 'resources.json', dict(status=status, wall_seconds=time.time() - launch['start_epoch'],
              in_script_wall_seconds=time.monotonic() - start, cpu_user_seconds=r.ru_utime,
              cpu_system_seconds=r.ru_stime, maxrss_kib=r.ru_maxrss, inference_runs=0, training_runs=0, gpu_calls=0))


if __name__ == '__main__':
    main()
