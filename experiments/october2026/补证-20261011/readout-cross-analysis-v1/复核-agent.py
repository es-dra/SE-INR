"""Independent NumPy reconstruction. Never import the original analysis."""
import argparse
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-run', required=True)
    parser.add_argument('--cross-run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--launch-record', required=True)
    a = parser.parse_args()
    launch = json.loads(Path(a.launch_record).read_text())
    assert 0 < launch['absolute_deadline_epoch'] - launch['start_epoch'] <= 120
    start = time.monotonic()
    destination = Path(a.out)
    assert not destination.exists()
    report = {'reviewer': 'probe_review independent reconstruction', 'status': 'failed',
              'analysis_imported': False, 'training_runs': 0, 'inference_runs': 0}

    def guard():
        if time.monotonic() - start > 120 or time.time() > launch['absolute_deadline_epoch']:
            raise TimeoutError('independent review fixed deadline exceeded')

    try:
        import numpy as np
        base, cross = Path(a.base_run), Path(a.cross_run)
        assert (cross / 'COMPLETE').exists()
        assert json.loads((cross / 'resources.json').read_text())['status'] == 'complete'
        f = json.loads((base / 'prediction_freeze.json').read_text())
        assert f['cohort_sha256'] == digest(base / 'prediction_cohort.json')
        cohort = json.loads((base / 'prediction_cohort.json').read_text())
        assert f['files'] == len(cohort) == 152
        for n, h in cohort.items():
            guard()
            assert digest(base / n) == h
        cases = [c for c in json.loads((base / 'cases.json').read_text()) if c['kind'] == 'synthetic']
        assert len(cases) == 72 and len({c['id'] for c in cases}) == 72
        fixed_cases = {(n, k, d, p) for n in (16, 32) for k in (4, 6, 10, 14, 18, 26)
                       for d in ((1, 0), (0, 1), (1, 1)) for p in (0, 1)}
        assert {(c['n'], c['k'], tuple(c['direction']), c['phase_index']) for c in cases} == fixed_cases
        assert all(abs(c['phase'] - (0 if c['phase_index'] == 0 else np.pi / 2)) < 1e-14 for c in cases)
        rows = json.loads((cross / '2304rows.json').read_text())
        summary = json.loads((cross / 'summary.json').read_text())
        assert len(rows) == summary['rows'] == 2304
        key = lambda r: (r['case'], r['model'], r['target'], r['hint'])
        saved = {key(r): r for r in rows}
        assert len(saved) == len(rows)
        footprints = {'square1': (1, 1), 'vertical': (2, 1), 'horizontal': (1, 2), 'square2': (2, 2)}
        models = ('lte', 'sc-inr')
        indices = np.arange(16, 48, dtype=float) + .5
        xx, yy = np.meshgrid(indices / 64, indices / 64, indexing='xy')
        positions = np.column_stack([yy.ravel(), xx.ravel()])
        expected = {}
        maximum = 0.0
        for c in cases:
            guard()
            dy, dx = c['direction']
            ky, kx = c['k'] * dy, c['k'] * dx
            angle = (positions[:, 0] * ky + positions[:, 1] * kx) * (2 * np.pi) + c['phase']
            cosine = np.exp(1j * angle).real
            raw_truth = .5 + .25 * cosine
            assert np.max(abs(raw_truth - np.load(base / 'truth' / (c['id'] + '.npy')))) < 1e-12
            for model in models:
                with np.load(base / 'predictions' / (c['id'] + '__' + model + '.npz')) as prediction:
                    point = np.asarray(prediction['point'], dtype=np.float64)
                    assert point.shape == (1024, 3)
                    for target_name, (height, width) in footprints.items():
                        transfer = float(np.sinc(ky * height / 64) * np.sinc(kx * width / 64))
                        truth = np.repeat((.5 + .25 * transfer * cosine)[:, None], 3, axis=1)
                        signal = np.sum((truth - .5) ** 2) / truth.size
                        point_error = point - truth
                        point_loss = float(np.sum(point_error ** 2) / truth.size)
                        for hint in footprints:
                            output = np.asarray(prediction['native_' + hint], dtype=np.float64)
                            change = output - point
                            error = output - truth
                            loss = float(np.sum(error ** 2) / truth.size)
                            gain = (point_loss - loss) / signal
                            energy = float(np.vdot(change, change).real / truth.size / signal)
                            align = float(2 * np.vdot(point_error, change).real / truth.size / signal)
                            r = dict(case=c['id'], model=model, n=c['n'], k=c['k'], direction=c['direction'],
                                     phase_index=c['phase_index'], target=target_name, hint=hint,
                                     matched=target_name == hint, band=1 if c['k'] < 8 else 2 if c['k'] < 16 else 3,
                                     signal_energy=float(signal), T=transfer, raw_mse=loss, point_raw_mse=point_loss,
                                     gain=gain, change_energy=energy, alignment=align,
                                     identity_residual=gain + energy + align)
                            k = key(r)
                            actual = saved[k]
                            for name in ['n', 'k', 'direction', 'phase_index', 'band', 'matched']:
                                assert actual[name] == r[name], (k, name)
                            for name in ['signal_energy', 'T', 'raw_mse', 'point_raw_mse', 'gain',
                                         'change_energy', 'alignment', 'identity_residual']:
                                delta = abs(actual[name] - r[name])
                                maximum = max(maximum, delta)
                                assert delta <= 1e-10 * (1 + abs(r[name])), (k, name, delta)
                            expected[k] = r
        assert len(expected) == 2304 and set(expected) == set(saved)
        target_results = {}
        for c in cases:
            for model in models:
                for target in footprints:
                    block = [expected[(c['id'], model, target, hint)] for hint in footprints]
                    minimum = min(b['raw_mse'] for b in block)
                    ties = [b['hint'] for b in block if b['raw_mse'] <= minimum + 1e-12]
                    matched = expected[(c['id'], model, target, target)]
                    t = dict(case=c['id'], model=model, n=c['n'], k=c['k'], direction=c['direction'],
                             phase_index=c['phase_index'], target=target, matched_best=target in ties,
                             best_hints=ties, matched_raw_regret=matched['raw_mse'] - minimum)
                    if target in ('horizontal', 'vertical'):
                        t['equal_area_raw_mse_contrast'] = expected[(c['id'], model, target, 'vertical')]['raw_mse'] - expected[(c['id'], model, target, 'horizontal')]['raw_mse']
                    target_results[(c['id'], model, target)] = t
        targets_saved = {(t['case'], t['model'], t['target']): t for t in summary['target_scores']}
        assert len(targets_saved) == len(summary['target_scores']) == len(target_results) == 576
        for k, t in target_results.items():
            u = targets_saved[k]
            for name in ['case', 'model', 'n', 'k', 'direction', 'phase_index', 'target', 'matched_best', 'best_hints']:
                assert t[name] == u[name], (k, name)
            for name in ['matched_raw_regret', 'equal_area_raw_mse_contrast']:
                if name in t:
                    assert abs(t[name] - u[name]) < 1e-10, (k, name)
        groups = {(g['model'], g['n'], g['phase_index']): g for g in summary['groups']}
        assert len(groups) == len(summary['groups']) == 8
        for model in models:
            for n in (16, 32):
                for phase in (0, 1):
                    guard()
                    rs = [r for r in expected.values() if (r['model'], r['n'], r['phase_index']) == (model, n, phase)]
                    ts = [t for t in target_results.values() if (t['model'], t['n'], t['phase_index']) == (model, n, phase)]
                    g = groups[(model, n, phase)]
                    assert g['rows'] == len(rs) == 288 and g['targets'] == len(ts) == 72
                    assert g['matched_best_count'] == sum(t['matched_best'] for t in ts)
                    assert abs(g['matched_best_fraction'] - sum(t['matched_best'] for t in ts) / len(ts)) < 1e-12
                    regrets = [t['matched_raw_regret'] for t in ts]
                    assert abs(g['matched_raw_regret_mean'] - sum(regrets) / len(regrets)) < 1e-10
                    assert abs(g['matched_raw_regret_max'] - max(regrets)) < 1e-10
                    assert len(g['signs']) == len(g['equal_area']) == 6
                    for sign in g['signs']:
                        sub = [r for r in rs if r['band'] == sign['band'] and (not sign['matched_only'] or r['matched'])]
                        positive = sum(r['gain'] > .001 for r in sub)
                        negative = sum(r['gain'] < -.001 for r in sub)
                        assert sign['count'] == len(sub) and sign['positive'] == positive and sign['negative'] == negative
                        assert sign['inactive'] == len(sub) - positive - negative and sign['both_signs'] == bool(positive and negative)
                    for area in g['equal_area']:
                        values = [t['equal_area_raw_mse_contrast'] for t in ts if t['target'] == area['target'] and t['direction'] == area['direction']]
                        assert area['count'] == len(values) == 6
                        for name, value in [('raw_mse_contrast_mean', sum(values) / len(values)),
                                            ('raw_mse_contrast_min', min(values)), ('raw_mse_contrast_max', max(values))]:
                            assert abs(area[name] - value) < 1e-10
        pair_keys = set()
        for pair in summary['equal_area_directional_pairs']:
            k = (pair['model'], pair['n'], pair['k'], pair['phase_index'], pair['target'])
            assert k not in pair_keys
            pair_keys.add(k)
            corresponding = [t for t in target_results.values() if
                             (t['model'], t['n'], t['k'], t['phase_index'], t['target']) == k]
            vertical = next(t for t in corresponding if t['direction'] == [1, 0])
            horizontal = next(t for t in corresponding if t['direction'] == [0, 1])
            y, x = vertical['equal_area_raw_mse_contrast'], horizontal['equal_area_raw_mse_contrast']
            assert abs(pair['contrast_y'] - y) < 1e-10 and abs(pair['contrast_x'] - x) < 1e-10
            assert abs(pair['directional_difference'] - (y - x)) < 1e-10
        assert len(pair_keys) == 96
        guard()
        report.update(status='passed', rows_recomputed=2304, target_scores_recomputed=576,
                      groups_recomputed=8, equal_area_directional_pairs_recomputed=96,
                      maximum_numeric_difference=maximum,
                      prediction_cohort_sha256=f['cohort_sha256'],
                      cross_rows_sha256=digest(cross / '2304rows.json'), cross_summary_sha256=digest(cross / 'summary.json'),
                      reviewer_source_sha256=digest(Path(__file__)),
                      boundary='Complete algebra and scoring replication only; no independent model inference, physical-point or generalization proof. Oracle target-aware scoring, correlated synthetic grid and unsupported query limits retained.')
    except Exception:
        report['failure'] = traceback.format_exc()
        raise
    finally:
        r = resource.getrusage(resource.RUSAGE_SELF)
        report.update(wall_seconds=time.time() - launch['start_epoch'], in_script_wall_seconds=time.monotonic() - start,
                      cpu_user_seconds=r.ru_utime, cpu_system_seconds=r.ru_stime, maxrss_kib=r.ru_maxrss)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
