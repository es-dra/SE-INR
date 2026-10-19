"""Independent NumPy audit of frozen decoder diagnostics; no model inference.

CPU1, nice10, fixed 120s wall/cumulative limit. Never overwrites an audit.
Only saved eight-query probes are independently differentiated; all pixels are
used to independently rescore the complete finite cohort.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
LIMIT = 120.0
SIGN = ('positive', 'negative', 'boundary')
MODES = ('ref', 'mid', 'native', 'full')
MAXERR = {}
COUNTS = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def close(actual, expected, name, tol=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    MAXERR[name] = max(MAXERR.get(name, 0.0), error)
    COUNTS[name] = COUNTS.get(name, 0) + 1
    assert error <= tol, (name, error, tol)


def compare(actual, expected, name):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (name, set(actual) ^ set(expected))
        for k, v in expected.items():
            compare(actual[k], v, name + '/' + k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), name
        for a, b in zip(actual, expected):
            compare(a, b, name)
    elif isinstance(expected, bool) or isinstance(expected, (str, int)):
        assert actual == expected, (name, actual, expected)
    elif isinstance(expected, float):
        close(actual, expected, name.split('/')[0] + '_float')
    else:
        assert actual == expected, name


def classify(value, threshold):
    if abs(value) <= threshold:
        return 'boundary'
    return 'positive' if value > 0 else 'negative'


def components(array, box):
    lo, hi = box
    x = np.array(array[lo:hi, lo:hi], dtype=np.float64)
    if x.ndim == 2:
        x = np.stack((x, x, x), axis=2)
    n = hi - lo
    freq = np.arange(n, dtype=float) / n
    freq[freq >= .5] -= 1
    radius = np.maximum(np.abs(freq[:, None]), np.abs(freq[None, :]))
    spectrum = np.fft.fft2(x, axes=(0, 1))
    result = [x]
    for left, right in ((0, .125), (.125, .25), (.25, .5001)):
        keep = (radius != 0) & (radius >= left) & (radius < right)
        filtered = spectrum.copy()
        filtered[~keep] = 0
        result.append(np.fft.ifft2(filtered, axes=(0, 1)).real)
    return result


def dense_numpy(z, delta, layers):
    h = z.copy()
    slope = delta.copy()
    for layer, (w, b) in enumerate(layers):
        pre = h @ w.T + b
        slope = slope @ w.T
        if layer == len(layers) - 1:
            return pre, slope
        active = pre > 0
        h = np.where(active, pre, 0.)
        slope = np.where(active, slope, 0.)
    raise AssertionError('empty decoder')


def row_key(row):
    return (row['id'], row['model'], row['roi'], row['band'])


def calculate_rows(case, model, truth, planes, rois):
    result = []
    for roi, box in rois.items():
        target = components(truth, box)
        p = planes[roi]
        for band in range(4):
            error = target[band] - p['point64'][band]
            r = dict(case, model=model, roi=roi, band=band,
                     reference_mse=float(np.mean(error ** 2)))
            for mode in MODES:
                step = p['d_' + mode][band]
                alignment = float(np.einsum('ijk,ijk->', error, step) / error.size)
                energy = float(np.sum(step ** 2) / step.size)
                gain = 2 * alignment - energy
                r.update({f'a_{mode}': alignment, f'b_{mode}': energy,
                          f'g_{mode}': gain,
                          f'direction_{mode}': classify(alignment, 1e-10),
                          f'gain_sign_{mode}': classify(gain, 1e-8)})
                # Direct MSE subtraction is independent of the gain formula.
                direct = np.mean(error ** 2) - np.mean((error - step) ** 2)
                close(gain, direct, 'direct_gain_identity')
            error32 = target[band] - p['point'][band]
            step32 = p['native'][band] - p['point'][band]
            a32 = float(np.mean(error32 * step32))
            g32 = float(2 * a32 - np.mean(step32 ** 2))
            residual = p['d_full'][band] - p['d_ref'][band]
            ra = float(np.mean(error * residual))
            rb = float(np.mean(residual ** 2))
            cross = float(2 * np.mean((error - p['d_ref'][band]) * residual))
            r.update(old_float32_alignment=a32, old_float32_gain=g32,
                     numerical_sensitive_direction=classify(a32, 1e-10) != r['direction_full'],
                     numerical_sensitive_gain=classify(g32, 1e-8) != r['gain_sign_full'],
                     residual_alignment=ra, residual_energy=rb,
                     residual_gain_cross=cross, residual_gain_penalty=rb,
                     gain_decomposition_residual=r['g_full'] - r['g_ref'] - cross + rb)
            close(r['gain_decomposition_residual'], 0., 'residual_gain_identity')
            close(r['a_full'] - r['a_ref'], ra, 'residual_alignment_identity')
            result.append(r)
    return result


def aggregate(rows, protocol, cases):
    groups = {}
    for model in protocol['models']:
        for roi in protocol['evaluation']['ROIs']:
            for band in range(4):
                subset = [r for r in rows if (r['model'], r['roi'], r['band']) == (model, roi, band)]
                bad = [r for r in subset if r['direction_full'] == 'negative' and not r['numerical_sensitive_direction']]
                g = dict(n=len(subset), full_direction_negative=len(bad),
                         ref_direction_among_full_negative={s: sum(r['direction_ref'] == s for r in bad) for s in SIGN},
                         all_anchor_negative_among_full_negative=sum(all(r['direction_' + m] == 'negative' for m in MODES[:3]) for r in bad),
                         all_anchor_positive_among_full_negative=sum(all(r['direction_' + m] == 'positive' for m in MODES[:3]) for r in bad),
                         mean_gain={m: float(np.mean([r['g_' + m] for r in subset])) for m in MODES},
                         numerical_sensitive_direction=sum(r['numerical_sensitive_direction'] for r in subset),
                         numerical_sensitive_gain=sum(r['numerical_sensitive_gain'] for r in subset),
                         transition_note='Unqualified transition matrices include sensitive rows; stable matrices below exclude corresponding sensitive rows.')
                for label, prefix in (('direction', 'direction_'), ('gain', 'gain_sign_')):
                    for stable in (False, True):
                        use = [r for r in subset if not (stable and r['numerical_sensitive_' + label])]
                        counts = {a + '->' + b: 0 for a in SIGN for b in SIGN}
                        for r in use:
                            counts[r[prefix + 'ref'] + '->' + r[prefix + 'full']] += 1
                        g[label + '_transition' + ('_stable' if stable else '')] = counts
                groups[f'{model}/{roi}/b{band}'] = g
    by_eta = {(r['family'], r['model'], r['roi'], r['band'], r['eta']): r for r in rows}
    assert len(by_eta) == len(rows)
    pairs = []
    for a in rows:
        if a['eta'] != 0:
            continue
        b = by_eta[(a['family'], a['model'], a['roi'], a['band'], 1)]
        p = {k: a[k] for k in ('family', 'template', 'phase', 'shift', 'model', 'roi', 'band')}
        for label in ('direction', 'gain'):
            p['numerical_sensitive_' + label] = a['numerical_sensitive_' + label] or b['numerical_sensitive_' + label]
        p['old32_positive_to_negative'] = a['old_float32_alignment'] > 1e-10 and b['old_float32_alignment'] < -1e-10
        for mode in MODES:
            p[mode + '_positive_to_negative'] = a['a_' + mode] > 1e-10 and b['a_' + mode] < -1e-10
            p['a0_' + mode], p['a1_' + mode] = a['a_' + mode], b['a_' + mode]
            p['delta_a_' + mode] = b['a_' + mode] - a['a_' + mode]
        p['delta_a_residual'] = b['residual_alignment'] - a['residual_alignment']
        pairs.append(p)
    oldmid = {}
    for model in protocol['models']:
        subset = [r for r in pairs if r['model'] == model and r['roi'] == 'center' and r['band'] == 2 and r['template'].startswith('mid_')]
        oldmid[model] = {k: sum(r[k] for r in subset) for k in ('old32_positive_to_negative', 'full_positive_to_negative', 'numerical_sensitive_direction')}
    summary = dict(groups=groups, midband_center_reproduction=oldmid, rows=len(rows),
                   conditions=len(cases), scope=protocol['scope'], training=0, GPU=0)
    return summary, pairs


def execute(args, check):
    root, source = args.run.resolve(), args.source.resolve()
    protocol = read(HERE / 'protocol.json')
    freeze = read(root / 'input_freeze.json')
    assert freeze['source_sha256'] == digest(HERE / 'run.py')
    assert freeze['protocol_sha256'] == digest(HERE / 'protocol.json')
    which = 'preflight' if freeze['preflight'] else 'formal'
    for filename, expected in protocol['source_manifests'][which].items():
        check()
        assert digest(source / filename) == expected, filename
    for filename in ('input_manifest.json', 'prediction_cohort.json'):
        for relative, expected in read(source / filename).items():
            check()
            assert digest(source / relative) == expected, relative
    assert freeze['input_manifest_sha256'] == digest(source / 'input_manifest.json')
    assert freeze['old_prediction_cohort_sha256'] == digest(source / 'prediction_cohort.json')
    assert freeze['inherited_manifest_sha256'] == digest(source / 'inherited_manifest.json')
    manifest = read(root / 'prediction_cohort.json')
    actual_files = {str(p.relative_to(root)) for folder in ('predictions', 'probes') for p in (root / folder).iterdir()}
    assert set(manifest) == actual_files
    for relative, expected in manifest.items():
        check()
        assert digest(root / relative) == expected, relative
    prediction_freeze = read(root / 'prediction_freeze.json')
    assert prediction_freeze['manifest_sha256'] == digest(root / 'prediction_cohort.json')
    complete = read(root / 'COMPLETE.json')
    assert complete['rows_sha256'] == digest(root / 'rows.json')
    assert complete['summary_sha256'] == digest(root / 'summary.json')
    launch, resources = read(root / 'launch.json'), read(root / 'resources.json')
    assert launch['start_epoch'] <= freeze['epoch'] <= prediction_freeze['epoch'] <= complete['epoch'] <= resources['end_epoch'] <= launch['deadline_epoch']
    budget = protocol['budget'][which + '_seconds']
    assert launch['deadline_epoch'] - launch['start_epoch'] <= budget + 1e-6
    assert resources['wall_seconds'] <= budget and resources['script_seconds'] <= budget
    assert resources['status'] == 'complete' and resources['CPU_threads'] == 1
    assert resources['GPU'] == resources['training'] == 0 and resources['nice'] == 10
    cases = read(root / 'cases.json')
    assert cases == read(source / 'cases.json')
    expected_cases, expected_inputs = (3, 3) if which == 'preflight' else (240, 148)
    assert len(cases) == expected_cases and len({c['observation'] for c in cases}) == expected_inputs
    names = {c['observation'] + '__' + model for c in cases for model in protocol['models']}
    assert len(names) == expected_inputs * 2
    assert {p.stem for p in (root / 'predictions').glob('*.npz')} == names
    assert prediction_freeze['prediction_files'] == len(names)
    journal = read(root / 'inference_journal.json')
    assert len(journal) == len(names)
    assert {r['observation'] + '__' + r['model'] for r in journal} == names
    for record in journal:
        assert record['status'] == 'complete'
        assert record['endpoint_float32_maxdiff'] <= 2e-6 and record['rebuild_double_maxdiff'] <= 2e-6
        for probe in record['probe_checks']:
            for values in probe['checks'].values():
                assert max(values.values()) <= 1e-10
    saved_rows = read(root / 'rows.json')
    saved_lookup = {row_key(r): r for r in saved_rows}
    assert len(saved_lookup) == len(saved_rows) == expected_cases * 16
    truths = {}
    recomputed = {}
    layers = {}
    for model in protocol['models']:
        with np.load(root / 'probes' / (model + '__weights.npz')) as w:
            layers[model] = [(w[f'w{i}'], w[f'b{i}']) for i in range(4)]
    for filename in sorted(names):
        check()
        observation, model = filename.rsplit('__', 1)
        with np.load(root / 'predictions' / (filename + '.npz')) as f:
            pred = {k: f[k] for k in f.files}
        assert all(x.shape == (64, 64, 3) and np.isfinite(x).all() for x in pred.values())
        with np.load(source / 'predictions' / (filename + '.npz')) as old:
            for k in ('point', 'native'):
                close(pred[k], old[k], 'old_endpoint', 2e-6)
                close(pred[k + '64'], pred[k], 'double_endpoint', 2e-6)
        close(pred['native64'] - pred['point64'], pred['d_full'], 'full_pixel_step')
        close(pred['d_full'] - pred['d_ref'], pred['residual'], 'full_pixel_residual')
        sums = {k: np.zeros((8, 3)) for k in MODES}
        total_weights = np.zeros((8, 1))
        with np.load(root / 'probes' / (filename + '.npz')) as probe:
            for branch in range(4):
                z0, zn, weight = (probe[f'{key}_{branch}'] for key in ('z0', 'zn', 'weight'))
                assert z0.shape == zn.shape == (8, 256) and weight.shape == (8, 1)
                assert np.isfinite(z0).all() and np.isfinite(zn).all() and (weight >= 0).all()
                delta = zn - z0
                endpoints = {}
                for mode, anchor in (('ref', z0), ('mid', (z0 + zn) / 2), ('native', zn)):
                    value, derivative = dense_numpy(anchor, delta, layers[model])
                    sums[mode] += .5 * weight * derivative
                    endpoints[mode] = value
                sums['full'] += .5 * weight * (endpoints['native'] - endpoints['ref'])
                total_weights += weight
        close(total_weights, np.ones((8, 1)), 'weight_sum', 1e-7)
        for mode in MODES:
            close(sums[mode], pred['d_' + mode].reshape(-1, 3)[:8], 'numpy_probe_' + mode, 1e-10)
        plane_sets = {roi: {k: components(pred[k], box) for k in ('point', 'native', 'point64', 'd_ref', 'd_mid', 'd_native', 'd_full')}
                      for roi, box in protocol['evaluation']['ROIs'].items()}
        for case in cases:
            if case['observation'] != observation:
                continue
            if case['family'] not in truths:
                with np.load(source / 'truth' / (case['family'] + '.npz')) as target:
                    truths[case['family']] = target['target']
            for row in calculate_rows(case, model, truths[case['family']], plane_sets, protocol['evaluation']['ROIs']):
                key = row_key(row)
                assert key not in recomputed
                compare(saved_lookup[key], row, 'rows')
                recomputed[key] = row
    assert set(recomputed) == set(saved_lookup)
    rows = [recomputed[row_key(r)] for r in saved_rows]
    summary, pairs = aggregate(rows, protocol, cases)
    compare(read(root / 'summary.json'), summary, 'summary')
    compare(read(root / 'eta_pairs.json'), pairs, 'eta_pairs')
    compare(read(root / 'numerical_sensitive.json'),
            {label: [r for r in rows if r['numerical_sensitive_' + label]] for label in ('direction', 'gain')}, 'sensitive')
    assert len(pairs) == expected_cases // 3 * 16
    if which == 'formal':
        assert summary['midband_center_reproduction']['lte']['old32_positive_to_negative'] == 14
        assert summary['midband_center_reproduction']['sc-inr']['old32_positive_to_negative'] == 16
    return dict(run=str(root), source=str(source), conditions=len(cases), unique_inputs=expected_inputs,
                prediction_files=len(names), rows=len(rows), eta_pairs=len(pairs),
                independently_differentiated_queries=len(names) * 4 * 8,
                probe_anchor_count=3, primary_scope='All saved 8-query branch probes; all-pixel complete-cohort rescoring; no full-pixel Jacobian replay',
                inherited_artifact_scope='Inherited manifest bytes verified; remote source/checkpoint file hashes are recorded by original run and not independently re-read by this local audit',
                freeze_scope='Hash and recorded epoch ordering verified; truth-gate placement statically reviewed, not an operating-system file-access trace',
                source_sha256=freeze['source_sha256'], protocol_sha256=freeze['protocol_sha256'],
                rows_sha256=digest(root / 'rows.json'), summary_sha256=digest(root / 'summary.json'),
                midband_center_reproduction=summary['midband_center_reproduction'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    output = args.out or HERE / (args.run.name + '-independent-audit.json')
    assert not output.exists(), 'Never overwrite an audit'
    started_wall, started_mono = time.time(), time.monotonic()
    def expired(*unused):
        raise TimeoutError('Independent audit fixed 120s absolute/cumulative budget exhausted')
    def check():
        if time.time() >= started_wall + LIMIT or time.monotonic() - started_mono >= LIMIT:
            expired()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    report = dict(status='running', start_epoch=started_wall, deadline_epoch=started_wall + LIMIT,
                  budget_seconds=LIMIT, audit_sha256=digest(__file__), CPU_threads=1, GPU=0, training=0)
    save(output, report)
    try:
        assert os.getpriority(os.PRIO_PROCESS, 0) == 10
        report.update(execute(args, check))
        check()
        report['status'] = 'passed'
    except BaseException:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        u = resource.getrusage(resource.RUSAGE_SELF)
        report.update(end_epoch=time.time(), wall_seconds=time.time() - started_wall,
                      script_seconds=time.monotonic() - started_mono,
                      cpu_seconds=u.ru_utime + u.ru_stime,
                      peak_rss_native=u.ru_maxrss, nice=os.getpriority(os.PRIO_PROCESS, 0),
                      check_counts=COUNTS, maximum_errors=MAXERR)
        save(output, report)
        print(json.dumps({k: report.get(k) for k in ('status', 'wall_seconds', 'rows', 'eta_pairs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
