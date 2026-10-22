"""Frozen, complete-grid scoring; paired-image bootstrap, never adaptive exclusion."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from run import STAT_NAMES, condition, sha


def verify_freeze(root):
    entries = []
    for line in (root / 'FROZEN.sha256').read_text().splitlines():
        digest, rel = line.split('  ', 1)
        assert sha(root / rel) == digest, ('Frozen input changed', rel)
        entries.append(rel)
    return len(entries)


def bootstrap_weights(n, count, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(count, n))
    weights = np.array([np.bincount(row, minlength=n) for row in indices], dtype=np.float64)/n
    return weights, hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()


def distribution(x, weights):
    """Image is axis 0; retain all other pairing axes."""
    # Explicit contraction avoids platform-specific BLAS floating-exception flags.
    draws = np.einsum('bi,ij->bj', weights, x.reshape(len(x), -1), optimize=False)
    assert np.isfinite(draws).all()
    return x.mean(axis=0), draws.reshape((len(weights),)+x.shape[1:])


def interval(point, draws, name):
    low, high = np.quantile(draws, [.025, .975])
    return {name: float(point), name+'_ci_low': float(low), name+'_ci_high': float(high)}


def fields(point, draws):
    result = {}
    for k, name in enumerate(STAT_NAMES):
        result.update(interval(point[k], draws[:, k], name))
    result.update(interval(-2*point[2], -2*draws[:, 2], 'alignment_cost'))
    result.update(interval(point[1]/point[0] if point[0] else 0,
                           np.divide(draws[:, 1], draws[:, 0], out=np.zeros(len(draws)), where=draws[:, 0] != 0),
                           'D_over_J'))
    return result


def classify_adjacent(point, draws):
    """Input scale-by-J vectors; intervals concern ratios of image means."""
    ratios = point[1:] / point[:-1]
    boot = draws[:, 1:] / draws[:, :-1]
    return ratios, boot, np.quantile(boot, .05/36, axis=0), np.quantile(boot, .05/3, axis=0), np.quantile(boot, 1-.05/3, axis=0)


def load(root, protocol, manifest):
    models, seeds, scales, steps = [protocol[k] for k in ['models', 'seeds', 'scales', 'steps']]
    names = {ds: [i['image'] for i in records] for ds, records in manifest['images'].items()}
    arrays, seen = {}, set()
    for ds, images in names.items():
        shape = (len(images), len(models), len(seeds), len(scales), len(steps))
        arrays[ds] = {name: np.full(shape+tail, np.nan) for name, tail in
                      [('total', (5,)), ('dc', (5,)), ('r_bins', (17, 5)), ('u_bins', (24, 5))]}
    source_files, checks = {}, {'identity_max_abs': 0., 'parseval_max_abs': 0., 'records': 0}
    for model in models:
        for seed in seeds:
            shard = root / 'outputs' / f'{model}-seed{seed}'
            complete = json.loads((shard / 'COMPLETE.json').read_text())
            assert complete['frozen_sha256'] == sha(root / 'FROZEN.sha256')
            records = shard / 'records.jsonl.gz'
            source_files[str(records)] = sha(records)
            assert source_files[str(records)] == complete['records_sha256']
            shard_count = 0
            with gzip.open(records, 'rt') as stream:
                for line in stream:
                    r = json.loads(line)
                    assert r['model'] == model and r['seed'] == seed
                    ds, image, scale, h = [r[k] for k in ['dataset', 'image', 'scale', 'h']]
                    key = (ds, image, model, seed, scale, h)
                    assert key not in seen, ('Duplicate record', key)
                    seen.add(key)
                    idx = (names[ds].index(image), models.index(model), seeds.index(seed), scales.index(scale), steps.index(h))
                    for k, v in condition(scale, h).items():
                        assert r[k] == v, (k, r[k], v)
                    expected_hw = manifest['images'][ds][idx[0]]['crop_hw']
                    assert r['hr_hw'] == expected_hw
                    for name in ['total', 'dc', 'r_bins']:
                        arrays[ds][name][idx] = r[name]
                    assert np.asarray(r['u_bins']).shape == (4*scale, 5)
                    arrays[ds]['u_bins'][idx] = np.pad(np.asarray(r['u_bins']), ((0, 24-4*scale), (0, 0)))
                    total = np.asarray(r['total'])
                    assert total[0] >= 0 and total[3] > 0 and total[4] > 0
                    err = abs(total[1] - total[0] + 2*total[2])
                    assert err < 2e-12
                    checks['identity_max_abs'] = max(checks['identity_max_abs'], err)
                    for name in ['r_bins', 'u_bins']:
                        np.testing.assert_allclose(np.sum(r[name], axis=0)+r['dc'], total, atol=2e-12, rtol=2e-9)
                    checks['parseval_max_abs'] = max(checks['parseval_max_abs'], r['parseval_max_abs'])
                    checks['records'] += 1
                    shard_count += 1
            assert shard_count == complete['records']
    assert checks['records'] == sum(len(v) for v in names.values())*len(models)*len(seeds)*len(scales)*len(steps)
    for ds, values in arrays.items():
        for name, x in values.items():
            assert np.isfinite(x).all(), ('Incomplete or non-finite grid', ds, name)
        # Native error and target are invariant to intervention; target also to model/scale.
        v = values['total'][..., 3]
        np.testing.assert_allclose(v, np.broadcast_to(v[..., :1], v.shape), atol=2e-12, rtol=2e-9)
        e = values['total'][..., 4]
        np.testing.assert_allclose(e, np.broadcast_to(e[:, :1, :1, :1, :1], e.shape), atol=2e-12, rtol=2e-9)
    return arrays, names, source_files, checks


def score_arrays(arrays, names, protocol):
    scales, steps = protocol['scales'], protocol['steps']
    count, seed = protocol['bootstrap']['replicates'], protocol['bootstrap']['seed']
    summary = {'conditions': [], 'scale_totals': [], 'adjacent_contrasts': [],
               'endpoint_contrasts': [], 'bootstrap': {}, 'stratum_totals': []}
    frequency_rows = []
    for di, (ds, data) in enumerate(arrays.items()):
        weights, index_hash = bootstrap_weights(len(names[ds]), count, seed+di)
        summary['bootstrap'][ds] = {'images': len(names[ds]), 'replicates': count,
                                    'seed': seed+di, 'indices_sha256': index_hash}
        # Model and seed are fixed factors, averaged before image resampling.
        average = {k: v.mean(axis=(1, 2)) for k, v in data.items()}
        point, draws = distribution(average['total'], weights)
        for hi, h in enumerate(steps):
            ratio, rb, global_low, local_low, local_high = classify_adjacent(point[:, hi, 0], draws[:, :, hi, 0])
            local_pass = bool(np.any(local_low > 1.10))
            decreases = bool(np.all(local_high < .90))
            summary['conditions'].append({'dataset': ds, 'h': h,
                 'within_condition_pass': local_pass,
                 'family_wide_supported_reversal': bool(np.any(global_low > 1.10)),
                 'diagnostic': 'supported_material_reversal' if local_pass else
                               'supported_material_monotone_decrease' if decreases else 'unresolved_or_small',
                 'point_sequence_nonincreasing': bool(np.all(np.diff(point[:, hi, 0]) <= 0))})
            for pi, (s0, s1) in enumerate(zip(scales[:-1], scales[1:])):
                summary['adjacent_contrasts'].append({'dataset': ds, 'h': h, 'start_scale': s0,
                     'end_scale': s1, **interval(ratio[pi], rb[:, pi], 'J_ratio'),
                     'lower_bound_family_36': float(global_low[pi]),
                     'lower_bound_local_3': float(local_low[pi]),
                     'upper_bound_local_3': float(local_high[pi]),
                     'primary_supported_reversal': bool(global_low[pi] > 1.10)})
            endpoint = {'dataset': ds, 'h': h, 'start_scale': scales[0], 'end_scale': scales[-1],
                **interval(point[-1, hi, 0]/point[0, hi, 0], draws[:, -1, hi, 0]/draws[:, 0, hi, 0], 'J_ratio')}
            for k, stat in enumerate(STAT_NAMES):
                endpoint.update(interval(point[-1, hi, k]-point[0, hi, k],
                                         draws[:, -1, hi, k]-draws[:, 0, hi, k], stat+'_difference'))
            summary['endpoint_contrasts'].append(endpoint)
            for si, scale in enumerate(scales):
                summary['scale_totals'].append({'dataset': ds, **condition(scale, h),
                                                **fields(point[si, hi], draws[:, si, hi])})
        del draws
        for coordinate in ['dc', 'r_bins', 'u_bins']:
            band_input = average[coordinate]
            if coordinate == 'dc':
                band_input = band_input[..., None, :]
            bp, bd = distribution(band_input, weights)
            for si, scale in enumerate(scales):
                bins = 1 if coordinate == 'dc' else 17 if coordinate == 'r_bins' else 4*scale
                for hi, h in enumerate(steps):
                    for bi in range(bins):
                        frequency_rows.append({'dataset': ds, 'scale': scale, 'h': h,
                            'coordinate': coordinate, 'bin': bi,
                            'lo': 0 if coordinate == 'dc' else bi/32 if coordinate == 'r_bins' else bi/4,
                            'hi': 0 if coordinate == 'dc' else min((bi+1)/32, .5) if coordinate == 'r_bins' else (bi+1)/4,
                            **fields(bp[si, hi, bi], bd[:, si, hi, bi])})
            del bd
        for mi, model in enumerate(protocol['models']):
            for ki, checkpoint_seed in enumerate(protocol['seeds']):
                sp, sd = distribution(data['total'][:, mi, ki], weights)
                for si, scale in enumerate(scales):
                    for hi, h in enumerate(steps):
                        summary['stratum_totals'].append({'dataset': ds, 'model': model,
                             'seed': checkpoint_seed, 'scale': scale, 'h': h,
                             **fields(sp[si, hi], sd[:, si, hi])})
                del sd
    primary = [r for r in summary['adjacent_contrasts'] if r['primary_supported_reversal']]
    local = [r for r in summary['conditions'] if r['within_condition_pass']]
    directions = [r for r in summary['endpoint_contrasts'] if r['dataset'] == 'Set14' and r['h'] > 0]
    assert len(summary['adjacent_contrasts']) == 36 and len(summary['conditions']) == 12 and len(directions) == 2
    summary['prediction_results'] = {
        'P17-equal-step': {'status': 'PASS' if primary else 'FAIL', 'supported_pairs': len(primary),
                           'family_size': 36, 'lower_quantile': .05/36, 'threshold': 1.10},
        'P17-all-conditions': {'status': 'PASS' if len(local) == 12 else 'FAIL', 'supported_conditions': len(local),
                              'condition_count': 12, 'within_condition_lower_quantile': .05/3, 'threshold': 1.10},
        'P17-Set14-positive': {'status': 'PASS' if all(r['J_ratio_ci_low'] > 1 for r in directions) else 'FAIL',
                             'endpoint_details': directions, 'lower_quantile': .025, 'threshold': 1.0}}
    return summary, frequency_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', required=True)
    args = ap.parse_args()
    root = Path(args.run_root).resolve()
    frozen_count = verify_freeze(root)
    assert (root / 'INFERENCE_COMPLETE.json').exists(), 'Inference has not completed'
    protocol = json.loads((root / 'protocol.json').read_text())
    manifest = json.loads((root / 'input_manifest.json').read_text())
    arrays, names, source_files, checks = load(root, protocol, manifest)
    summary, frequency = score_arrays(arrays, names, protocol)
    summary['provenance'] = {'run_root': str(root), 'frozen_sha256': sha(root/'FROZEN.sha256'),
        'frozen_files_checked': frozen_count, 'source_files': source_files,
        'protocol_sha256': sha(root/'protocol.json'), 'prediction_sha256': sha(root/'predictions.json'),
        'scorer_sha256': sha(Path(__file__)), 'image_names': names}
    summary['checks'] = checks
    out = root / 'score'
    out.mkdir(exist_ok=False)
    strata = summary.pop('stratum_totals')
    for filename, rows in [('frequency.csv', frequency), ('strata.csv', strata)]:
        with (out/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary['detail_files'] = {name: {'path': str(out/name), 'sha256': sha(out/name)}
                               for name in ['frequency.csv', 'strata.csv']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps({'summary_sha256': sha(out/'summary.json'),
            'prediction_results': summary['prediction_results'], 'checks': checks}, indent=2)+'\n')
    print(json.dumps(summary['prediction_results'], indent=2))


if __name__ == '__main__':
    main()
