"""Independent post-completion check; never imports the frozen scorer or runs a model.

Run on the server after COMPLETE.txt exists:
  python3 verify_completed.py --run-root RUN > independent_validation.json
Synthetic-only checks: python3 verify_completed.py --self-test
The output distinguishes frozen gate replication from supplemental descriptions.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_FREEZE = 'b601cf3d197cc81bfe0e6691f1c310457c8c936e49985e7e073062e4e06c1fb1'
ARMS = ['native', 'nocell', 'constant', 'shuffle']
DATASETS = ['Set5', 'Set14', 'BSD100', 'Urban100']
SCALES = [2, 3, 4, 6]
SEEDS = [1, 2]
MODES = ['native', 'mean'] + [f'marginal_{i}' for i in range(4)]
REGIONS = [('low_u_lt_0.75', 0., .75), ('transition_0.75_to_1', .75, 1.),
           ('high_1_to_2', 1., 2.), ('tail_u_ge_2', 2., float('inf'))]


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            d.update(block)
    return d.hexdigest()


def interval(values, coverage=.95):
    tail = (1-coverage)/2
    return np.quantile(values, [tail, 1-tail], axis=0).tolist()


def gates(primary, equivalent):
    result = {}
    for arm, x in primary.items():
        result['T1_'+arm] = bool(x['difference_db'] >= .05 and x['ci97_5'][0] > 0
                                 and min(x['seed_means']) > 0)
    for arm, x in equivalent.items():
        result['T2_'+arm+'_equivalent_nocell'] = bool(x['ci95'][0] > -.03 and x['ci95'][1] < .03)
    return result


def metric_policy(data, arm, ds, policy):
    if policy == 'marginal':
        # Average the four PSNR metrics, never predictions or MSEs.
        return np.mean([data[arm][ds][f'marginal_{i}'] for i in range(4)], axis=0)
    return data[arm][ds][policy]


def balanced_policy(data, arm, policy, resamples):
    points, boot, per_seed = [], [], []
    for ds in DATASETS:
        values = metric_policy(data, arm, ds, policy)
        points.append(values.mean())
        per_image = values.mean(axis=(1, 2))
        boot.append(per_image[resamples[ds]].mean(axis=1))
        per_seed.append(values.mean(axis=(0, 2)))
    return float(np.mean(points)), np.mean(boot, axis=0), np.mean(per_seed, axis=0).tolist()


def compare(data, arm, reference, policy, resamples):
    per_dataset, bootstrap, seed_means, details = [], [], [], []
    for ds in DATASETS:
        diff = metric_policy(data, reference, ds, 'native')-metric_policy(data, arm, ds, policy)
        per_dataset.append(diff.mean())
        # Explicit direct image resampling, independent of frozen implementation.
        boot = diff[resamples[ds]].mean(axis=(1, 2, 3))
        bootstrap.append(boot)
        seed_means.append(diff.mean(axis=(0, 2)))
        for si, scale in enumerate(SCALES):
            image_differences = diff[:, :, si].mean(axis=1)
            draws = image_differences[resamples[ds]].mean(axis=1)
            details.append(dict(dataset=ds, scale=scale,
                                difference_db=float(image_differences.mean()), ci95=interval(draws)))
    draws = np.mean(bootstrap, axis=0)
    return dict(difference_db=float(np.mean(per_dataset)), ci95=interval(draws),
                ci97_5=interval(draws, .975), seed_means=np.mean(seed_means, axis=0).tolist(),
                per_dataset_scale=details)


def recovery_comparison(native, other, indices):
    """Shapes image, seed, bin, [target,error]; seed-specific aggregate ratios."""
    np.testing.assert_array_equal(native[..., 0], other[..., 0])
    energy = native[..., 0].sum(axis=0)
    assert np.all(energy > 0)
    native_r = 1-native[..., 1].sum(axis=0)/energy
    other_r = 1-other[..., 1].sum(axis=0)/energy
    delta = native_r-other_r
    sampled_native, sampled_other = native[indices].sum(axis=1), other[indices].sum(axis=1)
    bootstrap = ((sampled_other[..., 1]-sampled_native[..., 1])/sampled_native[..., 0]).mean(axis=1)
    return native_r.mean(axis=0), other_r.mean(axis=0), delta, bootstrap


def verify_completed(root):
    # Do not inspect training/evaluation values until the entire fixed pipeline is complete.
    if not (root/'COMPLETE.txt').exists() or not (root/'scores/summary.json').exists():
        raise SystemExit('NOT_READY: complete pipeline and scores are required; no partial outcome was read')
    assert (root/'COMPLETE.txt').read_text().strip() == 'complete'
    assert not (root/'PIPELINE_FAILED.txt').exists()
    assert sha(root/'FROZEN.sha256') == EXPECTED_FREEZE
    for line in (root/'FROZEN.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        assert sha(root/relative) == digest, relative
    protocol = json.loads((root/'protocol.json').read_text())
    declared = json.loads((root/'predictions.json').read_text())
    assert declared == protocol['predictions']
    assert protocol['scales'] == SCALES and protocol['recipe']['seed'] == SEEDS
    original = json.loads((root/'scores/summary.json').read_text())
    manifest = json.loads((root/'data_manifest.json').read_text())
    image_paths = {ds: sorted(manifest[ds+'/HR']) for ds in DATASETS}
    names = {ds: [Path(p).stem for p in paths] for ds, paths in image_paths.items()}
    assert [len(names[x]) for x in DATASETS] == [5, 14, 100, 100]
    recovery_names = [f'{i:04d}' for i in range(869, 901)]
    recovery_paths = {Path(p).stem: p for p in manifest['DIV2K_valid_HR']
                      if 869 <= int(Path(p).stem) <= 900}
    assert sorted(recovery_paths) == recovery_names
    benchmark = {arm: {ds: {} for ds in DATASETS} for arm in ARMS}
    recovery, receipts, checkpoint_details, policy_audits = {}, {}, {}, {}
    raw_rows = 0
    import torch  # CPU checkpoint metadata only; no model is instantiated.
    for arm in ARMS:
        for ds in DATASETS:
            for mode in (MODES if arm == 'shuffle' else ['native']):
                benchmark[arm][ds][mode] = np.full((len(names[ds]), 2, 4), np.nan)
        recovery[arm] = {mode: {s: np.full((32, 2, 4*s, 2), np.nan) for s in SCALES}
                         for mode in (MODES if arm == 'shuffle' else ['native'])}
        for seed in SEEDS:
            label = f'{arm}_s{seed}'
            folder = root/'evaluation'/label
            receipt = json.loads((folder/'COMPLETE.json').read_text())
            path = folder/'rows.jsonl.gz'
            assert receipt['rows_sha256'] == sha(path)
            assert receipt['frozen_sha256'] == EXPECTED_FREEZE
            assert original['inputs'][str(path.relative_to(root))] == receipt['rows_sha256']
            modes = MODES if arm == 'shuffle' else ['native']
            assert set(receipt['modes']) == set(modes)
            ckpath = Path(receipt['checkpoint'])
            digest = sha(ckpath)
            assert digest == receipt['checkpoint_sha256']
            if arm in ('native', 'nocell'):
                expected = protocol['baseline_checkpoints'][label]
                assert str(ckpath) == expected['path'] and digest == expected['sha256']
            else:
                assert ckpath == root/'save'/label/'epoch-last.pth'
                trained = json.loads((ckpath.parent/'COMPLETE.json').read_text())
                assert trained['epoch'] == 200 and trained['checkpoint_sha256'] == digest
                assert trained['seconds'] < 24*3600
                assert (root/'statuses'/f'{label}.exit').read_text().strip() == '0'
                audit = json.loads((root/'logs'/f'cell_audit_{label}.json').read_text())
                assert audit['batches'] == 1000 and audit['samples'] == 16000
                c0 = sum((.5 if k in (48, 192) else 1.)*96/k for k in range(48, 193))/144
                assert abs(audit['constant_expected']-c0) < 1e-12
                if arm == 'shuffle':
                    assert audit['fixed_index_count'] == 0
                    assert abs(audit['exact_marginal_sum_error']) < 1e-9
                    assert abs(audit['exact_marginal_square_error']) < 1e-9
                else:
                    assert abs(audit['assigned_mean']-c0) < 2e-7
                    assert abs(audit['assigned_variance']) < 1e-12
                policy_audits[label] = audit
            checkpoint = torch.load(ckpath, map_location='cpu', weights_only=True)
            assert checkpoint['epoch'] == 200
            args = checkpoint['model']['args']
            assert args['imnet_spec']['args']['hidden_list'] == [256]*4
            assert args['encoder_spec']['name'] == 'edsr-baseline'
            assert args['encoder_spec']['args']['no_upsampling']
            assert args.get('cell_decode', True) == (arm != 'nocell')
            if arm in ('constant', 'shuffle'):
                assert checkpoint['model']['name'] == 'liif-cell-source'
                assert args['cell_mode'] == arm and args['cell_seed'] == 180000+seed
            checkpoint_details[label] = dict(epoch=200, sha256=digest,
                state_values=sum(v.numel() for v in checkpoint['model']['sd'].values()))
            del checkpoint
            expected_images = {p: manifest[ds+'/HR'][p] for ds in DATASETS for p in image_paths[ds]}
            expected_images.update({p: manifest['DIV2K_valid_HR'][p] for p in recovery_paths.values()})
            assert receipt['image_sha256'] == expected_images
            expected_count = (219*4+32*60)*len(modes)
            seen, count = set(), 0
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    assert row['arm'] == arm and row['seed'] == seed
                    scale, mode, img = row['scale'], row['eval_mode'], row['image']
                    assert scale in SCALES and mode in modes
                    key = (row['kind'], row['dataset'], img, scale, mode, row.get('lo'))
                    assert key not in seen, ('duplicate', label, key)
                    seen.add(key)
                    if row['kind'] == 'benchmark':
                        ds = row['dataset']; assert ds in DATASETS and img in names[ds]
                        assert np.isfinite(row['y_psnr'])
                        benchmark[arm][ds][mode][names[ds].index(img), seed-1, SCALES.index(scale)] = row['y_psnr']
                    else:
                        assert row['kind'] == 'recovery' and row['dataset'] == 'DIV2K-fresh'
                        assert img in recovery_names
                        lo = row['lo']; bi = int(round(lo*4))
                        assert lo == bi/4 and 0 <= bi < 4*scale
                        assert row['hi'] == (scale+1e-6 if bi == 4*scale-1 else lo+.25)
                        values = [row['target'], row['error']]
                        assert np.isfinite(values).all() and min(values) >= 0
                        recovery[arm][mode][scale][recovery_names.index(img), seed-1, bi] = values
                    count += 1
            assert count == receipt['rows'] == expected_count
            raw_rows += count
            receipts[label] = dict(rows=count, rows_sha256=receipt['rows_sha256'], checkpoint_sha256=digest)
    assert raw_rows == 50328
    for arm in ARMS:
        for ds in DATASETS:
            for values in benchmark[arm][ds].values():
                assert np.isfinite(values).all(), ('incomplete benchmark grid', arm, ds)
        for mode, by_scale in recovery[arm].items():
            for scale, values in by_scale.items():
                assert np.isfinite(values).all(), ('incomplete recovery grid', arm, mode, scale)
                np.testing.assert_array_equal(values[..., 0], recovery['native']['native'][scale][..., 0])
    for seed in SEEDS:
        native_n = checkpoint_details[f'native_s{seed}']['state_values']
        for arm in ('constant', 'shuffle'):
            assert checkpoint_details[f'{arm}_s{seed}']['state_values'] == native_n
        assert native_n-checkpoint_details[f'nocell_s{seed}']['state_values'] == 512

    # Independently recompute target Fourier power from the unchanged HR inputs.
    from PIL import Image
    target_error = 0.
    N = 232; radius = np.maximum(abs(np.fft.fftfreq(N))[:, None], abs(np.fft.fftfreq(N))[None, :])
    for ii, img in enumerate(recovery_names):
        path = Path(recovery_paths[img]); assert sha(path) == manifest['DIV2K_valid_HR'][str(path)]
        with Image.open(path) as image:
            image = image.convert('RGB'); w, h = image.size
            crop = np.asarray(image.crop(((w-240)//2, (h-240)//2, (w-240)//2+240, (h-240)//2+240)), dtype=np.float32)/255
        power = abs(np.fft.fft2(crop[4:-4, 4:-4].astype(np.float64), axes=(0, 1)))**2
        for scale in SCALES:
            u = 2*scale*radius
            for bi in range(4*scale):
                hi = scale+1e-6 if bi == 4*scale-1 else (bi+1)/4
                value = float(power[(u >= bi/4) & (u < hi) & (radius > 0)].sum()/(3*N**4))
                target_error = max(target_error, abs(value-recovery['native']['native'][scale][ii, 0, bi, 0]))
    assert target_error < 2e-12

    rng = np.random.default_rng(protocol['bootstrap']['seed'])
    nboot = protocol['bootstrap']['replicates']; assert nboot == 2000
    indices = {ds: rng.integers(0, len(names[ds]), size=(nboot, len(names[ds]))) for ds in DATASETS}
    recovery_indices = rng.integers(0, 32, size=(nboot, 32))
    delta_max = 0.
    def check(a, b):
        nonlocal delta_max
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        assert a.shape == b.shape
        delta_max = max(delta_max, float(np.max(np.abs(a-b))))
        np.testing.assert_allclose(a, b, atol=3e-10, rtol=1e-11)
    def check_contrast(actual, expected):
        for field in ['difference_db', 'ci95', 'ci97_5', 'seed_means']:
            check(actual[field], expected[field])
        for row in actual['per_dataset_scale']:
            old = next(x for x in expected['per_dataset_scale'] if x['dataset'] == row['dataset'] and x['scale'] == row['scale'])
            check(row['difference_db'], old['difference_db']); check(row['ci95'], old['ci95'])
    primary = {arm: compare(benchmark, arm, 'native', 'native', indices) for arm in ['constant', 'shuffle']}
    equiv = {arm: compare(benchmark, arm, 'nocell', 'native', indices) for arm in ['constant', 'shuffle']}
    for arm in primary:
        check_contrast(primary[arm], original['primary_contrasts'][arm])
        check_contrast(equiv[arm], original['equivalence'][arm])
    for policy in ['mean', 'marginal']:
        check_contrast(compare(benchmark, 'shuffle', 'native', policy, indices), original['shuffle_policy_sensitivity'][policy])
    computed_gates = gates(primary, equiv)
    for name, passed in computed_gates.items():
        assert passed == original['predictions'][name]['pass'], name
    for row in original['absolute_y_psnr']:
        x = metric_policy(benchmark, row['arm'], row['dataset'], row['eval_mode'])[:, :, SCALES.index(row['scale'])]
        check(float(x.mean()), row['psnr']); check(x.mean(axis=0), row['seed_means'])

    # Predeclared descriptive additions only: policies and equal-bin regional CIs.
    policy_points, policy_draws, policy_seed_means = [], [], []
    for mode in MODES:
        point, draws, seed_values = balanced_policy(benchmark, 'shuffle', mode, indices)
        policy_points.append(point); policy_draws.append(draws); policy_seed_means.append(seed_values)
    policy_draws = np.stack(policy_draws, axis=1)
    marginal_point, marginal_draws, marginal_seeds = balanced_policy(benchmark, 'shuffle', 'marginal', indices)
    check(marginal_point, np.mean(policy_points[2:]))
    policy_summary = {
        'definition': 'Dataset/scale-balanced mean of image PSNR metrics; six fixed inference policies. SD is descriptive dispersion across policies, not uncertainty across trained seeds.',
        'six_policies': [dict(policy=mode, balanced_y_psnr=policy_points[i], ci95=interval(policy_draws[:, i]),
                             seed_means=policy_seed_means[i]) for i, mode in enumerate(MODES)],
        'range_db': float(np.ptp(policy_points)), 'range_ci95': interval(np.ptp(policy_draws, axis=1)),
        'sd_db_ddof0': float(np.std(policy_points)), 'sd_ci95': interval(np.std(policy_draws, axis=1)),
        'marginal_four_draw_metric_mean': dict(balanced_y_psnr=marginal_point, ci95=interval(marginal_draws), seed_means=marginal_seeds),
        'inference_rule': 'Each marginal c is the SHA256-derived 96/round(48*(1+3*v)) from frozen evaluate.py, held constant across scales for that image. This is source-verified, not a logged per-query readback.'}
    region_rows, shuffle_policy_regions = [], []
    for arm in ['nocell', 'constant', 'shuffle']:
        policies = MODES + ['marginal'] if arm == 'shuffle' else ['native']
        for mode in policies:
            for scale in SCALES:
                native = recovery['native']['native'][scale]
                other = np.mean([recovery[arm][f'marginal_{i}'][scale] for i in range(4)], axis=0) if mode == 'marginal' else recovery[arm][mode][scale]
                nr, ar, seed_delta, bootstrap = recovery_comparison(native, other, recovery_indices)
                point = seed_delta.mean(axis=0)
                if mode == 'native':
                    for bi in range(scale*4):
                        old = next(x for x in original['recovery'] if x['arm'] == arm and x['scale'] == scale and x['lo'] == bi/4)
                        check([nr[bi], ar[bi], point[bi]], [old['native_R'], old['arm_R'], old['native_minus_arm_R']])
                        check(seed_delta[:, bi], old['seed_deltas']); check(interval(bootstrap[:, bi]), old['ci95'])
                lo_values = np.arange(scale*4)/4
                for label, lo, hi in REGIONS:
                    selected = (lo_values >= lo) & (lo_values < hi)
                    if not selected.any():
                        continue
                    row = dict(arm=arm, eval_mode=mode, scale=scale, region=label,
                        bins=int(selected.sum()), mean_bin_delta_R=float(point[selected].mean()),
                        ci95=interval(bootstrap[:, selected].mean(axis=1)),
                        max_abs_bin_delta_R=float(np.max(abs(point[selected]))))
                    if mode == 'native':
                        old = next(x for x in original['recovery_regions'] if x['arm'] == arm and x['scale'] == scale and x['region'] == label)
                        check(row['mean_bin_delta_R'], old['mean_bin_delta_R'])
                        check(row['max_abs_bin_delta_R'], old['max_abs_bin_delta_R'])
                        region_rows.append(row)
                    if arm == 'shuffle':
                        shuffle_policy_regions.append(row)
    assert len(region_rows) == 45 and len(shuffle_policy_regions) == 105
    return {
        'status': 'verified_completed_run', 'run_root': str(root), 'frozen_sha256': EXPECTED_FREEZE,
        'formal_summary_sha256': sha(root/'scores/summary.json'), 'raw_rows': raw_rows,
        'evaluation_slots': receipts, 'checkpoint_metadata_cpu_checked': checkpoint_details,
        'training_cell_audits': policy_audits, 'target_fft_max_error': target_error,
        'max_formal_statistic_difference': delta_max, 'prediction_results': computed_gates,
        'geometry': {
            'benchmark': 'Full Set5/Set14/BSD100/Urban100; independent top-left modcrop per scale, PIL bicubic, output clamp [0,1], Y-PSNR using coefficients [65.738,129.057,25.064]/256 and shave=scale. Not stage17 common-modcrop torch-bicubic geometry.',
            'recovery': 'DIV2K0869-0900 center240 RGB crops, torch antialiased bicubic/clamped LR, unclipped prediction, shave4, non-DC FFT with width .25 in u=2*s*r; includes exact Nyquist outer ring.',
            'R_aggregation': 'For each seed/bin first sum error and target over 32 images, compute 1-sum(error)/sum(target), then average two seeds. Regional metric remains the unweighted mean of quarter-u-bin delta_R; no hidden region-energy pooling.',
            'policy_labels': 'Rows called native in the constant arm still execute its fixed training constant internally; no-cell ignores the passed cell.'},
        'interpretation_limits': [
            'T1 point estimate must be at least .05 dB, but its confidence interval only has to exclude zero; not a confidence lower bound of .05 dB.',
            'T2 requires an entire 95% interval within +/- .03 dB. Failure is not evidence of difference or equivalence.',
            'Image uncertainty conditions on two trained seeds; dataset-balanced means give Set5 and Urban100 equal total weight.',
            'Per-image Y-PSNR aggregation and stored spectral statistics are independently recomputed. No saved predictions exist to independently rerun the Y conversion numerically; that portion is frozen-source/geometry verification.',
            'Correct pairing can help this recipe without identifying a unique internal mechanism; constant changes the feature distribution and shuffle injects noise.'],
        'supplemental_descriptive_only': {
            'no_new_gates': True,
            'shuffle_benchmark_policy_variation': policy_summary,
            'equal_bin_recovery_region_intervals': region_rows,
            'shuffle_recovery_policy_regions': shuffle_policy_regions,
            'CI_scope': 'Pointwise paired percentile95%, descriptive and unadjusted; not new confirmatory tests.'}}


def self_test():
    rng = np.random.default_rng(18)
    data = {arm: {ds: {} for ds in DATASETS} for arm in ARMS}
    counts = dict(zip(DATASETS, [5, 14, 100, 100]))
    indices = {ds: rng.integers(counts[ds], size=(2000, counts[ds])) for ds in DATASETS}
    for ds in DATASETS:
        n = counts[ds]
        base = 30+np.arange(n)[:, None, None]*.001+np.array(SEEDS)[None, :, None]*.01+np.array(SCALES)[None, None, :]*.001
        for arm in ARMS:
            for mode in MODES if arm == 'shuffle' else ['native']:
                penalty = {'native': 0., 'nocell': .10, 'constant': .10, 'shuffle': .08}[arm]
                if arm == 'shuffle' and mode.startswith('marginal_'):
                    penalty += int(mode[-1])*.01
                if mode == 'mean':
                    penalty += .03
                data[arm][ds][mode] = base-penalty
    main = {arm: compare(data, arm, 'native', 'native', indices) for arm in ['constant', 'shuffle']}
    equivalent = {arm: compare(data, arm, 'nocell', 'native', indices) for arm in ['constant', 'shuffle']}
    assert all(gates(main, equivalent).values())
    assert abs(main['constant']['difference_db']-.10) < 1e-12
    # Explicit failing and unresolved cases cannot turn into a pass.
    altered = json.loads(json.dumps(main)); altered['constant']['ci97_5'][0] = -.001
    assert not gates(altered, equivalent)['T1_constant']
    altered = json.loads(json.dumps(equivalent)); altered['shuffle']['ci95'] = [-.031, .02]
    assert not gates(main, altered)['T2_shuffle_equivalent_nocell']
    altered = json.loads(json.dumps(main)); altered['shuffle']['seed_means'][1] = -.001
    assert not gates(altered, equivalent)['T1_shuffle']
    values = [balanced_policy(data, 'shuffle', mode, indices)[0] for mode in MODES]
    averaged = balanced_policy(data, 'shuffle', 'marginal', indices)[0]
    assert abs(averaged-np.mean(values[2:])) < 1e-12
    # Energy-pooled recovery must differ from a mean of image-wise ratios.
    native = np.array([[[[1., .5], [9., .9]], [[1., .5], [9., .9]]],
                       [[[9., 0.], [1., .1]], [[9., 0.], [1., .1]]]])
    other = native.copy(); other[1, :, 0, 1] += 4.5
    rdraw = rng.integers(2, size=(2000, 2))
    _, _, delta, boot = recovery_comparison(native, other, rdraw)
    np.testing.assert_allclose(delta[:, 0], .45)
    assert abs(np.mean((other[..., 1]-native[..., 1])/native[..., 0])-.125) < 1e-12
    assert boot.shape == (2000, 2)
    return {'status': 'synthetic_pass', 'experimental_results_read': False,
            'models_or_checkpoints_loaded': False, 'checks': ['balanced paired PSNR', 'T1/T2 failure boundaries',
            'four-draw metric mean', 'seed-specific energy-pooled R', 'paired regional-bootstrap shape']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--self-test', action='store_true')
    arguments = parser.parse_args()
    if arguments.self_test:
        result = self_test()
    else:
        assert arguments.run_root is not None, '--run-root is required'
        result = verify_completed(arguments.run_root.resolve())
    print(json.dumps(result, indent=2))
