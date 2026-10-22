"""Synthetic-only implementation checks; no experimental images or model calls."""
import json

import numpy as np

from run import condition, spectral_stats
from score import bootstrap_weights, classify_adjacent, distribution


def main():
    rng = np.random.default_rng(170023)
    errors = []
    for H, W in [(24, 36), (25, 31), (72, 48)]:
        target = rng.random((H, W, 3))
        native = target + rng.normal(0, .03, target.shape)
        variants = [native.copy(), native+.02, native+rng.normal(0, .02, native.shape),
                    native+.5*(target-native), native-.5*(target-native)]
        for changed in variants:
            for scale in [2, 3, 4, 6]:
                result = spectral_stats(target, native, changed, scale)
                J, D, A, V, E = result['total']
                assert abs(D-J+2*A) < 1e-12
                errors.append(result['parseval_max_abs'])
        native_test = spectral_stats(target, native, native, 2)
        assert native_test['total'][0] == 0 and native_test['total'][1] == 0
        constant = spectral_stats(target, native, native+.02, 2)
        np.testing.assert_allclose(constant['dc'][0], .02**2, atol=1e-14)
        assert np.sum(np.array(constant['r_bins'])[:, 0]) < 1e-28
        correcting = spectral_stats(target, native, native+.5*(target-native), 2)
        assert correcting['total'][1] < 0 and correcting['total'][2] > 0
    for scale in [2, 3, 4, 6]:
        for h in [-.25, .25, .5]:
            c = condition(scale, h)
            np.testing.assert_allclose(2*(c['multiplier']-1)/scale, h, atol=1e-15)
    assert not condition(3, -.25)['perturbed_cell_in_training_interval']
    assert condition(6, .25)['perturbed_cell_in_training_interval']
    assert not condition(6, .25)['native_cell_in_training_interval']
    weights, digest = bootstrap_weights(14, 10000, 20261023)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    # All paired image multipliers cancel in ratios, so direction gates must be exact.
    for sequence, expect in [([1., 1.2, .8, 1.], True), ([1., .8, .6, .4], False),
                             ([1., 1., 1., 1.], False), ([1., 1.05, 1.02, 1.03], False)]:
        images = np.arange(1., 15.)[:, None]*np.array(sequence)[None, :]
        point, draws = distribution(images, weights)
        for bi in range(20):
            np.testing.assert_allclose(draws[bi], np.sum(weights[bi, :, None]*images, axis=0), atol=1e-14)
        ratios, boots, lo36, lo3, hi3 = classify_adjacent(point, draws)
        assert bool(np.any(lo36 > 1.10)) == expect
        np.testing.assert_allclose(ratios, np.array(sequence[1:])/sequence[:-1], atol=1e-14)
        np.testing.assert_allclose(boots, np.broadcast_to(ratios, boots.shape), atol=1e-14)
        if sequence == [1., .8, .6, .4]:
            assert np.all(hi3 < .9)
    # A naive unpaired calculation would lose this exact ratio cancellation.
    print(json.dumps({'synthetic_only': True, 'model_calls': 0, 'parseval_cases': len(errors),
                       'parseval_max_abs': max(errors), 'bootstrap_replicates': 10000,
                       'bootstrap_indices_sha256': digest, 'direction_gate_cases': 4,
                       'paired_ratio_check': 'PASS', 'DC_and_Nyquist_included': True}, indent=2))


if __name__ == '__main__':
    main()
