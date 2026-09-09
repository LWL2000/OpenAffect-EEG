import numpy as np
import pytest

from openaffect_eeg.block_validation import block_primary_draws, simulate_blocks
from openaffect_eeg.exposure_statistics import analyze_predictions


@pytest.mark.parametrize('target_names', [('valence',), ('valence', 'arousal')])
def test_matches_production_estimator_and_intervals(target_names):
    supports = [dict(n_participants=4, n_stimuli_or_conditions=5, n_trials=20)]*5
    table, _ = simulate_blocks(supports=supports, seed=71, scenario='trial_signal',
        noise_sd=.5, doses=[0, 2], target_names=target_names)
    point, draws, _, info = block_primary_draws(table, iterations=100, seed=97,
                                               target_names=target_names)
    _, contrast, summary, _ = analyze_predictions(table, iterations=100, seed=97,
                                                  target_names=target_names)
    selected = contrast.loc[contrast.contrast.eq('eeg_added_matched_calibration')]
    np.testing.assert_allclose(point, selected.ccc_delta, atol=1e-12)
    row = summary.loc[summary.contrast.eq('matched_eeg_added_uniform_mean_ccc')].iloc[0]
    np.testing.assert_allclose(point.mean(), row.estimate, atol=1e-12)
    np.testing.assert_allclose(np.quantile(draws.mean(1), [.025, .975]),
                               [row.ci_low, row.ci_high], atol=1e-12)
    assert info['valid_bootstrap'] == row.valid_bootstrap
    assert info['n_trials'] == 100


def test_labels_only_cancels_and_wrong_support_is_rejected():
    supports = [dict(n_participants=4, n_stimuli_or_conditions=5, n_trials=20)]*5
    table, _ = simulate_blocks(supports=supports, seed=71, scenario='labels_only',
        noise_sd=.5, doses=[0, 2], target_names=['valence'])
    point, draws, _, _ = block_primary_draws(table, iterations=100, seed=97,
                                             target_names=['valence'])
    np.testing.assert_allclose(point, 0, atol=1e-12)
    np.testing.assert_allclose(draws, 0, atol=1e-12)
    with pytest.raises(ValueError, match='Different test support'):
        block_primary_draws(table.iloc[1:], iterations=100, seed=97, target_names=['valence'])
