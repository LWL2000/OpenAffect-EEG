from __future__ import annotations

import numpy as np
import pandas as pd

from openaffect_eeg.training_uncertainty import analyze_training_uncertainty


def synthetic_predictions(effect: float = 0.02) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(17)
    subjects = [f"p{i}" for i in range(6)]
    stimuli = [f"s{i}" for i in range(6)]
    truth = {
        (participant, stimulus): rng.normal()
        for participant in subjects
        for stimulus in stimuli
    }
    for training_seed in (101, 102, 103):
        seed_noise = (training_seed - 102) * 0.002
        for participant_dose in (0, 1):
            for stimulus_dose in (0, 1):
                for participant_index, participant in enumerate(subjects):
                    for stimulus_index, stimulus in enumerate(stimuli):
                        target = truth[(participant, stimulus)]
                        prior = target + rng.normal(scale=0.5)
                        combined = (1.0 - effect - seed_noise) * prior + (
                            effect + seed_noise
                        ) * target
                        rows.append(
                            {
                                "assignment_seed": 900 + participant_index % 3,
                                "training_seed": training_seed,
                                "participant_dose": participant_dose,
                                "stimulus_dose": stimulus_dose,
                                "trial_uid": f"{participant}:{stimulus}",
                                "subject_uid": participant,
                                "stimulus_uid": stimulus,
                                "target_valence": target,
                                "prior_personalized_valence": prior,
                                "combined_personalized_valence": combined,
                            }
                        )
    return pd.DataFrame(rows)


def test_joint_identity_and_seed_analysis_is_reproducible() -> None:
    first_cells, first = analyze_training_uncertainty(
        synthetic_predictions(), iterations=120, seed=31
    )
    second_cells, second = analyze_training_uncertainty(
        synthetic_predictions(), iterations=120, seed=31
    )
    pd.testing.assert_frame_equal(first_cells, second_cells)
    assert first == second
    assert len(first_cells) == 4
    assert first["training_seed_count"] == 3
    assert first["primary"]["estimate"] > 0
    assert first["primary"]["equivalence_primary"]["margin"] == 0.05
    assert first["primary"]["equivalence_primary"]["alpha_each_side"] == 0.05
    assert len(first["primary"]["equivalence_primary"]["decision_interval_90"]) == 2


def test_training_uncertainty_rejects_single_seed() -> None:
    table = synthetic_predictions().loc[lambda frame: frame.training_seed.eq(101)]
    try:
        analyze_training_uncertainty(table, iterations=100)
    except ValueError as error:
        assert "At least two" in str(error)
    else:
        raise AssertionError("single-seed input should fail")
