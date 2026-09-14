from __future__ import annotations

import pandas as pd

from openaffect_eeg.amigos_confirmation import crossed_outer_assignments
from openaffect_eeg.identity_exposure import compile_exposure_cell


def synthetic_amigos() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset_id": "amigos_confirmation_v14",
            "trial_uid": f"p{participant:02d}:s{stimulus:02d}",
            "subject_uid": f"p{participant:02d}",
            "stimulus_uid": f"s{stimulus:02d}",
            "target_valence": (stimulus - 7.5) / 8,
            "target_arousal": (participant - 12) / 13,
        }
        for participant in range(25)
        for stimulus in range(16)
    ])


def test_crossed_split_supports_locked_resource_grid() -> None:
    trials = synthetic_amigos()
    metadata, strict = crossed_outer_assignments(trials, rotation_seed=20260914)[0]
    hashes, train_counts, test_counts = set(), set(), set()
    for participant_dose in (0, 1, 2, 4, 8):
        for stimulus_dose in (0, 1, 2, 4, 8):
            _, audit = compile_exposure_cell(
                trials,
                strict,
                participant_dose=participant_dose,
                stimulus_dose=stimulus_dose,
                maximum_participant_dose=8,
                maximum_stimulus_dose=8,
                seed=914000,
                target_columns=("target_valence", "target_arousal"),
            )
            hashes.add(audit["fixed_test_support_sha256"])
            train_counts.add(audit["counts"]["train"])
            test_counts.add(audit["counts"]["test"])
    assert metadata["participant_block"] == 0
    assert len(hashes) == len(train_counts) == len(test_counts) == 1


def test_each_rotation_covers_every_trial_once() -> None:
    trials = synthetic_amigos()
    observed = []
    for _, strict in crossed_outer_assignments(trials, rotation_seed=31):
        observed.extend(strict.loc[strict.split.eq("test"), "trial_uid"])
    assert len(observed) == len(set(observed)) == len(trials)
