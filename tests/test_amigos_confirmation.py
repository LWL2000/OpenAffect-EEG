from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import savemat

from openaffect_eeg.amigos_confirmation import (
    crossed_outer_assignments,
    ingest_amigos,
    structural_qc,
)


def test_label_blind_qc_and_ingestion(tmp_path) -> None:
    signals = np.empty((1, 16), dtype=object)
    labels = np.empty((1, 16), dtype=object)
    rng = np.random.default_rng(4)
    for trial in range(16):
        signals[0, trial] = rng.normal(scale=12.0, size=(4480, 17))
        rating = np.zeros((1, 12))
        rating[0, 0] = 5.0 + (trial % 3 - 1)
        rating[0, 1] = 5.0 + (trial % 5 - 2) / 2
        labels[0, trial] = rating
    savemat(tmp_path / "Data_Preprocessed_P01.mat", {
        "joined_data": signals, "labels_selfassessment": labels
    })
    blind = structural_qc(tmp_path, subject_ids=[1])
    assert blind["status"] == "pass"
    assert blind["outcome_values_loaded"] is False
    assert "target" not in json_text(blind).lower()
    table, tensors, uids, features, report = ingest_amigos(
        tmp_path, blind, minimum_participants=1
    )
    assert table.shape[0] == 16
    assert tensors.shape == (16, 14, 3000)
    assert features.shape == (16, 14, 5)
    assert len(set(uids)) == 16
    assert report["participants"] == 1


def json_text(value) -> str:
    import json
    return json.dumps(value, sort_keys=True)


def test_crossed_assignments_cover_every_trial_once() -> None:
    table = pd.DataFrame([
        {"trial_uid": f"p{p}:s{s}", "subject_uid": f"p{p}", "stimulus_uid": f"s{s}"}
        for p in range(10) for s in range(8)
    ])
    assignments = crossed_outer_assignments(table, rotation_seed=17)
    assert len(assignments) == 20
    test_ids = [
        uid for _, assignment in assignments
        for uid in assignment.loc[assignment.split.eq("test"), "trial_uid"]
    ]
    assert len(test_ids) == len(set(test_ids)) == len(table)

