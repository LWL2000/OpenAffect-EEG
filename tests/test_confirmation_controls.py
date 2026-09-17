from __future__ import annotations

import numpy as np

import pandas as pd

from openaffect_eeg.confirmation_controls import (
    residual_signal_targets,
    residual_training_targets,
    synthetic_target_signal,
)


def resources() -> dict:
    return {
        "train_y": np.arange(12, dtype=float).reshape(6, 2),
        "train_loo": np.zeros((6, 2)),
        "val_y": np.arange(8, dtype=float).reshape(4, 2),
        "val_prior": np.zeros((4, 2)),
    }


def test_label_permutation_is_deterministic_and_preserves_split_values() -> None:
    first = residual_training_targets(resources(), control="label_permutation", seed=31)
    second = residual_training_targets(resources(), control="label_permutation", seed=31)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert sorted(first["train"][:, 0]) == sorted(resources()["train_y"][:, 0])
    assert np.array_equal(first["fit"], np.concatenate([first["train"], first["validation"]]))


def test_synthetic_signal_encodes_both_targets_without_touching_other_channels() -> None:
    targets = np.array([[-1.0, 0.5], [0.25, -0.75]])
    signal = synthetic_target_signal(targets, channels=4, samples=100)
    assert signal.shape == (2, 4, 100)
    assert not np.allclose(signal[:, 0], 0)
    assert not np.allclose(signal[:, 1], 0)
    assert np.allclose(signal[:, 2:], 0)


def test_residual_signal_targets_match_split_training_objective() -> None:
    trials = pd.DataFrame(
        {
            "target_valence": [1.0, 2.0, 3.0, 4.0],
            "target_arousal": [4.0, 3.0, 2.0, 1.0],
        }
    )
    split_resources = {
        "all_prior": np.ones((4, 2)),
        "train": pd.DataFrame({"tensor_index": [0, 2]}),
        "validation": pd.DataFrame({"tensor_index": [1]}),
        "train_y": np.array([[1.0, 4.0], [3.0, 2.0]]),
        "train_loo": np.array([[0.25, 0.5], [0.75, 0.25]]),
        "val_y": np.array([[2.0, 3.0]]),
        "val_prior": np.array([[0.5, 0.75]]),
    }
    result = residual_signal_targets(
        split_resources, trials, ("target_valence", "target_arousal")
    )
    assert np.allclose(result[[0, 2]], split_resources["train_y"] - split_resources["train_loo"])
    assert np.allclose(result[1], split_resources["val_y"][0] - split_resources["val_prior"][0])
    assert np.allclose(result[3], [3.0, 0.0])
