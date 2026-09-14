from __future__ import annotations

import numpy as np

from openaffect_eeg.confirmation_controls import residual_training_targets, synthetic_target_signal


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
