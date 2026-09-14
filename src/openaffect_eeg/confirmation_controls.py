"""Prespecified negative and positive controls for v14 neural training."""
from __future__ import annotations

import numpy as np


def residual_training_targets(resources: dict, *, control: str, seed: int) -> dict[str, np.ndarray]:
    """Return observed or split-preserving permuted residual targets."""
    train = np.asarray(resources["train_y"] - resources["train_loo"], dtype=float)
    validation = np.asarray(resources["val_y"] - resources["val_prior"], dtype=float)
    if control in {"observed", "synthetic_signal"}:
        return {"train": train, "validation": validation, "fit": np.concatenate([train, validation])}
    if control != "label_permutation":
        raise ValueError(f"Unknown confirmation control {control}")
    rng = np.random.default_rng(seed)
    train_permuted = train[rng.permutation(len(train))]
    validation_permuted = validation[rng.permutation(len(validation))]
    return {
        "train": train_permuted,
        "validation": validation_permuted,
        "fit": np.concatenate([train_permuted, validation_permuted]),
    }


def synthetic_target_signal(
    targets: np.ndarray,
    *,
    channels: int,
    samples: int,
    sampling_hz: float = 100.0,
    amplitude: float = 100.0,
) -> np.ndarray:
    """Encode two known targets in two channels for a sensitivity control."""
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Synthetic control requires finite valence and arousal targets")
    if channels < 2 or samples < 10 or sampling_hz <= 0 or amplitude <= 0:
        raise ValueError("Invalid synthetic signal dimensions")
    time = np.arange(samples, dtype=np.float32) / float(sampling_hz)
    signal = np.zeros((len(values), channels, samples), dtype=np.float32)
    signal[:, 0, :] = (
        amplitude * values[:, 0, None] * np.sin(2 * np.pi * 10.0 * time)[None, :]
    )
    signal[:, 1, :] = (
        amplitude * values[:, 1, None] * np.sin(2 * np.pi * 15.0 * time)[None, :]
    )
    return signal
