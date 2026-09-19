"""Prespecified negative and positive controls for v14 neural training."""
from __future__ import annotations

import numpy as np


def residual_training_targets(resources: dict, *, control: str, seed: int) -> dict[str, np.ndarray]:
    """Return observed or split-preserving permuted residual targets."""
    train = np.asarray(resources["train_y"] - resources["train_loo"], dtype=float)
    validation = np.asarray(resources["val_y"] - resources["val_prior"], dtype=float)
    if control in {
        "observed",
        "synthetic_signal",
        "synthetic_residual_signal",
        "synthetic_signed_power_residual",
    }:
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


def residual_signal_targets(resources: dict, trials, targets) -> np.ndarray:
    """Build split-aligned oracle residuals for a positive-control signal.

    Training and validation rows exactly match their fitted residual targets.
    Other rows use the fit-only population prior that is used when predictions
    are scored. This avoids injecting raw outcomes into a residual learner,
    which would double-count the population prior at evaluation time.
    """
    columns = list(targets)
    values = np.asarray(trials[columns].to_numpy(float) - resources["all_prior"], dtype=float)
    train_index = resources["train"].tensor_index.to_numpy(int)
    validation_index = resources["validation"].tensor_index.to_numpy(int)
    train = np.asarray(resources["train_y"] - resources["train_loo"], dtype=float)
    validation = np.asarray(resources["val_y"] - resources["val_prior"], dtype=float)
    if values.shape != (len(trials), len(columns)):
        raise ValueError("Malformed split-aligned residual signal targets")
    values[train_index] = train
    values[validation_index] = validation
    if not np.isfinite(values).all():
        raise ValueError("Split-aligned residual signal targets must be finite")
    return values


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


def synthetic_signed_power_signal(
    targets: np.ndarray,
    *,
    channels: int,
    samples: int,
    sampling_hz: float = 100.0,
    amplitude: float = 100.0,
) -> np.ndarray:
    """Encode signed targets as nonnegative carrier power in four channels.

    Positive and negative values occupy different channels, so random crop
    phase and phase-invariant temporal features cannot erase the target sign.
    This is an exploratory diagnostic designed after the two signed-carrier
    controls failed; it must not be presented as prespecified confirmation.
    """
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Signed-power control requires two finite targets")
    if channels < 4 or samples < 10 or sampling_hz <= 0 or amplitude <= 0:
        raise ValueError("Invalid signed-power signal dimensions")
    time = np.arange(samples, dtype=np.float32) / float(sampling_hz)
    carriers = (
        np.sin(2 * np.pi * 10.0 * time),
        np.sin(2 * np.pi * 10.0 * time),
        np.sin(2 * np.pi * 15.0 * time),
        np.sin(2 * np.pi * 15.0 * time),
    )
    magnitudes = np.stack(
        [
            np.clip(values[:, 0], 0.0, None),
            np.clip(-values[:, 0], 0.0, None),
            np.clip(values[:, 1], 0.0, None),
            np.clip(-values[:, 1], 0.0, None),
        ],
        axis=1,
    )
    signal = np.zeros((len(values), channels, samples), dtype=np.float32)
    for channel, carrier in enumerate(carriers):
        signal[:, channel, :] = amplitude * magnitudes[:, channel, None] * carrier[None, :]
    return signal
