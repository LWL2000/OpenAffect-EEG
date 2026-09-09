"""Memory-mapped raw-trial inputs for end-to-end EmoEEG-MC models."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from openaffect_eeg.foundation import (
    EMO_CHANNEL_COUNT,
    EMO_SAMPLING_FREQUENCY,
    EMO_TRIAL_SECONDS,
)


def eligible_emo_raw_trials(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return uniquely aligned, supervised EmoEEG-MC rows in stable order."""
    required = {
        "dataset_id",
        "trial_uid",
        "de_path",
        "de_trial_index",
        "label_available",
        "feature_alignment_status",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Trial manifest is missing columns: {sorted(missing)}")
    selected = manifest.loc[
        manifest["dataset_id"].eq("ds005540")
        & manifest["label_available"].astype(bool)
        & manifest["feature_alignment_status"].eq("verified")
    ].copy()
    if selected["trial_uid"].duplicated().any():
        raise ValueError("Eligible raw EEG trial identifiers must be unique")
    return selected.sort_values("trial_uid").reset_index(drop=True)


def trial_uid_array(values: object) -> np.ndarray:
    """Convert identifiers to a fixed-width Unicode array that needs no pickle."""
    identifiers = np.asarray([str(value) for value in values], dtype=str)
    if identifiers.ndim != 1:
        raise ValueError("Trial identifiers must be one-dimensional")
    return identifiers


def resample_emo_trial(
    values: np.ndarray,
    *,
    target_frequency: int,
) -> np.ndarray:
    """Anti-alias and resample one canonical 30-second EmoEEG-MC trial."""
    trial = np.asarray(values)
    expected_shape = (
        EMO_CHANNEL_COUNT,
        EMO_SAMPLING_FREQUENCY * EMO_TRIAL_SECONDS,
    )
    if trial.shape != expected_shape:
        raise ValueError(f"EmoEEG-MC trial must have shape {expected_shape}")
    if not np.isfinite(trial).all():
        raise ValueError("EmoEEG-MC trial contains non-finite values")
    if not isinstance(target_frequency, int) or target_frequency < 1:
        raise ValueError("Target frequency must be a positive integer")
    divisor = gcd(EMO_SAMPLING_FREQUENCY, target_frequency)
    resampled = resample_poly(
        trial,
        target_frequency // divisor,
        EMO_SAMPLING_FREQUENCY // divisor,
        axis=-1,
    )
    expected_samples = target_frequency * EMO_TRIAL_SECONDS
    if resampled.shape != (EMO_CHANNEL_COUNT, expected_samples):
        raise RuntimeError(
            f"Resampling produced {resampled.shape}; expected "
            f"({EMO_CHANNEL_COUNT}, {expected_samples})"
        )
    return resampled.astype(np.float32, copy=False)


def load_raw_trial_archive(
    tensor_path: str | Path,
    trial_uid_path: str | Path,
) -> tuple[np.ndarray, np.memmap]:
    """Load and validate a trial-aligned memory-mapped raw EEG archive."""
    try:
        tensors = np.load(tensor_path, mmap_mode="r", allow_pickle=False)
        trial_uids = np.load(trial_uid_path, allow_pickle=False).astype(str)
    except (OSError, ValueError) as error:
        raise ValueError("Cannot load raw EEG trial archive") from error
    if not isinstance(tensors, np.memmap) or tensors.ndim != 3:
        raise ValueError("Raw EEG tensors must be a memory-mapped 3D array")
    if tensors.shape[1] != EMO_CHANNEL_COUNT:
        raise ValueError(f"Raw EEG tensors must contain {EMO_CHANNEL_COUNT} channels")
    if tensors.dtype != np.float32:
        raise ValueError("Raw EEG tensors must use float32 storage")
    if trial_uids.ndim != 1 or len(trial_uids) != len(tensors):
        raise ValueError("Raw EEG tensors and trial identifiers are not aligned")
    if len(np.unique(trial_uids)) != len(trial_uids):
        raise ValueError("Raw EEG trial identifiers must be unique")
    if not np.isfinite(tensors).all():
        raise ValueError("Raw EEG tensors contain non-finite values")
    return trial_uids, tensors
