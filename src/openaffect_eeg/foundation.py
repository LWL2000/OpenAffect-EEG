"""Input validation and pooling for frozen EEG foundation-model features."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EMO_CHANNEL_COUNT = 64
EMO_SAMPLING_FREQUENCY = 200
EMO_TRIAL_SECONDS = 30
CBRAMOD_PATCH_SAMPLES = 200


class FoundationFeatureError(ValueError):
    pass


def load_foundation_archive(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an auditable trial-aligned foundation-feature archive."""
    source = Path(path)
    if not source.is_file():
        raise FoundationFeatureError(f"Foundation feature archive is unavailable: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "trial_uids" not in archive or "features" not in archive:
                raise FoundationFeatureError(
                    "Foundation feature archive must contain trial_uids and features"
                )
            trial_uids = np.asarray(archive["trial_uids"]).astype(str)
            features = np.asarray(archive["features"], dtype=np.float32)
    except (OSError, ValueError) as error:
        raise FoundationFeatureError(
            f"Cannot load foundation feature archive: {source}"
        ) from error

    if trial_uids.ndim != 1 or features.ndim != 2:
        raise FoundationFeatureError(
            "Foundation trial_uids must be one-dimensional and features two-dimensional"
        )
    if len(trial_uids) != len(features):
        raise FoundationFeatureError(
            "Foundation trial_uids and features must contain the same number of rows"
        )
    if len(np.unique(trial_uids)) != len(trial_uids):
        raise FoundationFeatureError("Foundation trial_uids must be unique")
    if not np.isfinite(features).all():
        raise FoundationFeatureError("Foundation features contain non-finite values")
    return trial_uids, features


def canonicalize_emo_reorder(values: np.ndarray) -> np.ndarray:
    """Return an EmoEEG-MC reorder array as trials x channels x samples."""
    array = np.asarray(values)
    expected_samples = EMO_SAMPLING_FREQUENCY * EMO_TRIAL_SECONDS
    if array.ndim != 3 or array.shape[-1] != expected_samples:
        raise FoundationFeatureError(
            "EmoEEG-MC reorder data must be three-dimensional with "
            f"{expected_samples} samples per trial"
        )
    if array.shape[0] == EMO_CHANNEL_COUNT:
        array = array.transpose(1, 0, 2)
    elif array.shape[1] != EMO_CHANNEL_COUNT:
        raise FoundationFeatureError(
            f"EmoEEG-MC reorder data must contain {EMO_CHANNEL_COUNT} EEG channels"
        )
    return array


def load_emo_reorder(path: str | Path) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FoundationFeatureError(f"Reorder data is unavailable: {source}")
    try:
        values = np.load(source, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise FoundationFeatureError(f"Cannot load reorder data: {source}") from error
    return canonicalize_emo_reorder(values)


def prepare_cbramod_trial(
    values: np.ndarray,
    *,
    scale_factor: float = 100.0,
) -> np.ndarray:
    """Convert one 30-second 200-Hz trial to CBraMod one-second patches."""
    trial = np.asarray(values, dtype=np.float32)
    expected_samples = EMO_SAMPLING_FREQUENCY * EMO_TRIAL_SECONDS
    if trial.shape != (EMO_CHANNEL_COUNT, expected_samples):
        raise FoundationFeatureError(
            "CBraMod EmoEEG-MC input must have shape "
            f"({EMO_CHANNEL_COUNT}, {expected_samples})"
        )
    if not np.isfinite(trial).all():
        raise FoundationFeatureError("EEG trial contains non-finite values")
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise FoundationFeatureError("Scale factor must be finite and positive")
    return (trial / scale_factor).reshape(
        EMO_CHANNEL_COUNT,
        EMO_TRIAL_SECONDS,
        CBRAMOD_PATCH_SAMPLES,
    )


def pool_foundation_tokens(values: np.ndarray) -> np.ndarray:
    """Mean-pool channel and temporal tokens while retaining embedding features."""
    tokens = np.asarray(values)
    if tokens.ndim != 4:
        raise FoundationFeatureError(
            "Foundation-model tokens must have shape batch x channels x patches x dim"
        )
    if not np.isfinite(tokens).all():
        raise FoundationFeatureError("Foundation-model tokens contain non-finite values")
    return tokens.mean(axis=(1, 2))


def reorder_path_from_de_path(path: str) -> str:
    suffix = "_de.npy"
    if not isinstance(path, str) or not path.endswith(suffix):
        raise FoundationFeatureError(f"Unexpected DE feature path: {path!r}")
    return f"{path[:-len(suffix)]}_reorder.npy"
