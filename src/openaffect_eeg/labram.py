"""Auditable input geometry for the official LaBraM EEG foundation model."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

LABRAM_PATCH_SAMPLES = 200
LABRAM_MAX_TIME_PATCHES = 16
EMO_TRIAL_PATCHES = 30

# Pinned from 935963004/LaBraM utils.py. Position zero is reserved for CLS.
LABRAM_STANDARD_1020 = (
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ",
    "AF2", "AF4", "AF6", "AF8", "AF10", "F9", "F7", "F5", "F3", "F1",
    "FZ", "F2", "F4", "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3",
    "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10", "T9", "T7", "C5",
    "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10", "TP9", "TP7",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6",
    "PO8", "PO10", "O1", "OZ", "O2", "O9", "CB1", "CB2", "IZ", "O10",
    "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2", "CFC1", "CFC2",
    "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8", "CCP1", "CCP2",
    "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8", "T1", "T2", "FTT9H",
    "TTP7H", "TPP9H", "FTT10H", "TPP8H", "TPP10H", "FP1-F7", "F7-T7",
    "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3",
    "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)

EMO_64_CHANNELS = (
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1",
    "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
    "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2",
    "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2",
    "F9", "F10", "TP9", "TP10",
)


class LaBraMError(ValueError):
    pass


def prepare_labram_trial(
    values: np.ndarray,
    channel_names: Sequence[str],
    *,
    scale_factor: float = 1.0,
) -> tuple[np.ndarray, list[int]]:
    """Return a 30-second LaBraM trial as one-second patches and position IDs."""

    patches, position_ids = prepare_labram_segment(
        values,
        channel_names,
        scale_factor=scale_factor,
    )
    if patches.shape[1] != EMO_TRIAL_PATCHES:
        raise LaBraMError(
            f"LaBraM trial must contain {EMO_TRIAL_PATCHES} patches, "
            f"got {patches.shape[1]}"
        )
    return patches, position_ids


def prepare_labram_segment(
    values: np.ndarray,
    channel_names: Sequence[str],
    *,
    scale_factor: float = 1.0,
) -> tuple[np.ndarray, list[int]]:
    """Return a whole-second 200-Hz segment as channel x patch x sample input."""

    channels = [str(name).upper() for name in channel_names]
    if not channels or len(channels) != len(set(channels)):
        raise LaBraMError("LaBraM channel names must be unique and non-empty")
    trial = np.asarray(values, dtype=np.float32)
    if trial.ndim != 2 or trial.shape[0] != len(channels):
        raise LaBraMError(
            "LaBraM segment must have one row per named channel; "
            f"got {trial.shape} for {len(channels)} channels"
        )
    if trial.shape[1] < LABRAM_PATCH_SAMPLES or trial.shape[1] % LABRAM_PATCH_SAMPLES:
        raise LaBraMError(
            "LaBraM segment sample count must be a positive multiple of "
            f"{LABRAM_PATCH_SAMPLES}, got {trial.shape[1]}"
        )
    if not np.isfinite(trial).all():
        raise LaBraMError("LaBraM trial contains non-finite values")
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise LaBraMError("LaBraM scale factor must be finite and positive")
    position = {name.upper(): index + 1 for index, name in enumerate(LABRAM_STANDARD_1020)}
    unknown = [name for name in channels if name not in position]
    if unknown:
        raise LaBraMError(
            f"Channels are absent from the official position table: {unknown}"
        )
    patches = (trial / scale_factor).reshape(
        len(channels), trial.shape[1] // LABRAM_PATCH_SAMPLES, LABRAM_PATCH_SAMPLES
    )
    return patches, [position[name] for name in channels]


def iter_labram_windows(
    patches: np.ndarray,
    *,
    window_patches: int = 8,
) -> Iterator[np.ndarray]:
    """Yield contiguous windows within LaBraM's finite time-position table."""
    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != LABRAM_PATCH_SAMPLES:
        raise LaBraMError(
            "LaBraM patches must have shape channel x patch x 200 samples"
        )
    if not 1 <= window_patches <= LABRAM_MAX_TIME_PATCHES:
        raise LaBraMError(
            f"LaBraM window patches must be between 1 and {LABRAM_MAX_TIME_PATCHES}"
        )
    for start in range(0, values.shape[1], window_patches):
        yield values[:, start : start + window_patches, :]


def pool_labram_tokens(values: np.ndarray) -> np.ndarray:
    """Mean-pool channel and patch tokens while retaining embedding features."""
    tokens = np.asarray(values, dtype=np.float32)
    if tokens.ndim != 4:
        raise LaBraMError(
            "LaBraM tokens must be batch x channel x patch x dim"
        )
    if not np.isfinite(tokens).all():
        raise LaBraMError("LaBraM tokens contain non-finite values")
    return tokens.mean(axis=(1, 2))
