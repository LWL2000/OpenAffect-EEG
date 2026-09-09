"""Training-independent EEG features for transparent benchmark baselines."""

from __future__ import annotations

import re

import numpy as np
from scipy.signal import welch

BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)
_DENS_EEG_CHANNEL = re.compile(r"E(\d+)$")
_DS006866_AUXILIARY_CHANNELS = {
    "HEO",
    "VEO",
    "EKG",
    "GSR",
    "EMG",
    "Trigger",
}


class FeatureError(ValueError):
    pass


def available_sample_frames(
    file_size_bytes: int,
    channel_count: int,
    *,
    sample_bytes: int = 4,
) -> tuple[int, int]:
    if file_size_bytes < 0 or channel_count <= 0 or sample_bytes <= 0:
        raise FeatureError("File size and sample geometry must be positive")
    frame_bytes = channel_count * sample_bytes
    return divmod(file_size_bytes, frame_bytes)


def log_bandpower(
    data: np.ndarray,
    sampling_frequency: float,
    *,
    window_s: float = 2.0,
) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise FeatureError("EEG data must have shape (channels, samples)")
    if values.shape[1] < 2:
        raise FeatureError("EEG data must contain at least two samples")
    if not np.isfinite(values).all():
        raise FeatureError("EEG data contains non-finite values")
    if sampling_frequency <= 2 * max(high for _, _, high in BANDS):
        raise FeatureError("Sampling frequency is too low for the configured bands")

    segment_samples = min(values.shape[1], round(window_s * sampling_frequency))
    frequencies, density = welch(
        values,
        fs=sampling_frequency,
        nperseg=segment_samples,
        noverlap=segment_samples // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    powers = []
    for _, low, high in BANDS:
        selected = frequencies[(frequencies >= low) & (frequencies < high)]
        if len(selected) < 2:
            raise FeatureError(f"Insufficient frequency bins for band {low}-{high} Hz")
        mask = (frequencies >= low) & (frequencies < high)
        powers.append(integrator(density[:, mask], selected, axis=-1))
    bandpower = np.stack(powers, axis=1)
    return np.log(np.maximum(bandpower, np.finfo(np.float64).tiny))


def global_bandpower_summary(channel_features: np.ndarray) -> np.ndarray:
    values = np.asarray(channel_features, dtype=np.float64)
    if values.ndim != 2:
        raise FeatureError("Channel features must have shape (channels, bands)")
    return np.concatenate([values.mean(axis=0), values.std(axis=0)])


def select_eeg_channels(dataset_id: str, channel_names: list[str]) -> list[int]:
    if dataset_id == "ds003751":
        selected = []
        for index, name in enumerate(channel_names):
            match = _DENS_EEG_CHANNEL.fullmatch(name)
            if match is not None and 1 <= int(match.group(1)) <= 128:
                selected.append(index)
        if len(selected) != 128:
            raise FeatureError(
                f"DENS must expose E1-E128 exactly; selected {len(selected)} channels"
            )
        return selected
    if dataset_id == "ds002721":
        if len(channel_names) != 19:
            raise FeatureError(
                f"MusicEEG must expose 19 EEG channels; found {len(channel_names)}"
            )
        return list(range(19))
    if dataset_id == "ds006866":
        selected = [
            index
            for index, name in enumerate(channel_names)
            if name not in _DS006866_AUXILIARY_CHANNELS
        ]
        if len(selected) != 64:
            raise FeatureError(
                "ds006866 must expose 64 EEG channels after excluding "
                f"HEO/VEO/EKG/GSR/Trigger; selected {len(selected)} channels"
            )
        return selected
    raise FeatureError(f"No audited channel selection for dataset {dataset_id}")
