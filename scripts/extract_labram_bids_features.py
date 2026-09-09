#!/usr/bin/env python3
"""Extract trial-aligned LaBraM features from BIDS EEG without padding."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from extract_labram_features import (
    OFFICIAL_MODEL_VARIANT,
    OFFICIAL_REPOSITORY,
    load_labram,
    repository_commit,
    sha256_file,
)
from scipy.signal import resample_poly

from openaffect_eeg.foundation import FoundationFeatureError
from openaffect_eeg.labram import (
    LABRAM_PATCH_SAMPLES,
    iter_labram_windows,
    pool_labram_tokens,
    prepare_labram_segment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model_repository", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--preprocessing-device",
        help="Device for common-average reference and anti-aliased resampling",
    )
    parser.add_argument("--target-sampling-frequency", type=float, default=200.0)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--window-patches", type=int, default=8)
    parser.add_argument("--exclude-channels", nargs="*", default=())
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--random-init-seed", type=int, default=20260821)
    return parser.parse_args()


def eligible_trials(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trial_uid",
        "dataset_id",
        "eeg_path",
        "eeg_onset_s",
        "eeg_duration_s",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise FoundationFeatureError(
            f"Manifest is missing required columns: {missing}"
        )
    selected = manifest.copy()
    if "label_available" in selected:
        selected = selected.loc[selected["label_available"].astype(bool)].copy()
    if selected["trial_uid"].duplicated().any():
        raise FoundationFeatureError("Eligible trial IDs must be unique")
    if (selected["eeg_duration_s"].astype(float) <= 0).any():
        raise FoundationFeatureError("All selected trial durations must be positive")
    return selected.sort_values("trial_uid").reset_index(drop=True)


def read_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
    raise FoundationFeatureError(f"Unsupported BIDS EEG file: {path.name}")


def resample_to_target(
    values: np.ndarray,
    source_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:
    """Resample an EEG segment exactly when both rates have a rational ratio."""

    if not np.isfinite(source_sfreq) or source_sfreq <= 0:
        raise FoundationFeatureError("Source sampling frequency must be positive")
    if not np.isfinite(target_sfreq) or target_sfreq <= 0:
        raise FoundationFeatureError("Target sampling frequency must be positive")
    if np.isclose(source_sfreq, target_sfreq):
        return np.asarray(values, dtype=np.float32)
    ratio = Fraction(str(target_sfreq)) / Fraction(str(source_sfreq))
    resampled = resample_poly(
        np.asarray(values, dtype=np.float32), ratio.numerator, ratio.denominator, axis=1
    )
    return np.asarray(resampled, dtype=np.float32)


def select_channel_indices(
    channel_names: list[str], excluded_channels: tuple[str, ...] | list[str]
) -> tuple[list[int], list[str]]:
    """Select channels case-insensitively while preserving source order."""

    excluded = {str(name).upper() for name in excluded_channels}
    selected = [
        (index, str(name).upper())
        for index, name in enumerate(channel_names)
        if str(name).upper() not in excluded
    ]
    if not selected:
        raise FoundationFeatureError("Channel exclusion removed every channel")
    names = [name for _, name in selected]
    if len(names) != len(set(names)):
        raise FoundationFeatureError("Selected channel names must be unique")
    return [index for index, _ in selected], names


def _trial_patches(
    raw: mne.io.BaseRaw,
    row: object,
    *,
    channel_indices: list[int],
    channel_names: list[str],
    target_sfreq: float,
    scale_factor: float,
) -> tuple[np.ndarray, list[int]]:
    source_sfreq = float(raw.info["sfreq"])
    start = round(float(row.eeg_onset_s) * source_sfreq)
    source_samples = round(float(row.eeg_duration_s) * source_sfreq)
    stop = start + source_samples
    if start < 0 or stop > raw.n_times:
        raise FoundationFeatureError(
            f"Trial {row.trial_uid} exceeds recording bounds: [{start}, {stop})"
        )
    # MNE returns volts; LaBraM receives microvolt-valued EEG, as in its source data.
    values = (
        raw.get_data(picks=channel_indices, start=start, stop=stop).astype(np.float32)
        * 1_000_000.0
    )
    values -= values.mean(axis=0, keepdims=True)
    values = resample_to_target(values, source_sfreq, target_sfreq)
    expected_samples = round(float(row.eeg_duration_s) * target_sfreq)
    if values.shape[1] != expected_samples:
        raise FoundationFeatureError(
            f"Trial {row.trial_uid} resampled to {values.shape[1]} samples; "
            f"expected {expected_samples}"
        )
    if expected_samples % LABRAM_PATCH_SAMPLES:
        raise FoundationFeatureError(
            f"Trial {row.trial_uid} is not a whole number of 200-sample patches"
        )
    return prepare_labram_segment(values, channel_names, scale_factor=scale_factor)


def _recording_segments(
    raw: mne.io.BaseRaw,
    rows: pd.DataFrame,
    *,
    channel_indices: list[int],
    target_sfreq: float,
    torch: object | None = None,
    preprocessing_device: str = "cpu",
) -> dict[int, np.ndarray]:
    """Read a recording once, then crop and resample equal-length trials in bulk."""

    source_sfreq = float(raw.info["sfreq"])
    indexed_bounds: list[tuple[int, int, int]] = []
    for row in rows.itertuples(index=True):
        start = round(float(row.eeg_onset_s) * source_sfreq)
        source_samples = round(float(row.eeg_duration_s) * source_sfreq)
        stop = start + source_samples
        if start < 0 or stop > raw.n_times:
            raise FoundationFeatureError(
                f"Trial {row.trial_uid} exceeds recording bounds: [{start}, {stop})"
            )
        indexed_bounds.append((row.Index, start, stop))
    if not indexed_bounds:
        return {}

    read_start = min(start for _, start, _ in indexed_bounds)
    read_stop = max(stop for _, _, stop in indexed_bounds)
    recording = np.asarray(
        raw.get_data(picks=channel_indices, start=read_start, stop=read_stop),
        dtype=np.float32,
    )
    recording *= 1_000_000.0
    grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for output_index, start, stop in indexed_bounds:
        grouped[stop - start].append((output_index, start - read_start, stop - read_start))

    ratio = Fraction(str(target_sfreq)) / Fraction(str(source_sfreq))
    output: dict[int, np.ndarray] = {}
    for source_samples, bounds in grouped.items():
        segments = np.stack(
            [recording[:, start:stop] for _, start, stop in bounds]
        )
        if preprocessing_device.startswith("cuda"):
            if torch is None:
                raise FoundationFeatureError(
                    "CUDA preprocessing requires the loaded PyTorch module"
                )
            try:
                from torchaudio.functional import resample as torch_resample
            except ImportError as error:
                raise FoundationFeatureError(
                    "CUDA preprocessing requires torchaudio"
                ) from error
            tensor = torch.from_numpy(segments).to(preprocessing_device)
            tensor -= tensor.mean(dim=1, keepdim=True)
            if not np.isclose(source_sfreq, target_sfreq):
                tensor = torch_resample(tensor, source_sfreq, target_sfreq)
            segments = tensor.cpu().numpy()
        else:
            segments -= segments.mean(axis=1, keepdims=True)
            if not np.isclose(source_sfreq, target_sfreq):
                segments = resample_poly(
                    segments, ratio.numerator, ratio.denominator, axis=2
                ).astype(np.float32, copy=False)
        expected_samples = round(source_samples * target_sfreq / source_sfreq)
        if segments.shape[2] != expected_samples:
            raise FoundationFeatureError(
                f"Recording trials resampled to {segments.shape[2]} samples; "
                f"expected {expected_samples}"
            )
        for (output_index, _, _), segment in zip(bounds, segments):
            output[output_index] = segment
    return output


def extract_trials(
    model: object,
    torch: object,
    trials: pd.DataFrame,
    dataset_root: Path,
    *,
    batch_size: int,
    device: str,
    target_sfreq: float,
    scale_factor: float,
    window_patches: int,
    excluded_channels: tuple[str, ...],
    preprocessing_device: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    feature_sums: list[np.ndarray | None] = [None] * len(trials)
    patch_weights = np.zeros(len(trials), dtype=int)
    expected_weights = np.zeros(len(trials), dtype=int)
    channel_names: list[str] | None = None

    grouped_recordings = trials.groupby("eeg_path", sort=True)
    recording_count = grouped_recordings.ngroups
    started_at = time.perf_counter()
    for recording_index, (relative_path, rows) in enumerate(
        grouped_recordings, start=1
    ):
        recording = dataset_root / str(relative_path)
        raw = read_raw(recording)
        by_patch_count: dict[
            int, list[tuple[int, np.ndarray, list[int]]]
        ] = defaultdict(list)
        try:
            channel_indices, current_channels = select_channel_indices(
                list(raw.ch_names), excluded_channels
            )
            if channel_names is None:
                channel_names = current_channels
            elif channel_names != current_channels:
                raise FoundationFeatureError(
                    f"EEG channel order differs in {recording.name}"
                )
            segments = _recording_segments(
                raw,
                rows,
                channel_indices=channel_indices,
                target_sfreq=target_sfreq,
                torch=torch,
                preprocessing_device=preprocessing_device,
            )
            for row in rows.itertuples(index=True):
                patches, channel_ids = prepare_labram_segment(
                    segments[row.Index],
                    current_channels,
                    scale_factor=scale_factor,
                )
                expected_weights[row.Index] = patches.shape[1]
                for window in iter_labram_windows(patches, window_patches=window_patches):
                    by_patch_count[window.shape[1]].append((row.Index, window, channel_ids))
        finally:
            raw.close()

        for patch_count, records in sorted(by_patch_count.items()):
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                channel_id_lists = {tuple(item[2]) for item in batch}
                if len(channel_id_lists) != 1:
                    raise FoundationFeatureError(
                        "LaBraM channel position IDs differ within a batch"
                    )
                tensor = (
                    torch.from_numpy(np.stack([item[1] for item in batch]))
                    .float()
                    .to(device)
                )
                with torch.inference_mode():
                    flat_tokens = model(
                        tensor,
                        input_chans=[0, *batch[0][2]],
                        return_patch_tokens=True,
                    )
                tokens = flat_tokens.detach().cpu().numpy()
                expected_tokens = len(current_channels) * patch_count
                if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
                    raise FoundationFeatureError(
                        "Official LaBraM returned unexpected patch-token geometry: "
                        f"{tokens.shape}, expected {expected_tokens} tokens"
                    )
                pooled = pool_labram_tokens(
                    tokens.reshape(
                        len(batch), len(current_channels), patch_count, -1
                    )
                )
                for (output_index, _, _), feature in zip(batch, pooled):
                    contribution = feature * patch_count
                    feature_sums[output_index] = (
                        contribution
                        if feature_sums[output_index] is None
                        else feature_sums[output_index] + contribution
                    )
                    patch_weights[output_index] += patch_count
        if recording_index == 1 or recording_index % 10 == 0 or recording_index == recording_count:
            elapsed = time.perf_counter() - started_at
            print(
                f"Processed recordings {recording_index}/{recording_count} "
                f"in {elapsed:.1f}s",
                flush=True,
            )

    if channel_names is None:
        raise FoundationFeatureError("No BIDS EEG recordings were selected")

    if not np.array_equal(patch_weights, expected_weights) or any(
        value is None for value in feature_sums
    ):
        raise FoundationFeatureError("LaBraM extraction did not cover every selected patch")
    features = np.stack(
        [np.asarray(total / weight, dtype=np.float32) for total, weight in zip(feature_sums, patch_weights)]
    )
    return (
        trials["trial_uid"].to_numpy(dtype=str),
        features,
        channel_names,
        {str(value): int(count) for value, count in pd.Series(expected_weights).value_counts().sort_index().items()},
    )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    preprocessing_device = args.preprocessing_device or (
        "cuda" if str(args.device).startswith("cuda") else "cpu"
    )
    if not preprocessing_device.startswith(("cpu", "cuda")):
        raise ValueError("Preprocessing device must be CPU or CUDA")
    if preprocessing_device.startswith("cuda") and not str(args.device).startswith(
        "cuda"
    ):
        raise ValueError("CUDA preprocessing requires a CUDA model device")
    manifest = pd.read_csv(args.manifest, sep="\t")
    trials = eligible_trials(manifest)
    dataset_ids = sorted(trials["dataset_id"].astype(str).unique())
    if len(dataset_ids) != 1:
        raise FoundationFeatureError("Feature extraction expects one dataset per manifest")
    model, torch, load_metadata = load_labram(
        args.model_repository.resolve(),
        args.checkpoint.resolve(),
        args.device,
        random_init=args.random_init,
        seed=args.random_init_seed,
    )
    trial_uids, features, channel_names, patch_counts = extract_trials(
        model,
        torch,
        trials,
        args.dataset_root.resolve(),
        batch_size=args.batch_size,
        device=args.device,
        target_sfreq=args.target_sampling_frequency,
        scale_factor=args.scale_factor,
        window_patches=args.window_patches,
        excluded_channels=tuple(args.exclude_channels),
        preprocessing_device=preprocessing_device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        trial_uids=trial_uids,
        feature_names=np.asarray([f"labram_pool_{index:03d}" for index in range(features.shape[1])]),
    )
    metadata = {
        "dataset_id": dataset_ids[0],
        "feature_shape": list(features.shape),
        "manifest_sha256": sha256_file(args.manifest),
        "model": "LaBraM",
        "model_variant": OFFICIAL_MODEL_VARIANT,
        "model_repository": OFFICIAL_REPOSITORY,
        "model_repository_commit": repository_commit(args.model_repository.resolve()),
        "model_initialization": load_metadata["model_initialization"],
        "checkpoint_sha256": load_metadata["checkpoint_sha256"],
        "random_init_seed": load_metadata["random_init_seed"],
        "channel_order": channel_names,
        "excluded_channels": list(args.exclude_channels),
        "sampling_frequency_hz": args.target_sampling_frequency,
        "patch_count_distribution": patch_counts,
        "source_preprocessing": "per-trial crop, volts-to-microvolts conversion, common-average reference, anti-aliased resampling; no padding or cross-trial windows",
        "preprocessing_device": preprocessing_device,
        "resampling_backend": (
            "torchaudio_bandlimited_sinc"
            if preprocessing_device.startswith("cuda")
            else "scipy_resample_poly"
        ),
        "input_scale_factor": args.scale_factor,
        "pooling": "mean over final channel and patch tokens, weighted by represented patches",
        "memory_strategy": "recording-streamed inference",
        "eligible_trial_count": len(trials),
        "extracted_trial_count": len(features),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Features: {args.output} {features.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
