"""Extract frozen LaBraM features from strict ds006850 image epochs on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from extract_labram_features import (
    OFFICIAL_MODEL_VARIANT,
    OFFICIAL_REPOSITORY,
    load_labram,
    repository_commit,
    sha256_file,
)

from openaffect_eeg.labram import pool_labram_tokens, prepare_labram_segment
from openaffect_eeg.urban_brainvision import (
    nominal_sampling_frequency,
    open_urban_eeg,
    read_urban_segment,
    trial_sample_bounds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model_repository", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0 or not str(args.device).startswith("cuda"):
        raise ValueError("This frozen confirmation requires a positive CUDA batch size")

    trials = (
        pd.read_csv(args.manifest, sep="\t")
        .sort_values("trial_uid")
        .reset_index(drop=True)
    )
    model, torch, load_metadata = load_labram(
        args.model_repository.resolve(),
        args.checkpoint.resolve(),
        args.device,
        random_init=False,
        seed=None,
    )
    try:
        from torchaudio.functional import resample as torch_resample
    except ImportError as error:
        raise RuntimeError("CUDA preprocessing requires torchaudio") from error

    features: list[np.ndarray | None] = [None] * len(trials)
    pending: list[tuple[int, np.ndarray]] = []
    channel_order = None
    channel_ids = None
    started = time.perf_counter()

    def flush() -> None:
        nonlocal pending
        while len(pending) >= args.batch_size:
            infer(pending[: args.batch_size])
            pending = pending[args.batch_size :]

    def infer(records: list[tuple[int, np.ndarray]]) -> None:
        values = np.stack([record[1] for record in records])
        tensor = torch.from_numpy(values).float().to(args.device)
        tensor -= tensor.mean(dim=1, keepdim=True)
        tensor = torch_resample(tensor, 500, 200)
        if tensor.shape[-1] != 600:
            raise ValueError(f"LaBraM resampling produced {tensor.shape[-1]} samples")
        tensor = tensor.reshape(len(records), len(channel_order), 3, 200)
        with torch.inference_mode():
            tokens = model(
                tensor,
                input_chans=[0, *channel_ids],
                return_patch_tokens=True,
            )
        expected_tokens = len(channel_order) * 3
        if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Official LaBraM returned {tuple(tokens.shape)}; "
                f"expected {expected_tokens} patch tokens"
            )
        pooled = pool_labram_tokens(
            tokens.detach()
            .float()
            .cpu()
            .numpy()
            .reshape(len(records), len(channel_order), 3, -1)
        )
        for (index, _), feature in zip(records, pooled, strict=True):
            features[index] = feature.astype(np.float32)

    groups = trials.groupby("eeg_path", sort=True)
    for recording_index, (relative, rows) in enumerate(groups, start=1):
        raw, factors, _ = open_urban_eeg(args.dataset_root / str(relative))
        try:
            nominal, _ = nominal_sampling_frequency(float(raw.info["sfreq"]))
            channels = list(raw.ch_names)
            if channel_order is None:
                channel_order = channels
                _, channel_ids = prepare_labram_segment(
                    np.zeros((len(channels), 600), dtype=np.float32), channels
                )
            elif channels != channel_order:
                raise ValueError(f"EEG channel order differs in {relative}")
            for row in rows.itertuples(index=True):
                start, stop, _ = trial_sample_bounds(
                    row, sampling_frequency=nominal, available_samples=raw.n_times
                )
                values = read_urban_segment(raw, factors, start=start, stop=stop)
                if values.shape != (len(channel_order), 1500):
                    raise ValueError(
                        f"Trial {row.trial_uid} has unsupported shape {values.shape}"
                    )
                pending.append((row.Index, (values * 1_000_000.0).astype(np.float32)))
                flush()
        finally:
            raw.close()
        if (
            recording_index == 1
            or recording_index % 10 == 0
            or recording_index == groups.ngroups
        ):
            elapsed = time.perf_counter() - started
            print(
                f"LaBraM input recordings {recording_index}/{groups.ngroups}; "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )
    if pending:
        infer(pending)
        pending = []
    if any(feature is None for feature in features):
        raise ValueError("LaBraM extraction did not cover every trial")

    feature_array = np.stack(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        trial_uids=trials["trial_uid"].to_numpy(str),
        features=feature_array,
        feature_names=np.asarray(
            [f"labram_pool_{index:03d}" for index in range(feature_array.shape[1])]
        ),
    )
    temporary.replace(args.output)
    metadata = {
        "dataset_id": "ds006850",
        "feature_shape": list(feature_array.shape),
        "manifest_sha256": sha256_file(args.manifest),
        "output_sha256": sha256_file(args.output),
        "model": "LaBraM",
        "model_variant": OFFICIAL_MODEL_VARIANT,
        "model_repository": OFFICIAL_REPOSITORY,
        "model_repository_commit": repository_commit(args.model_repository.resolve()),
        "model_implementation_sha256": sha256_file(
            args.model_repository.resolve() / "modeling_finetune.py"
        ),
        "model_initialization": load_metadata["model_initialization"],
        "checkpoint_sha256": load_metadata["checkpoint_sha256"],
        "missing_checkpoint_keys": load_metadata["missing_checkpoint_keys"],
        "unexpected_checkpoint_keys": load_metadata["unexpected_checkpoint_keys"],
        "device": str(args.device),
        "cuda_device_name": torch.cuda.get_device_name(args.device),
        "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        "batch_size": args.batch_size,
        "source_sampling_frequency_hz": 500.0,
        "target_sampling_frequency_hz": 200.0,
        "resampling_backend": "torchaudio_bandlimited_sinc_cuda_500_to_200",
        "patches_per_trial": 3,
        "padding": "none",
        "input_units": "microvolt-valued EEG after verified volts conversion",
        "channel_reference": "per-sample common average on CUDA",
        "channel_order": channel_order,
        "pooling": "mean over 64 channels and three one-second patch tokens",
        "trial_count": len(trials),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in metadata.items() if key != "channel_order"}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
