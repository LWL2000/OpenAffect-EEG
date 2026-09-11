#!/usr/bin/env python3
"""Extract official frozen LaBraM representations from EmoEEG-MC trials."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from openaffect_eeg.foundation import (
    FoundationFeatureError,
    load_emo_reorder,
    reorder_path_from_de_path,
)
from openaffect_eeg.labram import (
    EMO_64_CHANNELS,
    iter_labram_windows,
    pool_labram_tokens,
    prepare_labram_trial,
)

OFFICIAL_REPOSITORY = "https://github.com/935963004/LaBraM"
OFFICIAL_MODEL_VARIANT = "labram_base_patch200_200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model_repository", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--window-patches", type=int, default=8)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--random-init-seed", type=int, default=20260820)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def import_official_labram(repository: Path):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "LaBraM extraction requires the project's foundation dependencies"
        ) from error
    source = repository / "modeling_finetune.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official LaBraM model source is unavailable: {source}")
    spec = importlib.util.spec_from_file_location(
        "openaffect_official_labram_modeling_finetune", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official LaBraM model source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(spec.name) is module:
            del sys.modules[spec.name]
        raise
    return getattr(module, OFFICIAL_MODEL_VARIANT), torch


def _new_model(model_factory):
    return model_factory(
        pretrained=False,
        num_classes=0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        use_mean_pooling=False,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )


def _checkpoint_state(checkpoint: object) -> OrderedDict:
    if not isinstance(checkpoint, dict):
        raise TypeError("Official LaBraM checkpoint must contain a state mapping")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Official LaBraM checkpoint model entry is not a mapping")
    has_student_prefix = any(str(key).startswith("student.") for key in state)
    filtered = OrderedDict()
    for raw_key, value in state.items():
        key = str(raw_key)
        if has_student_prefix:
            if not key.startswith("student."):
                continue
            key = key[len("student.") :]
        if key in {
            "head.weight",
            "head.bias",
            "lm_head.weight",
            "lm_head.bias",
            "mask_token",
        } or "relative_position_index" in key:
            continue
        filtered[key] = value
    if not filtered:
        raise RuntimeError("No LaBraM student parameters were found in checkpoint")
    return filtered


def _load_checkpoint_safely(torch, checkpoint: Path):
    try:
        numpy_multiarray = importlib.import_module("numpy._core.multiarray")
    except ModuleNotFoundError:
        numpy_multiarray = importlib.import_module("numpy.core.multiarray")
    safe_globals = [
        (numpy_multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float64)),
        argparse.Namespace,
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(checkpoint, map_location="cpu", weights_only=True)


def load_labram(
    repository: Path,
    checkpoint: Path,
    device: str,
    *,
    random_init: bool = False,
    seed: int = 20260820,
):
    if not random_init and not checkpoint.is_file():
        raise FileNotFoundError(f"Official LaBraM checkpoint is unavailable: {checkpoint}")
    model_factory, torch = import_official_labram(repository)
    if random_init:
        torch.manual_seed(seed)
    model = _new_model(model_factory)
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    checkpoint_hash = None
    if not random_init:
        checkpoint_hash = sha256_file(checkpoint)
        loaded = _load_checkpoint_safely(torch, checkpoint)
        incompatible = model.load_state_dict(_checkpoint_state(loaded), strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
    model.eval().to(device)
    metadata = {
        "model_initialization": "random" if random_init else "pretrained",
        "checkpoint_sha256": checkpoint_hash,
        "random_init_seed": seed if random_init else None,
        "missing_checkpoint_keys": missing_keys,
        "unexpected_checkpoint_keys": unexpected_keys,
    }
    return model, torch, metadata


def eligible_trials(manifest: pd.DataFrame) -> pd.DataFrame:
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
        raise FoundationFeatureError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )
    selected = manifest[
        manifest["dataset_id"].eq("ds005540")
        & manifest["label_available"].astype(bool)
        & manifest["feature_alignment_status"].eq("verified")
    ].copy()
    if selected["trial_uid"].duplicated().any():
        raise FoundationFeatureError("Eligible trial IDs must be unique")
    return selected.sort_values("trial_uid").reset_index(drop=True)


def extract_session(
    model,
    torch,
    trials: np.ndarray,
    rows: pd.DataFrame,
    *,
    batch_size: int,
    device: str,
    scale_factor: float,
    window_patches: int,
) -> tuple[list[str], list[np.ndarray]]:
    grouped_windows: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    channel_ids: list[int] | None = None
    trial_ids = rows["trial_uid"].astype(str).tolist()
    for output_index, row in enumerate(rows.itertuples(index=False)):
        source_index = int(row.de_trial_index) - 1
        if not 0 <= source_index < len(trials):
            raise FoundationFeatureError(
                f"Trial index {source_index + 1} is outside reorder data for {row.trial_uid}"
            )
        patches, current_ids = prepare_labram_trial(
            trials[source_index], EMO_64_CHANNELS, scale_factor=scale_factor
        )
        if channel_ids is None:
            channel_ids = current_ids
        elif current_ids != channel_ids:
            raise RuntimeError("LaBraM channel positions changed within a session")
        for window in iter_labram_windows(
            patches, window_patches=window_patches
        ):
            grouped_windows[window.shape[1]].append((output_index, window))
    assert channel_ids is not None

    sums: list[np.ndarray | None] = [None] * len(rows)
    weights = np.zeros(len(rows), dtype=int)
    input_chans = [0, *channel_ids]
    for patch_count, records in sorted(grouped_windows.items()):
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            tensor = torch.from_numpy(
                np.stack([window for _, window in batch_records])
            ).float().to(device)
            with torch.inference_mode():
                flat_tokens = model(
                    tensor,
                    input_chans=input_chans,
                    return_patch_tokens=True,
                )
            flat_tokens = flat_tokens.detach().cpu().numpy()
            expected_tokens = len(channel_ids) * patch_count
            if flat_tokens.ndim != 3 or flat_tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    "Official LaBraM returned unexpected patch-token geometry: "
                    f"{flat_tokens.shape}, expected second axis {expected_tokens}"
                )
            tokens = flat_tokens.reshape(
                len(batch_records), len(channel_ids), patch_count, -1
            )
            pooled = pool_labram_tokens(tokens)
            for (output_index, _), feature in zip(batch_records, pooled):
                contribution = feature * patch_count
                sums[output_index] = (
                    contribution
                    if sums[output_index] is None
                    else sums[output_index] + contribution
                )
                weights[output_index] += patch_count
    if np.any(weights != 30) or any(value is None for value in sums):
        raise RuntimeError("LaBraM extraction did not cover all 30 trial patches")
    features = [
        np.asarray(value / weight, dtype=np.float32)
        for value, weight in zip(sums, weights)
    ]
    return trial_ids, features


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    manifest = pd.read_csv(args.manifest, sep="\t")
    selected = eligible_trials(manifest)
    model, torch, load_metadata = load_labram(
        args.model_repository.resolve(),
        args.checkpoint.resolve(),
        args.device,
        random_init=args.random_init,
        seed=args.random_init_seed,
    )

    all_ids: list[str] = []
    all_features: list[np.ndarray] = []
    skipped_sessions: list[str] = []
    selected["reorder_path"] = selected["de_path"].map(reorder_path_from_de_path)
    for relative_path, rows in selected.groupby("reorder_path", sort=True):
        path = args.dataset_root / relative_path
        try:
            trials = load_emo_reorder(path)
        except FoundationFeatureError:
            if not args.skip_missing:
                raise
            skipped_sessions.append(str(relative_path))
            continue
        trial_ids, features = extract_session(
            model,
            torch,
            trials,
            rows,
            batch_size=args.batch_size,
            device=args.device,
            scale_factor=args.scale_factor,
            window_patches=args.window_patches,
        )
        all_ids.extend(trial_ids)
        all_features.extend(features)
        print(f"{relative_path}: {len(trial_ids)} trials", flush=True)

    if not all_features:
        raise RuntimeError("No LaBraM features were extracted")
    order = np.argsort(np.asarray(all_ids, dtype=str))
    trial_uids = np.asarray(all_ids, dtype=str)[order]
    feature_array = np.stack(all_features)[order]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=feature_array,
        trial_uids=trial_uids,
        feature_names=np.asarray(
            [f"labram_pool_{index:03d}" for index in range(feature_array.shape[1])]
        ),
    )
    metadata = {
        "dataset_id": "ds005540",
        "feature_shape": list(feature_array.shape),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "model": "LaBraM",
        "model_variant": OFFICIAL_MODEL_VARIANT,
        "model_repository": OFFICIAL_REPOSITORY,
        "model_repository_commit": repository_commit(args.model_repository.resolve()),
        "model_source": f"{OFFICIAL_REPOSITORY}/tree/main/checkpoints",
        **load_metadata,
        "device": args.device,
        "sampling_frequency_hz": 200,
        "trial_seconds": 30,
        "patch_samples": 200,
        "window_patches": args.window_patches,
        "channel_order": list(EMO_64_CHANNELS),
        "channel_position_ids": prepare_labram_trial(
            np.zeros((64, 6000), dtype=np.float32), EMO_64_CHANNELS
        )[1],
        "source_preprocessing": (
            "EmoEEG-MC published reorder arrays: 0.1-47 Hz, 200 Hz, artifact "
            "cleaned, common-average referenced, and reordered"
        ),
        "input_unit": "microvolt",
        "input_scale_factor": args.scale_factor,
        "pooling": (
            "mean over final channel/patch tokens; 8-patch windows and a final "
            "6-patch window weighted by represented patch count"
        ),
        "eligible_trial_count": len(selected),
        "extracted_trial_count": len(feature_array),
        "skipped_session_count": len(skipped_sessions),
        "skipped_sessions": skipped_sessions,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Features: {args.output} {feature_array.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
