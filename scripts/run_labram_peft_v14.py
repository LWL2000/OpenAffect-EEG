#!/usr/bin/env python3
"""Run a resumable five-seed, full-grid LaBraM tail-adaptation analysis.

The EEG residual is fitted separately for every evaluation fold, stimulus-label
dose, and training seed. Participant calibration remains post-fit and therefore
does not trigger a redundant EEG refit. Every completed fit writes predictions
and metadata before the next fit starts so an interrupted run can resume without
silently changing its seed or selection history.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from extract_labram_features import repository_commit, sha256_file
from openaffect_eeg.confirmatory_execution import split_records
from openaffect_eeg.confirmation_controls import residual_training_targets, synthetic_target_signal
from openaffect_eeg.final_robustness import population_targets, prediction_table
from openaffect_eeg.labram import EMO_64_CHANNELS
from run_labram_finetune_v13 import fit_residual, write_json


DEFAULT_TRAINING_SEEDS = (
    2026091401,
    2026091402,
    2026091403,
    2026091404,
    2026091405,
)


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("assignments_root", type=Path)
    parser.add_argument("model_repository", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trainable-final-blocks", type=int, default=4)
    parser.add_argument("--training-seed", type=int, action="append")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--control",
        choices=("observed", "label_permutation", "synthetic_signal"),
        default="observed",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Retain large fitted state files after predictions and metadata are sealed.",
    )
    return parser.parse_args()


def dataset_tensors(args: argparse.Namespace, spec: dict, table: pd.DataFrame):
    raw_uids = np.load(args.data_root / spec["uids"], allow_pickle=False).astype(str)
    raw_index = pd.Index(raw_uids).get_indexer(table.trial_uid)
    if (raw_index < 0).any():
        raise ValueError("Fine-tuning trial identity mismatch")
    raw = np.load(args.data_root / spec["tensors"], mmap_mode="r", allow_pickle=False)
    import torch

    tensors = torch.as_tensor(np.asarray(raw[raw_index], dtype=np.float32), device="cuda")
    if "channel_order" in spec and "units" in spec:
        channels = list(spec["channel_order"])
        if spec["units"] == "microvolts":
            scale_factor = 1.0
        elif spec["units"] == "physical volts":
            scale_factor = 1_000_000.0
        else:
            raise ValueError("Configured tensor units must be microvolts or physical volts")
    elif args.dataset == "ds005540":
        channels = list(EMO_64_CHANNELS)
        scale_factor = 1.0
    else:
        meta_path = args.data_root / "derived/final_robustness_v9_inputs/urban_100hz.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        channels = meta["channel_order"]
        if meta.get("units") != "physical volts":
            raise ValueError("Unexpected Urban tensor units")
        scale_factor = 1_000_000.0
    if len(channels) != tensors.shape[1]:
        raise ValueError("Channel metadata does not match tensors")
    return tensors, channels, scale_factor


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset not in cfg["datasets"]:
        raise ValueError(f"Unknown dataset {args.dataset}")
    if args.control != "observed" and args.dataset != "amigos_confirmation_v14":
        raise ValueError("v14 neural controls are prespecified for AMIGOS only")
    spec = cfg["datasets"][args.dataset]
    doses = tuple(int(value) for value in cfg.get("doses", (0, 1, 2, 4, 8)))
    if doses != (0, 1, 2, 4, 8):
        raise ValueError("v14 confirmation requires the locked 5x5 dose grid")
    training_seeds = tuple(args.training_seed or DEFAULT_TRAINING_SEEDS)
    if len(training_seeds) != 5 or len(set(training_seeds)) != 5:
        raise ValueError("Exactly five distinct training seeds are required")

    targets = tuple(spec["targets"])
    base = args.assignments_root / args.dataset
    table = (
        pd.read_csv(base / "eligible_trials.tsv.gz", sep="\t")
        .sort_values("trial_uid")
        .reset_index(drop=True)
    )
    table["tensor_index"] = np.arange(len(table))
    tensors, channels, scale_factor = dataset_tensors(args, spec, table)
    if args.control == "synthetic_signal":
        import torch
        addition = synthetic_target_signal(
            table[list(spec["targets"])].to_numpy(float),
            channels=tensors.shape[1],
            samples=tensors.shape[2],
            amplitude=100.0 / scale_factor,
        )
        tensors = tensors + torch.as_tensor(addition, device=tensors.device)

    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [
        args.config,
        Path(__file__),
        Path(inspect.getfile(population_targets)),
        Path(inspect.getfile(prediction_table)),
        args.checkpoint,
    ]
    manifest = {
        "dataset_id": args.dataset,
        "control": args.control,
        "model_repository_commit": repository_commit(args.model_repository),
        "sources": {str(path): sha256_file(path) for path in source_paths},
        "scope": "Full 5x5 resource grid with five-seed LaBraM tail adaptation.",
        "training_seeds": list(training_seeds),
        "trainable_final_blocks": args.trainable_final_blocks,
        "keep_checkpoints": args.keep_checkpoints,
        "doses": list(doses),
        "split_count": len(split_records(cfg, base)),
        "input_sha256": {
            "tensors": sha256_file(args.data_root / spec["tensors"]),
            "uids": sha256_file(args.data_root / spec["uids"]),
            **({"split_manifest": sha256_file(base / "split_manifest.csv")}
               if (base / "split_manifest.csv").exists() else {}),
        },
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Frozen LaBraM inputs or source changed; use a new output directory")
    write_json(manifest_path, manifest)

    rows: list[pd.DataFrame] = []
    training: list[dict] = []
    for split in split_records(cfg, base):
        fold = int(split["split_index"])
        assignment_seed = int(split["assignment_seed"])
        folder = Path(split["folder_path"])
        for stimulus_dose in doses:
            assignment0 = pd.read_csv(
                folder / f"participant-00_stimulus-{stimulus_dose:02d}.tsv.gz",
                sep="\t",
            )
            resources = population_targets(table, assignment0, targets)
            for training_seed in training_seeds:
                fit_seed = training_seed + assignment_seed + stimulus_dose
                training_targets = residual_training_targets(
                    resources, control=args.control, seed=fit_seed
                )
                fit_id = (
                    f"fold-{fold}_stimulus-{stimulus_dose}_"
                    f"training-seed-{training_seed}"
                )
                prediction_path = args.output / "fit_cache" / f"{fit_id}.npy"
                metadata_path = args.output / "fit_cache" / f"{fit_id}.json"
                model_path = args.output / "checkpoints" / f"{fit_id}.pt"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if prediction_path.exists() and metadata_path.exists():
                    prediction = np.load(prediction_path, allow_pickle=False)
                    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if prediction.shape != (len(table), len(targets)):
                        raise ValueError(f"Malformed cached predictions: {prediction_path}")
                else:
                    prediction, meta = fit_residual(
                        tensors,
                        resources["train"].tensor_index.to_numpy(),
                        training_targets["train"],
                        resources["validation"].tensor_index.to_numpy(),
                        training_targets["validation"],
                        resources["fit"].tensor_index.to_numpy(),
                        training_targets["fit"],
                        channels=channels,
                        repository=args.model_repository,
                        checkpoint=args.checkpoint,
                        seed=fit_seed,
                        crop_samples=int(spec["crop_samples"]),
                        scale_factor=scale_factor,
                        output=model_path,
                        trainable_final_blocks=args.trainable_final_blocks,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                    )
                    np.save(prediction_path, prediction, allow_pickle=False)
                    write_json(metadata_path, meta)
                    if not args.keep_checkpoints and model_path.exists():
                        model_path.unlink()
                training.append(
                    {
                        "fold": fold,
                        "assignment_seed": assignment_seed,
                        **({"fold_rotation_seed": int(split["fold_rotation_seed"])}
                           if "fold_rotation_seed" in split else {}),
                        "stimulus_dose": stimulus_dose,
                        "training_seed": training_seed,
                        "prediction_sha256": sha256_file(prediction_path),
                        **meta,
                    }
                )
                for participant_dose in doses:
                    assignment = pd.read_csv(
                        folder
                        / f"participant-{participant_dose:02d}_stimulus-{stimulus_dose:02d}.tsv.gz",
                        sep="\t",
                    )
                    zero = np.zeros_like(prediction)
                    frame = prediction_table(
                        table,
                        assignment,
                        targets,
                        resources["all_prior"],
                        zero,
                        prediction,
                    )
                    frame.insert(0, "representation", f"labram_final4_residual_{args.control}")
                    frame.insert(1, "assignment_seed", assignment_seed)
                    frame.insert(2, "training_seed", training_seed)
                    offset = 3
                    if "fold_rotation_seed" in split:
                        frame.insert(offset, "fold_rotation_seed", int(split["fold_rotation_seed"]))
                        offset += 1
                    frame.insert(offset, "participant_dose", participant_dose)
                    frame.insert(offset + 1, "stimulus_dose", stimulus_dose)
                    rows.append(frame)
                write_json(args.output / "training.json", training)
                print(
                    args.dataset,
                    fold,
                    stimulus_dose,
                    training_seed,
                    meta["selected_epochs"],
                    flush=True,
                )

    result = pd.concat(rows, ignore_index=True)
    destination = args.output / "identity_exposure_predictions.tsv.gz"
    result.to_csv(destination, sep="\t", index=False)
    write_json(
        args.output / "completion.json",
        {
            "status": "complete",
            "rows": len(result),
            "fit_count": len(training),
            "predictions_sha256": sha256_file(destination),
            "boundary": (
                "Training-seed uncertainty is represented; arbitrary tuning-policy "
                "uncertainty is outside scope."
            ),
        },
    )


if __name__ == "__main__":
    main()
