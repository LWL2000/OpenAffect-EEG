#!/usr/bin/env python3
"""Run the locked five-seed EEGNet confirmation with validation LR selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import yaml

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.confirmatory_execution import split_records, write_json
from openaffect_eeg.confirmation_controls import (
    residual_signal_targets,
    residual_training_targets,
    synthetic_target_signal,
)
from openaffect_eeg.final_robustness import (
    GPURegressionConfig,
    fit_gpu_regression,
    population_targets,
    prediction_table,
)


def load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("assignments_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training-seed", type=int, action="append")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument(
        "--control",
        choices=("observed", "label_permutation", "synthetic_signal", "synthetic_residual_signal"),
        default="observed",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    spec = config["datasets"][args.dataset]
    if args.control != "observed" and not spec.get("prespecified_neural_controls"):
        raise ValueError("Neural controls are not prespecified for this dataset")
    doses = tuple(int(value) for value in config["doses"])
    training_seeds = tuple(args.training_seed or config["training_seeds"])
    if doses != (0, 1, 2, 4, 8):
        raise ValueError("v14 confirmation requires the locked 5x5 grid")
    if len(training_seeds) != 5 or len(set(training_seeds)) != 5:
        raise ValueError("Exactly five distinct training seeds are required")
    base = args.assignments_root / args.dataset
    splits = split_records(config, base)
    table = pd.read_csv(base / "eligible_trials.tsv.gz", sep="\t").sort_values("trial_uid").reset_index(drop=True)
    table["tensor_index"] = np.arange(len(table))
    uids = np.load(args.data_root / spec["uids"], allow_pickle=False).astype(str)
    raw_index = pd.Index(uids).get_indexer(table.trial_uid)
    if (raw_index < 0).any():
        raise ValueError("EEG tensor and trial identities do not match")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    raw = np.load(args.data_root / spec["tensors"], mmap_mode="r", allow_pickle=False)
    tensor_array = np.asarray(raw[raw_index], dtype=np.float32)
    if args.control == "synthetic_signal":
        tensor_array = tensor_array + synthetic_target_signal(
            table[list(spec["targets"])].to_numpy(float),
            channels=tensor_array.shape[1],
            samples=tensor_array.shape[2],
        )
    tensors = torch.as_tensor(tensor_array, device="cuda")
    if not torch.isfinite(tensors).all():
        raise ValueError("EEG tensors contain non-finite values")

    args.output.mkdir(parents=True, exist_ok=True)
    sources = [args.config, Path(__file__), Path(inspect.getfile(fit_gpu_regression))]
    manifest = {
        "dataset_id": args.dataset,
        "control": args.control,
        "scope": "Locked 5x5 resource grid, five EEGNet seeds, validation-selected learning rate.",
        "training_seeds": list(training_seeds),
        "doses": list(doses),
        "split_count": len(splits),
        "source_sha256": {str(path): sha256_file(path) for path in sources},
        "input_sha256": {
            "tensors": sha256_file(args.data_root / spec["tensors"]),
            "uids": sha256_file(args.data_root / spec["uids"]),
            "split_manifest": sha256_file(base / "split_manifest.csv"),
        },
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Frozen EEGNet inputs or source changed; use a new output directory")
    write_json(manifest_path, manifest)
    target_columns = tuple(spec["targets"])
    rows: list[pd.DataFrame] = []
    selections: list[dict] = []
    for split in splits:
        assignment_seed = int(split["assignment_seed"])
        folder = Path(split["folder_path"])
        for stimulus_dose in doses:
            assignment0 = pd.read_csv(
                folder / f"participant-00_stimulus-{stimulus_dose:02d}.tsv.gz", sep="\t"
            )
            resources = population_targets(table, assignment0, target_columns)
            fit_tensors = tensors
            if args.control == "synthetic_residual_signal":
                addition = synthetic_target_signal(
                    residual_signal_targets(resources, table, target_columns),
                    channels=tensor_array.shape[1],
                    samples=tensor_array.shape[2],
                )
                fit_tensors = tensors + torch.as_tensor(addition, device=tensors.device)
            for training_seed in training_seeds:
                fit_seed = training_seed + assignment_seed + stimulus_dose
                training_targets = residual_training_targets(
                    resources, control=args.control, seed=fit_seed
                )
                candidates: list[tuple[str, np.ndarray, dict]] = []
                for setting, learning_rate in config["eegnet_settings"].items():
                    fit_id = (
                        f"split-{int(split['split_index']):03d}_stimulus-{stimulus_dose}_"
                        f"seed-{training_seed}_{setting}"
                    )
                    prediction_path = args.output / "fit_cache" / f"{fit_id}.npy"
                    metadata_path = args.output / "fit_cache" / f"{fit_id}.json"
                    checkpoint_path = args.output / "checkpoints" / f"{fit_id}.pt"
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    if prediction_path.exists() and metadata_path.exists():
                        prediction = np.load(prediction_path, allow_pickle=False)
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    else:
                        train_config = GPURegressionConfig(
                            crop_samples=int(spec["crop_samples"]),
                            learning_rate=float(learning_rate),
                            **config["eegnet"],
                        )
                        try:
                            prediction, metadata = fit_gpu_regression(
                                fit_tensors,
                                resources["train"].tensor_index.to_numpy(),
                                training_targets["train"],
                                resources["validation"].tensor_index.to_numpy(),
                                training_targets["validation"],
                                resources["fit"].tensor_index.to_numpy(),
                                training_targets["fit"],
                                seed=fit_seed,
                                config=train_config,
                                checkpoint=checkpoint_path,
                            )
                        except Exception as error:
                            failure = args.output / "attempt_failures.jsonl"
                            with failure.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps({
                                    "fit_id": fit_id,
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                    "traceback": traceback.format_exc(),
                                }, sort_keys=True) + "\n")
                            raise
                        metadata.update(setting=setting, learning_rate=float(learning_rate), config=asdict(train_config))
                        np.save(prediction_path, prediction, allow_pickle=False)
                        metadata["prediction_sha256"] = sha256_file(prediction_path)
                        write_json(metadata_path, metadata)
                        if not args.keep_checkpoints and checkpoint_path.exists():
                            checkpoint_path.unlink()
                    if prediction.shape != (len(table), len(target_columns)):
                        raise ValueError(f"Malformed cached prediction {prediction_path}")
                    if metadata.get("prediction_sha256") != sha256_file(prediction_path):
                        raise ValueError(f"Cached prediction hash mismatch {prediction_path}")
                    candidates.append((setting, prediction, metadata))
                selected_setting, prediction, selected_meta = min(
                    candidates,
                    key=lambda item: (
                        float(item[2]["validation_mae"]),
                        int(item[2]["parameter_count"]),
                        str(item[0]),
                    ),
                )
                selection = {
                    "split_index": int(split["split_index"]),
                    "assignment_seed": assignment_seed,
                    **({"fold_rotation_seed": int(split["fold_rotation_seed"])} if "fold_rotation_seed" in split else {}),
                    "stimulus_dose": stimulus_dose,
                    "training_seed": training_seed,
                    "selected_setting": selected_setting,
                    "candidates": {
                        setting: {
                            "validation_mae": float(metadata["validation_mae"]),
                            "selected_epochs": int(metadata["selected_epochs"]),
                            "parameter_count": int(metadata["parameter_count"]),
                        }
                        for setting, _, metadata in candidates
                    },
                }
                selections.append(selection)
                for participant_dose in doses:
                    assignment = pd.read_csv(
                        folder / f"participant-{participant_dose:02d}_stimulus-{stimulus_dose:02d}.tsv.gz",
                        sep="\t",
                    )
                    frame = prediction_table(
                        table,
                        assignment,
                        target_columns,
                        resources["all_prior"],
                        np.zeros_like(prediction),
                        prediction,
                    )
                    frame.insert(0, "representation", f"eegnet_selected_residual_{args.control}")
                    frame.insert(1, "assignment_seed", assignment_seed)
                    frame.insert(2, "training_seed", training_seed)
                    offset = 3
                    if "fold_rotation_seed" in split:
                        frame.insert(offset, "fold_rotation_seed", int(split["fold_rotation_seed"]))
                        offset += 1
                    frame.insert(offset, "participant_dose", participant_dose)
                    frame.insert(offset + 1, "stimulus_dose", stimulus_dose)
                    rows.append(frame)
                write_json(args.output / "selection.json", selections)
                print(args.dataset, split["split_index"], stimulus_dose, training_seed, selected_setting, flush=True)

    result = pd.concat(rows, ignore_index=True)
    destination = args.output / "identity_exposure_predictions.tsv.gz"
    result.to_csv(destination, sep="\t", index=False)
    write_json(args.output / "completion.json", {
        "status": "complete",
        "fit_candidates": len(splits) * len(doses) * len(training_seeds) * len(config["eegnet_settings"]),
        "selected_fits": len(selections),
        "rows": len(result),
        "predictions_sha256": sha256_file(destination),
    })


if __name__ == "__main__":
    main()
