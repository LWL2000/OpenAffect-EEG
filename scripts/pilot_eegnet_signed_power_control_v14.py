"""Run one locked FACED cell to test a crop-robust signed-power positive control."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from openaffect_eeg.baselines import regression_metrics
from openaffect_eeg.confirmation_controls import (
    residual_signal_targets,
    residual_training_targets,
    synthetic_signed_power_signal,
)
from openaffect_eeg.confirmatory_execution import split_records
from openaffect_eeg.final_robustness import (
    GPURegressionConfig,
    fit_gpu_regression,
    population_targets,
    prediction_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("assignments_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", default="faced_confirmation_v14")
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--stimulus-dose", type=int, default=8)
    parser.add_argument("--training-seed", type=int, default=2026091401)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = config["datasets"][args.dataset]
    base = args.assignments_root / args.dataset
    splits = {int(item["split_index"]): item for item in split_records(config, base)}
    split = splits[args.split_index]
    folder = Path(split["folder_path"])
    trials = (
        pd.read_csv(base / "eligible_trials.tsv.gz", sep="\t")
        .sort_values("trial_uid")
        .reset_index(drop=True)
    )
    trials["tensor_index"] = np.arange(len(trials))
    uids = np.load(args.data_root / spec["uids"], allow_pickle=False).astype(str)
    raw_index = pd.Index(uids).get_indexer(trials.trial_uid)
    if (raw_index < 0).any():
        raise ValueError("Tensor and trial identities do not match")

    import torch

    raw = np.load(args.data_root / spec["tensors"], mmap_mode="r", allow_pickle=False)
    tensor_array = np.asarray(raw[raw_index], dtype=np.float32)
    tensors = torch.as_tensor(tensor_array, device="cuda")
    targets = tuple(spec["targets"])
    assignment0 = pd.read_csv(
        folder / f"participant-00_stimulus-{args.stimulus_dose:02d}.tsv.gz",
        sep="\t",
    )
    resources = population_targets(trials, assignment0, targets)
    residual_targets = residual_signal_targets(resources, trials, targets)
    signal = synthetic_signed_power_signal(
        residual_targets,
        channels=tensors.shape[1],
        samples=tensors.shape[2],
    )
    injected = tensors.clone()
    injected[:, :4, :] = torch.as_tensor(signal[:, :4, :], device=tensors.device)
    fit_seed = args.training_seed + int(split["assignment_seed"]) + args.stimulus_dose
    training_targets = residual_training_targets(
        resources, control="synthetic_residual_signal", seed=fit_seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for setting, learning_rate in config["eegnet_settings"].items():
        train_config = GPURegressionConfig(
            crop_samples=int(spec["crop_samples"]),
            learning_rate=float(learning_rate),
            **config["eegnet"],
        )
        checkpoint = args.output / f"{setting}.pt"
        prediction, metadata = fit_gpu_regression(
            injected,
            resources["train"].tensor_index.to_numpy(),
            training_targets["train"],
            resources["validation"].tensor_index.to_numpy(),
            training_targets["validation"],
            resources["fit"].tensor_index.to_numpy(),
            training_targets["fit"],
            seed=fit_seed,
            config=train_config,
            checkpoint=checkpoint,
        )
        if checkpoint.exists():
            checkpoint.unlink()
        candidates.append((setting, prediction, metadata, train_config))

    selected_setting, prediction, metadata, train_config = min(
        candidates, key=lambda item: (float(item[2]["validation_mae"]), item[0])
    )
    evaluations = {}
    for participant_dose in (0, 8):
        assignment = pd.read_csv(
            folder
            / f"participant-{participant_dose:02d}_stimulus-{args.stimulus_dose:02d}.tsv.gz",
            sep="\t",
        )
        frame = prediction_table(
            trials,
            assignment,
            targets,
            resources["all_prior"],
            np.zeros_like(prediction),
            prediction,
        )
        truth = frame[[*targets]].to_numpy(float)
        prior = frame[["prior_personalized_valence", "prior_personalized_arousal"]].to_numpy(float)
        combined = frame[["combined_personalized_valence", "combined_personalized_arousal"]].to_numpy(float)
        evaluations[str(participant_dose)] = {
            "prior": regression_metrics(truth, prior),
            "combined": regression_metrics(truth, combined),
        }

    report = {
        "status": "complete",
        "scope": "exploratory one-cell diagnostic after two failed sinusoidal controls",
        "signal": "first four channels replaced by positive/negative residual carrier power",
        "split_index": args.split_index,
        "assignment_seed": int(split["assignment_seed"]),
        "stimulus_dose": args.stimulus_dose,
        "training_seed": args.training_seed,
        "fit_seed": fit_seed,
        "selected_setting": selected_setting,
        "selected_validation_mae": float(metadata["validation_mae"]),
        "selected_epochs": int(metadata["selected_epochs"]),
        "selected_config": asdict(train_config),
        "candidates": {
            setting: {
                "validation_mae": float(meta["validation_mae"]),
                "selected_epochs": int(meta["selected_epochs"]),
                "fit_metrics": meta["fit_metrics"],
            }
            for setting, _, meta, _ in candidates
        },
        "evaluations": evaluations,
    }
    (args.output / "pilot_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
