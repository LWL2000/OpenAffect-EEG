"""Check whether the v14 positive-control threshold is attainable by an oracle residual."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from openaffect_eeg.confirmatory_execution import split_records
from openaffect_eeg.final_robustness import population_targets, prediction_table
from openaffect_eeg.training_uncertainty import analyze_training_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("assignments_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", default="faced_confirmation_v14")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026091400)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = config["datasets"][args.dataset]
    doses = tuple(int(value) for value in config["doses"])
    training_seeds = tuple(int(value) for value in config["training_seeds"])
    base = args.assignments_root / args.dataset
    trials = (
        pd.read_csv(base / "eligible_trials.tsv.gz", sep="\t")
        .sort_values("trial_uid")
        .reset_index(drop=True)
    )
    targets = tuple(spec["targets"])
    observed = trials[list(targets)].to_numpy(float)
    rows: list[pd.DataFrame] = []

    for split in split_records(config, base):
        folder = Path(split["folder_path"])
        for stimulus_dose in doses:
            assignment0 = pd.read_csv(
                folder / f"participant-00_stimulus-{stimulus_dose:02d}.tsv.gz",
                sep="\t",
            )
            resources = population_targets(trials, assignment0, targets)
            oracle_residual = observed - resources["all_prior"]
            for participant_dose in doses:
                assignment = pd.read_csv(
                    folder / f"participant-{participant_dose:02d}_stimulus-{stimulus_dose:02d}.tsv.gz",
                    sep="\t",
                )
                base_frame = prediction_table(
                    trials,
                    assignment,
                    targets,
                    resources["all_prior"],
                    np.zeros_like(oracle_residual),
                    oracle_residual,
                )
                for training_seed in training_seeds:
                    frame = base_frame.copy()
                    frame.insert(0, "representation", "oracle_residual")
                    frame.insert(1, "assignment_seed", int(split["assignment_seed"]))
                    frame.insert(2, "training_seed", training_seed)
                    offset = 3
                    if "fold_rotation_seed" in split:
                        frame.insert(offset, "fold_rotation_seed", int(split["fold_rotation_seed"]))
                        offset += 1
                    frame.insert(offset, "participant_dose", participant_dose)
                    frame.insert(offset + 1, "stimulus_dose", stimulus_dose)
                    rows.append(frame)

    predictions = pd.concat(rows, ignore_index=True)
    cells, report = analyze_training_uncertainty(
        predictions,
        iterations=args.bootstrap,
        seed=args.seed,
        equivalence_margin=0.05,
        strict_margin=0.025,
    )
    report["diagnostic"] = {
        "kind": "oracle_residual_threshold_attainability",
        "expected_minimum_increment": 0.10,
        "passes": bool(report["primary"]["estimate"] >= 0.10),
        "note": "The five repeated seed labels represent the same deterministic oracle; this is a threshold diagnostic, not training evidence.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "oracle_cells.csv", index=False)
    (args.output / "oracle_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary": report["primary"], "diagnostic": report["diagnostic"]}, sort_keys=True))


if __name__ == "__main__":
    main()
