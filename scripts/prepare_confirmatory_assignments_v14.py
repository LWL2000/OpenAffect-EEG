#!/usr/bin/env python3
"""Compile a locked confirmation dataset's crossed-fold resource assignments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from openaffect_eeg.amigos_confirmation import crossed_outer_assignments
from openaffect_eeg.identity_exposure import compile_exposure_cell


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(config_path: Path, data_root: Path, assignments_root: Path, dataset: str) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if dataset not in config["datasets"]:
        raise ValueError(f"Unknown dataset {dataset}")
    spec = config["datasets"][dataset]
    doses = tuple(int(value) for value in config["doses"])
    rotations = tuple(int(value) for value in config["fold_rotation_seeds"])
    if doses != (0, 1, 2, 4, 8) or len(rotations) != 5 or len(set(rotations)) != 5:
        raise ValueError("Configuration differs from the locked confirmation design")
    trial_path = data_root / spec["trials"]
    trials = pd.read_csv(trial_path, sep="\t").sort_values("trial_uid").reset_index(drop=True)
    required = {"trial_uid", "subject_uid", "stimulus_uid", *spec["targets"]}
    if required - set(trials) or trials.trial_uid.duplicated().any():
        raise ValueError("Malformed eligible confirmation trial table")
    minimum_participants = int(spec.get("minimum_participants", 25))
    required_stimuli = int(spec.get("required_stimuli", 16))
    if (
        trials.subject_uid.nunique() < minimum_participants
        or trials.stimulus_uid.nunique() != required_stimuli
    ):
        raise ValueError(
            f"{dataset} requires at least {minimum_participants} complete "
            f"participants and exactly {required_stimuli} stimuli"
        )

    base = assignments_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    trials.to_csv(base / "eligible_trials.tsv.gz", sep="\t", index=False)
    manifest_rows: list[dict[str, object]] = []
    support_by_rotation: dict[int, set[str]] = {rotation: set() for rotation in rotations}
    split_index = 0
    for rotation_seed in rotations:
        crossed = crossed_outer_assignments(trials, rotation_seed=rotation_seed)
        for metadata, strict in crossed:
            assignment_seed = int(config["design_seed"]) + split_index
            folder_name = f"assignments/split-{split_index:03d}"
            folder = base / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            strict.to_csv(folder / "strict.tsv.gz", sep="\t", index=False)
            support_hashes: set[str] = set()
            train_counts: set[int] = set()
            for participant_dose in doses:
                for stimulus_dose in doses:
                    assignment, audit = compile_exposure_cell(
                        trials,
                        strict,
                        participant_dose=participant_dose,
                        stimulus_dose=stimulus_dose,
                        maximum_participant_dose=max(doses),
                        maximum_stimulus_dose=max(doses),
                        seed=assignment_seed,
                        target_columns=tuple(spec["targets"]),
                    )
                    path = folder / (
                        f"participant-{participant_dose:02d}_"
                        f"stimulus-{stimulus_dose:02d}.tsv.gz"
                    )
                    assignment.to_csv(path, sep="\t", index=False)
                    write_json(path.with_suffix(".audit.json"), audit)
                    support_hashes.add(str(audit["fixed_test_support_sha256"]))
                    train_counts.add(int(audit["counts"]["train"]))
            if len(support_hashes) != 1 or len(train_counts) != 1:
                raise ValueError("Resource cells changed test support or population train size")
            test_ids = set(strict.loc[strict.split.eq("test"), "trial_uid"].astype(str))
            if support_by_rotation[rotation_seed] & test_ids:
                raise ValueError("Test support overlaps within a fold rotation")
            support_by_rotation[rotation_seed].update(test_ids)
            manifest_rows.append({
                "split_index": split_index,
                "assignment_seed": assignment_seed,
                "fold_rotation_seed": rotation_seed,
                "participant_block": int(metadata["participant_block"]),
                "stimulus_block": int(metadata["stimulus_block"]),
                "folder": folder_name,
                "test_trials": len(test_ids),
                "test_support_sha256": next(iter(support_hashes)),
                "population_train_rows": next(iter(train_counts)),
            })
            split_index += 1
    expected = set(trials.trial_uid.astype(str))
    if split_index != 100 or any(support != expected for support in support_by_rotation.values()):
        raise ValueError("Each rotation must cover every trial exactly once across 20 cross-products")
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(base / "split_manifest.csv", index=False)
    report = {
        "status": "complete",
        "dataset": dataset,
        "participants": int(trials.subject_uid.nunique()),
        "stimuli": int(trials.stimulus_uid.nunique()),
        "trials": len(trials),
        "rotations": list(rotations),
        "split_count": len(manifest),
        "resource_cell_count": len(manifest) * len(doses) ** 2,
        "input_sha256": sha256_file(trial_path),
        "manifest_sha256": sha256_file(base / "split_manifest.csv"),
        "outcome_blind": False,
        "boundary": (
            "Assignments are deterministic and the design was specified before "
            f"{dataset} outcomes were accessed."
        ),
    }
    write_json(base / "preparation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("assignments_root", type=Path)
    parser.add_argument("--dataset", default="amigos_confirmation_v14")
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.data_root, args.assignments_root, args.dataset), sort_keys=True))


if __name__ == "__main__":
    main()
