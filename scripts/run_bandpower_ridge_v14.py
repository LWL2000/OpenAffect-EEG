#!/usr/bin/env python3
"""Run the deterministic locked AMIGOS band-power Ridge grid."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import pandas as pd
import yaml

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.confirmatory_execution import feature_table, split_records, write_json
from openaffect_eeg.identity_exposure import evaluate_exposure_cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("assignments_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", default="amigos_confirmation_v14")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = config["datasets"][args.dataset]
    doses = tuple(int(value) for value in config["doses"])
    if doses != (0, 1, 2, 4, 8) or set(config["ridge_settings"]) != {"ridge_locked"}:
        raise ValueError("Ridge configuration differs from the locked confirmation design")
    base = args.assignments_root / args.dataset
    splits = split_records(config, base)
    trials = pd.read_csv(base / "eligible_trials.tsv.gz", sep="\t").sort_values("trial_uid").reset_index(drop=True)
    feature_spec = spec["features"]["bandpower"]
    features = feature_table(args.data_root / feature_spec["path"], feature_spec["array"])
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": args.dataset,
        "scope": "Deterministic locked 5x5 band-power Ridge sanity baseline.",
        "split_count": len(splits),
        "doses": list(doses),
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (args.config, Path(__file__), Path(inspect.getfile(evaluate_exposure_cell)))
        },
        "input_sha256": {
            "features": sha256_file(args.data_root / feature_spec["path"]),
            "split_manifest": sha256_file(base / "split_manifest.csv"),
        },
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Frozen Ridge inputs or source changed; use a new output directory")
    write_json(manifest_path, manifest)
    rows: list[pd.DataFrame] = []
    records: list[dict] = []
    for split in splits:
        folder = Path(split["folder_path"])
        for participant_dose in doses:
            for stimulus_dose in doses:
                cache = args.output / "cells" / (
                    f"split-{int(split['split_index']):03d}_p-{participant_dose}_s-{stimulus_dose}"
                )
                prediction_path = cache.with_suffix(".tsv.gz")
                record_path = cache.with_suffix(".json")
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                if prediction_path.exists() and record_path.exists():
                    prediction = pd.read_csv(prediction_path, sep="\t")
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    if record["predictions_sha256"] != sha256_file(prediction_path):
                        raise ValueError(f"Ridge cache hash mismatch {prediction_path}")
                else:
                    assignment_path = folder / (
                        f"participant-{participant_dose:02d}_stimulus-{stimulus_dose:02d}.tsv.gz"
                    )
                    assignment = pd.read_csv(assignment_path, sep="\t")
                    record, prediction = evaluate_exposure_cell(
                        trials,
                        assignment,
                        features,
                        backend="numpy",
                        alphas=tuple(config["ridge_settings"]["ridge_locked"]),
                        target_columns=tuple(spec["targets"]),
                    )
                    prediction = prediction.rename(columns={"seed": "assignment_seed"})
                    prediction.insert(0, "representation", "bandpower_ridge")
                    if "fold_rotation_seed" in split:
                        prediction.insert(2, "fold_rotation_seed", int(split["fold_rotation_seed"]))
                    prediction.to_csv(prediction_path, sep="\t", index=False)
                    record.update({
                        "split_index": int(split["split_index"]),
                        "assignment_seed": int(split["assignment_seed"]),
                        **({"fold_rotation_seed": int(split["fold_rotation_seed"])} if "fold_rotation_seed" in split else {}),
                        "predictions_sha256": sha256_file(prediction_path),
                    })
                    write_json(record_path, record)
                rows.append(prediction)
                records.append(record)
        print(args.dataset, "ridge", split["split_index"], flush=True)
    result = pd.concat(rows, ignore_index=True)
    destination = args.output / "identity_exposure_predictions.tsv.gz"
    result.to_csv(destination, sep="\t", index=False)
    write_json(args.output / "completion.json", {
        "status": "complete",
        "split_count": len(splits),
        "cell_count": len(records),
        "rows": len(result),
        "predictions_sha256": sha256_file(destination),
    })


if __name__ == "__main__":
    main()
