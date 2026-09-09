"""Deterministic command-line compiler for identity-aware EEG audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from openaffect_eeg.baselines import harmonize_affect_targets
from openaffect_eeg.exposure_statistics import analyze_predictions
from openaffect_eeg.audit_evidence import review_evidence
from openaffect_eeg.splits import build_double_holdout_split
from openaffect_eeg.identity_exposure import (
    ExposureCell,
    compile_exposure_cell,
    evaluate_exposure_cell,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_archive(path: Path, view: str) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as archive:
        uid_key = "trial_uid" if "trial_uid" in archive.files else "trial_uids"
        trial_uids = archive[uid_key].astype(str)
        if view == "foundation":
            values = archive["features"].astype(float)
        elif view == "channel":
            values = archive["channel_log_bandpower"].astype(float)
            values = values.reshape(len(values), -1)
        elif view == "global":
            values = archive["global_log_bandpower"].astype(float)
        else:
            raise ValueError(f"Unsupported feature view: {view}")
    if values.ndim != 2 or trial_uids.ndim != 1 or len(values) != len(trial_uids):
        raise ValueError("Feature dimensions and trial identifiers do not match")
    if len(set(trial_uids)) != len(trial_uids) or not np.isfinite(values).all():
        raise ValueError("Duplicate feature trials or non-finite features")
    table = pd.DataFrame(
        values,
        columns=[f"feature_{index:04d}" for index in range(values.shape[1])],
    )
    table.insert(0, "trial_uid", trial_uids)
    return table


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compile_audit_bundle(
    trials: pd.DataFrame,
    features: pd.DataFrame,
    strict_assignment: pd.DataFrame | None,
    contract: dict[str, object],
    output: Path,
    *,
    input_paths: dict[str, Path] | None = None,
) -> dict[str, object]:
    """Compile an auditable response surface and its deterministic claim cards."""

    if (output / "reproduction_manifest.json").exists():
        raise ValueError("Refuse to mix a new audit with a previously completed bundle")
    if not isinstance(contract, dict):
        raise ValueError("Deployment contract must be a mapping")
    supplied_evidence = contract.get("audit_evidence", {})
    if not isinstance(supplied_evidence, dict) or not isinstance(supplied_evidence.get("cells", {}), dict):
        raise ValueError("Audit evidence and its cells must be mappings")
    if any(not isinstance(value, dict) for value in supplied_evidence.get("cells", {}).values()):
        raise ValueError("Each audit evidence cell must be a mapping")
    output.mkdir(parents=True, exist_ok=True)
    if trials["trial_uid"].duplicated().any() or features["trial_uid"].duplicated().any():
        raise ValueError("Duplicate trial identifiers")
    feature_columns = [c for c in features if c.startswith("feature_")]
    if not feature_columns or not np.isfinite(features[feature_columns].to_numpy(float)).all():
        raise ValueError("Missing or non-finite features")
    original_trial_count = len(trials)
    original_feature_count = len(features)
    if "target_valence" not in trials or "target_arousal" not in trials:
        trials = harmonize_affect_targets(trials)
    if "target_available" in trials:
        trials = trials.loc[trials["target_available"]].copy()
    trials = trials.loc[np.isfinite(trials[["target_valence", "target_arousal"]].to_numpy(float)).all(axis=1)].copy()
    labelled_count = len(trials)
    subject_column = str(contract.get("subject_column", "subject_uid"))
    stimulus_column = str(contract.get("stimulus_column", "stimulus_uid"))
    if subject_column == "subject_uid" and subject_column not in trials:
        trials[subject_column] = trials["dataset_id"].astype(str) + ":" + trials["subject_id"].astype(str)
    common = set(trials["trial_uid"].astype(str)) & set(features["trial_uid"].astype(str))
    trials = trials.loc[trials["trial_uid"].astype(str).isin(common)].copy()
    features = features.loc[features["trial_uid"].astype(str).isin(common)].copy()
    participant_doses = tuple(int(value) for value in contract["participant_doses"])
    stimulus_doses = tuple(int(value) for value in contract["stimulus_doses"])
    seed = int(contract["seed"])
    backend = str(contract.get("backend", "numpy"))
    alphas = tuple(float(value) for value in contract.get("ridge_alphas", (0.1, 1.0, 10.0)))
    if not participant_doses or not stimulus_doses or len(set(participant_doses)) != len(participant_doses) or len(set(stimulus_doses)) != len(stimulus_doses):
        raise ValueError("Dose axes must be nonempty and unique")
    if strict_assignment is None:
        strict_assignment = build_double_holdout_split(trials, first_group=subject_column,
            second_group=stimulus_column, seed=seed,
            validation_fraction=float(contract.get("validation_fraction", 0.1)),
            test_fraction=float(contract.get("test_fraction", 0.2)))
    strict_path = output / "strict_assignment.tsv.gz"
    strict_assignment.to_csv(strict_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    audits = []
    results = []
    predictions = []
    assignment_records = []
    support_hash = None
    evidence_reports = {}
    for participant_dose in participant_doses:
        for stimulus_dose in stimulus_doses:
            cell = ExposureCell(participant_dose, stimulus_dose)
            assignment, audit = compile_exposure_cell(
                trials,
                strict_assignment,
                participant_dose=participant_dose,
                stimulus_dose=stimulus_dose,
                maximum_participant_dose=max(participant_doses),
                maximum_stimulus_dose=max(stimulus_doses),
                seed=seed,
                subject_column=subject_column,
                stimulus_column=stimulus_column,
                stimulus_label=str(contract.get("stimulus_label", "stimulus")),
                deployment_context=dict(contract.get("deployment_context", {})),
            )
            if support_hash is None:
                support_hash = audit["fixed_test_support_sha256"]
            elif support_hash != audit["fixed_test_support_sha256"]:
                raise ValueError("Fixed test support changed across audit cells")
            supplied_evidence = dict(contract.get("audit_evidence", {}))
            cell_evidence = dict(supplied_evidence.get("cells", {}).get(cell.name, {}))
            cell_evidence = {**{k: v for k, v in supplied_evidence.items() if k != "cells"}, **cell_evidence}
            # These lists come from this evaluator's actual assignment, not a user checkbox.
            if "predictor_calibration_trial_uids" not in cell_evidence:
                calibration_ids = sorted(assignment.loc[assignment["split"].eq("calibration"), "trial_uid"].astype(str))
                cell_evidence["predictor_calibration_trial_uids"] = {
                    name: calibration_ids for name in ("prior_personalized", "combined_personalized")}
            actual_prior_resources = ["population_training_labels"]
            if assignment["split"].eq("calibration").any():
                actual_prior_resources.append("participant_calibration_labels")
            if {"stimulus_prior_valence", "stimulus_prior_arousal"}.issubset(trials.columns):
                actual_prior_resources.append("released_normative_prior")
            if "prior_resources" not in cell_evidence:
                cell_evidence["prior_resources"] = actual_prior_resources
            feature_path = (input_paths or {}).get("features")
            reviewed = review_evidence(assignment, feature_sha256=sha256_file(feature_path) if feature_path else None,
                                       evidence=cell_evidence, actual_prior_resources=actual_prior_resources)
            evidence_reports[cell.name] = reviewed
            if reviewed["status"] == "block":
                _write_json(output / "evidence_review.json", evidence_reports)
                raise ValueError(f"Supplied evidence contradicts the allowed resources in {cell.name}")
            result, prediction = evaluate_exposure_cell(
                trials,
                assignment,
                features,
                backend=backend,
                subject_column=subject_column,
                stimulus_column=stimulus_column,
                alphas=alphas,
            )
            audits.append(audit)
            results.append(result)
            prediction.insert(0, "cell", cell.name)
            predictions.append(prediction)
            assignment.insert(0, "cell", cell.name)
            assignment_records.append(assignment)

    score_rows = []
    probe_rows = []
    for result in results:
        for predictor, metrics in result["metrics"].items():
            score_rows.append(
                {
                    "participant_dose": result["participant_dose"],
                    "stimulus_dose": result["stimulus_dose"],
                    "predictor": predictor,
                    "macro_ccc": metrics["macro"]["ccc"],
                    "macro_mae": metrics["macro"]["mae"],
                }
            )
        for identity_axis, probe in result["identity_probes"].items():
            probe_rows.append(
                {
                    "participant_dose": result["participant_dose"],
                    "stimulus_dose": result["stimulus_dose"],
                    "identity_axis": identity_axis,
                    **probe,
                }
            )
    score_path = output / "score_table.tsv"
    probe_path = output / "identity_probe_report.tsv"
    assignment_path = output / "split_manifest.tsv.gz"
    prediction_path = output / "predictions.tsv.gz"
    pd.DataFrame(score_rows).to_csv(score_path, sep="\t", index=False)
    pd.DataFrame(probe_rows).to_csv(probe_path, sep="\t", index=False)
    pd.concat(assignment_records, ignore_index=True).to_csv(
        assignment_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0}
    )
    prediction_table = pd.concat(predictions, ignore_index=True)
    prediction_table.to_csv(
        prediction_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0}
    )
    statistics_input = prediction_table.copy()
    statistics_input["representation"] = "supplied_features_ridge"
    if subject_column != "subject_uid":
        statistics_input = statistics_input.rename(columns={subject_column: "subject_uid"})
    if stimulus_column != "stimulus_uid":
        statistics_input = statistics_input.rename(columns={stimulus_column: "stimulus_uid"})
    statistics = analyze_predictions(statistics_input,
        iterations=int(contract.get("bootstrap_iterations", 2000)), seed=seed)
    for name, table in zip(("scores", "contrasts", "summary"), statistics[:3], strict=True):
        table.to_csv(output / f"added_value_{name}.tsv", sep="\t", index=False)
    _write_json(output / "added_value_statistics.json", statistics[3])
    _write_json(output / "identity_certificate.json", audits)
    _write_json(output / "evidence_review.json", evidence_reports)
    _write_json(
        output / "coverage_report.json",
        {
            "eligible_trial_count": len(trials),
            "feature_trial_count": len(features),
            "input_trial_count": original_trial_count,
            "input_feature_count": original_feature_count,
            "missing_target_count": original_trial_count - labelled_count,
            "labelled_trials_without_features": labelled_count - len(trials),
            "feature_rows_outside_labelled_support": original_feature_count - len(features),
            "fixed_test_support_sha256": support_hash,
            "cells": [audit["counts"] | {"cell": audit["cell"]} for audit in audits],
        },
    )
    _write_json(
        output / "claim_cards.json",
        {audit["cell"]: audit["claim_card"] for audit in audits},
    )
    supplied_paths = input_paths or {}
    manifest = {
        "schema_version": 3,
        "contract": contract,
        "input_sha256": {
            name: sha256_file(path) for name, path in supplied_paths.items()
        },
        "output_sha256": {
            path.name: sha256_file(path)
            for path in (score_path, probe_path, assignment_path, prediction_path, strict_path,
                         output / "identity_certificate.json", output / "coverage_report.json",
                         output / "claim_cards.json", output / "added_value_statistics.json", output / "evidence_review.json",
                         *(output / f"added_value_{name}.tsv" for name in ("scores", "contrasts", "summary")))
        },
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "PyYAML")},
        "evaluator_source_sha256": {path.name: sha256_file(path) for path in
            (Path(__file__), Path(__file__).with_name("identity_exposure.py"), Path(__file__).with_name("exposure_statistics.py"), Path(__file__).with_name("audit_evidence.py"))},
        "fixed_test_support_sha256": support_hash,
        "interpretation_boundary": (
            "The bundle evaluates observed reports under declared identity exposure. "
            "It does not establish latent-emotion disentanglement or causal reliance."
        ),
    }
    _write_json(output / "reproduction_manifest.json", manifest)
    return manifest


def _toy_inputs(output: Path) -> tuple[Path, Path, Path, Path]:
    toy = output / "toy_inputs"
    toy.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject_index in range(8):
        for stimulus_index in range(8):
            rows.append(
                {
                    "trial_uid": f"s{subject_index}-x{stimulus_index}",
                    "dataset_id": "toy",
                    "subject_uid": f"s{subject_index}",
                    "stimulus_uid": f"x{stimulus_index}",
                    "context": "toy",
                    "target_valence": 0.1 + 0.07 * subject_index + 0.03 * stimulus_index,
                    "target_arousal": 0.9 - 0.06 * subject_index - 0.02 * stimulus_index,
                }
            )
    trials = pd.DataFrame(rows)
    assignments = []
    for row in trials.itertuples(index=False):
        subject_split = "test" if row.subject_uid in {"s0", "s1"} else "validation" if row.subject_uid in {"s2", "s3"} else "train"
        stimulus_split = "test" if row.stimulus_uid in {"x0", "x1"} else "validation" if row.stimulus_uid in {"x2", "x3"} else "train"
        assignments.append(subject_split if subject_split == stimulus_split else "excluded")
    strict = pd.DataFrame(
        {
            "trial_uid": trials["trial_uid"],
            "split": assignments,
            "protocol": "subject_stimulus_holdout",
            "seed": 11,
        }
    )
    rng = np.random.default_rng(11)
    features = rng.normal(size=(len(trials), 12))
    features[:, 0] += trials["subject_uid"].str[1:].astype(float).to_numpy()
    features[:, 1] += trials["stimulus_uid"].str[1:].astype(float).to_numpy()
    trial_path = toy / "trials.tsv"
    feature_path = toy / "features.npz"
    split_path = toy / "strict_assignment.tsv"
    contract_path = toy / "deployment.yaml"
    trials.to_csv(trial_path, sep="\t", index=False)
    np.savez_compressed(feature_path, trial_uid=trials["trial_uid"].to_numpy(dtype=str), features=features)
    strict.to_csv(split_path, sep="\t", index=False)
    contract = {
        "seed": 11,
        "subject_column": "subject_uid",
        "stimulus_column": "stimulus_uid",
        "stimulus_label": "stimulus",
        "participant_doses": [0, 1, 2],
        "stimulus_doses": [0, 1, 2],
        "ridge_alphas": [0.1, 1.0, 10.0],
        "backend": "numpy",
        "bootstrap_iterations": 200,
        "validation_fraction": 0.25,
        "test_fraction": 0.25,
    }
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return trial_path, feature_path, split_path, contract_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openaffect")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="compile an identity-exposure audit bundle")
    audit.add_argument("--trials", type=Path, required=True)
    audit.add_argument("--features", type=Path, required=True)
    audit.add_argument("--strict-assignment", type=Path)
    audit.add_argument("--contract", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--feature-view", choices=("foundation", "channel", "global"), default="foundation")
    toy = subparsers.add_parser("toy", help="run the public synthetic acceptance test")
    toy.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.output / "reproduction_manifest.json").exists():
        raise ValueError("Refuse to modify a previously completed bundle")
    if args.command == "toy":
        trial_path, feature_path, split_path, contract_path = _toy_inputs(args.output)
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        compile_audit_bundle(
            pd.read_csv(trial_path, sep="\t"),
            load_feature_archive(feature_path, "foundation"),
            pd.read_csv(split_path, sep="\t"),
            contract,
            args.output,
            input_paths={
                "trials": trial_path,
                "features": feature_path,
                "strict_assignment": split_path,
                "contract": contract_path,
            },
        )
        print(f"Toy audit: {args.output}")
        return 0
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    compile_audit_bundle(
        pd.read_csv(args.trials, sep="\t"),
        load_feature_archive(args.features, args.feature_view),
        pd.read_csv(args.strict_assignment, sep="\t") if args.strict_assignment else None,
        contract,
        args.output,
        input_paths={
            "trials": args.trials,
            "features": args.features,
            **({"strict_assignment": args.strict_assignment} if args.strict_assignment else {}),
            "contract": args.contract,
        },
    )
    print(f"Audit bundle: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
