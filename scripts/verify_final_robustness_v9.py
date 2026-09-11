"""Independently verify complete v9 outputs; publish diagnostics without trial IDs."""
import argparse
from importlib.metadata import version
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file


def verify_learning(record):
    history, refit = record["history"], record["refit_history"]
    selected, best = 0, float("inf")
    for row in history:
        if not np.isfinite([row["objective"], row["gradient_norm_sum"], row["validation_mae"]]).all():
            raise ValueError("Non-finite learning history")
        if row["validation_mae"] < best - 1e-6:
            selected, best = row["epoch"], row["validation_mae"]
    if selected != record["selected_epochs"] or len(refit) != selected:
        raise ValueError("Validation selection/refit length mismatch")
    if not np.isclose(best, record["validation_mae"], atol=1e-12):
        raise ValueError("Selected validation score mismatch")
    if not all(np.isfinite([r["objective"], r["gradient_norm_sum"]]).all() for r in refit):
        raise ValueError("Non-finite refit history")
    if not any(r["gradient_norm_sum"] > 0 for r in history):
        raise ValueError("No nonzero training gradient")
    return dict(selected_epochs=selected, selection_epochs=len(history), validation_mae=best,
                initial_train_objective=history[0]["objective"], final_train_objective=history[-1]["objective"],
                minimum_fit_prediction_std=min(record["fit_prediction_std"]),
                target_count=record["target_count"], parameter_count=record["parameter_count"])


def verify_dataset(root, spec, config):
    manifest = json.loads((root / "input_manifest.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    response = json.loads((root / "identity_exposure_response.json").read_text())
    predictions_path = root / "identity_exposure_predictions.tsv.gz"
    if sha256_file(predictions_path) != response["predictions_sha256"]:
        raise ValueError("Aggregate predictions changed")
    table = pd.read_csv(predictions_path, sep="\t")
    identities = pd.read_csv(root / "eligible_trials.tsv.gz", sep="\t").set_index("trial_uid")
    names = [name.removeprefix("target_") for name in spec["targets"]]
    if not coverage["all_identities_covered"] or response["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Incomplete identity coverage or inconsistent fingerprint")
    expected_models = {f"{feature}_{setting}" for feature in spec["features"] for setting in config["ridge_settings"]}
    expected_models.update(config["eegnet_settings"])
    if set(table.representation) != expected_models:
        raise ValueError("Missing or unexpected representation")
    target_columns = list(spec["targets"])
    value_columns = target_columns + [c for c in table if any(c.endswith("_" + name) for name in names) and c not in target_columns]
    if not np.isfinite(table[value_columns].to_numpy(float)).all():
        raise ValueError("Non-finite final prediction")
    all_test, verified_cells = set(), 0
    support = []
    for fold in range(config["fold_count"]):
        seed = config["design_seed"] + fold
        fold_table = table.loc[table.seed.eq(seed)]
        fixed_test, counts = None, set()
        for p in config["doses"]:
            for s in config["doses"]:
                assignment_path = root / "assignments" / f"seed-{seed}" / f"participant-{p:02d}_stimulus-{s:02d}.tsv.gz"
                assignment = pd.read_csv(assignment_path, sep="\t")
                if assignment.trial_uid.duplicated().any() or set(assignment.trial_uid) != set(identities.index):
                    raise ValueError("Malformed assignment support")
                parts = {name: set(assignment.loc[assignment.split.eq(name), "trial_uid"])
                         for name in ("train", "validation", "test", "calibration")}
                current_test = parts["test"]
                if fixed_test is None:
                    fixed_test = current_test
                if fixed_test != current_test:
                    raise ValueError("Within-block test support changed")
                if parts["test"] & (parts["train"] | parts["validation"] | parts["calibration"]):
                    raise ValueError("Test trial used for fitting or calibration")
                counts.add(len(parts["train"]))
                test_id = identities.loc[sorted(current_test)]
                population_id = identities.loc[sorted(parts["train"] | parts["validation"])]
                cal = identities.loc[sorted(parts["calibration"])]
                if set(population_id.subject_uid) & set(test_id.subject_uid):
                    raise ValueError("Test participant entered population fit")
                cal_counts = cal.groupby("subject_uid").size().reindex(test_id.subject_uid.unique(), fill_value=0)
                exposure_counts = population_id.groupby("stimulus_uid").size().reindex(test_id.stimulus_uid.unique(), fill_value=0)
                if not cal_counts.eq(p).all() or not exposure_counts.eq(s).all():
                    raise ValueError("Actual exposure does not equal requested dose")
                if set(cal.stimulus_uid) & set(test_id.stimulus_uid):
                    raise ValueError("Calibration uses a tested stimulus")
                cell = fold_table.loc[fold_table.participant_dose.eq(p) & fold_table.stimulus_dose.eq(s)]
                priors = None
                for model in sorted(expected_models):
                    current = cell.loc[cell.representation.eq(model)].sort_values("trial_uid")
                    if current.trial_uid.duplicated().any() or set(current.trial_uid) != current_test:
                        raise ValueError("Model test support mismatch")
                    if not np.array_equal(current[target_columns].to_numpy(), test_id[target_columns].to_numpy()):
                        raise ValueError("Prediction truth differs from frozen trial table")
                    current_prior = current[[f"{prefix}_{name}" for prefix in ("prior", "prior_personalized") for name in names]].to_numpy()
                    if priors is not None and not np.allclose(current_prior, priors, rtol=0, atol=1e-10):
                        raise ValueError("Models do not receive the same prior/personalization")
                    priors = current_prior
                    if p == 0:
                        for base, personal in (("prior", "prior_personalized"), ("combined_population", "combined_personalized")):
                            if not np.allclose(current[[f"{base}_{n}" for n in names]], current[[f"{personal}_{n}" for n in names]], rtol=0, atol=1e-10):
                                raise ValueError("Zero-calibration predictor mismatch")
                    source = root / "cells" / f"fold-{fold}_p-{p}_s-{s}" / f"{model}.json"
                    record = json.loads(source.read_text())
                    if record["fingerprint"] != manifest["fingerprint"] or record["assignment_sha256"] != sha256_file(assignment_path):
                        raise ValueError("Cell source/assignment fingerprint mismatch")
                    if record["predictions_sha256"] != sha256_file(source.with_suffix(".tsv.gz")):
                        raise ValueError("Cell predictions changed")
                    verified_cells += 1
        if len(counts) != 1 or all_test & fixed_test:
            raise ValueError("Population count drift or repeated outer test trial")
        all_test.update(fixed_test)
        support.append(dict(fold=fold, test_trials=len(fixed_test), population_train_trials=counts.pop()))
    learning = []
    for path in sorted((root / "fits").glob("*.json")):
        record = json.loads(path.read_text())
        if record["fingerprint"] != manifest["fingerprint"] or sha256_file(path.with_suffix(".npz")) != record["prediction_sha256"]:
            raise ValueError("Fit cache changed")
        for branch, history in record["training"].items():
            result = verify_learning(history)
            if result["target_count"] != len(names):
                raise ValueError("Fabricated or missing target dimension")
            learning.append(dict(fit=path.stem, branch=branch, **result))
    expected_branches = set(config.get("eegnet_branches", ["direct", "residual"]))
    expected_fit_count = (
        config["fold_count"] * len(config["doses"])
        * len(config["eegnet_settings"]) * len(expected_branches)
    )
    if len(learning) != expected_fit_count or {item["branch"] for item in learning} != expected_branches:
        raise ValueError("Missing neural fits")
    return dict(status="passed", coverage=coverage, verified_model_cells=verified_cells,
                verified_prediction_rows=len(table), tested_unique_trials=len(all_test), support=support,
                input_fingerprint=manifest["fingerprint"], predictions_sha256=response["predictions_sha256"],
                targets=names, learning=learning,
                boundary="Observed trial resources and this executed training adapter only; not upstream pretraining provenance or latent emotion identification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    from braindecode.models import EEGNet
    import torch
    runtime = dict(packages={name: version(name) for name in ("torch", "braindecode", "numpy", "pandas", "scikit-learn", "scipy")},
                   eegnet_source_sha256=sha256_file(Path(inspect.getfile(EEGNet))),
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                   verifier_sha256=sha256_file(Path(__file__)), config_sha256=sha256_file(args.config))
    reports = {}
    for dataset, spec in config["datasets"].items():
        reports[dataset] = verify_dataset(args.response / dataset, spec, config)
        print(dataset, "independent output verification passed", flush=True)
    (args.output / "verification.json").write_text(json.dumps(dict(runtime=runtime, datasets=reports), indent=2) + "\n")


if __name__ == "__main__":
    main()
