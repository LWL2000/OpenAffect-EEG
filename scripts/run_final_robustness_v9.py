"""Execute the frozen v9 identity-coverage sensitivity without replacing v7/v8."""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.baselines import harmonize_affect_targets, regression_metrics
from openaffect_eeg.final_robustness import (
    GPURegressionConfig, coverage_assignments, fit_gpu_regression, population_targets, prediction_table,
)
from openaffect_eeg.identity_exposure import compile_exposure_cell, evaluate_exposure_cell


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".incomplete.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def feature_table(path, key):
    with np.load(path, allow_pickle=False) as data:
        uid = data["trial_uid" if "trial_uid" in data else "trial_uids"].astype(str)
        values = np.asarray(data[key], dtype=float).reshape(len(uid), -1)
    if not np.isfinite(values).all() or len(set(uid)) != len(uid):
        raise ValueError("Malformed feature archive")
    table = pd.DataFrame(values, columns=[f"feature_{i:04d}" for i in range(values.shape[1])])
    table.insert(0, "trial_uid", uid)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", choices=["prepare", "ridge", "eegnet", "aggregate"], required=True)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    training_seed_offset = int(config.get("training_seed_offset", 0))
    eegnet_branches = tuple(config.get("eegnet_branches", ("direct", "residual")))
    if not eegnet_branches or not set(eegnet_branches) <= {"direct", "residual"}:
        raise ValueError("eegnet_branches must contain direct and/or residual")
    spec = config["datasets"][args.dataset]
    targets = tuple(spec["targets"])
    names = tuple(t.removeprefix("target_") for t in targets)
    output = args.output / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    features = {name: feature_table(args.data_root / value["path"], value["array"])
                for name, value in spec["features"].items()}
    table = pd.read_csv(args.data_root / spec["trials"], sep="\t")
    source_rows = len(table)
    if spec["harmonize"]:
        table = harmonize_affect_targets(table)
        table = table.loc[table.target_available].copy()
    table = table.loc[table.dataset_id.eq(args.dataset) & np.isfinite(table[list(targets)].to_numpy(float)).all(1)].copy()
    if "subject_uid" not in table:
        table["subject_uid"] = table.dataset_id.astype(str) + ":" + table.subject_id.astype(str)
    all_uids = np.load(args.data_root / spec["uids"], allow_pickle=False).astype(str)
    if len(all_uids) != len(set(all_uids)):
        raise ValueError("Duplicate raw EEG UID")
    available = set(all_uids)
    for values in features.values():
        available &= set(values.trial_uid)
    table = table.loc[table.trial_uid.isin(available)].sort_values("trial_uid").reset_index(drop=True)
    table["tensor_index"] = np.arange(len(table))
    raw_indices = pd.Index(all_uids).get_indexer(table.trial_uid)
    if (raw_indices < 0).any() or table.trial_uid.duplicated().any():
        raise ValueError("Raw/label identity mismatch")
    paths = [args.config, args.data_root / spec["trials"], args.data_root / spec["tensors"], args.data_root / spec["uids"],
             Path(__file__), Path(inspect.getfile(coverage_assignments)),
             Path(inspect.getfile(compile_exposure_cell)),
             Path(inspect.getfile(regression_metrics))]
    paths += [args.data_root / value["path"] for value in spec["features"].values()]
    hashes = {str(path): sha256_file(path) for path in paths}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "input_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Frozen input/source fingerprint changed; do not reuse results")
    else:
        write_json(manifest_path, dict(fingerprint=fingerprint, input_sha256=hashes, source_rows=source_rows,
            eligible_rows=len(table), participants=table.subject_uid.nunique(), stimuli=table.stimulus_uid.nunique(),
            targets=list(targets), design_seed=config["design_seed"], coverage="identity diagonal blocks, not all crossed pairs"))
    strict_assignments = coverage_assignments(table, folds=config["fold_count"], seed=config["design_seed"])
    doses = config["doses"]
    if args.phase == "prepare":
        table.to_csv(output / "eligible_trials.tsv.gz", sep="\t", index=False)
        support_p, support_s, support_trials = set(), set(), set()
        records = []
        for fold, strict in enumerate(strict_assignments):
            seed = config["design_seed"] + fold
            folder = output / "assignments" / f"seed-{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            strict.to_csv(folder / "strict.tsv.gz", sep="\t", index=False)
            test_hashes, train_counts = set(), set()
            for p in doses:
                for s in doses:
                    assigned, audit = compile_exposure_cell(table, strict, participant_dose=p, stimulus_dose=s,
                        maximum_participant_dose=max(doses), maximum_stimulus_dose=max(doses), seed=seed,
                        target_columns=targets)
                    path = folder / f"participant-{p:02d}_stimulus-{s:02d}.tsv.gz"
                    assigned.to_csv(path, sep="\t", index=False)
                    write_json(path.with_suffix(".audit.json"), audit)
                    test_hashes.add(audit["fixed_test_support_sha256"])
                    train_counts.add(audit["counts"]["train"])
                    if p == s == 0:
                        test = table.loc[table.trial_uid.isin(assigned.loc[assigned.split.eq("test"), "trial_uid"])]
                        support_p.update(test.subject_uid)
                        support_s.update(test.stimulus_uid)
                        if support_trials & set(test.trial_uid):
                            raise ValueError("Outer test trial support overlaps")
                        support_trials.update(test.trial_uid)
                        records.append(dict(fold=fold, seed=seed, test_trials=len(test),
                            test_participants=test.subject_uid.nunique(), test_stimuli=test.stimulus_uid.nunique()))
            if len(test_hashes) != 1 or len(train_counts) != 1:
                raise ValueError("Support or population size changed within coverage block")
        coverage = dict(folds=records, tested_participants=len(support_p), tested_stimuli=len(support_s),
                        tested_trials=len(support_trials), total_participants=table.subject_uid.nunique(),
                        total_stimuli=table.stimulus_uid.nunique(), total_trials=len(table),
                        all_identities_covered=len(support_p) == table.subject_uid.nunique() and len(support_s) == table.stimulus_uid.nunique())
        write_json(output / "coverage.json", coverage)
        print(args.dataset, json.dumps(coverage), flush=True)
        return
    if not (output / "coverage.json").exists():
        raise ValueError("Prepare and inspect the support contract before training")
    if args.phase == "aggregate":
        paths = sorted((output / "cells").glob("*/*.tsv.gz"))
        predictions = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
        expected = len(spec["features"]) * len(config["ridge_settings"]) + len(config["eegnet_settings"])
        observed = predictions.groupby(["representation", "seed", "participant_dose", "stimulus_dose"]).size()
        if len(observed) != expected * config["fold_count"] * len(doses) ** 2:
            raise ValueError("Incomplete grid: aggregate refuses to hide missing experiments")
        uid_sets = predictions.groupby(["seed", "participant_dose", "stimulus_dose", "representation"]).trial_uid.apply(lambda x: frozenset(x))
        for seed in predictions.seed.unique():
            if len(set(uid_sets.loc[seed])) != 1:
                raise ValueError("Model/grid test support mismatch")
        destination = output / "identity_exposure_predictions.tsv.gz"
        predictions.to_csv(destination, sep="\t", index=False)
        records = [json.loads(path.read_text()) for path in sorted((output / "cells").glob("*/*.json"))]
        write_json(output / "identity_exposure_response.json", dict(results=records, fingerprint=fingerprint,
            predictions_sha256=sha256_file(destination), target_names=names,
            training_seed_offset=training_seed_offset,
            adaptation="bounded v9 robustness, not a new cohort"))
        print(args.dataset, "complete verified grid", len(observed), flush=True)
        return
    gpu_tensors = None
    if args.phase == "eegnet":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required")
        raw = np.load(args.data_root / spec["tensors"], mmap_mode="r", allow_pickle=False)
        data = np.asarray(raw[raw_indices], dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("Non-finite raw EEG")
        gpu_tensors = torch.as_tensor(data, device="cuda")
        del data
    for fold in range(config["fold_count"]):
        if args.fold is not None and args.fold != fold:
            continue
        seed = config["design_seed"] + fold
        training_seed = seed + training_seed_offset
        folder = output / "assignments" / f"seed-{seed}"
        for s in doses:
            base_assignment = pd.read_csv(folder / f"participant-00_stimulus-{s:02d}.tsv.gz", sep="\t")
            neural = {}
            if args.phase == "eegnet":
                base = population_targets(table, base_assignment, targets)
                for setting, lr in config["eegnet_settings"].items():
                    train_config = GPURegressionConfig(crop_samples=spec["crop_samples"], learning_rate=lr, **config["eegnet"])
                    cache = output / "fits" / f"{setting}_fold-{fold}_stimulus-{s}"
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache_json = cache.with_suffix(".json")
                    cache_npz = cache.with_suffix(".npz")
                    if cache_json.exists():
                        metadata = json.loads(cache_json.read_text())
                        if metadata["fingerprint"] != fingerprint or metadata["prediction_sha256"] != sha256_file(cache_npz):
                            raise ValueError("Neural checkpoint fingerprint mismatch")
                        with np.load(cache_npz, allow_pickle=False) as stored:
                            direct, residual = stored["direct"], stored["residual"]
                    else:
                        predictions, training = {}, {}
                        for kind in eegnet_branches:
                            is_residual = kind == "residual"
                            predictions[kind], training[kind] = fit_gpu_regression(gpu_tensors,
                                base["train"].tensor_index.to_numpy(), base["train_y"] - base["train_loo"] if is_residual else base["train_y"],
                                base["validation"].tensor_index.to_numpy(), base["val_y"] - base["val_prior"] if is_residual else base["val_y"],
                                base["fit"].tensor_index.to_numpy(), base["fit_y"] - base["fit_loo"] if is_residual else base["fit_y"],
                                seed=training_seed, config=train_config, checkpoint=cache.parent / f"{cache.name}_{kind}.pt")
                        template = next(iter(predictions.values()))
                        direct = predictions.get("direct", np.zeros_like(template))
                        residual = predictions.get("residual", np.zeros_like(template))
                        np.savez_compressed(cache_npz, direct=direct, residual=residual)
                        metadata = dict(fingerprint=fingerprint, prediction_sha256=sha256_file(cache_npz),
                                        training_seed=training_seed,
                                        config=asdict(train_config), training=training)
                        write_json(cache_json, metadata)
                    neural[setting] = (base["all_prior"], direct, residual, metadata)
                    print(args.dataset, "neural", fold, s, setting,
                          {k: v["selected_epochs"] for k, v in metadata["training"].items()}, flush=True)
            for p in doses:
                path = folder / f"participant-{p:02d}_stimulus-{s:02d}.tsv.gz"
                assignment = pd.read_csv(path, sep="\t")
                cell_dir = output / "cells" / f"fold-{fold}_p-{p}_s-{s}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                experiments = []
                if args.phase == "ridge":
                    for feature_name, values in features.items():
                        for setting, alphas in config["ridge_settings"].items():
                            label = feature_name + "_" + setting
                            if (cell_dir / f"{label}.json").exists():
                                cached = json.loads((cell_dir / f"{label}.json").read_text())
                                if cached["fingerprint"] != fingerprint or cached["predictions_sha256"] != sha256_file(cell_dir / f"{label}.tsv.gz"):
                                    raise ValueError("Ridge resume fingerprint mismatch")
                                continue
                            result, prediction = evaluate_exposure_cell(table, assignment, values, backend="torch",
                                                                        alphas=tuple(alphas), target_columns=targets)
                            experiments.append((label, result, prediction))
                else:
                    for label, (prior, direct, residual, metadata) in neural.items():
                        prediction = prediction_table(table, assignment, targets, prior, direct, residual)
                        prediction.insert(0, "seed", seed)
                        prediction.insert(1, "participant_dose", p)
                        prediction.insert(2, "stimulus_dose", s)
                        truth = prediction[list(targets)].to_numpy(float)
                        predictors = ("prior", "prior_personalized", "eeg_population", "eeg_personalized", "combined_population", "combined_personalized")
                        metrics = {name: regression_metrics(truth, prediction[[f"{name}_{target}" for target in names]].to_numpy(float), target_names=names)
                                   for name in predictors}
                        result = dict(seed=seed, participant_dose=p, stimulus_dose=s, metrics=metrics, training=metadata)
                        experiments.append((label, result, prediction))
                for label, result, prediction in experiments:
                    prediction.insert(0, "representation", label)
                    dest = cell_dir / f"{label}.tsv.gz"
                    prediction.to_csv(dest, sep="\t", index=False)
                    result.update(representation=label, fingerprint=fingerprint,
                                  assignment_sha256=sha256_file(path), predictions_sha256=sha256_file(dest))
                    write_json(cell_dir / f"{label}.json", result)
            print(args.dataset, args.phase, "verified fold/stimulus", fold, s, flush=True)


if __name__ == "__main__":
    main()
