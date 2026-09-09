"""Trace official split/control-flow functions; separate source behavior from injected faults."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.workflow_audit import audit_workflow


SOURCES = {
    "EEGain": dict(commit="d6892f5586181f345969651a9baedabd828bb1c5", url="https://github.com/EmotionLab/EEGain"),
    "LibEER": dict(commit="dddff9776dbdae21195fe320dff0a5ba61628a18", url="https://github.com/XJTU-EEG/LibEER"),
}
SOURCE_FILES = {
    ("LibEER", "LibEER/data_utils/split.py"): "49757dd41a50a0d53aa5d90c7208e7c1885471198e520c66368109cb675315e4",
    ("LibEER", "LibEER/EEGNet_train.py"): "1e42fead0c095e325066b79e6068c667997bc270546faae070ed06dc315db244",
    ("EEGain", "eegain/data/loader.py"): "a1b73f7dc1985ad73397be35d58d66f99b3307cc7b862537ca6705ddebc38299",
}


def compile_definition(path, name, namespace):
    """Execute the exact pinned AST definition with explicitly supplied imports."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one official definition: {name}")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name], dict(file=path.name, function=name, source_sha256=sha256_file(path),
                                 first_line=nodes[0].lineno, last_line=nodes[0].end_lineno)


def support_hash(uids):
    return hashlib.sha256("\n".join(sorted(set(uids))).encode()).hexdigest()


def execution_record(train, validation, calibration, test, *, preprocessing=None, selection=None):
    train, validation, calibration, test = [sorted(set(x)) for x in (train, validation, calibration, test)]
    resources = dict(calibration_trial_uids=calibration, population_label_trial_uids=sorted(set(train) | set(validation)),
                     evaluation_trial_uids=test, additional_label_resources=[])
    return dict(train_trial_uids=train, validation_trial_uids=validation, calibration_trial_uids=calibration,
        test_trial_uids=test, preprocessing_fit_trial_uids=list(train) if preprocessing is None else sorted(set(preprocessing)),
        model_selection_trial_uids=list(validation) if selection is None else sorted(set(selection)),
        predictor_resources={"no_eeg": copy.deepcopy(resources), "eeg": copy.deepcopy(resources)},
        expected_test_support_sha256=support_hash(test))


def record_case(name, origin, table, execution, contract, expected, source, output, rows):
    reports = {}
    for checker, extended in (("elementary_split", False), ("resource_contract", True)):
        started = time.perf_counter()
        report = audit_workflow(table, execution, contract, extended=extended)
        reports[checker] = report
        rows.append(dict(case=name, origin=origin, source=source, checker=checker,
                         expected_observable_status=expected, observed_status=report["observable_status"],
                         runtime_ms=(time.perf_counter() - started) * 1000,
                         blocked_checks=";".join(c["check"] for c in report["checks"] if c["status"] == "block")))
    # Execution identities remain private. Publication contains aggregate diagnostics only.
    (output / f"{name}.json").write_text(json.dumps(dict(execution=execution, contract=contract, reports=reports), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    parser.add_argument("eegain_source", type=Path)
    parser.add_argument("libeer_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    roots = {"EEGain": args.eegain_source, "LibEER": args.libeer_source}
    for (project, name), digest in SOURCE_FILES.items():
        if sha256_file(roots[project] / name) != digest:
            raise ValueError(f"Expected exact pinned source bytes: {project}/{name}; use git archive to avoid line-ending conversion")
    args.output.mkdir(parents=True, exist_ok=True)
    private = args.output / "private_traces"
    private.mkdir(exist_ok=True)
    table = pd.read_csv(args.trials, sep="\t").sort_values("trial_uid").reset_index(drop=True)
    if table.trial_uid.duplicated().any():
        raise ValueError("Duplicate source trials")
    subjects = sorted(table.subject_uid.unique())
    rows, provenance = [], []
    # Actual eligible trial identities, with three synthetic window-index markers
    # per trial. No EEG classifier is trained or scored in this control-flow audit.
    lib_namespace = dict(np=np, random=random, KFold=KFold, LeaveOneOut=LeaveOneOut,
                         StratifiedKFold=StratifiedKFold, train_test_split=train_test_split)
    split_path = args.libeer_source / "LibEER/data_utils/split.py"
    for name in ("get_split_index", "index_to_data"):
        _, source = compile_definition(split_path, name, lib_namespace)
        provenance.append(dict(project="LibEER", **source))
    # Execute the unchanged official empty-validation branch, not a paraphrase.
    entry = args.libeer_source / "LibEER/EEGNet_train.py"
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.dump(node.test) == ast.dump(ast.parse("len(val_data) == 0", mode="eval").body)]
    if len(candidates) != 1:
        raise ValueError("Pinned LibEER no-validation branch changed")
    fallback = compile(ast.Module(body=candidates, type_ignores=[]), str(entry), "exec")
    provenance.append(dict(project="LibEER", file=entry.name, source_sha256=sha256_file(entry),
                           first_line=candidates[0].lineno, last_line=candidates[0].end_lineno,
                           function="exact empty-validation branch; classifier training not executed"))
    # Only the loader class is required; source annotation types are supplied.
    import torch
    import logging
    from itertools import combinations
    from typing import Any, Dict, List, Tuple
    namespace = dict(np=np, copy=copy, torch=torch, logger=logging.getLogger("external_audit"),
                     combinations=combinations, Any=Any, Dict=Dict, List=List, Tuple=Tuple,
                     EEGDataset=object, EEGDatasetBase=object, DataLoader=object)
    loader, source = compile_definition(args.eegain_source / "eegain/data/loader.py", "EEGDataloader", namespace)
    provenance.append(dict(project="EEGain", **source))
    native_instance = loader(dataset=None, batch_size=16)
    for fold in range(5):
        random.seed(20260908 + fold)
        np.random.seed(20260908 + fold)
        settings = SimpleNamespace(split_type="leave-one-out", experiment_mode="subject-independent", sr=[fold + 1])
        data = [table.index[table.subject_uid.eq(subject)].to_numpy().reshape(-1, 1) for subject in subjects]
        labels = [np.zeros(len(values), dtype=int) for values in data]
        split = lib_namespace["get_split_index"](data, labels, settings)
        arrays = lib_namespace["index_to_data"](data, labels, split["train"][0], split["test"][0], split["val"][0])
        train_data, train_label, val_data, val_label, test_data, test_label = arrays
        state = dict(val_data=val_data, val_label=val_label, test_data=test_data, test_label=test_label)
        exec(fallback, state)
        train_uids = table.iloc[train_data.reshape(-1).astype(int)].trial_uid.tolist()
        test_uids = table.iloc[test_data.reshape(-1).astype(int)].trial_uid.tolist()
        val_uids = table.iloc[state["val_data"].reshape(-1).astype(int)].trial_uid.tolist()
        native = execution_record(train_uids, val_uids, [], test_uids, preprocessing=[])
        record_case(f"libeer_no_validation_fold{fold}", "observed_pinned_source_control_flow", table, native,
                    dict(subject_holdout=True, validation_trial_holdout=True, require_matched_predictors=False),
                    "block", "LibEER", private, rows)
        # Correct use of the native explicit train/validation/test subject split.
        settings = SimpleNamespace(split_type="train-val-test", experiment_mode="subject-independent", sr=None,
                                   test_size=0.2, val_size=0.2)
        split = lib_namespace["get_split_index"](data, labels, settings)
        arrays = lib_namespace["index_to_data"](data, labels, split["train"][0], split["test"][0], split["val"][0])
        train_uids = table.iloc[arrays[0].reshape(-1).astype(int)].trial_uid.tolist()
        val_uids = table.iloc[arrays[2].reshape(-1).astype(int)].trial_uid.tolist()
        test_uids = table.iloc[arrays[4].reshape(-1).astype(int)].trial_uid.tolist()
        clean = execution_record(train_uids, val_uids, [], test_uids, preprocessing=[])
        record_case(f"libeer_explicit_validation_fold{fold}", "observed_pinned_source_split", table, clean,
                    dict(subject_holdout=True, validation_trial_holdout=True, require_matched_predictors=False),
                    "allow", "LibEER", private, rows)
        # EEGain stratifies windows. Markers retain actual trial identity after
        # native shuffling; this does not fabricate independent EEG observations.
        development = table.loc[~table.subject_uid.eq(subjects[fold])].copy()
        marker = np.repeat(development.index.to_numpy(), 3)
        x = torch.tensor(marker[:, None], dtype=torch.float64)
        label = np.repeat((development.target_valence.to_numpy() > 0.5).astype(int), 3)
        split_x, _, split_val, _, _, _ = native_instance.split_train_val(x, label, train_ratio=0.8, videos=marker.tolist())
        train_uids = table.iloc[split_x.numpy().reshape(-1).astype(int)].trial_uid.tolist()
        val_uids = table.iloc[split_val.numpy().reshape(-1).astype(int)].trial_uid.tolist()
        test_uids = table.loc[table.subject_uid.eq(subjects[fold]), "trial_uid"].tolist()
        window_execution = execution_record(train_uids, val_uids, [], test_uids, preprocessing=development.trial_uid.tolist())
        record_case(f"eegain_window_validation_fold{fold}", "native_function_on_three_marker_windows_per_real_trial", table,
                    window_execution, dict(subject_holdout=True, validation_trial_holdout=True, require_matched_predictors=False),
                    "block", "EEGain", private, rows)
        record_case(f"eegain_subject_test_fold{fold}", "same_native_trace_without_unclaimed_validation_trial_holdout", table,
                    window_execution, dict(subject_holdout=True, validation_trial_holdout=False, require_matched_predictors=False),
                    "allow", "EEGain", private, rows)
        # Independently declared trace mutations exercise the broader resource
        # contract. They are not defects attributed to either upstream library.
        for source_name, base in (("LibEER", clean), ("EEGain", window_execution)):
            baseline = copy.deepcopy(base)
            # Use executed calibration from one non-test development trial; the
            # marker scenario is a resource checker fixture, not a study result.
            calibration = baseline["train_trial_uids"].pop()
            baseline["validation_trial_uids"] = [u for u in baseline["validation_trial_uids"] if u != calibration]
            baseline = execution_record(baseline["train_trial_uids"], baseline["validation_trial_uids"], [calibration], baseline["test_trial_uids"])
            contract = dict(subject_holdout=True, validation_trial_holdout=False, require_matched_predictors=True)
            variants = {"matched": (copy.deepcopy(baseline), "allow")}
            variants["unequal_calibration"] = (copy.deepcopy(baseline), "block")
            variants["unequal_calibration"][0]["predictor_resources"]["no_eeg"]["calibration_trial_uids"] = []
            variants["unequal_population_labels"] = (copy.deepcopy(baseline), "block")
            variants["unequal_population_labels"][0]["predictor_resources"]["eeg"]["population_label_trial_uids"] = []
            variants["different_evaluation_support"] = (copy.deepcopy(baseline), "block")
            variants["different_evaluation_support"][0]["predictor_resources"]["eeg"]["evaluation_trial_uids"].pop()
            variants["missing_provenance"] = (copy.deepcopy(baseline), "unverifiable")
            del variants["missing_provenance"][0]["preprocessing_fit_trial_uids"]
            del variants["missing_provenance"][0]["predictor_resources"]
            variants["compound_fit_and_support"] = (copy.deepcopy(baseline), "block")
            variants["compound_fit_and_support"][0]["preprocessing_fit_trial_uids"].append(baseline["test_trial_uids"][0])
            variants["compound_fit_and_support"][0]["expected_test_support_sha256"] = "different"
            variants["selection_test_not_declared_as_validation"] = (copy.deepcopy(baseline), "block")
            variants["selection_test_not_declared_as_validation"][0]["model_selection_trial_uids"] = baseline["test_trial_uids"]
            for variant, (execution, expected) in variants.items():
                record_case(f"{source_name.lower()}_{variant}_fold{fold}", "injected_contract_fixture_not_upstream_defect",
                            table, execution, contract, expected, source_name, private, rows)
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "audit_cases.csv", index=False)
    summary = []
    for (origin, checker), values in results.groupby(["origin", "checker"]):
        faulty = values.expected_observable_status.eq("block")
        clean = values.expected_observable_status.eq("allow")
        unknown = values.expected_observable_status.eq("unverifiable")
        summary.append(dict(origin=origin, checker=checker, cases=len(values),
            violations=int(faulty.sum()), detected_violations=int((faulty & values.observed_status.eq("block")).sum()),
            false_observable_allows=int((~clean & values.observed_status.eq("allow")).sum()),
            clean_cases=int(clean.sum()), false_blocks=int((clean & values.observed_status.eq("block")).sum()),
            unknown_cases=int(unknown.sum()), retained_unknown=int((unknown & values.observed_status.eq("unverifiable")).sum()),
            median_runtime_ms=float(values.runtime_ms.median())))
    public = dict(sources=SOURCES, source_functions=provenance, input_manifest_sha256=sha256_file(args.trials),
                  implementation_sha256={"runner": sha256_file(Path(__file__)), "workflow_audit": sha256_file(Path(inspect.getfile(audit_workflow)))},
                  cases_sha256=sha256_file(args.output / "audit_cases.csv"), summary=summary,
                  boundaries=["Control-flow/source-split execution, not official emotion classifier reproduction",
                    "Actual eligible trial identities; marker windows are generated, not measured independent EEG",
                    "Injected resource faults are not attributed to upstream libraries",
                    "Developer-designed finite fixtures do not estimate literature-wide failure prevalence",
                    "No independent human user acceptance was performed", "No published benchmark result is invalidated by these traces"])
    (args.output / "public_summary.json").write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
