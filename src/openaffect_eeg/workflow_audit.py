"""Observable execution-contract audit, not an upstream authenticity certificate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def audit_workflow(trials, execution, contract, *, extended=True):
    """Check executed UID sets, not the advertised name of a split protocol.

    The elementary comparator uses all the same observable split/isolation
    checks. The extended tool additionally checks predictor resource matching,
    preprocessing provenance, and fixed evaluation support.
    """
    checks = []

    def add(name, status, **detail):
        checks.append(dict(check=name, status=status, detail=detail))

    def uid_set(value, label, *, known):
        if value is None:
            add(label, "unverifiable", reason="UID list was not supplied")
            return None
        if not isinstance(value, list) or any(not isinstance(uid, str) or not uid.strip() for uid in value):
            add(label, "block", reason="UIDs must be nonempty strings in a list")
            return None
        ids = set(value)
        if len(ids) != len(value) or not ids.issubset(known):
            add(label, "block", duplicates=len(value) - len(ids), unknown_count=len(ids - known))
            return None
        return ids

    if not isinstance(execution, dict) or not isinstance(contract, dict):
        return dict(observable_status="block", authenticity_status="unverifiable", checks=[
            dict(check="schema", status="block", detail="Expected execution and contract mappings")])
    supported = {"subject_holdout", "stimulus_holdout", "validation_trial_holdout", "require_matched_predictors"}
    if set(contract) - supported or any(type(value) is not bool for value in contract.values()):
        return dict(observable_status="block", authenticity_status="unverifiable", checks=[
            dict(check="contract_schema", status="block", detail="Only supported boolean contract fields are accepted")])
    if ("trial_uid" not in trials or trials.trial_uid.isna().any() or trials.trial_uid.duplicated().any()
            or not trials.trial_uid.map(lambda uid: isinstance(uid, str) and bool(uid.strip())).all()):
        return dict(observable_status="block", authenticity_status="unverifiable", checks=[
            dict(check="trial_schema", status="block", detail="Unique trial identity table required")])
    known = set(trials.trial_uid.astype(str))
    roles = {}
    for role in ("train", "validation", "calibration", "test"):
        roles[role] = uid_set(execution.get(role + "_trial_uids"), role + "_uid_schema", known=known)
    test = roles["test"]
    if test is not None and not test:
        add("nonempty_test", "block", reason="No evaluation trials")
    for role in ("train", "validation", "calibration"):
        ids = roles[role]
        if ids is not None and test is not None:
            overlap = ids & test
            add(role + "_test_trial_isolation", "block" if overlap else "allow", overlap_count=len(overlap))
    if contract.get("validation_trial_holdout", False) and roles["train"] is not None and roles["validation"] is not None:
        overlap = roles["train"] & roles["validation"]
        add("training_validation_trial_isolation", "block" if overlap else "allow", overlap_count=len(overlap))
    for axis in ("subject", "stimulus"):
        if not contract.get(axis + "_holdout", False):
            continue
        column = axis + "_uid"
        if (column not in trials or trials[column].isna().any()
                or not trials[column].map(lambda uid: isinstance(uid, str) and bool(uid.strip())).all()
                or roles["train"] is None or test is None):
            add(axis + "_holdout", "unverifiable", reason="Complete source identities and roles are required")
            continue
        development = roles["train"] | (roles["validation"] or set()) | (roles["calibration"] or set())
        left = set(trials.loc[trials.trial_uid.isin(development), column].astype(str))
        right = set(trials.loc[trials.trial_uid.isin(test), column].astype(str))
        add(axis + "_holdout", "block" if left & right else "allow", overlap_count=len(left & right))
    if extended:
        fit = uid_set(execution.get("preprocessing_fit_trial_uids"), "preprocessing_evidence", known=known)
        if fit is not None and test is not None:
            add("preprocessing_test_exclusion", "block" if fit & test else "allow", overlap_count=len(fit & test))
        selection = uid_set(execution.get("model_selection_trial_uids"), "selection_evidence", known=known)
        if selection is not None and test is not None:
            add("model_selection_test_exclusion", "block" if selection & test else "allow", overlap_count=len(selection & test))
        for axis in ("subject", "stimulus"):
            column = axis + "_uid"
            if contract.get(axis + "_holdout", False) and column in trials and test is not None:
                if fit is None or selection is None:
                    add(axis + "_preprocessing_selection_holdout", "unverifiable", reason="Execution evidence missing")
                else:
                    left = set(trials.loc[trials.trial_uid.isin(fit | selection), column].astype(str))
                    right = set(trials.loc[trials.trial_uid.isin(test), column].astype(str))
                    add(axis + "_preprocessing_selection_holdout", "block" if left & right else "allow", overlap_count=len(left & right))
        predictors = execution.get("predictor_resources")
        if contract.get("require_matched_predictors", False):
            if not isinstance(predictors, dict) or not {"no_eeg", "eeg"}.issubset(predictors):
                add("matched_predictor_evidence", "unverifiable", reason="Both executed predictor resource maps are needed")
            else:
                for resource in ("calibration_trial_uids", "population_label_trial_uids", "evaluation_trial_uids"):
                    pair = [uid_set(predictors[name].get(resource) if isinstance(predictors[name], dict) else None,
                                    name + "_" + resource, known=known) for name in ("no_eeg", "eeg")]
                    if any(value is None for value in pair):
                        continue
                    matched = pair[0] == pair[1]
                    if resource == "calibration_trial_uids":
                        matched = matched and pair[0] == roles["calibration"]
                    if resource == "evaluation_trial_uids":
                        matched = matched and pair[0] == test
                    if resource == "population_label_trial_uids" and roles["train"] is not None and roles["validation"] is not None:
                        matched = matched and pair[0] == roles["train"] | roles["validation"]
                    if resource != "evaluation_trial_uids" and test is not None:
                        matched = matched and not (pair[0] | pair[1]) & test
                    add("matched_" + resource, "allow" if matched else "block",
                        left_count=len(pair[0]), right_count=len(pair[1]), symmetric_difference_count=len(pair[0] ^ pair[1]))
                left = predictors["no_eeg"].get("additional_label_resources") if isinstance(predictors["no_eeg"], dict) else None
                right = predictors["eeg"].get("additional_label_resources") if isinstance(predictors["eeg"], dict) else None
                if not isinstance(left, list) or not isinstance(right, list):
                    add("matched_additional_label_resources", "unverifiable", reason="External label-resource declarations missing")
                elif any(not isinstance(x, str) or not x for x in left + right):
                    add("matched_additional_label_resources", "block", reason="Malformed resource names")
                else:
                    add("matched_additional_label_resources", "allow" if set(left) == set(right) else "block",
                        different_resource_count=len(set(left) ^ set(right)))
        expected_support = execution.get("expected_test_support_sha256")
        if expected_support is None:
            add("fixed_evaluation_support", "unverifiable", reason="No predeclared evaluation-support digest")
        elif test is not None:
            actual = hashlib.sha256("\n".join(sorted(test)).encode()).hexdigest()
            add("fixed_evaluation_support", "allow" if actual == expected_support else "block")
    status = "block" if any(c["status"] == "block" for c in checks) else (
        "unverifiable" if any(c["status"] == "unverifiable" for c in checks) else "allow")
    return dict(observable_status=status, authenticity_status="unverifiable", checks=checks,
                boundary="allow applies only to supplied observable execution evidence; identities, aliases and pretraining authenticity remain unverifiable")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_workflow(pd.read_csv(args.trials, sep="\t"), json.loads(args.execution.read_text()), json.loads(args.contract.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 2 if result["observable_status"] == "block" else 3 if result["observable_status"] == "unverifiable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
