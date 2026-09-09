"""Run four deterministic synthetic workflow checks; no EEG or model weights required."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

from openaffect_eeg.workflow_audit import audit_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(dict(trial_uid=["train", "validation", "calibration", "test"],
                              subject_uid=["p0", "p1", "p2", "p2"], stimulus_uid=["s0", "s1", "s2", "s3"]))
    resources = dict(calibration_trial_uids=["calibration"], population_label_trial_uids=["train", "validation"],
                     evaluation_trial_uids=["test"], additional_label_resources=[])
    execution = dict(train_trial_uids=["train"], validation_trial_uids=["validation"],
                     calibration_trial_uids=["calibration"], test_trial_uids=["test"],
                     preprocessing_fit_trial_uids=["train"], model_selection_trial_uids=["validation"],
                     predictor_resources={"no_eeg": copy.deepcopy(resources), "eeg": copy.deepcopy(resources)},
                     expected_test_support_sha256=hashlib.sha256(b"test").hexdigest())
    contract = dict(subject_holdout=False, stimulus_holdout=True, validation_trial_holdout=True, require_matched_predictors=True)
    examples = {"matched": (copy.deepcopy(execution), contract, "allow")}
    mismatch = copy.deepcopy(execution)
    mismatch["predictor_resources"]["no_eeg"]["calibration_trial_uids"] = []
    examples["unmatched_calibration"] = (mismatch, contract, "block")
    unknown = copy.deepcopy(execution)
    del unknown["preprocessing_fit_trial_uids"]
    examples["missing_provenance"] = (unknown, contract, "unverifiable")
    examples["false_unseen_subject_claim"] = (execution, {**contract, "subject_holdout": True}, "block")
    table.to_csv(args.output / "trials.tsv", sep="\t", index=False)
    summary = []
    for name, (trace, claims, expected) in examples.items():
        report = audit_workflow(table, trace, claims)
        if report["observable_status"] != expected or report["authenticity_status"] != "unverifiable":
            raise AssertionError(f"Unexpected toy result: {name}")
        for kind, value in (("execution", trace), ("contract", claims), ("report", report)):
            (args.output / f"{name}_{kind}.json").write_text(json.dumps(value, indent=2) + "\n")
        summary.append(dict(case=name, expected=expected, observed=report["observable_status"]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
