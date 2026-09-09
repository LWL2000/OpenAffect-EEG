import copy
import hashlib

import pandas as pd
import pytest

from openaffect_eeg.workflow_audit import audit_workflow


def example():
    trials = pd.DataFrame(dict(trial_uid=["t0", "t1", "v", "c", "z0", "z1"],
        subject_uid=["p0", "p1", "p2", "p3", "p3", "p3"], stimulus_uid=["s0", "s0", "s1", "s2", "s3", "s4"]))
    resources = dict(calibration_trial_uids=["c"], population_label_trial_uids=["t0", "t1", "v"],
                     evaluation_trial_uids=["z0", "z1"], additional_label_resources=[])
    execution = dict(train_trial_uids=["t0", "t1"], validation_trial_uids=["v"], calibration_trial_uids=["c"], test_trial_uids=["z0", "z1"],
        preprocessing_fit_trial_uids=["t0", "t1"], model_selection_trial_uids=["v"],
        predictor_resources={"no_eeg": copy.deepcopy(resources), "eeg": copy.deepcopy(resources)},
        expected_test_support_sha256=hashlib.sha256(b"z0\nz1").hexdigest())
    return trials, execution, dict(require_matched_predictors=True, validation_trial_holdout=True)


def test_pass_is_observable_only_not_authenticity():
    trials, execution, contract = example()
    for extended in (False, True):
        result = audit_workflow(trials, execution, contract, extended=extended)
        assert result["observable_status"] == "allow"
        assert result["authenticity_status"] == "unverifiable"


@pytest.mark.parametrize("role", ["train", "validation", "calibration"])
def test_elementary_comparator_is_not_weakened_to_ignore_validation_or_calibration(role):
    trials, execution, contract = example()
    execution[role + "_trial_uids"].append("z0")
    for extended in (False, True):
        assert audit_workflow(trials, execution, contract, extended=extended)["observable_status"] == "block"


def test_comparator_also_checks_declared_identity_holdouts():
    trials, execution, contract = example()
    contract["subject_holdout"] = True
    for extended in (False, True):
        assert audit_workflow(trials, execution, contract, extended=extended)["observable_status"] == "block"


def test_matched_budget_missingness_is_not_a_clean_pass():
    trials, execution, contract = example()
    del execution["predictor_resources"]["eeg"]["calibration_trial_uids"]
    assert audit_workflow(trials, execution, contract)["observable_status"] == "unverifiable"


def test_calibration_resource_mismatch_is_distinct_from_split_leakage():
    trials, execution, contract = example()
    execution["predictor_resources"]["no_eeg"]["calibration_trial_uids"] = []
    assert audit_workflow(trials, execution, contract, extended=False)["observable_status"] == "allow"
    assert audit_workflow(trials, execution, contract)["observable_status"] == "block"


def test_fixed_support_drift_and_preprocessing_fit_are_traced():
    trials, execution, contract = example()
    execution["expected_test_support_sha256"] = "different"
    execution["preprocessing_fit_trial_uids"].append("z0")
    blocked = {r["check"] for r in audit_workflow(trials, execution, contract)["checks"] if r["status"] == "block"}
    assert {"fixed_evaluation_support", "preprocessing_test_exclusion"}.issubset(blocked)


def test_malformed_predictor_is_unknown_and_never_crashes():
    trials, execution, contract = example()
    execution["predictor_resources"]["no_eeg"] = None
    assert audit_workflow(trials, execution, contract)["observable_status"] == "unverifiable"


def test_execution_marker_fields_do_not_alias_during_fault_injection():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "scripts/run_external_workflow_audit_v9.py"
    spec = importlib.util.spec_from_file_location("external_workflow_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    execution = module.execution_record(["t"], ["v"], [], ["z"])
    execution["preprocessing_fit_trial_uids"].append("z")
    execution["model_selection_trial_uids"].append("z")
    assert execution["train_trial_uids"] == ["t"]
    assert execution["validation_trial_uids"] == ["v"]


@pytest.mark.parametrize("identifier", [123, "", "   "])
def test_noncanonical_trial_identity_fails_closed(identifier):
    trials, execution, contract = example()
    trials["trial_uid"] = trials.trial_uid.astype(object)
    trials.loc[0, "trial_uid"] = identifier
    assert audit_workflow(trials, execution, contract)["observable_status"] == "block"


@pytest.mark.parametrize("field,value", [("session_holdout", True), ("subject_holdout", "false"), ("stimulus_holdout", 0)])
def test_unsupported_or_nonboolean_contract_is_not_silently_accepted(field, value):
    trials, execution, contract = example()
    contract[field] = value
    assert audit_workflow(trials, execution, contract)["observable_status"] == "block"


def test_equal_but_wrong_population_resource_sets_do_not_pass():
    trials, execution, contract = example()
    for resources in execution["predictor_resources"].values():
        resources["population_label_trial_uids"] = ["t0", "t1"]
    assert audit_workflow(trials, execution, contract)["observable_status"] == "block"


def test_unlabelled_preprocessing_exposure_cannot_claim_unseen_participant():
    trials, execution, contract = example()
    execution["calibration_trial_uids"] = []
    for resources in execution["predictor_resources"].values():
        resources["calibration_trial_uids"] = []
    execution["preprocessing_fit_trial_uids"].append("c")
    contract["subject_holdout"] = True
    result = audit_workflow(trials, execution, contract)
    assert result["observable_status"] == "block"
    assert any(c["check"] == "subject_preprocessing_selection_holdout" and c["status"] == "block" for c in result["checks"])
