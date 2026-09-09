import pandas as pd
import pytest

from openaffect_eeg.audit_evidence import review_evidence


def fixture():
    assignment = pd.DataFrame({"trial_uid": ["tr", "v", "c", "t"],
                               "split": ["train", "validation", "calibration", "test"]})
    evidence = {"feature_sha256": "abc", "preprocessing_fit_trial_uids": ["tr", "v"],
                "predictor_calibration_trial_uids": {"prior_personalized": ["c"], "combined_personalized": ["c"]},
                "prior_resources": ["population_training_labels", "participant_calibration_labels"]}
    return assignment, evidence


def test_missing_provenance_is_not_clean():
    assignment, _ = fixture()
    report = review_evidence(assignment)
    assert report["status"] == "unverifiable"
    assert any(c["check"] == "declared_preprocessing_test_exclusion" and c["status"] == "unverifiable" for c in report["checks"])


def test_supplied_evidence_cannot_certify_authenticity():
    assignment, evidence = fixture()
    report = review_evidence(assignment, feature_sha256="abc", evidence=evidence)
    assert report["status"] == "unverifiable"
    assert not any(c["status"] == "block" for c in report["checks"])


@pytest.mark.parametrize("fault", ["hash", "preprocess_test", "unequal_calibration", "extra_calibration", "unknown_prior", "duplicate_trial"])
def test_known_faults_are_blocked(fault):
    assignment, evidence = fixture()
    if fault == "hash":
        evidence["feature_sha256"] = "changed"
    elif fault == "preprocess_test":
        evidence["preprocessing_fit_trial_uids"].append("t")
    elif fault == "unequal_calibration":
        evidence["predictor_calibration_trial_uids"]["prior_personalized"] = []
    elif fault == "extra_calibration":
        evidence["predictor_calibration_trial_uids"] = {k: ["c", "t"] for k in ("prior_personalized", "combined_personalized")}
    elif fault == "unknown_prior":
        evidence["prior_resources"].append("test_ratings")
    else:
        assignment = pd.concat([assignment, assignment.iloc[[-1]]])
    report = review_evidence(assignment, feature_sha256="abc", evidence=evidence)
    assert report["status"] == "block"


def test_cli_does_not_reinterpret_missing_provenance_as_pass(tmp_path):
    import json
    from openaffect_eeg.audit_cli import main
    assert main(["toy", "--output", str(tmp_path)]) == 0
    reports = json.loads((tmp_path / "evidence_review.json").read_text())
    assert all(r["status"] == "unverifiable" for r in reports.values())
    for report in reports.values():
        matched = next(c for c in report["checks"] if c["check"] == "matched_predictor_calibration")
        assert matched["status"] == "allow"


@pytest.mark.parametrize("field,value", [
    ("feature_sha256", ["abc"]), ("preprocessing_fit_trial_uids", "tr"),
    ("prior_resources", [["population_training_labels"]]),
    ("predictor_calibration_trial_uids", {"prior_personalized": ["c", "c"], "combined_personalized": ["c"]}),
])
def test_malformed_evidence_fails_closed(field, value):
    assignment, evidence = fixture()
    evidence[field] = value
    assert review_evidence(assignment, feature_sha256="abc", evidence=evidence)["status"] == "block"


def test_malformed_assignment_and_top_level_evidence_are_blocked():
    assignment, evidence = fixture()
    assert review_evidence(assignment, evidence=["not-a-mapping"])["status"] == "block"
    assignment.loc[0, "split"] = "secret_test"
    assert review_evidence(assignment, evidence=evidence)["status"] == "block"


def test_valid_resource_names_cannot_hide_actual_evaluator_resources():
    assignment, evidence = fixture()
    evidence["prior_resources"] = []
    report = review_evidence(assignment, evidence=evidence,
        actual_prior_resources=["population_training_labels", "participant_calibration_labels"])
    assert report["status"] == "block"
    check = next(x for x in report["checks"] if x["check"] == "prior_resources_match_evaluator")
    assert check["status"] == "block"
