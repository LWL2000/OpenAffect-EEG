"""Three-state checks of supplied evidence, with explicit unverifiable boundaries."""
from __future__ import annotations


def review_evidence(assignment, *, feature_sha256=None, evidence=None, actual_prior_resources=None):
    """Validate declared preprocessing and predictor resources against a split.

    A matching hash binds a declaration to a file; it does not establish that
    the declaration truthfully describes the computation that produced it.
    """
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    if evidence is None:
        evidence = {}
    elif not isinstance(evidence, dict):
        add("evidence_schema", "block", "Evidence must be a mapping")
        evidence = {}
    allowed_splits = {"train", "validation", "calibration", "test", "excluded"}
    valid_ids = assignment["trial_uid"].map(lambda x: isinstance(x, str) and bool(x.strip())).all()
    valid_splits = assignment["split"].isin(allowed_splits).all()
    add("assignment_schema", "allow" if valid_ids and valid_splits else "block",
        "Trial IDs must be nonempty strings and split roles must be recognized")
    if assignment["trial_uid"].duplicated().any():
        add("unique_trial_assignment", "block", "A trial has more than one assigned row")
    else:
        add("unique_trial_assignment", "allow", "One split row per supplied trial ID")
    ids = {part: set(assignment.loc[assignment["split"].eq(part), "trial_uid"].astype(str))
           for part in ("train", "validation", "calibration", "test")}
    declared_hash = evidence.get("feature_sha256")
    if not declared_hash or not feature_sha256:
        add("feature_file_binding", "unverifiable", "No supplied hash and actual feature-file hash pair")
    elif not isinstance(declared_hash, str) or not isinstance(feature_sha256, str):
        add("feature_file_binding", "block", "Feature digests must be strings")
    else:
        add("feature_file_binding", "allow" if declared_hash == feature_sha256 else "block",
            "Declared feature digest compared with the actual input file")
    fit = evidence.get("preprocessing_fit_trial_uids")
    if fit is None:
        add("declared_preprocessing_test_exclusion", "unverifiable", "Upstream fit trial IDs were not supplied")
    elif not isinstance(fit, list) or any(not isinstance(x, str) or not x for x in fit):
        add("declared_preprocessing_test_exclusion", "block", "Malformed fit trial ID list")
    else:
        overlap = sorted(set(fit) & ids["test"])
        add("declared_preprocessing_test_exclusion", "block" if overlap else "allow",
            {"test_trial_overlap": overlap, "scope": "declared IDs only"})
    resources = evidence.get("predictor_calibration_trial_uids")
    if not isinstance(resources, dict) or not {"prior_personalized", "combined_personalized"}.issubset(resources):
        add("matched_predictor_calibration", "unverifiable", "Both predictor resource lists are required")
    else:
        lists = [resources[k] for k in ("prior_personalized", "combined_personalized")]
        valid = all(isinstance(x, list) and all(isinstance(v, str) and v for v in x) and len(x) == len(set(x)) for x in lists)
        matched = valid and set(lists[0]) == set(lists[1]) == ids["calibration"]
        add("matched_predictor_calibration", "allow" if matched else "block",
            "Both declared calibration sets must exactly match the compiler's allowed complete trials")
    prior_resources = evidence.get("prior_resources")
    if prior_resources is None:
        add("prior_resource_declaration", "unverifiable", "Normative or stimulus-label resources not declared")
    elif not isinstance(prior_resources, list):
        add("prior_resource_declaration", "block", "Prior resources must be a list")
    elif any(not isinstance(x, str) or not x for x in prior_resources):
        add("prior_resource_declaration", "block", "Prior resource names must be nonempty strings")
    else:
        unknown = set(prior_resources) - {"population_training_labels", "released_normative_prior", "participant_calibration_labels"}
        add("prior_resource_declaration", "block" if unknown else "allow",
            {"unknown_resources": sorted(unknown), "scope": "declaration only, not source authenticity"})
        if actual_prior_resources is not None:
            add("prior_resources_match_evaluator", "allow" if set(prior_resources) == set(actual_prior_resources) else "block",
                {"declared": prior_resources, "evaluator_resources": actual_prior_resources})
    add("upstream_authenticity_and_pretraining_exposure", "unverifiable",
        "Arrays, IDs and hashes cannot prove raw-trial identity, absence of aliases, or complete pretraining provenance")
    status = "block" if any(x["status"] == "block" for x in checks) else "unverifiable"
    return {"status": status, "checks": checks,
            "boundary": "allow means the named observable check passed; it is not a global leakage-free certificate"}
