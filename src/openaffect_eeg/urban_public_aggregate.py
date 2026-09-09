"""Publication-safe aggregate for the ds006850 confirmation experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class UrbanAggregateError(ValueError):
    """Raised when the validated private artifact is incomplete."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UrbanAggregateError(f"{name} must be a JSON object")
    return value


def _required(value: Mapping[str, Any], name: str) -> Any:
    if name not in value:
        raise UrbanAggregateError(f"Validated artifact is missing {name!r}")
    return value[name]


def build_public_urban_aggregate(validation: Mapping[str, Any]) -> dict[str, object]:
    """Select only counts, aggregate estimates, and provenance hashes."""
    source = _mapping(validation, "validation")
    if _required(source, "status") != (
        "validated_artifact_chain_no_claim_of_positive_eeg_added_value"
    ):
        raise UrbanAggregateError("Private artifact chain has not passed validation")
    statistics = _mapping(_required(source, "statistics"), "statistics")
    response = _mapping(_required(source, "response"), "response")
    return {
        "schema_version": 1,
        "dataset": _required(source, "dataset"),
        "eligible_counts": {
            "participants": _required(source, "eligible_participants"),
            "stimuli": _required(source, "eligible_stimuli"),
            "trials": _required(source, "eligible_trials"),
        },
        "source_qc": _required(source, "source_qc"),
        "design": _required(source, "design"),
        "features": _required(source, "features"),
        "aggregate_input_sha256": {
            "full_source_audit": _required(source, "full_source_audit_sha256"),
            "eligible_input_audit": _required(source, "eligible_input_audit_sha256"),
            "eligible_manifest": _required(source, "manifest_sha256"),
            "fitted_response": _required(response, "response_sha256"),
            "private_predictions": _required(response, "predictions_sha256"),
        },
        "primary_endpoint": _required(statistics, "primary_endpoint"),
        "interpretation_boundary": (
            "The primary endpoint is the resource-matched EC-PC CCC difference. "
            "Intervals condition on fitted predictions and do not establish equivalence, "
            "causal mechanisms, preprocessing optimality, or absence of EEG information."
        ),
        "release_boundary": (
            "Aggregate counts, estimates, intervals, and hashes only. No raw EEG, feature "
            "arrays, trial identifiers, participant identifiers, trial predictions, model "
            "weights, or stimulus media are included."
        ),
    }
