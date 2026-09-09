"""Build source-safe aggregate evidence for the independent MusicEEG audits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class MusicAggregateError(ValueError):
    """Raised when a private result artifact lacks the required aggregate fields."""


def _mapping(source: object, name: str) -> Mapping[str, Any]:
    if not isinstance(source, Mapping):
        raise MusicAggregateError(f"{name} must be a JSON object")
    return source


def _required(source: Mapping[str, Any], name: str) -> Any:
    if name not in source:
        raise MusicAggregateError(f"Result artifact is missing {name!r}")
    return source[name]


def _safe_probe_seed_metrics(probes: object) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for label, rows in _mapping(probes, "identity_probe_results").items():
        if not isinstance(rows, list):
            raise MusicAggregateError("Identity probe rows must be a list")
        safe_rows = []
        for row in rows:
            item = _mapping(row, "identity probe row")
            safe = {
                key: item[key]
                for key in (
                    "seed",
                    "balanced_accuracy",
                    "permuted_balanced_accuracy",
                    "uniform_chance_accuracy",
                    "fit_class_count",
                    "fit_trial_count",
                    "test_trial_count",
                    "covered_test_trial_count",
                    "seen_test_fraction",
                )
                if key in item
            }
            if "seed" not in safe or "balanced_accuracy" not in safe:
                raise MusicAggregateError("Identity probe rows need seed and balanced accuracy")
            safe_rows.append(safe)
        result[str(label)] = safe_rows
    return result


def _safe_foundation_probe_seed_metrics(
    rows: object,
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(rows, list):
        raise MusicAggregateError("Foundation subject probe rows must be a list")
    result: dict[str, list[dict[str, object]]] = {"pretrained": [], "random": []}
    for row in rows:
        item = _mapping(row, "foundation subject probe row")
        if "seed" not in item:
            raise MusicAggregateError("Foundation subject probe rows need a seed")
        for representation, metrics in result.items():
            values = _mapping(item.get(representation), representation)
            safe = {
                key: values[key]
                for key in (
                    "balanced_accuracy",
                    "permuted_balanced_accuracy",
                    "uniform_chance_accuracy",
                    "fit_participant_count",
                    "test_trial_count",
                )
                if key in values
            }
            if "balanced_accuracy" not in safe:
                raise MusicAggregateError(
                    "Foundation subject probes need balanced accuracy"
                )
            metrics.append({"seed": item["seed"], **safe})
    return result


def build_public_music_aggregate(
    protocol_audit: Mapping[str, Any],
    foundation_audit: Mapping[str, Any],
    self_report_diagnostics: Mapping[str, Any],
) -> dict[str, object]:
    """Select only publication-safe summary fields from private result JSON."""
    protocol = _mapping(protocol_audit, "protocol audit")
    foundation = _mapping(foundation_audit, "foundation audit")
    diagnostics = _mapping(self_report_diagnostics, "self-report diagnostics")
    dataset_ids = {
        _required(protocol, "dataset_id"),
        _required(foundation, "dataset_id"),
        _required(diagnostics, "dataset_id"),
    }
    if dataset_ids != {"ds002721"}:
        raise MusicAggregateError("All public MusicEEG evidence must describe ds002721")
    return {
        "schema_version": 1,
        "dataset_id": "ds002721",
        "available_labeled_trial_count": _required(
            protocol, "available_labeled_trial_count"
        ),
        "input_sha256": {
            "trial_table": _required(protocol, "input_sha256")["trial_table"],
            "bandpower_features": _required(protocol, "input_sha256")["features"],
            "labram_trial_table": _required(foundation, "trial_table_sha256"),
            "labram_pretrained_features": _required(
                foundation, "pretrained_feature_sha256"
            ),
            "labram_random_features": _required(
                foundation, "random_feature_sha256"
            ),
        },
        "transparent_bandpower": {
            "protocol_summary": _required(protocol, "protocol_summary"),
            "primary_paired_comparison": _required(
                protocol, "primary_paired_comparison"
            ),
            "identity_probe_summary": _required(
                protocol, "identity_probe_summary"
            ),
            "identity_probe_seed_metrics": _safe_probe_seed_metrics(
                _required(protocol, "identity_probe_results")
            ),
        },
        "frozen_labram": {
            "protocol_summary": _required(foundation, "protocol_summary"),
            "subject_probe_summary": _required(foundation, "subject_probe_summary"),
            "subject_probe_seed_metrics": _safe_foundation_probe_seed_metrics(
                _required(foundation, "subject_probe_results")
            ),
        },
        "self_report_diagnostics": {
            "results": _required(diagnostics, "results"),
            "interpretation_boundary": _required(
                diagnostics, "interpretation_boundary"
            ),
        },
        "release_boundary": (
            "Aggregate metrics, seed labels, input hashes, and source-safe counts only. "
            "No raw EEG, feature arrays, stimuli, trial identifiers, participant "
            "identifiers, trial predictions, checkpoints, or local paths are included."
        ),
    }
