"""Build source-linked summaries used by the final evaluation-method audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from openaffect_eeg.artifacts import sha256_file


class ClosureEvidenceError(RuntimeError):
    """Raised when a frozen source no longer supports the reported summary."""


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClosureEvidenceError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _float_range(rows: list[dict[str, str]], field: str) -> list[float]:
    values = [float(row[field]) for row in rows]
    if not values:
        raise ClosureEvidenceError(f"No values available for {field}")
    return [min(values), max(values)]


def build_submission_closure(project_root: str | Path, output: str | Path) -> dict:
    """Generate compact summaries from frozen simulation and external-audit rows."""

    root = Path(project_root).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = root / output

    oracle_dir = root / "results" / "evaluation_validation_v6" / "oracle_full"
    pipeline_dir = root / "results" / "evaluation_validation_v6" / "pipeline_full"
    interval_dir = root / "results" / "interval_diagnostics_v7"
    external_dir = root / "paper" / "generated" / "final_closure_v9" / "external_audit"
    catalog_path = root / "configs" / "audit_rule_catalog.json"

    oracle_completion = _read_json(oracle_dir / "completion.json")
    pipeline_completion = _read_json(pipeline_dir / "completion.json")
    interval_completion = _read_json(interval_dir / "completion.json")
    oracle_replicates = oracle_dir / "replicate_results.csv"
    pipeline_replicates = pipeline_dir / "replicate_results.csv"
    interval_replicates = interval_dir / "replicates.csv"
    for completion, source, key in (
        (oracle_completion, oracle_replicates, "result_sha256"),
        (pipeline_completion, pipeline_replicates, "result_sha256"),
        (interval_completion, interval_replicates, "predictions_sha256"),
    ):
        observed = sha256_file(source)
        if completion.get(key) != observed:
            raise ClosureEvidenceError(
                f"Frozen hash mismatch for {source}: {observed} != {completion.get(key)}"
            )

    operating_rows = [
        row
        for row in _read_csv(oracle_dir / "operating_characteristics.csv")
        if row["contrast"] == "eeg_added_matched_calibration"
    ]
    by_scenario = {
        name: [row for row in operating_rows if row["scenario"] == name]
        for name in (
            "labels_only",
            "independent_noise",
            "trial_signal",
            "participant_constant",
        )
    }
    if any(len(rows) != 16 for rows in by_scenario.values()):
        raise ClosureEvidenceError("Expected 16 matched-contrast cells per v6 scenario")

    interval_rows = _read_csv(interval_dir / "summary.csv")
    target_rows = [
        row
        for row in interval_rows
        if row["participants"] == "32"
        and row["stimuli"] == "24"
        and row["density"] == "1.0"
        and row["noise_sd"] == "0.5"
        and row["scenario"] == "participant_constant"
        and row["participant_dose"] == "0"
        and row["stimulus_dose"] == "0"
    ]
    if {row["method"] for row in target_rows} != {"percentile", "basic", "normal_t"}:
        raise ClosureEvidenceError("The prespecified v7 target diagnostic is incomplete")
    target_by_method = {row["method"]: row for row in target_rows}
    small_cluster = [
        row
        for row in interval_rows
        if row["participants"] == "8" and row["stimuli"] == "6"
    ]

    validation_summary = {
        "schema_version": "1.0",
        "scope": (
            "Known-truth and production-pipeline stress tests of the fixed-fit "
            "resource-matched evaluator; not biological validation or retraining uncertainty."
        ),
        "v6": {
            "oracle_settings": int(oracle_completion["settings"]),
            "replicates_per_setting": int(
                oracle_completion["requested_replicates_per_setting"]
            ),
            "completed_replicates": int(oracle_completion["completed_replicates"]),
            "labels_only_max_abs_mean_delta": max(
                abs(float(row["mean_estimate"])) for row in by_scenario["labels_only"]
            ),
            "independent_noise_positive_detection_rate_range": _float_range(
                by_scenario["independent_noise"], "positive_detection_rate"
            ),
            "trial_signal_positive_detection_rate_range": _float_range(
                by_scenario["trial_signal"], "positive_detection_rate"
            ),
            "trial_signal_pointwise_coverage_range": _float_range(
                by_scenario["trial_signal"], "coverage"
            ),
            "participant_constant_min_pointwise_coverage": min(
                float(row["coverage"])
                for row in by_scenario["participant_constant"]
            ),
            "pipeline_completed_replicates": int(
                pipeline_completion["completed_replicates"]
            ),
            "pipeline_failed_replicates": int(pipeline_completion["failed_replicates"]),
            "interpretation": (
                "The evaluator detects the prespecified trial-varying signal and preserves "
                "exact null identities, but one finite-sample setting undercovers."
            ),
        },
        "v7": {
            "planned_replicates": 9600,
            "valid_replicates": sum(int(row["replicates"]) for row in interval_rows) // 12,
            "rejected_replicates": int(interval_completion["failures"]),
            "target_32_by_24_coverage": {
                method: float(target_by_method[method]["coverage"])
                for method in ("percentile", "basic", "normal_t")
            },
            "small_8_by_6_min_coverage": {
                method: min(
                    float(row["coverage"])
                    for row in small_cluster
                    if row["method"] == method
                )
                for method in ("percentile", "basic", "normal_t")
            },
            "interpretation": (
                "Fresh simulations improve the original target diagnostic, while small "
                "crossed supports retain material undercoverage for percentile/basic intervals."
            ),
        },
        "source_sha256": {
            "v6_oracle_replicates": sha256_file(oracle_replicates),
            "v6_pipeline_replicates": sha256_file(pipeline_replicates),
            "v7_replicates": sha256_file(interval_replicates),
        },
    }

    external_summary = _read_json(external_dir / "public_summary.json")
    audit_rows = _read_csv(external_dir / "audit_cases.csv")
    strict_rows = [
        row
        for row in audit_rows
        if row["source"] == "EEGain"
        and row["checker"] == "resource_contract"
        and row["origin"] == "native_function_on_three_marker_windows_per_real_trial"
    ]
    scoped_rows = [
        row
        for row in audit_rows
        if row["source"] == "EEGain"
        and row["checker"] == "resource_contract"
        and row["origin"]
        == "same_native_trace_without_unclaimed_validation_trial_holdout"
    ]
    if len(strict_rows) != 5 or len(scoped_rows) != 5:
        raise ClosureEvidenceError("Expected five paired EEGain claim-repair folds")
    if {row["observed_status"] for row in strict_rows} != {"block"}:
        raise ClosureEvidenceError("Strict EEGain cases no longer all block")
    if {row["observed_status"] for row in scoped_rows} != {"allow"}:
        raise ClosureEvidenceError("Scoped EEGain cases no longer all allow")

    claim_repair = {
        "schema_version": "1.0",
        "source": "EEGain",
        "source_commit": external_summary["sources"]["EEGain"]["commit"],
        "shared_execution_basis": (
            "The same pinned split function and eligible real-trial identity universe; "
            "three generated marker windows per trial are lineage markers, not new EEG."
        ),
        "strict_claim": {
            "requested": "held-out test participants plus train/validation trial isolation",
            "folds": len(strict_rows),
            "status": "block",
            "failed_check": "training_validation_trial_isolation",
            "cases": [row["case"] for row in strict_rows],
        },
        "scoped_claim": {
            "requested": "held-out test participants; no validation-trial isolation claim",
            "folds": len(scoped_rows),
            "status": "allow",
            "cases": [row["case"] for row in scoped_rows],
        },
        "repair_options": [
            "Report only the held-out test-participant claim supported by the trace.",
            "If validation-trial isolation is required, split complete trials before window generation.",
        ],
        "boundary": (
            "This is a source-linked contract demonstration, not a reproduction or "
            "invalidation of EEGain's reported emotion-classification scores."
        ),
        "source_sha256": {
            "audit_cases": sha256_file(external_dir / "audit_cases.csv"),
            "public_summary": sha256_file(external_dir / "public_summary.json"),
            "rule_catalog": sha256_file(catalog_path),
        },
    }

    validation_path = output / "evaluation_method_validation.json"
    repair_path = output / "external_claim_repair.json"
    _write_json(validation_path, validation_summary)
    _write_json(repair_path, claim_repair)
    manifest = {
        "schema_version": "1.0",
        "status": "generated_from_hash_verified_frozen_sources",
        "outputs": {
            validation_path.name: sha256_file(validation_path),
            repair_path.name: sha256_file(repair_path),
        },
        "boundaries": [
            "No model was retrained by this summary build.",
            "Simulation validates behavior only under its specified data-generating processes.",
            "The external case checks observable execution records, not source authenticity.",
            "No independent-human usability test is claimed.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
