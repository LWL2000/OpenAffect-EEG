"""Sanitize aggregate SEED-V protocol-audit results for manuscript use."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


class SeedVCaseStudyError(ValueError):
    """Raised when an authorized SEED-V export cannot be safely summarized."""


_REQUIRED_FILES = (
    "main_results.csv",
    "statistics.json",
    "decision_gate.json",
    "probes.csv",
    "protocol_ladder.csv",
    "ablation.csv",
)
_RESTRICTED_SUFFIXES = {
    ".bdf",
    ".edf",
    ".fdt",
    ".mat",
    ".npy",
    ".npz",
    ".set",
    ".wav",
}
_RESULT_FIELDS = {
    "held_out_subject",
    "model",
    "n_trials",
    "protocol",
    "result_path",
    "seed",
    "trial_accuracy",
    "trial_macro_f1",
}
_PROTOCOLS = ("strict_loso", "joint_subject_stimulus_holdout")
_MODELS = ("raw_temporal", "sdsa_m3")
_STATISTIC_FIELDS = (
    "ci95_high",
    "ci95_low",
    "mean_difference",
    "metric",
    "n_boot",
    "n_subjects",
    "seed",
)
_PROBE_FIELDS = (
    "emotion_probe",
    "session_probe",
    "stimulus_probe",
    "subject_probe",
    "time_bin_probe",
)
_DECISION_FIELDS = (
    "checks",
    "joint_macro_f1_delta",
    "joint_n_subjects",
    "joint_positive",
    "passed",
    "probe_constraints_pass",
    "strict_improved_subjects",
    "strict_macro_f1_delta",
    "strict_n_subjects",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SeedVCaseStudyError(f"CSV has no header: {path.name}")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def _require_fields(rows: list[dict[str, str]], fields: set[str], name: str) -> None:
    if not rows:
        raise SeedVCaseStudyError(f"CSV has no rows: {name}")
    missing = sorted(fields.difference(rows[0]))
    if missing:
        raise SeedVCaseStudyError(f"{name} is missing column(s): {', '.join(missing)}")


def _as_float(value: str, *, field: str, source: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise SeedVCaseStudyError(
            f"{source} contains a non-numeric {field!r} value"
        ) from error


def _validate_export_directory(source: Path) -> dict[str, Path]:
    if not source.is_dir():
        raise SeedVCaseStudyError(f"SEED-V export directory does not exist: {source}")
    restricted = sorted(
        path.name for path in source.rglob("*") if path.is_file() and path.suffix in _RESTRICTED_SUFFIXES
    )
    if restricted:
        raise SeedVCaseStudyError(
            "SEED-V export contains restricted suffix member(s): " + ", ".join(restricted)
        )
    paths = {name: source / name for name in _REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SeedVCaseStudyError(
            "SEED-V export is missing required file(s): " + ", ".join(missing)
        )
    return paths


def _protocol_means(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float | int]]]:
    _require_fields(rows, _RESULT_FIELDS, "main_results.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        protocol = row["protocol"]
        model = row["model"]
        if protocol in _PROTOCOLS and model in _MODELS:
            grouped[(protocol, model)].append(row)

    expected = {(protocol, model) for protocol in _PROTOCOLS for model in _MODELS}
    absent = sorted(f"{protocol}/{model}" for protocol, model in expected.difference(grouped))
    if absent:
        raise SeedVCaseStudyError(
            "main_results.csv is missing required protocol/model cells: "
            + ", ".join(absent)
        )

    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for protocol in _PROTOCOLS:
        summary[protocol] = {}
        for model in _MODELS:
            cell = grouped[(protocol, model)]
            summary[protocol][model] = {
                "mean_trial_accuracy": sum(
                    _as_float(row["trial_accuracy"], field="trial_accuracy", source="main_results.csv")
                    for row in cell
                )
                / len(cell),
                "trial_macro_f1": sum(
                    _as_float(row["trial_macro_f1"], field="trial_macro_f1", source="main_results.csv")
                    for row in cell
                )
                / len(cell),
                "result_rows": len(cell),
                "participant_count": len(
                    {row["held_out_subject"] for row in cell if row["held_out_subject"]}
                ),
                "seed_count": len({row["seed"] for row in cell if row["seed"]}),
            }
    return summary


def _statistics(path: Path) -> dict[str, dict[str, float | int | str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SeedVCaseStudyError("statistics.json must contain an object")
    result: dict[str, dict[str, float | int | str]] = {}
    for name in ("strict", "joint"):
        block = raw.get(name)
        if not isinstance(block, Mapping):
            raise SeedVCaseStudyError(f"statistics.json is missing {name!r} statistics")
        result[name] = {
            key: block[key]
            for key in _STATISTIC_FIELDS
            if key in block and isinstance(block[key], (float, int, str))
        }
        if "mean_difference" not in result[name] or "n_subjects" not in result[name]:
            raise SeedVCaseStudyError(
                f"statistics.json {name!r} block is missing required summary fields"
            )
    return result


def _metric_rows(path: Path, key: str, fields: tuple[str, ...]) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    _require_fields(rows, {key}, path.name)
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        name = row[key]
        if not name:
            raise SeedVCaseStudyError(f"{path.name} contains an empty {key!r}")
        values = {
            field: _as_float(row[field], field=field, source=path.name)
            for field in fields
            if row.get(field)
        }
        result[name] = values
    return dict(sorted(result.items()))


def _decision_gate(path: Path) -> dict[str, bool | float | int | dict[str, bool]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or "passed" not in raw:
        raise SeedVCaseStudyError("decision_gate.json is missing the passed field")
    result: dict[str, bool | float | int | dict[str, bool]] = {}
    for key in _DECISION_FIELDS:
        value = raw.get(key)
        if isinstance(value, (bool, float, int)):
            result[key] = value
        elif key == "checks" and isinstance(value, Mapping):
            checks = {str(name): bool(flag) for name, flag in value.items()}
            result[key] = dict(sorted(checks.items()))
    return result


def _validate_hashes(code_hashes: Mapping[str, str]) -> dict[str, str]:
    result = {str(name): str(value).lower() for name, value in code_hashes.items()}
    invalid = sorted(name for name, value in result.items() if not _SHA256.fullmatch(value))
    if invalid:
        raise SeedVCaseStudyError("Invalid code SHA-256 value(s): " + ", ".join(invalid))
    return dict(sorted(result.items()))


def build_seedv_case_study(
    source: Path,
    output: Path,
    *,
    code_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Create a source-safe aggregate case-study artifact from authorized results."""

    source = source.resolve()
    paths = _validate_export_directory(source)
    means = _protocol_means(_read_csv(paths["main_results.csv"]))
    ladder = _metric_rows(paths["protocol_ladder.csv"], "policy", ("trial_macro_f1",))
    required_ladder = {"s0_strict", "p_full_paperstyle"}
    if not required_ladder.issubset(ladder):
        raise SeedVCaseStudyError("protocol_ladder.csv is missing strict or PaperStyle row")

    result = {
        "schema_version": 1,
        "case_id": "seedv_authorized_protocol_audit_v1",
        "data_boundary": {
            "source": "SEED-V",
            "access": "authorized source access; no recordings, features, stimuli, or trial predictions are redistributed",
            "role": "independent protocol-audit case study; not a fifth public OpenAffect-EEG source",
        },
        "protocol_means": means,
        "statistics": _statistics(paths["statistics.json"]),
        "decision_gate": _decision_gate(paths["decision_gate.json"]),
        "probes": _metric_rows(paths["probes.csv"], "model", _PROBE_FIELDS),
        "protocol_ladder": {
            name: ladder[name]["trial_macro_f1"] for name in sorted(required_ladder)
        },
        "ablation": _metric_rows(paths["ablation.csv"], "model", ("trial_macro_f1",)),
        "input_sha256": {name: _sha256(path) for name, path in sorted(paths.items())},
        "code_sha256": _validate_hashes(code_hashes or {}),
        "interpretation_boundary": (
            "The strict-to-PaperStyle contrast is a protocol sensitivity result, not a model-only gain; "
            "the SDSA gate did not pass and is reported as a negative control."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
