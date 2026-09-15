#!/usr/bin/env python3
"""Validate and hash-lock one human coder's literature audit files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


PAPER_FIELDS = (
    "paper_id", "coder_id", "document_version", "datasets_evaluated",
    "prediction_target", "claim_type", "no_eeg_comparator", "overall_parity",
    "confidence", "evidence_locator", "notes",
)
EVALUATION_FIELDS = (
    "paper_id", "coder_id", "dataset", "protocol_id", "target_label_source",
    "prediction_unit", "participant_isolation", "stimulus_isolation",
    "window_trial_leakage", "eeg_label_resources", "baseline_label_resources",
    "participant_calibration", "stimulus_label_count", "population_label_count",
    "test_label_model_selection", "comparator_predictions", "metric_rows",
    "overall_parity", "evidence_locator", "notes",
)
ALLOWED = {
    "claim_type": {
        "eeg_prediction_only", "eeg_incremental", "multimodal_incremental",
        "method_comparison", "unclear",
    },
    "no_eeg_comparator": {"present", "absent", "unclear"},
    "overall_parity": {
        "matched", "eeg_favored", "baseline_favored", "mixed",
        "no_comparator", "unclear",
    },
    "confidence": {"high", "medium", "low"},
    "target_label_source": {
        "recorded_participant_self_report", "external_stimulus_norm",
        "intended_stimulus_category", "hybrid", "unclear",
    },
    "participant_isolation": {"held_out", "overlap", "unclear"},
    "stimulus_isolation": {"held_out", "overlap", "unclear"},
    "window_trial_leakage": {"yes", "no", "unclear", "not_applicable"},
    "test_label_model_selection": {"yes", "no", "unclear"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(table: pd.DataFrame, fields: tuple[str, ...], name: str) -> None:
    missing = set(fields) - set(table.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    blanks = [field for field in fields if table[field].astype(str).str.strip().eq("").any()]
    if blanks:
        raise ValueError(f"{name} contains blank required fields: {blanks}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("paper_codes", type=Path)
    parser.add_argument("evaluation_codes", type=Path)
    parser.add_argument("lock_file", type=Path)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, keep_default_na=False)
    paper = pd.read_csv(args.paper_codes, keep_default_na=False)
    evaluation = pd.read_csv(args.evaluation_codes, keep_default_na=False)
    require_columns(paper, PAPER_FIELDS, "paper codes")
    require_columns(evaluation, EVALUATION_FIELDS, "evaluation codes")
    expected = set(cohort.paper_id)
    observed = set(paper.paper_id)
    if paper.paper_id.duplicated().any() or observed != expected or len(paper) != len(cohort):
        raise ValueError("Paper codes must contain every cohort paper exactly once")
    if not expected.issubset(set(evaluation.paper_id)):
        raise ValueError("Every cohort paper needs at least one evaluation row")
    if evaluation.duplicated(["paper_id", "dataset", "protocol_id"]).any():
        raise ValueError("Duplicate paper/dataset/protocol evaluation key")
    coder_ids = set(paper.coder_id) | set(evaluation.coder_id)
    if len(coder_ids) != 1 or next(iter(coder_ids)) not in {"A", "B"}:
        raise ValueError(f"Expected one coder ID A or B, got {sorted(coder_ids)}")
    for field, allowed in ALLOWED.items():
        for table, name in ((paper, "paper"), (evaluation, "evaluation")):
            if field not in table:
                continue
            invalid = set(table[field]) - allowed
            if invalid:
                raise ValueError(f"Invalid {name} {field}: {sorted(invalid)}")
    expected_versions = dict(zip(cohort.paper_id, "pdf_sha256:" + cohort.pdf_sha256))
    for row in paper.itertuples(index=False):
        if row.document_version != expected_versions[row.paper_id]:
            raise ValueError(f"Document hash mismatch for {row.paper_id}")

    result = {
        "status": "locked",
        "coder_id": next(iter(coder_ids)),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": sha256_file(args.cohort),
        "paper_codes_sha256": sha256_file(args.paper_codes),
        "evaluation_codes_sha256": sha256_file(args.evaluation_codes),
        "paper_count": len(paper),
        "evaluation_count": len(evaluation),
    }
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    args.lock_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
