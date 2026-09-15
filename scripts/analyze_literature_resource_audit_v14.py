#!/usr/bin/env python3
"""Compare two locked human coders and create agreement/adjudication outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from openaffect_eeg.literature_audit import nominal_agreement


PAPER_COMPARE_FIELDS = (
    "claim_type", "no_eeg_comparator", "overall_parity", "confidence"
)
EVALUATION_COMPARE_FIELDS = (
    "target_label_source", "participant_isolation", "stimulus_isolation",
    "window_trial_leakage", "test_label_model_selection", "overall_parity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lock(lock_path: Path, paper_path: Path, evaluation_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked":
        raise ValueError(f"Invalid lock status in {lock_path}")
    if lock.get("paper_codes_sha256") != sha256_file(paper_path):
        raise ValueError(f"Paper codes changed after lock: {paper_path}")
    if lock.get("evaluation_codes_sha256") != sha256_file(evaluation_path):
        raise ValueError(f"Evaluation codes changed after lock: {evaluation_path}")
    return lock


def compare(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], fields: tuple[str, ...], unit: str):
    merged = left.merge(right, on=keys, how="outer", suffixes=("_A", "_B"), indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError(f"Coder evaluation keys do not align for {unit}")
    agreement, disagreements = [], []
    for field in fields:
        a, b = merged[f"{field}_A"].astype(str), merged[f"{field}_B"].astype(str)
        agreement.append({"unit": unit, "field": field, **nominal_agreement(a, b)})
        for _, row in merged.loc[a.ne(b)].iterrows():
            disagreements.append({
                "unit": unit,
                "unit_key": "|".join(str(row[key]) for key in keys),
                "field": field,
                "coder_A": row[f"{field}_A"],
                "coder_B": row[f"{field}_B"],
                "final_value": "",
                "adjudicator": "",
                "evidence_locator": "",
                "reason": "",
            })
    return agreement, disagreements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_A", type=Path)
    parser.add_argument("evaluations_A", type=Path)
    parser.add_argument("lock_A", type=Path)
    parser.add_argument("paper_B", type=Path)
    parser.add_argument("evaluations_B", type=Path)
    parser.add_argument("lock_B", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lock_a = verify_lock(args.lock_A, args.paper_A, args.evaluations_A)
    lock_b = verify_lock(args.lock_B, args.paper_B, args.evaluations_B)
    if lock_a["coder_id"] != "A" or lock_b["coder_id"] != "B":
        raise ValueError("Locks must be supplied in A then B order")
    if lock_a["cohort_sha256"] != lock_b["cohort_sha256"]:
        raise ValueError("Coders did not use the same cohort")

    paper_a = pd.read_csv(args.paper_A, keep_default_na=False)
    paper_b = pd.read_csv(args.paper_B, keep_default_na=False)
    eval_a = pd.read_csv(args.evaluations_A, keep_default_na=False)
    eval_b = pd.read_csv(args.evaluations_B, keep_default_na=False)
    agreement_p, disagreement_p = compare(
        paper_a, paper_b, ["paper_id"], PAPER_COMPARE_FIELDS, "paper"
    )
    agreement_e, disagreement_e = compare(
        eval_a, eval_b, ["paper_id", "dataset", "protocol_id"],
        EVALUATION_COMPARE_FIELDS, "paper_dataset_protocol"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    agreement = pd.DataFrame(agreement_p + agreement_e)
    disagreements = pd.DataFrame(disagreement_p + disagreement_e)
    agreement.to_csv(args.output / "pre_adjudication_agreement.csv", index=False)
    disagreements.to_csv(args.output / "adjudication_template.csv", index=False)
    summary = {
        "status": "agreement_complete_adjudication_pending" if len(disagreements) else "complete_no_disagreements",
        "paper_count": len(paper_a),
        "evaluation_count": len(eval_a),
        "disagreement_count": len(disagreements),
        "agreement_sha256": sha256_file(args.output / "pre_adjudication_agreement.csv"),
        "adjudication_template_sha256": sha256_file(args.output / "adjudication_template.csv"),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
