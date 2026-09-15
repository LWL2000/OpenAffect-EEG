#!/usr/bin/env python3
"""Summarize two locked AI literature codings without calling them human coders."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import beta

from openaffect_eeg.literature_audit import nominal_agreement


PAPER_FIELDS = ("claim_type", "no_eeg_comparator", "overall_parity", "confidence")
EVALUATION_FIELDS = (
    "target_label_source",
    "participant_isolation",
    "stimulus_isolation",
    "window_trial_leakage",
    "test_label_model_selection",
    "overall_parity",
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
        raise ValueError(f"Invalid lock status: {lock_path}")
    if lock.get("paper_codes_sha256") != sha256_file(paper_path):
        raise ValueError(f"Paper codes changed after lock: {paper_path}")
    if lock.get("evaluation_codes_sha256") != sha256_file(evaluation_path):
        raise ValueError(f"Evaluation codes changed after lock: {evaluation_path}")
    return lock


def exact_binomial_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


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
    if lock_a.get("coder_id") != "A" or lock_b.get("coder_id") != "B":
        raise ValueError("Locks must be supplied in A then B order")
    if lock_a.get("cohort_sha256") != lock_b.get("cohort_sha256"):
        raise ValueError("Coders did not use the same cohort")

    paper_a = pd.read_csv(args.paper_A, keep_default_na=False)
    paper_b = pd.read_csv(args.paper_B, keep_default_na=False)
    eval_a = pd.read_csv(args.evaluations_A, keep_default_na=False)
    eval_b = pd.read_csv(args.evaluations_B, keep_default_na=False)
    merged = paper_a.merge(paper_b, on="paper_id", how="outer", suffixes=("_A", "_B"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Paper IDs do not align")

    agreement_rows: list[dict] = []
    disagreement_rows: list[dict] = []
    for field in PAPER_FIELDS:
        result = nominal_agreement(merged[f"{field}_A"], merged[f"{field}_B"])
        agreement_rows.append({"field": field, **{key: json_safe(value) for key, value in result.items()}})
        different = merged[f"{field}_A"].ne(merged[f"{field}_B"])
        for _, row in merged.loc[different].iterrows():
            disagreement_rows.append(
                {
                    "paper_id": row.paper_id,
                    "field": field,
                    "coder_A": row[f"{field}_A"],
                    "coder_B": row[f"{field}_B"],
                }
            )

    distribution_rows: list[dict] = []
    for coder, frame in (("A", eval_a), ("B", eval_b)):
        for field in EVALUATION_FIELDS:
            for value, count in frame[field].value_counts(dropna=False).items():
                distribution_rows.append(
                    {
                        "coder": coder,
                        "field": field,
                        "value": value,
                        "count": int(count),
                        "total": len(frame),
                        "proportion": float(count / len(frame)),
                    }
                )

    counts_a = eval_a.groupby("paper_id").size()
    counts_b = eval_b.groupby("paper_id").size()
    unit_counts = pd.concat([counts_a.rename("coder_A_units"), counts_b.rename("coder_B_units")], axis=1).fillna(0)
    unit_counts = unit_counts.astype(int).reset_index()
    unit_counts["same_unit_count"] = unit_counts.coder_A_units.eq(unit_counts.coder_B_units)
    exact_keys_a = set(map(tuple, eval_a[["paper_id", "dataset", "protocol_id"]].to_numpy()))
    exact_keys_b = set(map(tuple, eval_b[["paper_id", "dataset", "protocol_id"]].to_numpy()))

    both_absent = merged.no_eeg_comparator_A.eq("absent") & merged.no_eeg_comparator_B.eq("absent")
    absent_count = int(both_absent.sum())
    interval = exact_binomial_interval(absent_count, len(merged))

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agreement_rows).to_csv(args.output / "paper_level_agreement.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(args.output / "paper_level_disagreements.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(args.output / "evaluation_field_distributions.csv", index=False)
    unit_counts.to_csv(args.output / "evaluation_unit_count_diagnostic.csv", index=False)

    summary = {
        "status": "dual_ai_exploratory_complete_protocol_alignment_pending",
        "audit_type": "two_independent_ai_coders",
        "human_coder_requirement_satisfied": False,
        "paper_count": len(merged),
        "coder_A_evaluation_count": len(eval_a),
        "coder_B_evaluation_count": len(eval_b),
        "both_coders_no_learned_no_eeg_comparator_count": absent_count,
        "both_coders_no_learned_no_eeg_comparator_fraction": absent_count / len(merged),
        "accessible_cohort_exact_95_interval": list(interval),
        "paper_level_disagreement_count": len(disagreement_rows),
        "papers_with_different_evaluation_unit_counts": int((~unit_counts.same_unit_count).sum()),
        "exact_evaluation_key_overlap": len(exact_keys_a & exact_keys_b),
        "interpretation_limits": [
            "The cohort was conditioned on validated public full-text access and is not an unbiased field-prevalence sample.",
            "AI coders do not satisfy the preregistered requirement for two independent human coders.",
            "Evaluation-level kappa is withheld until dataset/protocol units are harmonized after lock.",
            "A constant category makes Cohen's kappa undefined even when raw agreement is 100 percent.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# Dual-AI exploratory literature audit (v14)

This is a sensitivity analysis by two independent AI coders. It is not a
two-human-coder audit and does not satisfy that preregistered completion condition.

- Both coders classified **{absent_count}/{len(merged)}** accessible-cohort papers as having no eligible learned no-EEG comparator.
- Raw agreement for `no_eeg_comparator` was **100%**. Cohen's kappa is undefined because both coders used one constant category.
- The descriptive exact 95% interval within this access-conditioned cohort is **[{interval[0]:.3f}, {interval[1]:.3f}]**. It must not be presented as an unbiased field prevalence interval.
- Coder A created **{len(eval_a)}** evaluation units and coder B created **{len(eval_b)}**; **{int((~unit_counts.same_unit_count).sum())}** papers had different unit counts. Evaluation-level agreement awaits post-lock unit harmonization.

The locked source files remain unchanged. See the CSV outputs for paper-level
agreement, disagreements, evaluation-field distributions, and unit-count diagnostics.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
