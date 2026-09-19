#!/usr/bin/env python3
"""Analyze matched-resource predictions across executed training seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from openaffect_eeg.training_uncertainty import analyze_training_uncertainty


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026091400)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
    parser.add_argument("--strict-margin", type=float, default=0.025)
    parser.add_argument(
        "--control",
        choices=(
            "observed",
            "label_permutation",
            "synthetic_signal",
            "synthetic_residual_signal",
            "synthetic_signed_power_residual",
        ),
        default="observed",
    )
    parser.add_argument(
        "--equivalence-scope",
        choices=("exploratory", "confirmatory"),
        default="exploratory",
    )
    args = parser.parse_args()
    if args.control != "observed" and args.equivalence_scope == "confirmatory":
        raise ValueError("Synthetic controls cannot receive a confirmatory equivalence label")

    predictions = pd.read_csv(args.predictions, sep="\t")
    cells, report = analyze_training_uncertainty(
        predictions,
        iterations=args.bootstrap,
        seed=args.seed,
        equivalence_margin=args.equivalence_margin,
        strict_margin=args.strict_margin,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "training_uncertainty_cells.csv"
    cells.to_csv(cells_path, index=False)
    report["input_predictions_sha256"] = sha256_file(args.predictions)
    report["cell_table_sha256"] = sha256_file(cells_path)
    report["equivalence_scope"] = args.equivalence_scope
    estimate = float(report["primary"]["estimate"])
    if args.control == "label_permutation":
        report["control_diagnostic"] = {
            "control": args.control,
            "warning_threshold": 0.10,
            "warning": bool(estimate > 0.10),
            "rule": "warn if the primary matched CCC increment exceeds +0.10",
        }
    elif args.control in {
        "synthetic_signal",
        "synthetic_residual_signal",
        "synthetic_signed_power_residual",
    }:
        report["control_diagnostic"] = {
            "control": args.control,
            "minimum_expected_increment": 0.10,
            "warning": bool(estimate < 0.10),
            "rule": "warn if the primary matched CCC increment is below +0.10",
        }
    else:
        report["control_diagnostic"] = {"control": "observed", "warning": False}
    (args.output / "training_uncertainty_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["primary"], sort_keys=True))


if __name__ == "__main__":
    main()
