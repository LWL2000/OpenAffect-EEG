"""Derive the five-block t candidate from frozen normal-t validation rows.

Both candidates use the same crossed-bootstrap standard error.  The only
change is the prespecified critical value: ``normal_t`` uses the smallest
within-block cluster degrees of freedom, whereas ``block_t`` uses the five
independent identity blocks (df = 4).  This script makes that deterministic
transformation auditable without rerunning the 2,400 simulations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from scipy.stats import t

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.evaluation_validation import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()

    source = args.input / "replicates.csv"
    data = pd.read_csv(source)
    data = data.loc[data.method.eq("normal_t")].copy()
    if data.empty:
        raise ValueError("No normal_t rows found")
    if not (data.n_blocks == 5).all():
        raise ValueError("This derivation is restricted to five-block designs")

    old_radius = (data.ci_high - data.ci_low) / 2
    scale = t.ppf(0.975, data.n_blocks - 1) / t.ppf(0.975, data.cluster_df)
    new_radius = old_radius * scale
    data["method"] = "block_t"
    data["ci_low"] = data.estimate - new_radius
    data["ci_high"] = data.estimate + new_radius
    data["covered"] = (
        (data.ci_low - 1e-12 <= data.truth) &
        (data.truth <= data.ci_high + 1e-12)
    )
    data["positive"] = data.ci_low > 1e-12

    args.output.mkdir(parents=True, exist_ok=False)
    data.to_csv(args.output / "replicates.csv", index=False)
    summary = []
    keys = ["dataset_id", "scenario", "method", "estimand"]
    for key, group in data.groupby(keys):
        lo, hi = wilson_interval(int(group.covered.sum()), len(group))
        summary.append(dict(
            zip(keys, key), replicates=len(group), coverage=group.covered.mean(),
            coverage_mc_low=lo, coverage_mc_high=hi,
            bias=(group.estimate - group.truth).mean(),
            positive_rate=group.positive.mean(),
            mean_width=(group.ci_high - group.ci_low).mean(),
        ))
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    manifest = {
        "status": "executed_requires_interpretation",
        "source_replicates": str(source.as_posix()),
        "source_replicates_sha256": sha256_file(source),
        "replicates_sha256": sha256_file(args.output / "replicates.csv"),
        "transformation": (
            "Preserve the normal_t crossed-bootstrap standard error and replace "
            "its cluster critical value by t_(0.975, n_identity_blocks-1)."
        ),
        "boundary": (
            "Validates the fixed-fit uniform-grid mean on four simulated signal "
            "families with the observed five-block topology; cell intervals remain "
            "descriptive and retraining uncertainty is outside scope."
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "completion.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
