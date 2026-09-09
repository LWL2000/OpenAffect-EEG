"""Build split-size- and test-support-matched identity-overlap controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openaffect_eeg.baselines import harmonize_affect_targets
from openaffect_eeg.reviewer_controls import build_support_matched_overlap_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_table", type=Path)
    parser.add_argument("reference_split_root", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_trial_uids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "trial_uid" in archive.files:
            return archive["trial_uid"].astype(str)
        if "trial_uids" in archive.files:
            return archive["trial_uids"].astype(str)
    raise ValueError("Feature archive has no trial UID array")


def main() -> int:
    args = parse_args()
    trials = harmonize_affect_targets(pd.read_csv(args.trial_table, sep="\t"))
    trials = trials.loc[trials["dataset_id"].eq("ds005540")].copy()
    if "subject_uid" not in trials:
        trials["subject_uid"] = (
            trials["dataset_id"].astype(str) + ":" + trials["subject_id"].astype(str)
        )
    feature_trial_uids = set(load_feature_trial_uids(args.features))
    eligible = trials.loc[
        trials["target_available"] & trials["trial_uid"].isin(feature_trial_uids),
        ["trial_uid", "subject_uid", "stimulus_uid"],
    ].copy()

    reference_paths = sorted(
        (args.reference_split_root / "subject_stimulus_holdout").glob(
            "seed-*.tsv.gz"
        )
    )
    if not reference_paths:
        raise ValueError("Joint subject--stimulus assignments are required")

    protocol_root = args.output_root / "support_matched_identity_overlap"
    protocol_root.mkdir(parents=True, exist_ok=True)
    records = []
    for reference_path in reference_paths:
        reference = pd.read_csv(reference_path, sep="\t")
        seed = int(reference["seed"].iloc[0])
        assignment, audit = build_support_matched_overlap_split(
            eligible,
            reference,
            subject_column="subject_uid",
            stimulus_column="stimulus_uid",
            seed=seed,
        )
        assignment_path = protocol_root / f"seed-{seed:03d}.tsv.gz"
        audit_path = protocol_root / f"seed-{seed:03d}.audit.json"
        assignment.to_csv(assignment_path, sep="\t", index=False)
        audit.update(
            {
                "assignment": str(assignment_path.resolve()),
                "assignment_sha256": sha256(assignment_path),
                "reference_assignment": str(reference_path.resolve()),
                "reference_assignment_sha256": sha256(reference_path),
            }
        )
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append(audit)
        print(
            seed,
            audit["matched_counts"],
            "subject overlap",
            audit["subject_overlap_fraction"],
            "stimulus overlap",
            audit["stimulus_overlap_fraction"],
            flush=True,
        )

    manifest = {
        "analysis": "support_matched_identity_overlap_control",
        "dataset_id": "ds005540",
        "eligible_trial_count": len(eligible),
        "trial_table": str(args.trial_table.resolve()),
        "trial_table_sha256": sha256(args.trial_table),
        "features": str(args.features.resolve()),
        "features_sha256": sha256(args.features),
        "records": records,
    }
    manifest_path = args.output_root / "support_matched_control_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
