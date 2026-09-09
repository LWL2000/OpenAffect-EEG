"""Build deterministic leakage-resistant splits before EEG windowing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openaffect_eeg.splits import (
    build_double_holdout_split,
    build_group_holdout_split,
    load_trial_manifests,
    split_audit,
)

DEFAULT_SEEDS = [11, 23, 47, 71, 101]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    table = load_trial_manifests(args.manifests)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        args.output_dir / "core_trials.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    input_hashes = {str(path.resolve()): sha256(path) for path in args.manifests}

    protocols: list[tuple[str, list[str]]] = [
        (
            "trial_random_holdout",
            ["trial_uid", "subject_uid", "stimulus_uid"],
        ),
        ("subject_holdout", ["subject_uid"]),
        ("stimulus_holdout", ["stimulus_uid"]),
        ("subject_stimulus_holdout", ["subject_uid", "stimulus_uid"]),
    ]
    if table["context"].nunique() >= 3:
        protocols.append(("context_holdout", ["context"]))
    if table["dataset_id"].nunique() >= 3:
        protocols.append(("dataset_holdout", ["dataset_id"]))

    summary: dict[str, object] = {
        "trial_count": len(table),
        "labeled_trial_count": int(table["label_available"].sum()),
        "dataset_count": int(table["dataset_id"].nunique()),
        "subject_count": int(table["subject_uid"].nunique()),
        "stimulus_count": int(table["stimulus_uid"].nunique()),
        "input_sha256": input_hashes,
        "seeds": args.seeds,
        "protocols": {},
    }
    protocol_summary: dict[str, list[dict[str, object]]] = {}
    for protocol, isolation_columns in protocols:
        protocol_dir = args.output_dir / protocol
        protocol_dir.mkdir(parents=True, exist_ok=True)
        protocol_summary[protocol] = []
        for seed in args.seeds:
            if protocol == "trial_random_holdout":
                assignments = build_group_holdout_split(
                    table,
                    "trial_uid",
                    seed=seed,
                    stratify_column="dataset_id",
                )
            elif protocol == "subject_stimulus_holdout":
                assignments = build_double_holdout_split(
                    table,
                    first_group="subject_uid",
                    second_group="stimulus_uid",
                    seed=seed,
                    validation_fraction=0.2,
                    stratify_column="dataset_id",
                )
            else:
                stratify_column = (
                    "dataset_id"
                    if protocol in {"subject_holdout", "stimulus_holdout"}
                    else None
                )
                assignments = build_group_holdout_split(
                    table,
                    isolation_columns[0],
                    seed=seed,
                    stratify_column=stratify_column,
                )
            assignments["protocol"] = protocol
            assignment_path = protocol_dir / f"seed-{seed:03d}.tsv.gz"
            audit_path = protocol_dir / f"seed-{seed:03d}.audit.json"
            assignments.to_csv(
                assignment_path,
                sep="\t",
                index=False,
                compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
            )
            audit = split_audit(
                table,
                assignments,
                isolation_columns=isolation_columns,
            )
            labeled = assignments.merge(
                table[["trial_uid", "label_available"]],
                on="trial_uid",
                validate="1:1",
            )
            audit.update(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "labeled_split_counts": {
                        str(split): int(count)
                        for split, count in labeled.loc[
                            labeled["label_available"], "split"
                        ]
                        .value_counts()
                        .items()
                    },
                    "input_sha256": input_hashes,
                }
            )
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            protocol_summary[protocol].append(
                {"seed": seed, "split_counts": audit["split_counts"]}
            )
    summary["protocols"] = protocol_summary
    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Unified trials: {args.output_dir / 'core_trials.tsv.gz'}")
    print(f"Split summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
