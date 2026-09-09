"""Build bidirectional EmoEEG-MC context-transfer splits with unseen subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openaffect_eeg.splits import (
    build_context_transfer_split,
    load_trial_manifests,
    split_audit,
)

DEFAULT_SEEDS = [11, 23, 47, 71, 101]
DIRECTIONS = (
    ("video_to_imagery", "video_viewing", "guided_imagery"),
    ("imagery_to_video", "guided_imagery", "video_viewing"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
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
    table = load_trial_manifests([args.manifest])
    eligible = table["label_available"].fillna(False).astype(bool)
    if "feature_alignment_status" in table:
        eligible &= table["feature_alignment_status"].eq("verified")
    table = table.loc[eligible].copy()
    if table["dataset_id"].nunique() != 1 or table["dataset_id"].iloc[0] != "ds005540":
        raise ValueError("Context-transfer splits require an EmoEEG-MC manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_hash = sha256(args.manifest)
    summary: dict[str, object] = {
        "dataset_id": "ds005540",
        "eligible_trial_count": len(table),
        "eligible_subject_count": int(table["subject_uid"].nunique()),
        "input_sha256": input_hash,
        "directions": {},
    }
    direction_summary: dict[str, list[dict[str, object]]] = {}
    for name, source_context, target_context in DIRECTIONS:
        output_dir = args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        direction_summary[name] = []
        for seed in args.seeds:
            assignments = build_context_transfer_split(
                table,
                subject_column="subject_uid",
                context_column="context",
                source_context=source_context,
                target_context=target_context,
                seed=seed,
            )
            assignments["protocol"] = name
            assignment_path = output_dir / f"seed-{seed:03d}.tsv.gz"
            assignments.to_csv(
                assignment_path,
                sep="\t",
                index=False,
                compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
            )
            audit = split_audit(
                table,
                assignments,
                isolation_columns=["subject_uid"],
            )
            audit.update(
                {
                    "protocol": name,
                    "source_context": source_context,
                    "target_context": target_context,
                    "seed": seed,
                    "input_sha256": input_hash,
                }
            )
            audit_path = output_dir / f"seed-{seed:03d}.audit.json"
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            direction_summary[name].append(
                {"seed": seed, "split_counts": audit["split_counts"]}
            )
    summary["directions"] = direction_summary
    summary_path = args.output_dir / "context_transfer_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
