"""Publish paired resource-matched contrasts without exporting individual predictions."""
import argparse
import json
from pathlib import Path

import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.exposure_statistics import analyze_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--interval-method", choices=["percentile", "basic", "normal_t", "block_t"], default="percentile")
    parser.add_argument(
        "--paired-model-contrast",
        action="append",
        nargs=3,
        metavar=("NAME", "RIGHT", "LEFT"),
        default=[],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    buckets = [[], [], []]
    reports = []
    for path in sorted(args.result_root.glob("ds*/identity_exposure_predictions.tsv.gz")):
        tables = analyze_predictions(
            pd.read_csv(path, sep="\t"),
            iterations=args.bootstrap,
            seed=args.seed,
            interval_method=args.interval_method,
            paired_models=[tuple(values) for values in args.paired_model_contrast],
        )
        for bucket, table in zip(buckets, tables[:3], strict=True):
            table.insert(0, "dataset_id", path.parent.name)
            bucket.append(table)
        reports.append(dict(dataset_id=path.parent.name, input_sha256=sha256_file(path), **tables[3]))
        print(path.parent.name, "paired contrasts complete", flush=True)
    if not reports:
        raise ValueError("No prediction files")
    outputs = {}
    for name, bucket in zip(("scores", "contrasts", "summary"), buckets, strict=True):
        dest = args.output / f"table_added_value_{name}.csv"
        pd.concat(bucket, ignore_index=True).to_csv(dest, index=False)
        outputs[dest.name] = sha256_file(dest)
    (args.output / "added_value_statistics.json").write_text(
        json.dumps(dict(datasets=reports, output_sha256=outputs), indent=2) + "\n")


if __name__ == "__main__":
    main()
