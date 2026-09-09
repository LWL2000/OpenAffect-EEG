"""Render both reading languages from identical verified aggregate values."""
import argparse
import json
from pathlib import Path

import pandas as pd

from openaffect_eeg.artifacts import sha256_file


LABELS = {"de_ridge_base": "Band-power R1", "de_ridge_strong": "Band-power R2",
          "bandpower_ridge_base": "Band-power R1", "bandpower_ridge_strong": "Band-power R2",
          "labram_ridge_base": "LaBraM R1", "labram_ridge_strong": "LaBraM R2",
          "eegnet_lr1e3": "EEGNet L1", "eegnet_lr1e4": "EEGNet L2"}
ORDER = ["Band-power R1", "Band-power R2", "LaBraM R1", "LaBraM R2", "EEGNet L1", "EEGNet L2"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    qa = json.loads((args.root / "verification/verification.json").read_text())
    tables = {}
    for method in ("percentile", "normal_t"):
        folder = args.root / method
        meta = json.loads((folder / "added_value_statistics.json").read_text())
        for filename, digest in meta["output_sha256"].items():
            if sha256_file(folder / filename) != digest:
                raise ValueError("Statistical table hash mismatch")
        for dataset in meta["datasets"]:
            expected = qa["datasets"][dataset["dataset_id"]]
            if expected["status"] != "passed" or dataset["input_sha256"] != expected["predictions_sha256"]:
                raise ValueError("Statistics and verified predictions do not match")
        tables[method] = pd.read_csv(folder / "table_added_value_summary.csv")
    key = ["dataset_id", "representation", "contrast"]
    joined = tables["percentile"].merge(tables["normal_t"], on=key, validate="1:1", suffixes=("_percentile", "_normal_t"))
    if ((joined.estimate_percentile - joined.estimate_normal_t).abs() > 1e-12).any():
        raise ValueError("Interval methods changed point estimates")
    selected = joined.loc[joined.contrast.isin(["matched_eeg_added_uniform_mean_ccc", "matched_eeg_added_uniform_mean_mae"])].copy()
    if len(selected) != 24:
        raise ValueError("Expected all two datasets, six settings, and two metrics")
    selected["configuration"] = selected.representation.map(LABELS)
    selected["metric"] = selected.contrast.map({"matched_eeg_added_uniform_mean_ccc": "CCC", "matched_eeg_added_uniform_mean_mae": "MAE"})
    selected.to_csv(args.root / "table_primary_robustness.csv", index=False)
    for language in ("en", "zh"):
        parts = []
        for dataset, name in (("ds005540", "EmoEEG-MC"), ("ds006850", "Urban Appraisal")):
            for metric in ("CCC", "MAE"):
                current = selected.loc[selected.dataset_id.eq(dataset) & selected.metric.eq(metric)].set_index("configuration").loc[ORDER]
                parts.append(f"### {name}: EC-PC {metric}\n")
                parts.append("| Configuration | Mean difference | Percentile 95% | t-normal 95% |" if language == "en" else
                             "| 配置 | 平均差值 | Percentile 95% | t-normal 95% |")
                parts.append("| --- | ---: | --- | --- |")
                for label, row in current.iterrows():
                    parts.append(f"| {label} | {row.estimate_percentile:+.4f} | [{row.ci_low_percentile:+.4f}, {row.ci_high_percentile:+.4f}] | [{row.ci_low_normal_t:+.4f}, {row.ci_high_normal_t:+.4f}] |")
                lo = int(current.valid_bootstrap_percentile.min())
                hi = int(current.valid_bootstrap_percentile.max())
                parts.append(f"\nFinite bootstrap draws: {lo} to {hi} of 2,000.\n" if language == "en" else
                             f"\n有效 bootstrap 次数：2,000 次中的 {lo} 至 {hi} 次。\n")
        (args.root / f"tables_{language}.md").write_text("\n".join(parts), encoding="utf-8")
    print(selected[["dataset_id", "configuration", "metric", "estimate_percentile", "ci_low_percentile", "ci_high_percentile", "ci_low_normal_t", "ci_high_normal_t"]].to_string(index=False))


if __name__ == "__main__":
    main()
