"""Summarize prespecified alternative-block, training-seed, and fine-tuning checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CONTRAST = "matched_eeg_added_uniform_mean_ccc"
LABELS = {
    "de_ridge_base": "Band-power R1", "de_ridge_strong": "Band-power R2",
    "bandpower_ridge_base": "Band-power R1", "bandpower_ridge_strong": "Band-power R2",
    "labram_ridge_base": "LaBraM R1", "labram_ridge_strong": "LaBraM R2",
    "eegnet_lr1e3": "EEGNet L1", "eegnet_lr1e4": "EEGNet L2",
}


def read(folder: Path) -> pd.DataFrame:
    table = pd.read_csv(folder / "table_added_value_summary.csv")
    result = table.loc[table.contrast.eq(CONTRAST)].copy()
    if result.empty:
        raise ValueError(f"Primary contrast missing from {folder}")
    return result


def covers_zero(table: pd.DataFrame) -> pd.Series:
    return (table.ci_low <= 0) & (table.ci_high >= 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    primary_full = read(root / "primary_full")
    primary = read(root / "primary_keycorners")
    alternative = read(root / "altblocks")
    seed_tables = [(0, primary)] + [
        (offset, read(root / f"trainseed_{offset}")) for offset in (100000, 200000)
    ]
    finetune = read(root / "finetune")

    key = ["dataset_id", "representation"]
    joined = primary_full[key + ["estimate"]].merge(
        alternative[key + ["estimate", "ci_low", "ci_high"]], on=key,
        suffixes=("_primary", "_alternative"), validate="1:1",
    )
    joined["absolute_shift"] = (joined.estimate_alternative - joined.estimate_primary).abs()
    joined["covers_zero"] = covers_zero(
        joined.rename(columns={"ci_low": "ci_low", "ci_high": "ci_high"})
    )

    records: dict[str, object] = {"alternative_blocks": [], "training_seeds": [], "fine_tuning": []}
    for dataset, group in joined.groupby("dataset_id"):
        records["alternative_blocks"].append({
            "dataset_id": dataset,
            "largest_absolute_point_shift": float(group.absolute_shift.max()),
            "intervals_covering_zero": int(group.covers_zero.sum()),
            "settings": len(group),
        })
    seeds = []
    for offset, table in seed_tables:
        current = table.loc[table.representation.str.startswith("eegnet_")].copy()
        current["training_seed_offset"] = offset
        seeds.append(current)
    seeds = pd.concat(seeds, ignore_index=True)
    for (dataset, representation), group in seeds.groupby(key):
        if len(group) != 3:
            raise ValueError("Expected three EEGNet initialization schedules")
        records["training_seeds"].append({
            "dataset_id": dataset, "representation": representation,
            "label": LABELS[representation],
            "estimate_min": float(group.estimate.min()),
            "estimate_max": float(group.estimate.max()),
            "point_span": float(group.estimate.max() - group.estimate.min()),
            "intervals_covering_zero": int(covers_zero(group).sum()),
            "runs": 3,
        })
    for row in finetune.itertuples(index=False):
        records["fine_tuning"].append({
            "dataset_id": row.dataset_id, "representation": row.representation,
            "estimate": float(row.estimate), "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high), "covers_zero": bool(row.ci_low <= 0 <= row.ci_high),
        })
    (root / "sensitivity_summary.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )

    task = {"ds005540": "Emo", "ds006850": "Urban"}
    lines = [
        r"\begin{table}[ht]\centering\small",
        r"\caption{Post-primary robustness checks. Block-$t$ intervals remain conditional on each fitted run. Alternative partitions and initialization schedules are sensitivity analyses, not independent cohorts.}",
        r"\begin{tabularx}{\linewidth}{p{0.09\linewidth}p{0.22\linewidth}p{0.18\linewidth}X}\toprule",
        r"Task & Check & Setting & Result\\\midrule",
    ]
    for item in records["alternative_blocks"]:
        lines.append(
            f"{task[item['dataset_id']]} & New blocks + refit & Six settings & "
            f"max point shift {item['largest_absolute_point_shift']:.4f}; "
            f"{item['intervals_covering_zero']}/{item['settings']} cover 0\\\\"
        )
    for item in records["training_seeds"]:
        lines.append(
            f"{task[item['dataset_id']]} & Three train seeds & {item['label']} & "
            f"[{item['estimate_min']:+.4f}, {item['estimate_max']:+.4f}]; "
            f"{item['intervals_covering_zero']}/3 cover 0\\\\"
        )
    for item in records["fine_tuning"]:
        lines.append(
            f"{task[item['dataset_id']]} & Last-block fine-tune & LaBraM residual & "
            f"{item['estimate']:+.4f} [{item['ci_low']:+.4f}, {item['ci_high']:+.4f}]\\\\"
        )
    lines += [r"\bottomrule\end{tabularx}", r"\end{table}"]
    (root / "sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
