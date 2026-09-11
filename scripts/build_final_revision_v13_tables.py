"""Render the V13 primary block-t table and topology-validation table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


LABELS = {
    "de_ridge_base": "Band-power R1",
    "de_ridge_strong": "Band-power R2",
    "bandpower_ridge_base": "Band-power R1",
    "bandpower_ridge_strong": "Band-power R2",
    "labram_ridge_base": "LaBraM R1",
    "labram_ridge_strong": "LaBraM R2",
    "eegnet_lr1e3": "EEGNet L1",
    "eegnet_lr1e4": "EEGNet L2",
}
ORDER = ["Band-power R1", "Band-power R2", "LaBraM R1", "LaBraM R2", "EEGNet L1", "EEGNet L2"]


def signed(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "paper/generated/final_revision_v13"
    output.mkdir(exist_ok=True)

    effects = pd.read_csv(output / "block_t/table_added_value_summary.csv")
    effects = effects.loc[effects.contrast.eq("matched_eeg_added_uniform_mean_ccc")].copy()
    effects["setting"] = effects.representation.map(LABELS)
    if len(effects) != 12 or effects.setting.isna().any():
        raise ValueError("Expected two tasks by six named settings")
    if not ((effects.ci_low <= 0) & (effects.ci_high >= 0)).all():
        raise ValueError("A primary interval excludes zero; prose requires revision")
    effects.to_csv(output / "primary_block_t.csv", index=False)

    by_task = {
        task: effects.loc[effects.dataset_id.eq(dataset)].set_index("setting").loc[ORDER]
        for dataset, task in (("ds005540", "EmoEEG-MC"), ("ds006850", "Urban"))
    }
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Primary uniform-grid mean matched EEG increment in CCC, with two-sided 95\% five-block $t$ intervals (df=4). Each task averages five fixed identity blocks and all 25 dose cells. Settings are sensitivity checks, not independent replications.}",
        r"\label{tab:primary-v13}", r"\begin{tabular}{@{}lll@{}}", r"\toprule",
        r"Setting & EmoEEG-MC: mean [interval] & Urban: mean [interval] \\", r"\midrule",
    ]
    for setting in ORDER:
        cells = []
        for task in ("EmoEEG-MC", "Urban"):
            row = by_task[task].loc[setting]
            cells.append(f"{signed(row.estimate)} [{signed(row.ci_low)}, {signed(row.ci_high)}]")
        lines.append(f"{setting} & {cells[0]} & {cells[1]} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "primary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    zh = ["| 设置 | EmoEEG-MC：均值 [区间] | Urban：均值 [区间] |", "| --- | --- | --- |"]
    for setting in ORDER:
        cells = []
        for task in ("EmoEEG-MC", "Urban"):
            row = by_task[task].loc[setting]
            cells.append(f"{signed(row.estimate)} [{signed(row.ci_low)}, {signed(row.ci_high)}]")
        zh.append(f"| {setting} | {cells[0]} | {cells[1]} |")
    (output / "primary_zh.md").write_text("\n".join(zh) + "\n", encoding="utf-8")

    old = pd.read_csv(root / "results/block_validation_v12/summary.csv")
    block = pd.read_csv(root / "results/block_t_validation_v13/summary.csv")
    summary = pd.concat([old, block], ignore_index=True)
    uniform = summary.loc[summary.estimand.eq("uniform_grid_mean")].copy()
    uniform.to_csv(output / "five_block_uniform_coverage.csv", index=False)
    lines = [
        r"\begin{table}[ht]\centering\small",
        r"\caption{Coverage of the primary uniform-grid mean under the observed five-block topology (300 replicates per row). Brackets give Wilson Monte Carlo intervals for the primary block-$t$ coverage.}",
        r"\begin{tabular}{llccc}\toprule",
        r"Task & Scenario & Block-$t$ [MC interval] & Percentile & Basic\\\midrule",
    ]
    for dataset, task in (("ds005540", "Emo"), ("ds006850", "Urban")):
        for scenario, label in (
            ("trial_signal", "Trial signal"),
            ("participant_constant", "Participant constant"),
            ("independent_noise", "Independent noise"),
            ("labels_only", "Labels only"),
        ):
            group = uniform.loc[
                uniform.dataset_id.eq(dataset) & uniform.scenario.eq(scenario)
            ].set_index("method")
            primary = group.loc["block_t"]
            lines.append(
                f"{task} & {label} & {primary.coverage:.3f} "
                f"[{primary.coverage_mc_low:.3f}, {primary.coverage_mc_high:.3f}] & "
                f"{group.loc['percentile'].coverage:.3f} & {group.loc['basic'].coverage:.3f}\\\\"
            )
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (output / "five_block_coverage.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
