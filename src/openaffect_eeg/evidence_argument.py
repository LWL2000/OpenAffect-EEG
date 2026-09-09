"""Trace existing aggregate scores into a resource-matched evaluation argument.

No fitting, resampling, or new significance test is performed here. Intervals
are copied from the frozen paired contrasts, never subtracted endpointwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file


DATASETS = {"ds005540": "EmoEEG-MC", "ds006850": "Urban"}
SETTINGS = {
    "de_ridge_base": "Band-power R1", "de_ridge_strong": "Band-power R2",
    "bandpower_ridge_base": "Band-power R1", "bandpower_ridge_strong": "Band-power R2",
    "labram_ridge_base": "LaBraM R1", "labram_ridge_strong": "LaBraM R2",
    "eegnet_lr1e3": "EEGNet L1", "eegnet_lr1e4": "EEGNet L2",
}
ORDER = ["Band-power R1", "Band-power R2", "LaBraM R1", "LaBraM R2", "EEGNet L1", "EEGNet L2"]
KEY = ["dataset_id", "representation", "participant_dose", "stimulus_dose"]
CONTRASTS = {
    "total": "total_minus_uncalibrated_prior",
    "calibration": "prior_calibration_gain",
    "matched": "eeg_added_matched_calibration",
}


def load_evidence(root: Path):
    root = Path(root)
    qa_path = root / "verification/verification.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    inputs = {"verification/verification.json": sha256_file(qa_path)}
    frames = {}
    for method in ("percentile", "normal_t"):
        meta_path = root / method / "added_value_statistics.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        inputs[f"{method}/added_value_statistics.json"] = sha256_file(meta_path)
        for name, digest in meta["output_sha256"].items():
            path = root / method / name
            if sha256_file(path) != digest:
                raise ValueError(f"Frozen statistical table hash mismatch: {method}/{name}")
            inputs[f"{method}/{name}"] = digest
        for data in meta["datasets"]:
            verified = qa["datasets"][data["dataset_id"]]
            if verified["status"] != "passed" or verified["predictions_sha256"] != data["input_sha256"]:
                raise ValueError("Statistics must derive from verified predictions")
        frames[method] = {
            kind: pd.read_csv(root / method / f"table_added_value_{kind}.csv")
            for kind in ("scores", "contrasts", "summary")
        }
    cells = validate_cells(frames["percentile"]["scores"], frames["percentile"]["contrasts"])
    for name, contrast in CONTRASTS.items():
        current = frames["normal_t"]["contrasts"].query("contrast == @contrast").set_index(KEY)
        for suffix, source in (("t_low", "ccc_ci_low"), ("t_high", "ccc_ci_high")):
            cells[f"{name}_{suffix}"] = current[source].reindex(pd.MultiIndex.from_frame(cells[KEY])).to_numpy()
        if not np.allclose(current.ccc_delta.reindex(pd.MultiIndex.from_frame(cells[KEY])), cells[name], atol=1e-12, rtol=0):
            raise ValueError("Interval methods changed contrast point estimates")
    return cells, frames, inputs


def validate_cells(scores, contrasts):
    if scores.duplicated(KEY + ["predictor"]).any() or contrasts.duplicated(KEY + ["contrast"]).any():
        raise ValueError("Duplicate score or contrast key")
    wide = scores.pivot(index=KEY, columns="predictor", values="macro_ccc")
    required = {"prior", "prior_personalized", "combined_personalized"}
    if not required.issubset(wide) or not np.isfinite(wide[list(required)]).all().all():
        raise ValueError("Missing or nonfinite predictor scores")
    cells = wide.reset_index()
    if set(cells.dataset_id) != set(DATASETS) or len(cells) != 300:
        raise ValueError("Expected two datasets, six settings, and all 25 cells")
    grid = {(p, s) for p in (0, 1, 2, 4, 8) for s in (0, 1, 2, 4, 8)}
    for _, group in cells.groupby(["dataset_id", "representation"]):
        if set(zip(group.participant_dose, group.stimulus_dose)) != grid:
            raise ValueError("Incomplete dose grid")
    for _, group in cells.groupby(["dataset_id", "participant_dose", "stimulus_dose"]):
        for predictor in ("prior", "prior_personalized"):
            if np.ptp(group[predictor]) > 1e-12:
                raise ValueError("Prior predictions must yield identical scores across settings")
    for name, source in CONTRASTS.items():
        selected = contrasts.loc[contrasts.contrast.eq(source), KEY + ["ccc_delta", "ccc_ci_low", "ccc_ci_high", "valid_bootstrap"]]
        selected = selected.rename(columns={"ccc_delta": name, "ccc_ci_low": name + "_low", "ccc_ci_high": name + "_high", "valid_bootstrap": name + "_draws"})
        cells = cells.merge(selected, on=KEY, how="left", validate="1:1")
    values = cells[[name + suffix for name in CONTRASTS for suffix in ("", "_low", "_high", "_draws")]]
    if not np.isfinite(values).all().all():
        raise ValueError("Missing or nonfinite paired contrast")
    expected = {
        "total": cells.combined_personalized - cells.prior,
        "calibration": cells.prior_personalized - cells.prior,
        "matched": cells.combined_personalized - cells.prior_personalized,
    }
    for name, value in expected.items():
        if not np.allclose(cells[name], value, rtol=0, atol=1e-12):
            raise ValueError("Paired contrast does not match predictor scores")
        if (cells[name + "_low"] > cells[name + "_high"]).any():
            raise ValueError("Invalid interval ordering")
    cells["arithmetic_error"] = cells.total - cells.calibration - cells.matched
    if cells.arithmetic_error.abs().max() > 1e-12:
        raise ValueError("Score decomposition identity failed")
    cells["configuration"] = cells.representation.map(SETTINGS)
    if cells.configuration.isna().any():
        raise ValueError("Unknown representation setting")
    for _, group in cells.groupby("dataset_id"):
        if set(group.configuration) != set(ORDER):
            raise ValueError("Unexpected model-setting coverage")
    return cells


def _number(value):
    return f"{value:+.4f}" if abs(value) >= 0.00005 else "0.0000"


def _interval(row, prefix="matched", t=False):
    infix = "_t" if t else ""
    return f"[{_number(row[prefix + infix + '_low'])}, {_number(row[prefix + infix + '_high'])}]"


def build_argument(root: Path, output: Path):
    cells, frames, inputs = load_evidence(root)
    output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output / "decomposition_all_cells.csv", index=False)
    endpoint = cells.loc[cells.participant_dose.eq(8) & cells.stimulus_dose.eq(8)].copy()
    endpoint["order"] = endpoint.configuration.map({name: i for i, name in enumerate(ORDER)})
    endpoint = endpoint.sort_values(["dataset_id", "order"])
    endpoint.to_csv(output / "decomposition_maximum_dose.csv", index=False)
    latex = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Changing the comparison at the predeclared maximum dose $(p,s)=(8,8)$. Total system gain $EC-P$ includes the EEG-free calibration gain $PC-P$. The last column is the paired percentile interval for $EC-PC$, not a difference of interval endpoints. These descriptive endpoints do not replace the primary uniform-grid mean.}",
             r"\label{tab:resource-decomposition}", r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
             r"Task & Setting & $EC-P$ & $PC-P$ & $EC-PC$ & Paired interval \\", r"\midrule"]
    md = ["| Task | Setting | EC-P | PC-P | EC-PC | Paired percentile interval |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for _, row in endpoint.iterrows():
        fields = [DATASETS[row.dataset_id], row.configuration, *[_number(row[c]) for c in CONTRASTS], _interval(row)]
        latex.append(" & ".join(fields) + r" \\")
        md.append("| " + " | ".join(fields) + " |")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output / "decomposition.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    (output / "decomposition_en.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    md[0] = "| 任务 | 设置 | EC-P | PC-P | EC-PC | 配对 percentile 区间 |"
    (output / "decomposition_zh.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    summary = frames["percentile"]["summary"].query("contrast == 'matched_eeg_added_uniform_mean_ccc'").copy()
    summary["configuration"] = summary.representation.map(SETTINGS)
    latex = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Primary uniform-grid mean matched EEG increment in CCC, with paired percentile intervals. Each task averages five fixed identity blocks and all 25 dose cells. Settings are sensitivity checks, not independent replications.}",
             r"\label{tab:primary-v9}", r"\begin{tabular}{@{}lll@{}}", r"\toprule",
             r"Setting & EmoEEG-MC: mean [interval] & Urban: mean [interval] \\", r"\midrule"]
    reading = ["| 设置 | EmoEEG-MC：均值 [区间] | Urban：均值 [区间] |", "| --- | --- | --- |"]
    for label in ORDER:
        fields = [label]
        for dataset in DATASETS:
            row = summary.loc[summary.dataset_id.eq(dataset) & summary.configuration.eq(label)].iloc[0]
            fields.append(f"{_number(row.estimate)} [{_number(row.ci_low)}, {_number(row.ci_high)}]")
        latex.append(" & ".join(fields) + r" \\")
        reading.append("| " + " | ".join(fields) + " |")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output / "primary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    (output / "primary_zh.md").write_text("\n".join(reading) + "\n", encoding="utf-8")
    complete = frames["percentile"]["summary"].merge(
        frames["normal_t"]["summary"], on=["dataset_id", "representation", "contrast"],
        suffixes=("_p", "_t"), validate="1:1")
    detail = []
    for dataset, task in DATASETS.items():
        for metric in ("ccc", "mae"):
            current = complete.loc[complete.dataset_id.eq(dataset) & complete.contrast.eq("matched_eeg_added_uniform_mean_" + metric)].copy()
            current["configuration"] = current.representation.map(SETTINGS)
            current = current.set_index("configuration").loc[ORDER]
            detail.extend([r"\begin{table}[htbp]\centering\small",
                           r"\caption{" + task + ": uniform-grid EC-PC " + metric.upper() + r". Paired pointwise intervals condition on fixed fitted predictions; t-normal is a heuristic sensitivity analysis.}",
                           r"\begin{tabular}{@{}lrll@{}}\toprule",
                           r"Setting & Mean & Percentile interval & t-normal interval \\", r"\midrule"])
            for label, row in current.iterrows():
                detail.append(" & ".join([label, _number(row.estimate_p),
                    f"[{_number(row.ci_low_p)}, {_number(row.ci_high_p)}]",
                    f"[{_number(row.ci_low_t)}, {_number(row.ci_high_t)}]"]) + r" \\")
            detail.extend([r"\bottomrule\end{tabular}\end{table}"])
    (output / "full_sensitivity.tex").write_text("\n".join(detail) + "\n", encoding="utf-8")
    owned = ("decomposition_all_cells.csv", "decomposition_maximum_dose.csv",
             "decomposition.tex", "decomposition_en.md", "decomposition_zh.md",
             "primary.tex", "primary_zh.md", "full_sensitivity.tex")
    outputs = {name: sha256_file(output / name) for name in owned}
    report = dict(status="passed", source="final_closure_v9", inputs=inputs, output_sha256=outputs,
                  cells=len(cells), maximum_dose_rows=len(endpoint), primary_rows=len(summary),
                  maximum_arithmetic_error=float(cells.arithmetic_error.abs().max()),
                  intervals="Copied paired contrasts; no endpoint subtraction or new bootstrap",
                  interpretation="Predictive score decomposition, not causal or latent-information decomposition")
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
