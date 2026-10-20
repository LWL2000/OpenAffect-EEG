"""Publish and rebuild v14 manuscript evidence from already computed aggregates.

This module deliberately does not import a trainer, load EEG/predictions, sample
a bootstrap, or download data. Frozen source files are copied byte for byte.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
from pathlib import Path


ANALYSES = (
    ("faced_eegnet_observed", "FACED", "EEGNet", "observed", "derived/faced_eegnet_observed_analysis_v14"),
    ("faced_labram_observed", "FACED", "LaBraM tail", "observed", "derived/faced_labram_observed_analysis_v14"),
    ("faced_eegnet_permutation", "FACED", "EEGNet", "label permutation", "derived/faced_eegnet_label_permutation_analysis_v14"),
    ("faced_labram_permutation", "FACED", "LaBraM tail", "label permutation", "derived/faced_labram_label_permutation_analysis_v14"),
    ("faced_eegnet_raw_sinusoid", "FACED", "EEGNet", "raw-target sinusoid", "derived/faced_eegnet_synthetic_signal_analysis_v14"),
    ("faced_eegnet_residual_sinusoid", "FACED", "EEGNet", "residual sinusoid", "derived/faced_eegnet_synthetic_residual_signal_analysis_v14"),
    ("faced_eegnet_signed_power", "FACED subset", "EEGNet", "signed-power subset", "derived/faced_eegnet_synthetic_signed_power_residual_subset_analysis_v14"),
    ("faced_labram_signed_power", "FACED subset", "LaBraM tail", "signed-power subset", "derived/faced_labram_synthetic_signed_power_residual_subset_analysis_v14"),
    ("emo_labram_adapted", "EmoEEG-MC", "LaBraM tail", "observed sensitivity", "derived/training_uncertainty_v14/ds005540_labram"),
    ("urban_labram_adapted", "Urban", "LaBraM tail", "observed sensitivity", "derived/training_uncertainty_v14/ds006850_labram"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def archive_results(root: Path) -> dict:
    """Import immutable small aggregate files; no computation is launched."""
    destination = root / "paper/generated/evidence_v14"
    records = []
    for analysis_id, dataset, model, condition, relative in ANALYSES:
        source = root / relative
        summary = json.loads((source / "training_uncertainty_summary.json").read_text(encoding="utf-8-sig"))
        cells = source / "training_uncertainty_cells.csv"
        if digest(cells) != summary["cell_table_sha256"]:
            raise ValueError(f"Frozen cell hash mismatch: {analysis_id}")
        with cells.open(encoding="utf-8-sig", newline="") as handle:
            cell_rows = list(csv.DictReader(handle))
        expected = {(int(p), int(s)) for p, s in summary["grid"]}
        observed = {(int(r["participant_dose"]), int(r["stimulus_dose"])) for r in cell_rows}
        if observed != expected or len(cell_rows) != len(expected):
            raise ValueError(f"Resource grid mismatch: {analysis_id}")
        # These checks link the reported primary statistic to the released grid.
        mean = sum(float(r["ccc_delta"]) for r in cell_rows) / len(cell_rows)
        if not math.isclose(mean, summary["primary"]["estimate"], abs_tol=1e-12):
            raise ValueError(f"Primary mean mismatch: {analysis_id}")
        files = []
        for original, name in ((source / "training_uncertainty_summary.json", "summary.json"), (cells, "cells.csv")):
            target = destination / "archive" / analysis_id / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            files.append({"path": target.relative_to(root).as_posix(), "sha256": digest(target), "source": original.relative_to(root).as_posix()})
        records.append({"analysis_id": analysis_id, "dataset": dataset, "model": model, "condition": condition, "files": files})
    report = {
        "schema_version": "1.0", "status": "archived", "analyses": records,
        "data_policy": "Aggregate summaries and cell statistics only; no EEG, trial predictions, participant IDs or fitted weights.",
        "interpretation": "Archived equivalence_scope fields record the original analysis invocation. Failed locked controls prevent calling the complete FACED pipeline a fully preregistered confirmation.",
        "execution": "Reuses existing results; no fitting, simulation, bootstrap or download.",
    }
    write_json(destination / "archive_manifest.json", report)
    return report


def load_archive(root: Path) -> tuple[dict, list[dict]]:
    manifest_path = root / "paper/generated/evidence_v14/archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for record in manifest["analyses"]:
        for item in record["files"]:
            path = root / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"Archive hash mismatch: {item['path']}")
        folder = root / "paper/generated/evidence_v14/archive" / record["analysis_id"]
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8-sig"))
        if digest(folder / "cells.csv") != summary["cell_table_sha256"]:
            raise ValueError(f"Cell/summary mismatch: {record['analysis_id']}")
        with (folder / "cells.csv").open(encoding="utf-8-sig", newline="") as handle:
            cells = list(csv.DictReader(handle))
        keys = {(int(c["participant_dose"]), int(c["stimulus_dose"])) for c in cells}
        if keys != {tuple(g) for g in summary["grid"]} or len(keys) != len(cells):
            raise ValueError(f"Duplicate or missing cells: {record['analysis_id']}")
        primary = summary["primary"]
        estimate = sum(float(c["ccc_delta"]) for c in cells) / len(cells)
        if not math.isclose(estimate, primary["estimate"], abs_tol=1e-12):
            raise ValueError(f"Primary mean mismatch: {record['analysis_id']}")
        eq = primary["equivalence_primary"]
        strict = primary["equivalence_strict_sensitivity"]
        lo90, hi90 = eq["decision_interval_90"]
        for decision in (eq, strict):
            interval = decision["decision_interval_90"]
            expected = -decision["margin"] < interval[0] and interval[1] < decision["margin"]
            if expected != decision["established"]:
                raise ValueError(f"Equivalence decision mismatch: {record['analysis_id']}")
        rows.append({
            "analysis_id": record["analysis_id"], "dataset": record["dataset"], "model": record["model"], "condition": record["condition"],
            "estimate": primary["estimate"], "ci95_low": primary["ci_95"][0], "ci95_high": primary["ci_95"][1],
            "ci90_low": lo90, "ci90_high": hi90,
            "equivalence_005": eq["decision"], "equivalence_0025": strict["decision"],
            "participants": summary["participants"], "stimuli": summary["stimuli"],
            "training_seeds": summary["training_seed_count"], "rotations": summary["fold_rotation_count"], "cells": len(cells),
            "valid_bootstrap": summary["bootstrap_iterations_valid"], "requested_bootstrap": summary["bootstrap_iterations_requested"],
            "diagnostic_warning": summary["control_diagnostic"]["warning"],
            "input_predictions_sha256": summary["input_predictions_sha256"], "cell_table_sha256": summary["cell_table_sha256"],
        })
    return manifest, rows


def interval(row: dict, level: int = 95) -> str:
    return f"[{row[f'ci{level}_low']:+.4f}, {row[f'ci{level}_high']:+.4f}]"


def build_assets(root: Path, output: Path | None = None) -> dict:
    """Rebuild tables from the public archive, without the private derived tree."""
    root = root.resolve()
    output = output or root / "paper/generated/evidence_v14"
    output.mkdir(parents=True, exist_ok=True)
    manifest, rows = load_archive(root)
    lookup = {row["analysis_id"]: row for row in rows}
    ridge_folder = root / "paper/generated/evidence_v14/ridge"
    ridge = json.loads((ridge_folder / "summary.json").read_text(encoding="utf-8-sig"))
    if digest(ridge_folder / "executed_cell_metrics.csv") != ridge["executed_cell_metrics_sha256"]:
        raise ValueError("Ridge executed-metric archive hash mismatch")
    with (ridge_folder / "executed_cell_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        ridge_rows = list(csv.DictReader(handle))
    ridge_keys = {(int(r["split_index"]), int(r["participant_dose"]), int(r["stimulus_dose"])) for r in ridge_rows}
    if len(ridge_rows) != 2500 or ridge_keys != {(i, p, s) for i in range(100) for p in (0, 1, 2, 4, 8) for s in (0, 1, 2, 4, 8)}:
        raise ValueError("Ridge executed grid incomplete")
    for field, column in (("ccc_delta_uniform_mean", "ccc_delta"), ("valence_ccc_delta_mean", "valence_ccc_delta"), ("arousal_ccc_delta_mean", "arousal_ccc_delta")):
        if not math.isclose(sum(float(r[column]) for r in ridge_rows) / len(ridge_rows), ridge[field], abs_tol=1e-12):
            raise ValueError("Ridge descriptive mean mismatch")
    if ridge["joint_uncertainty_available"]:
        raise ValueError("This archived Ridge table has no joint uncertainty")
    (output / "ridge_descriptive.csv").write_text(csv_text([ridge]), encoding="utf-8")
    (output / "all_results.csv").write_text(csv_text(rows), encoding="utf-8")
    selected = [lookup[k] for k in ("faced_eegnet_observed", "faced_labram_observed", "emo_labram_adapted", "urban_labram_adapted")]
    table = [
        "% evidence: paper/generated/evidence_v14/all_results.csv",
        r"\begin{table}[t]\centering\small",
        r"\caption{Existing five-seed adapted-model evidence. FACED intervals jointly resample participants, stimuli, executed training seeds and five fully refitted rotations; development-task intervals hold their single rotation fixed. Equivalence decisions apply to the uniform-grid mean and use the 90\% interval. Failed locked controls limit the FACED confirmation claim.}",
        r"\label{tab:v14-primary}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{@{}l l r X l@{}}\toprule",
        r"Dataset & Model & $\Delta$CCC & 95\% interval & Decision\\\midrule",
    ]
    for row in selected:
        decision = "equivalent$^{*}$" if row["equivalence_005"] == "equivalent" else ("uncertain" if row["dataset"] == "FACED" else "negative")
        table.append(f"{row['dataset']} & {row['model']} & ${row['estimate']:+.4f}$ & ${interval(row)}$ & {decision} " + r"\\")
    table.extend([r"\bottomrule\end{tabularx}", r"\vspace{2pt}\parbox{\linewidth}{\footnotesize $^{*}$FACED LaBraM also meets the narrower $\pm0.025$ margin. Development-task margins were not prespecified and are not used for a confirmatory equivalence claim.}", r"\end{table}"])
    (output / "primary.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    controls = ["faced_eegnet_permutation", "faced_labram_permutation", "faced_eegnet_raw_sinusoid", "faced_eegnet_residual_sinusoid", "faced_eegnet_signed_power", "faced_labram_signed_power"]
    lines = ["% evidence: paper/generated/evidence_v14/all_results.csv", r"\begin{table}[htbp]\centering\small", r"\caption{All completed FACED neural controls, including failures. Signed-power rows use an exploratory subset with 85 participants, 23 observed stimulus identities, 15 resource cells and five sampled split indices; they do not replace either failed full-grid control.}", r"\label{tab:v14-controls}", r"\begin{tabularx}{\linewidth}{@{}l l r X l@{}}\toprule", r"Model & Control & $\Delta$CCC & 95\% interval & Gate\\\midrule"]
    for key in controls:
        row = lookup[key]
        status = "warning" if row["diagnostic_warning"] else "pass"
        lines.append(f"{row['model']} & {row['condition']} & ${row['estimate']:+.4f}$ & ${interval(row)}$ & {status} " + r"\\")
    lines.extend([r"\bottomrule\end{tabularx}", r"\end{table}"])
    (output / "controls.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # All four resource corners are included by design, never selected by sign.
    corners = []
    for key in ("faced_eegnet_observed", "faced_labram_observed"):
        path = root / "paper/generated/evidence_v14/archive" / key / "cells.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for cell in csv.DictReader(handle):
                if int(cell["participant_dose"]) in (0, 8) and int(cell["stimulus_dose"]) in (0, 8):
                    corners.append({"analysis_id": key, **cell})
    (output / "resource_corners.csv").write_text(csv_text(corners), encoding="utf-8")
    cases_source = root / "paper/generated/evidence_argument_v10/decomposition_maximum_dose.csv"
    with cases_source.open(encoding="utf-8-sig", newline="") as handle:
        cases = [row for row in csv.DictReader(handle) if row["representation"] == "labram_ridge_base"]
    if len(cases) != 2 or any(int(r["participant_dose"]) != 8 or int(r["stimulus_dose"]) != 8 for r in cases):
        raise ValueError("The two fixed maximum-dose cases are missing or changed")
    (output / "system_attribution_cases.csv").write_text(csv_text(cases), encoding="utf-8")
    case_lines = ["% evidence: paper/generated/evidence_v14/system_attribution_cases.csv", r"\begin{table}[t]\centering\small", r"\caption{Existing maximum-dose LaBraM R1 cases: point CCC contrasts. Changing the comparator changes the attribution. None of the total or matched endpoint ranges establishes a positive gain; full conditional ranges and other settings are retained in the development appendix.}", r"\label{tab:v14-cases}", r"\begin{tabularx}{\linewidth}{@{}lXXX@{}}\toprule", r"Task & System $EC-P$ & Calibration $PC-P$ & Matched EEG $EC-PC$\\\midrule"]
    for case in cases:
        name = "EmoEEG-MC" if case["dataset_id"] == "ds005540" else "Urban"
        case_lines.append(f"{name} & ${float(case['total']):+.4f}$ & ${float(case['calibration']):+.4f}$ & ${float(case['matched']):+.4f}$ " + r"\\")
    case_lines.extend([r"\bottomrule\end{tabularx}", r"\end{table}"])
    (output / "cases.tex").write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    values = {}
    values["FacedRidgeMean"] = f"{ridge['ccc_delta_uniform_mean']:+.4f}"
    for key, prefix in (("faced_eegnet_observed", "FacedEegnet"), ("faced_labram_observed", "FacedLabram")):
        row = lookup[key]
        values[prefix + "Mean"] = f"{row['estimate']:+.4f}"
        values[prefix + "CI"] = interval(row)
        values[prefix + "DecisionCI"] = interval(row, 90)
    (output / "values.tex").write_text("% Generated from archived v14 summaries; no refitting.\n" + "\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in values.items()) + "\n", encoding="utf-8")
    files = ["all_results.csv", "primary.tex", "controls.tex", "resource_corners.csv", "values.tex", "system_attribution_cases.csv", "cases.tex", "ridge_descriptive.csv"]
    report = {"status": "pass", "analysis_count": len(rows), "source_file_count": sum(len(r["files"]) for r in manifest["analyses"]), "ridge_executed_cells": len(ridge_rows), "experimental_computation_run": False, "outputs": {name: digest(output / name) for name in files}}
    write_json(output / "build_report.json", report)
    return report


def verify_assets(root: Path, output: Path) -> dict:
    report = build_assets(root, output)
    frozen = json.loads((root / "paper/generated/evidence_v14/build_report.json").read_text(encoding="utf-8"))
    if report != frozen:
        raise ValueError("Rebuilt v14 evidence differs from the frozen tables")
    return report
