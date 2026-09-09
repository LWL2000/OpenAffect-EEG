#!/usr/bin/env python3
"""Append hash-locked identity-exposure claims to the manuscript ledger."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import pandas as pd

from openaffect_eeg.manuscript_assets import write_claim_value_macros


EXPECTED = {
    "C105": -0.0051893142572676,
    "C106": 0.5748353004972577,
    "C107": 0.4604484159968499,
    "C108": 0.1025654246705688,
    "C109": 0.3070212969887674,
    "C110": 0.2301752189607718,
    "C111": 0.1788304234980595,
    "C112": 0.0712107124494336,
    "C113": 0.23065373272566153,
    "C114": 0.4728257981948536,
    "C115": 0.29004400127452096,
    "C116": 0.5419687903974225,
    "C117": 0.5800246147545256,
    "C118": 0.40717395375538756,
    "C119": 0.6219174207039093,
    "C120": 0.0953023032410538,
    "C121": -0.009665961720865889,
    "C122": 0.20372752702338381,
    "C123": 0.1059548838621458,
    "C124": -0.09500704570287377,
    "C125": 0.2798237027281157,
    "C126": 0.2044558723181986,
    "C127": 0.015664047877188173,
    "C128": 0.3319738280457475,
    "C129": 0.1468975295417781,
    "C130": 1.0,
    "C131": 0.0012286223064555,
    "C132": 4.0,
    "C133": -0.0037867018532815044,
    "C134": 0.014517551731394518,
    "C135": 3.0,
    "C136": 2.0,
    "C137": 0.2855764734985491,
    "C138": 0.21318462724275067,
    "C139": 0.34608975998708214,
    "C140": -0.0013800623644246,
    "C141": -0.0071179015820329655,
    "C142": 0.004172705518896348,
    "C143": 0.3256244934431868,
    "C144": 0.2693801776704382,
    "C145": 0.3684575289520286,
    "C146": 4.0,
    "C147": 5.0,
    "C148": 10.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=root / "paper" / "generated" / "identity_exposure_v1" / "analysis",
    )
    parser.add_argument(
        "--base-ledger",
        type=Path,
        default=root / "paper" / "generated" / "claim_ledger.tsv",
    )
    parser.add_argument(
        "--output-ledger",
        type=Path,
        default=root / "paper" / "generated" / "claim_ledger.tsv",
    )
    parser.add_argument(
        "--output-macros",
        type=Path,
        default=root / "paper" / "generated" / "claim_values.tex",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=root / "paper" / "generated" / "table_identity_exposure_claims.csv",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(table: pd.DataFrame, **where: object) -> pd.Series:
    selected = table.copy()
    for column, value in where.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Selector {where} resolved to {len(selected)} rows")
    return selected.iloc[0]


def _interval(row: pd.Series) -> tuple[float, float]:
    value = row["ccc_delta_crossed_bootstrap_95ci"]
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return float(parsed[0]), float(parsed[1])


def main() -> int:
    args = parse_args()
    paths = {
        "identity_exposure_deployment": args.analysis_root / "table_identity_exposure_deployment.csv",
        "identity_exposure_effects": args.analysis_root / "table_identity_exposure_effects.csv",
        "identity_exposure_seed_scores": args.analysis_root / "table_identity_exposure_seed_scores.csv",
        "identity_exposure_support": args.analysis_root / "table_identity_exposure_support.csv",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    deployment = pd.read_csv(paths["identity_exposure_deployment"])
    effects = pd.read_csv(paths["identity_exposure_effects"])
    seeds = pd.read_csv(paths["identity_exposure_seed_scores"])
    support = pd.read_csv(paths["identity_exposure_support"])
    rows: list[dict[str, object]] = []

    def add(
        claim_id: str,
        claim: str,
        value: float,
        *,
        table: str,
        selector: str,
        decimals: int = 4,
        wording: str,
    ) -> None:
        expected = EXPECTED[claim_id]
        if abs(float(value) - expected) > 1e-12:
            raise ValueError(f"{claim_id} drifted: observed {value}, expected {expected}")
        display = f"{float(value):.{decimals}f}" if decimals else str(int(round(value)))
        rows.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "numeric_value": float(value),
                "display_value": display,
                "table": table,
                "selector": selector,
                "source_assets": table,
                "source_sha256": hashes[table],
                "allowed_wording": wording,
            }
        )

    deployment_specs = (
        ("ds005540", "C105", "C106", "C107", "C129", "C130", "EmoEEG-MC"),
        ("ds002721", "C108", "C109", "C110", "C131", "C132", "MusicEEG"),
    )
    for dataset_id, zero_id, maximum_id, auc_id, first_id, half_id, label in deployment_specs:
        row = _one(
            deployment,
            dataset_id=dataset_id,
            representation="labram_pretrained",
            predictor="combined_personalized",
        )
        selector = f"dataset_id={dataset_id}; representation=labram_pretrained; predictor=combined_personalized"
        for claim_id, column, name in (
            (zero_id, "zero_exposure_ccc", "zero-resource CCC"),
            (maximum_id, "maximum_exposure_ccc", "maximum-resource CCC"),
            (auc_id, "normalized_surface_auc_ccc", "normalized response-surface AUC"),
            (first_id, "first_participant_trial_gain_ccc", "first participant-trial gain"),
        ):
            add(
                claim_id,
                f"{label} pretrained-LaBraM {name}",
                float(row[column]),
                table="identity_exposure_deployment",
                selector=f"{selector}; metric={column}",
                wording=f"{label} pretrained-LaBraM {name} was reported from the fixed-support surface.",
            )
        add(
            half_id,
            f"{label} participant half-gain dose",
            float(row["participant_half_gain_dose"]),
            table="identity_exposure_deployment",
            selector=f"{selector}; metric=participant_half_gain_dose",
            decimals=0,
            wording=f"{label} reached half of its positive participant-calibration endpoint gain at this dose.",
        )

    effect_ids = {
        ("ds005540", "participant_calibration"): ("C111", "C112", "C113"),
        ("ds005540", "stimulus_exposure"): ("C114", "C115", "C116"),
        ("ds005540", "joint_exposure"): ("C117", "C118", "C119"),
        ("ds002721", "participant_calibration"): ("C120", "C121", "C122"),
        ("ds002721", "stimulus_exposure"): ("C123", "C124", "C125"),
        ("ds002721", "joint_exposure"): ("C126", "C127", "C128"),
        ("ds006866", "participant_calibration"): ("C137", "C138", "C139"),
        ("ds006866", "stimulus_exposure"): ("C140", "C141", "C142"),
        ("ds006866", "joint_exposure"): ("C143", "C144", "C145"),
    }
    representation = {"ds005540": "labram_pretrained", "ds002721": "labram_pretrained", "ds006866": "bandpower"}
    for (dataset_id, effect), (mean_id, low_id, high_id) in effect_ids.items():
        row = _one(
            effects,
            dataset_id=dataset_id,
            representation=representation[dataset_id],
            effect=effect,
        )
        low, high = _interval(row)
        selector = f"dataset_id={dataset_id}; representation={representation[dataset_id]}; effect={effect}"
        for claim_id, value, metric in (
            (mean_id, float(row["ccc_delta_mean"]), "ccc_delta_mean"),
            (low_id, low, "ccc_delta_ci_low"),
            (high_id, high, "ccc_delta_ci_high"),
        ):
            add(
                claim_id,
                f"{dataset_id} {effect} {metric}",
                value,
                table="identity_exposure_effects",
                selector=f"{selector}; metric={metric}",
                wording=f"The fixed-support {effect} estimate and crossed interval were reported for {dataset_id}.",
            )

    for dataset_id, claim_id, label in (
        ("ds005540", "C133", "EmoEEG-MC"),
        ("ds002721", "C134", "MusicEEG"),
    ):
        selected = deployment.loc[
            deployment["dataset_id"].eq(dataset_id)
            & deployment["predictor"].eq("combined_personalized")
        ].set_index("representation")
        value = float(
            selected.loc["labram_pretrained", "normalized_surface_auc_ccc"]
            - selected.loc["labram_random", "normalized_surface_auc_ccc"]
        )
        add(
            claim_id,
            f"{label} pretrained-minus-random surface AUC",
            value,
            table="identity_exposure_deployment",
            selector=f"dataset_id={dataset_id}; predictor=combined_personalized; contrast=labram_pretrained-labram_random",
            wording=f"{label} pretrained-minus-random normalized surface AUC was computed from matched cells.",
        )

    for dataset_id, claim_id, label in (
        ("ds005540", "C135", "EmoEEG-MC"),
        ("ds002721", "C136", "MusicEEG"),
    ):
        table = seeds.loc[seeds["dataset_id"].eq(dataset_id)]
        maximum_participant = int(table["participant_dose"].max())
        maximum_stimulus = int(table["stimulus_dose"].max())
        changed = 0
        for seed in sorted(table["seed"].unique()):
            zero = table.loc[
                table["seed"].eq(seed)
                & table["participant_dose"].eq(0)
                & table["stimulus_dose"].eq(0)
                & table["predictor"].eq("combined_population")
            ].sort_values("macro_ccc", ascending=False)
            maximum = table.loc[
                table["seed"].eq(seed)
                & table["participant_dose"].eq(maximum_participant)
                & table["stimulus_dose"].eq(maximum_stimulus)
                & table["predictor"].eq("combined_personalized")
            ].sort_values("macro_ccc", ascending=False)
            changed += zero.iloc[0]["representation"] != maximum.iloc[0]["representation"]
        add(
            claim_id,
            f"{label} seeds whose winning representation changed",
            float(changed),
            table="identity_exposure_seed_scores",
            selector=f"dataset_id={dataset_id}; contrast=zero_vs_maximum_winner",
            decimals=0,
            wording=f"The winning representation changed between deployment endpoints in this many {label} split assignments.",
        )

    dens = support.loc[support["dataset_id"].eq("ds003751")]
    for claim_id, value, name in (
        ("C146", float(dens["seed"].nunique()), "supported seeds"),
        ("C147", float(dens["fixed_test_trials"].min()), "minimum fixed-test trials"),
        ("C148", float(dens["fixed_test_trials"].max()), "maximum fixed-test trials"),
    ):
        add(
            claim_id,
            f"DENS response-surface {name}",
            value,
            table="identity_exposure_support",
            selector=f"dataset_id=ds003751; metric={name.replace(' ', '_')}",
            decimals=0,
            wording="DENS response-surface support was restricted by the predeclared fixed-support rule.",
        )

    exposure = pd.DataFrame(rows).sort_values(
        "claim_id", key=lambda values: values.str.removeprefix("C").astype(int)
    )
    base = pd.read_csv(args.base_ledger, sep="\t")
    base = base.loc[~base["claim_id"].isin(exposure["claim_id"])].copy()
    ledger = pd.concat([base, exposure], ignore_index=True).sort_values(
        "claim_id", key=lambda values: values.str.removeprefix("C").astype(int)
    )
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    exposure.to_csv(args.output_table, index=False, float_format="%.12g")
    ledger.to_csv(args.output_ledger, sep="\t", index=False, float_format="%.12g")
    write_claim_value_macros(ledger, args.output_macros)
    print(f"Identity-exposure claims: {len(exposure)}")
    print(f"Combined ledger: {len(ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
