#!/usr/bin/env python3
"""Compare trial-aligned affect predictions with clustered paired bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from openaffect_eeg.baselines import regression_metrics
from openaffect_eeg.statistics import (
    cluster_bootstrap_indices,
    crossed_cluster_bootstrap_indices,
    exact_sign_flip_pvalue,
    percentile_interval,
)

TARGET_COLUMNS = ["target_valence", "target_arousal"]
CLUSTERS_BY_PROTOCOL = {
    "trial_random_holdout": ("subject_id", "stimulus_uid"),
    "subject_holdout": ("subject_id",),
    "stimulus_holdout": ("stimulus_uid",),
    "subject_stimulus_holdout": ("subject_id", "stimulus_uid"),
    "video_to_imagery": ("subject_id",),
    "imagery_to_video": ("subject_id",),
}


class PredictionSpec(NamedTuple):
    path: Path
    model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument(
        "--model", default="disentangled_prior_plus_eeg_residual"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_prediction_specs(
    specifications: list[str],
    *,
    default_model: str,
) -> dict[str, PredictionSpec]:
    parsed: dict[str, PredictionSpec] = {}
    for specification in specifications:
        label, separator, raw_path = specification.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"Prediction must use LABEL=PATH: {specification}")
        if label in parsed:
            raise ValueError(f"Duplicate prediction label: {label}")
        raw_archive, model_separator, model = raw_path.rpartition("::")
        if model_separator:
            if not raw_archive or not model:
                raise ValueError(f"Invalid model override: {specification}")
            path = Path(raw_archive)
        else:
            path = Path(raw_path)
            model = default_model
        parsed[label] = PredictionSpec(path=path, model=model)
    if len(parsed) < 2:
        raise ValueError("At least two prediction archives are required")
    return parsed


def load_aligned_predictions(
    specifications: dict[str, PredictionSpec],
) -> pd.DataFrame:
    key_columns = ["protocol", "seed", "trial_uid"]
    metadata_columns = [
        *key_columns,
        "dataset_id",
        "subject_id",
        "stimulus_uid",
        "context",
        *TARGET_COLUMNS,
    ]
    combined: pd.DataFrame | None = None
    reference_metadata: pd.DataFrame | None = None
    for label, specification in specifications.items():
        path = specification.path
        model = specification.model
        table = pd.read_csv(path, sep="\t")
        prediction_columns = [f"{model}_valence", f"{model}_arousal"]
        required = set(metadata_columns + prediction_columns)
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if table.duplicated(key_columns).any():
            raise ValueError(f"{path} contains duplicate prediction rows")
        metadata = table[metadata_columns].sort_values(key_columns).reset_index(drop=True)
        if reference_metadata is None:
            reference_metadata = metadata
            combined = metadata.copy()
        elif not metadata.equals(reference_metadata):
            raise ValueError(f"Prediction metadata is not trial-aligned: {path}")
        assert combined is not None
        ordered = table.sort_values(key_columns).reset_index(drop=True)
        combined[f"{label}_valence"] = ordered[prediction_columns[0]].to_numpy()
        combined[f"{label}_arousal"] = ordered[prediction_columns[1]].to_numpy()
    assert combined is not None
    return combined


def score(frame: pd.DataFrame, label: str) -> dict[str, float]:
    targets = frame[TARGET_COLUMNS].to_numpy()
    prediction = frame[[f"{label}_valence", f"{label}_arousal"]].to_numpy()
    return regression_metrics(targets, prediction)["macro"]


def protocol_analysis(
    table: pd.DataFrame,
    labels: list[str],
    *,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    protocol = str(table["protocol"].iloc[0])
    cluster_columns = CLUSTERS_BY_PROTOCOL.get(protocol)
    if cluster_columns is None:
        raise ValueError(f"No bootstrap cluster policy for {protocol}")
    seeds = sorted(int(seed) for seed in table["seed"].unique())
    seed_scores = {
        seed: {
            label: score(table.loc[table["seed"].eq(seed)], label)
            for label in labels
        }
        for seed in seeds
    }
    representation_summary = {}
    for label in labels:
        ccc = np.asarray([seed_scores[seed][label]["ccc"] for seed in seeds])
        mae = np.asarray([seed_scores[seed][label]["mae"] for seed in seeds])
        representation_summary[label] = {
            "ccc_mean": float(ccc.mean()),
            "ccc_sample_std": float(ccc.std(ddof=1)),
            "mae_mean": float(mae.mean()),
            "mae_sample_std": float(mae.std(ddof=1)),
        }

    comparisons = {}
    for first, second in itertools.combinations(labels, 2):
        seed_ccc_delta = np.asarray(
            [
                seed_scores[seed][first]["ccc"]
                - seed_scores[seed][second]["ccc"]
                for seed in seeds
            ]
        )
        seed_mae_delta = np.asarray(
            [
                seed_scores[seed][first]["mae"]
                - seed_scores[seed][second]["mae"]
                for seed in seeds
            ]
        )
        bootstrap_ccc = np.empty(iterations)
        bootstrap_mae = np.empty(iterations)
        for iteration in range(iterations):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            ccc_deltas = []
            mae_deltas = []
            for sampled_seed in sampled_seeds:
                seed_table = table.loc[table["seed"].eq(sampled_seed)].reset_index(
                    drop=True
                )
                if len(cluster_columns) == 1:
                    indices = cluster_bootstrap_indices(
                        seed_table[cluster_columns[0]].astype(str).to_numpy(), rng
                    )
                else:
                    indices = crossed_cluster_bootstrap_indices(
                        seed_table[cluster_columns[0]].astype(str).to_numpy(),
                        seed_table[cluster_columns[1]].astype(str).to_numpy(),
                        rng,
                    )
                sampled = seed_table.iloc[indices]
                first_score = score(sampled, first)
                second_score = score(sampled, second)
                ccc_deltas.append(first_score["ccc"] - second_score["ccc"])
                mae_deltas.append(first_score["mae"] - second_score["mae"])
            bootstrap_ccc[iteration] = np.mean(ccc_deltas)
            bootstrap_mae[iteration] = np.mean(mae_deltas)
        ccc_low, ccc_high = percentile_interval(bootstrap_ccc)
        mae_low, mae_high = percentile_interval(bootstrap_mae)
        comparisons[f"{first}_vs_{second}"] = {
            "interpretation": (
                "Positive CCC delta and negative MAE delta favor the first model"
            ),
            "ccc_delta_mean_over_seeds": float(seed_ccc_delta.mean()),
            "ccc_delta_cluster_bootstrap_95ci": [ccc_low, ccc_high],
            "ccc_exact_seed_sign_flip_p": exact_sign_flip_pvalue(seed_ccc_delta),
            "mae_delta_mean_over_seeds": float(seed_mae_delta.mean()),
            "mae_delta_cluster_bootstrap_95ci": [mae_low, mae_high],
            "mae_exact_seed_sign_flip_p": exact_sign_flip_pvalue(seed_mae_delta),
        }
    return {
        "protocol": protocol,
        "row_count": len(table),
        "seed_count": len(seeds),
        "cluster_columns": list(cluster_columns),
        "cluster_count_by_seed": {
            str(seed): {
                column: int(
                    table.loc[table["seed"].eq(seed), column].nunique()
                )
                for column in cluster_columns
            }
            for seed in seeds
        },
        "representations": representation_summary,
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    specifications = parse_prediction_specs(
        args.prediction, default_model=args.model
    )
    table = load_aligned_predictions(specifications)
    labels = list(specifications)
    rng = np.random.default_rng(args.seed)
    results = [
        protocol_analysis(
            table.loc[table["protocol"].eq(protocol)].copy(),
            labels,
            iterations=args.bootstrap,
            rng=rng,
        )
        for protocol in sorted(table["protocol"].unique())
    ]
    payload = {
        "default_model": args.model,
        "bootstrap_iterations": args.bootstrap,
        "random_seed": args.seed,
        "bootstrap_policy": (
            "Resample five evaluation seeds, then complete held-out subject or "
            "stimulus clusters within each sampled seed; independently resample "
            "both crossed factors for joint subject-stimulus holdout"
        ),
        "prediction_archives": {
            label: {
                "path": str(specification.path.resolve()),
                "sha256": sha256(specification.path),
                "model": specification.model,
            }
            for label, specification in specifications.items()
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for result in results:
        print(result["protocol"], result["representations"], flush=True)
    print(f"Statistics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
