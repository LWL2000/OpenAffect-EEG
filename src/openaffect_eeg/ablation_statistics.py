"""Paired clustered statistics for frequency and channel-region audits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openaffect_eeg.statistics import (
    cluster_bootstrap_indices,
    crossed_cluster_bootstrap_indices,
    exact_sign_flip_pvalue,
    percentile_interval,
)

CLUSTERS_BY_PROTOCOL = {
    "trial_random_holdout": ("subject_uid", "stimulus_uid"),
    "subject_holdout": ("subject_uid",),
    "stimulus_holdout": ("stimulus_uid",),
    "subject_stimulus_holdout": ("subject_uid", "stimulus_uid"),
}
TARGET_COLUMNS = ("target_valence", "target_arousal")
PREDICTION_COLUMNS = ("prediction_valence", "prediction_arousal")


class AblationAlignmentError(ValueError):
    """Raised when feature-view predictions are not trial aligned."""


def vectorized_regression_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score several prediction views against one two-dimensional target."""
    observed = np.asarray(truth, dtype=float)
    estimated = np.asarray(predictions, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("Truth must have shape (trials, 2)")
    if estimated.ndim != 3 or estimated.shape != (
        len(observed),
        estimated.shape[1],
        2,
    ):
        raise ValueError("Predictions must have shape (trials, views, 2)")
    if not len(observed) or not np.isfinite(observed).all():
        raise ValueError("Truth must be non-empty and finite")
    if not np.isfinite(estimated).all():
        raise ValueError("Predictions must be finite")

    ccc_by_target = []
    mae_by_target = []
    rmse_by_target = []
    for target_index in range(2):
        target = observed[:, target_index]
        prediction = estimated[:, :, target_index]
        target_centered = target - target.mean()
        prediction_centered = prediction - prediction.mean(axis=0)
        covariance = np.mean(target_centered[:, None] * prediction_centered, axis=0)
        denominator = (
            target.var()
            + prediction.var(axis=0)
            + np.square(target.mean() - prediction.mean(axis=0))
        )
        ccc = np.divide(
            2.0 * covariance,
            denominator,
            out=np.full_like(denominator, np.nan, dtype=float),
            where=denominator > 0,
        )
        error = prediction - target[:, None]
        ccc_by_target.append(ccc)
        mae_by_target.append(np.mean(np.abs(error), axis=0))
        rmse_by_target.append(np.sqrt(np.mean(np.square(error), axis=0)))
    return {
        "ccc": np.nanmean(np.stack(ccc_by_target), axis=0),
        "mae": np.mean(np.stack(mae_by_target), axis=0),
        "rmse": np.mean(np.stack(rmse_by_target), axis=0),
    }


def _aligned_seed_arrays(
    table: pd.DataFrame,
    views: list[str],
) -> dict[int, dict[str, object]]:
    metadata_columns = ["trial_uid", "subject_uid", "stimulus_uid", *TARGET_COLUMNS]
    aligned: dict[int, dict[str, object]] = {}
    for seed_value in sorted(table["seed"].unique()):
        seed = int(seed_value)
        seed_table = table.loc[table["seed"].eq(seed_value)]
        reference: pd.DataFrame | None = None
        predictions = []
        for view in views:
            selected = seed_table.loc[seed_table["view_id"].eq(view)]
            if selected["trial_uid"].duplicated().any():
                raise AblationAlignmentError(
                    f"Duplicate trial rows for seed {seed}, view {view}"
                )
            selected = selected.sort_values("trial_uid").reset_index(drop=True)
            metadata = selected[metadata_columns]
            if reference is None:
                reference = metadata
            elif not metadata.equals(reference):
                raise AblationAlignmentError(
                    f"Feature views are not trial-aligned for seed {seed}"
                )
            predictions.append(selected[list(PREDICTION_COLUMNS)].to_numpy(dtype=float))
        assert reference is not None
        aligned[seed] = {
            "metadata": reference,
            "truth": reference[list(TARGET_COLUMNS)].to_numpy(dtype=float),
            "predictions": np.stack(predictions, axis=1),
        }
    return aligned


def _bootstrap_indices(
    metadata: pd.DataFrame,
    cluster_columns: tuple[str, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    if len(cluster_columns) == 1:
        return cluster_bootstrap_indices(
            metadata[cluster_columns[0]].astype(str).to_numpy(), rng
        )
    first = metadata[cluster_columns[0]].astype(str).to_numpy()
    second = metadata[cluster_columns[1]].astype(str).to_numpy()
    for _ in range(100):
        indices = crossed_cluster_bootstrap_indices(first, second, rng)
        if len(indices):
            return indices
    raise ValueError("Crossed cluster bootstrap repeatedly produced no rows")


def analyze_ablation_group(
    table: pd.DataFrame,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Compare every feature view with full features using paired resampling."""
    required = {
        "dataset_id",
        "protocol",
        "seed",
        "view_id",
        "trial_uid",
        "subject_uid",
        "stimulus_uid",
        *TARGET_COLUMNS,
        *PREDICTION_COLUMNS,
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if iterations < 100:
        raise ValueError("Bootstrap requires at least 100 iterations")
    datasets = table["dataset_id"].drop_duplicates().tolist()
    protocols = table["protocol"].drop_duplicates().tolist()
    if len(datasets) != 1 or len(protocols) != 1:
        raise ValueError("Analyze one dataset and protocol at a time")
    protocol = str(protocols[0])
    if protocol not in CLUSTERS_BY_PROTOCOL:
        raise ValueError(f"No cluster policy for protocol: {protocol}")
    available_views = [str(view) for view in table["view_id"].unique()]
    if "full" not in available_views:
        raise ValueError("Ablation predictions have no full reference view")
    views = ["full", *sorted(view for view in available_views if view != "full")]
    aligned = _aligned_seed_arrays(table, views)
    seeds = sorted(aligned)
    cluster_columns = CLUSTERS_BY_PROTOCOL[protocol]

    seed_metrics = {metric: [] for metric in ("ccc", "mae", "rmse")}
    for seed in seeds:
        values = aligned[seed]
        metrics = vectorized_regression_metrics(
            values["truth"], values["predictions"]
        )
        for metric, stored_values in seed_metrics.items():
            stored_values.append(metrics[metric])
    metric_matrices = {
        metric: np.stack(values) for metric, values in seed_metrics.items()
    }
    representations = {}
    for view_index, view in enumerate(views):
        representations[view] = {}
        for metric, matrix in metric_matrices.items():
            representations[view][f"{metric}_mean"] = float(matrix[:, view_index].mean())
            representations[view][f"{metric}_sample_std"] = float(
                matrix[:, view_index].std(ddof=1)
            )

    bootstrap_ccc = np.empty((iterations, len(views)), dtype=float)
    bootstrap_mae = np.empty((iterations, len(views)), dtype=float)
    for iteration in range(iterations):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_ccc = []
        sampled_mae = []
        for sampled_seed in sampled_seeds:
            values = aligned[int(sampled_seed)]
            indices = _bootstrap_indices(
                values["metadata"], cluster_columns, rng
            )
            metrics = vectorized_regression_metrics(
                values["truth"][indices], values["predictions"][indices]
            )
            sampled_ccc.append(metrics["ccc"])
            sampled_mae.append(metrics["mae"])
        bootstrap_ccc[iteration] = np.nanmean(np.stack(sampled_ccc), axis=0)
        bootstrap_mae[iteration] = np.mean(np.stack(sampled_mae), axis=0)

    comparisons = {}
    for view_index, view in enumerate(views[1:], start=1):
        seed_ccc_delta = metric_matrices["ccc"][:, view_index] - metric_matrices[
            "ccc"
        ][:, 0]
        seed_mae_delta = metric_matrices["mae"][:, view_index] - metric_matrices[
            "mae"
        ][:, 0]
        bootstrap_ccc_delta = bootstrap_ccc[:, view_index] - bootstrap_ccc[:, 0]
        bootstrap_mae_delta = bootstrap_mae[:, view_index] - bootstrap_mae[:, 0]
        comparisons[f"{view}_vs_full"] = {
            "interpretation": (
                "Positive CCC delta and negative MAE delta favor the ablated view"
            ),
            "ccc_delta_mean_over_seeds": float(seed_ccc_delta.mean()),
            "ccc_delta_cluster_bootstrap_95ci": list(
                percentile_interval(bootstrap_ccc_delta)
            ),
            "ccc_exact_seed_sign_flip_p": exact_sign_flip_pvalue(seed_ccc_delta),
            "mae_delta_mean_over_seeds": float(seed_mae_delta.mean()),
            "mae_delta_cluster_bootstrap_95ci": list(
                percentile_interval(bootstrap_mae_delta)
            ),
            "mae_exact_seed_sign_flip_p": exact_sign_flip_pvalue(seed_mae_delta),
        }
    return {
        "dataset_id": str(datasets[0]),
        "protocol": protocol,
        "row_count_per_view": int(len(table) // len(views)),
        "seed_count": len(seeds),
        "view_count": len(views),
        "cluster_columns": list(cluster_columns),
        "cluster_count_by_seed": {
            str(seed): {
                column: int(aligned[seed]["metadata"][column].nunique())
                for column in cluster_columns
            }
            for seed in seeds
        },
        "representations": representations,
        "comparisons": comparisons,
    }
