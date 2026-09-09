"""Deterministic tables, figures, and claim ledgers for the manuscript."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import ResultRegistry, compile_artifacts, sha256_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROTOCOL_ORDER = (
    "trial_random_holdout",
    "stimulus_holdout",
    "subject_holdout",
    "subject_stimulus_holdout",
    "imagery_to_video",
    "video_to_imagery",
)
PROTOCOL_LABELS = {
    "trial_random_holdout": "Trial random\n(audit control)",
    "stimulus_holdout": "Unseen\nstimulus",
    "subject_holdout": "Unseen\nsubject",
    "subject_stimulus_holdout": "Unseen subject\n+ stimulus",
    "imagery_to_video": "Imagery to\nvideo",
    "video_to_imagery": "Video to\nimagery",
}
REPRESENTATION_LABELS = {
    "de": "DE",
    "cbramod": "CBraMod",
    "labram_pretrained": "LaBraM pretrained",
    "labram_random": "LaBraM random",
}
REPRESENTATION_ORDER = tuple(REPRESENTATION_LABELS.values())
IDENTITY_LABELS = {
    "identity_de": "DE",
    "identity_cbramod_pretrained": "CBraMod pretrained",
    "identity_cbramod_random": "CBraMod random",
    "identity_labram_pretrained": "LaBraM pretrained",
    "identity_labram_random": "LaBraM random",
}
EXTERNAL_ASSET_NAMES = {
    "external_bandpower_summary",
    "external_bandpower_global",
    "external_condition_probe_channel",
    "external_condition_probe_global",
    "external_condition_effects_channel",
    "external_paired_channel",
    "external_paired_feature_view",
    "external_ds006866_v2",
}
SCOPE_ASSET_NAMES = {"benchmark_scope_summary"}
REVIEWER_CONTROL_ASSET_NAMES = {
    "semantic_prior_ablation_statistics",
    "support_matched_statistics",
    "subject_subspace_statistics",
}
DATA_QUALITY_ASSET_NAMES = {"dens_feature_missingness"}
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000")
LINE_STYLES = ("-", "--", "-.", ":", "-")
MARKERS = ("o", "s", "^", "D", "v")


class ManuscriptAssetError(ValueError):
    """Raised when a manuscript asset cannot be traced to frozen evidence."""


def build_reviewer_control_table(
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> pd.DataFrame:
    columns = [
        "analysis",
        "variant",
        "endpoint",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "sign_flip_p",
        "source_asset",
        "source_sha256",
    ]
    rows: list[dict[str, object]] = []
    if "support_matched_statistics" in sources:
        payload, source_hash = sources["support_matched_statistics"]
        results_by_model = payload.get("results_by_model")
        if results_by_model is None:
            model = str(
                payload.get("model", "disentangled_prior_plus_eeg_residual")
            )
            results_by_model = {model: payload["result"]}
        for model, result in results_by_model.items():
            effect = result["matched_minus_joint"]
            for metric in ("ccc", "mae"):
                interval = effect[f"{metric}_delta_cluster_bootstrap_95ci"]
                rows.append(
                    {
                        "analysis": "support_matched",
                        "variant": model,
                        "endpoint": "matched_minus_joint",
                        "metric": f"{metric}_delta",
                        "value": effect[f"{metric}_delta_mean"],
                        "ci_low": interval[0],
                        "ci_high": interval[1],
                        "sign_flip_p": effect[
                            f"{metric}_exact_seed_sign_flip_p"
                        ],
                        "source_asset": "support_matched_statistics",
                        "source_sha256": source_hash,
                    }
                )

        primary = results_by_model.get(
            "disentangled_prior_plus_eeg_residual"
        )
        if primary is not None:
            support_fields = {
                "test_rows": primary.get("test_rows_by_seed", {}),
                "subject_clusters": primary.get(
                    "subject_clusters_by_seed", {}
                ),
                "stimulus_clusters": primary.get(
                    "stimulus_clusters_by_seed", {}
                ),
            }
            for metric, values_by_seed in support_fields.items():
                if not values_by_seed:
                    continue
                values = [int(value) for value in values_by_seed.values()]
                for reduction, value in (
                    ("min", min(values)),
                    ("max", max(values)),
                ):
                    rows.append(
                        {
                            "analysis": "support_matched_support",
                            "variant": "test",
                            "endpoint": "across_seed_range",
                            "metric": f"{metric}_{reduction}",
                            "value": value,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "sign_flip_p": np.nan,
                            "source_asset": "support_matched_statistics",
                            "source_sha256": source_hash,
                        }
                    )

    if "subject_subspace_statistics" in sources:
        payload, source_hash = sources["subject_subspace_statistics"]
        summary = payload["summary"]
        summaries_by_kind = summary.get(
            "by_projection_kind", {"subject_centroid": summary}
        )
        intervals_by_kind = summary.get(
            "paired_cluster_bootstrap_by_projection_kind"
        )
        if intervals_by_kind is None:
            intervals_by_kind = {
                "subject_centroid": summary["paired_cluster_bootstrap"]
            }
        for projection_kind, kind_summary in summaries_by_kind.items():
            for endpoint in ("baseline", "max_removal"):
                values = kind_summary[endpoint]
                for metric, key in (
                    ("combined_ccc", "combined_ccc_mean"),
                    ("combined_mae", "combined_mae_mean"),
                    (
                        "subject_balanced_accuracy",
                        "subject_balanced_accuracy_mean",
                    ),
                ):
                    rows.append(
                        {
                            "analysis": "subject_subspace",
                            "variant": projection_kind,
                            "endpoint": endpoint,
                            "metric": metric,
                            "value": values[key],
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "sign_flip_p": np.nan,
                            "source_asset": "subject_subspace_statistics",
                            "source_sha256": source_hash,
                        }
                    )

            delta = kind_summary["max_removal_minus_baseline"]
            intervals = intervals_by_kind[projection_kind]
            for metric, value_key, interval_key, p_key in (
                (
                    "ccc_delta",
                    "combined_ccc_delta_mean",
                    "combined_ccc_delta_cluster_bootstrap_95ci",
                    "combined_ccc_exact_seed_sign_flip_p",
                ),
                (
                    "mae_delta",
                    "combined_mae_delta_mean",
                    "combined_mae_delta_cluster_bootstrap_95ci",
                    "combined_mae_exact_seed_sign_flip_p",
                ),
            ):
                interval = intervals[interval_key]
                rows.append(
                    {
                        "analysis": "subject_subspace",
                        "variant": projection_kind,
                        "endpoint": "max_minus_baseline",
                        "metric": metric,
                        "value": delta[value_key],
                        "ci_low": interval[0],
                        "ci_high": interval[1],
                        "sign_flip_p": delta[p_key],
                        "source_asset": "subject_subspace_statistics",
                        "source_sha256": source_hash,
                    }
                )
            rows.extend(
                [
                    {
                        "analysis": "subject_subspace",
                        "variant": projection_kind,
                        "endpoint": "max_minus_baseline",
                        "metric": "subject_balanced_accuracy_delta",
                        "value": delta["subject_balanced_accuracy_delta_mean"],
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "sign_flip_p": delta[
                            "subject_balanced_accuracy_exact_seed_sign_flip_p"
                        ],
                        "source_asset": "subject_subspace_statistics",
                        "source_sha256": source_hash,
                    },
                    {
                        "analysis": "subject_subspace",
                        "variant": projection_kind,
                        "endpoint": "full_curve",
                        "metric": "within_seed_identity_ccc_correlation",
                        "value": kind_summary[
                            "within_seed_identity_ccc_correlation"
                        ],
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "sign_flip_p": np.nan,
                        "source_asset": "subject_subspace_statistics",
                        "source_sha256": source_hash,
                    },
                ]
            )

        specificity_intervals = summary.get("specificity_bootstrap", {})
        for control_kind, specificity in summary.get(
            "specificity_vs_controls", {}
        ).items():
            interval = specificity_intervals[control_kind][
                "combined_ccc_difference_in_differences_95ci"
            ]
            rows.append(
                {
                    "analysis": "subject_subspace_specificity",
                    "variant": control_kind,
                    "endpoint": "identity_minus_control_endpoint_effect",
                    "metric": "ccc_difference_in_differences",
                    "value": specificity[
                        "combined_ccc_difference_in_differences_mean"
                    ],
                    "ci_low": interval[0],
                    "ci_high": interval[1],
                    "sign_flip_p": specificity[
                        "combined_ccc_exact_seed_sign_flip_p"
                    ],
                    "source_asset": "subject_subspace_statistics",
                    "source_sha256": source_hash,
                }
            )

    if "semantic_prior_ablation_statistics" in sources:
        payload, source_hash = sources["semantic_prior_ablation_statistics"]
        for result in payload["results"]:
            protocol = str(result["protocol"])
            for variant, metrics in result["representations"].items():
                for metric in ("ccc", "mae"):
                    rows.append(
                        {
                            "analysis": "semantic_prior_ablation",
                            "variant": variant,
                            "endpoint": protocol,
                            "metric": f"{metric}_mean",
                            "value": metrics[f"{metric}_mean"],
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "sign_flip_p": np.nan,
                            "source_asset": "semantic_prior_ablation_statistics",
                            "source_sha256": source_hash,
                        }
                    )
            for comparison, effect in result["comparisons"].items():
                if not comparison.startswith("full_vs_"):
                    continue
                variant = comparison[len("full_vs_") :]
                for metric in ("ccc", "mae"):
                    interval = effect[
                        f"{metric}_delta_cluster_bootstrap_95ci"
                    ]
                    rows.append(
                        {
                            "analysis": "semantic_prior_ablation",
                            "variant": variant,
                            "endpoint": protocol,
                            "metric": f"{metric}_full_minus_ablation",
                            "value": effect[f"{metric}_delta_mean_over_seeds"],
                            "ci_low": interval[0],
                            "ci_high": interval[1],
                            "sign_flip_p": effect[
                                f"{metric}_exact_seed_sign_flip_p"
                            ],
                            "source_asset": "semantic_prior_ablation_statistics",
                            "source_sha256": source_hash,
                        }
                    )
    return pd.DataFrame(rows, columns=columns)


def build_data_quality_table(
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "metric",
        "value",
        "source_asset",
        "source_sha256",
    ]
    rows: list[dict[str, object]] = []
    if "dens_feature_missingness" not in sources:
        return pd.DataFrame(rows, columns=columns)
    payload, source_hash = sources["dens_feature_missingness"]
    result = payload["result"]
    dataset_id = str(payload.get("dataset_id") or "ds003751")
    metrics = {
        "eligible_trial_count": result["eligible_trial_count"],
        "feature_available_count": result["feature_available_count"],
        "missing_count": result["missing_count"],
        "missing_fraction": result["missing_fraction"],
        "valence_standardized_mean_difference": result["target_distribution"][
            "target_valence"
        ]["standardized_mean_difference"],
        "arousal_standardized_mean_difference": result["target_distribution"][
            "target_arousal"
        ]["standardized_mean_difference"],
        "subjects_all_missing_count": sum(
            values["missing_fraction"] == 1.0
            for values in result["by_subject"].values()
        ),
        "subjects_no_missing_count": sum(
            values["missing_fraction"] == 0.0
            for values in result["by_subject"].values()
        ),
    }
    for metric, value in metrics.items():
        rows.append(
            {
                "dataset_id": dataset_id,
                "metric": metric,
                "value": value,
                "source_asset": "dens_feature_missingness",
                "source_sha256": source_hash,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_protocol_model_table(
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for asset_name, (payload, source_hash) in sources.items():
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ManuscriptAssetError(f"Asset {asset_name} has no result list")
        for result in results:
            if not isinstance(result, Mapping):
                continue
            protocol = str(result.get("protocol", ""))
            representations = result.get("representations")
            if not isinstance(representations, Mapping):
                continue
            for representation, metrics in representations.items():
                if not isinstance(metrics, Mapping):
                    continue
                label = REPRESENTATION_LABELS.get(
                    str(representation), str(representation)
                )
                rows.append(
                    {
                        "protocol": protocol,
                        "representation": label,
                        "ccc_mean": float(metrics["ccc_mean"]),
                        "ccc_sample_std": float(metrics["ccc_sample_std"]),
                        "mae_mean": float(metrics["mae_mean"]),
                        "mae_sample_std": float(metrics["mae_sample_std"]),
                        "seed_count": int(result.get("seed_count", 0)),
                        "source_asset": asset_name,
                        "source_sha256": source_hash,
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ManuscriptAssetError("Protocol model table has no rows")
    protocol_rank = {name: index for index, name in enumerate(PROTOCOL_ORDER)}
    representation_rank = {
        name: index for index, name in enumerate(REPRESENTATION_ORDER)
    }
    table["_protocol_rank"] = table["protocol"].map(protocol_rank).fillna(999)
    table["_representation_rank"] = (
        table["representation"].map(representation_rank).fillna(999)
    )
    return (
        table.sort_values(
            ["_protocol_rank", "_representation_rank", "source_asset"],
            kind="stable",
        )
        .drop(columns=["_protocol_rank", "_representation_rank"])
        .reset_index(drop=True)
    )


def _probe_records(
    asset_name: str,
    representation: str,
    payload: Mapping[str, object],
    source_hash: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ManuscriptAssetError(f"Identity asset {asset_name} has no result list")
    for result in results:
        if not isinstance(result, Mapping):
            continue
        seed = int(result["seed"])
        for probe in ("dataset", "stimulus", "subject"):
            raw_probe = result.get(f"{probe}_probe")
            if not isinstance(raw_probe, Mapping):
                continue
            if "metrics" in raw_probe:
                datasets: Mapping[str, object] = {"all": raw_probe}
            else:
                datasets = raw_probe
            for dataset_id, details in datasets.items():
                if not isinstance(details, Mapping):
                    continue
                metrics = details.get("metrics")
                permuted = details.get("permuted_label_metrics")
                if not isinstance(metrics, Mapping) or not isinstance(
                    permuted, Mapping
                ):
                    continue
                rows.append(
                    {
                        "representation": representation,
                        "probe": probe,
                        "dataset_id": str(dataset_id),
                        "seed": seed,
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "permuted_balanced_accuracy": float(
                            permuted["balanced_accuracy"]
                        ),
                        "chance_accuracy": float(
                            details["uniform_chance_accuracy"]
                        ),
                        "seen_test_fraction": (
                            float(details["seen_test_fraction"])
                            if "seen_test_fraction" in details
                            else None
                        ),
                        "source_asset": asset_name,
                        "source_sha256": source_hash,
                    }
                )
    return rows


def build_identity_probe_seed_table(
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for asset_name, (payload, source_hash) in sources.items():
        representation = IDENTITY_LABELS.get(asset_name, asset_name)
        records.extend(
            _probe_records(asset_name, representation, payload, source_hash)
        )
    raw = pd.DataFrame(records)
    if raw.empty:
        raise ManuscriptAssetError("Identity probe table has no rows")
    raw["identity_lift"] = (
        raw["balanced_accuracy"] - raw["permuted_balanced_accuracy"]
    )
    representation_rank = {
        name: index
        for index, name in enumerate(
            (
                "DE",
                "CBraMod pretrained",
                "CBraMod random",
                "LaBraM pretrained",
                "LaBraM random",
            )
        )
    }
    probe_rank = {"dataset": 0, "subject": 1, "stimulus": 2}
    raw["_representation_rank"] = (
        raw["representation"].map(representation_rank).fillna(999)
    )
    raw["_probe_rank"] = raw["probe"].map(probe_rank).fillna(999)
    return (
        raw.sort_values(
            ["_probe_rank", "_representation_rank", "dataset_id", "seed"],
            kind="stable",
        )
        .drop(columns=["_probe_rank", "_representation_rank"])
        .reset_index(drop=True)
    )


def build_identity_probe_table(
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> pd.DataFrame:
    raw = build_identity_probe_seed_table(sources)
    grouped = raw.groupby(
        [
            "representation",
            "probe",
            "dataset_id",
            "source_asset",
            "source_sha256",
        ],
        sort=False,
        dropna=False,
    )
    table = grouped.agg(
        seed_count=("seed", "nunique"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sample_std=("balanced_accuracy", "std"),
        permuted_balanced_accuracy_mean=("permuted_balanced_accuracy", "mean"),
        chance_accuracy=("chance_accuracy", "mean"),
        seen_test_fraction=("seen_test_fraction", "mean"),
    ).reset_index()
    representation_rank = {
        name: index
        for index, name in enumerate(
            (
                "DE",
                "CBraMod pretrained",
                "CBraMod random",
                "LaBraM pretrained",
                "LaBraM random",
            )
        )
    }
    probe_rank = {"dataset": 0, "subject": 1, "stimulus": 2}
    table["_representation_rank"] = (
        table["representation"].map(representation_rank).fillna(999)
    )
    table["_probe_rank"] = table["probe"].map(probe_rank).fillna(999)
    return (
        table.sort_values(
            ["_probe_rank", "_representation_rank", "dataset_id"], kind="stable"
        )
        .drop(columns=["_probe_rank", "_representation_rank"])
        .reset_index(drop=True)
    )


def build_representation_comparison_table(
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for asset_name, (payload, source_hash) in sources.items():
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ManuscriptAssetError(f"Asset {asset_name} has no result list")
        for result in results:
            if not isinstance(result, Mapping):
                continue
            comparisons = result.get("comparisons")
            if not isinstance(comparisons, Mapping):
                continue
            for comparison, statistics in comparisons.items():
                if not isinstance(statistics, Mapping):
                    continue
                ccc_interval = statistics.get("ccc_delta_cluster_bootstrap_95ci")
                mae_interval = statistics.get("mae_delta_cluster_bootstrap_95ci")
                if not (
                    isinstance(ccc_interval, list)
                    and len(ccc_interval) == 2
                    and isinstance(mae_interval, list)
                    and len(mae_interval) == 2
                ):
                    raise ManuscriptAssetError(
                        f"Comparison {comparison} in {asset_name} lacks paired intervals"
                    )
                rows.append(
                    {
                        "protocol": str(result.get("protocol", "")),
                        "comparison": str(comparison),
                        "ccc_delta_mean": float(
                            statistics["ccc_delta_mean_over_seeds"]
                        ),
                        "ccc_ci_low": float(ccc_interval[0]),
                        "ccc_ci_high": float(ccc_interval[1]),
                        "ccc_sign_flip_p": float(
                            statistics["ccc_exact_seed_sign_flip_p"]
                        ),
                        "mae_delta_mean": float(
                            statistics["mae_delta_mean_over_seeds"]
                        ),
                        "mae_ci_low": float(mae_interval[0]),
                        "mae_ci_high": float(mae_interval[1]),
                        "mae_sign_flip_p": float(
                            statistics["mae_exact_seed_sign_flip_p"]
                        ),
                        "source_asset": asset_name,
                        "source_sha256": source_hash,
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        columns = [
            "protocol",
            "comparison",
            "ccc_delta_mean",
            "ccc_ci_low",
            "ccc_ci_high",
            "ccc_sign_flip_p",
            "mae_delta_mean",
            "mae_ci_low",
            "mae_ci_high",
            "mae_sign_flip_p",
            "source_asset",
            "source_sha256",
        ]
        return pd.DataFrame(columns=columns)
    protocol_rank = {name: index for index, name in enumerate(PROTOCOL_ORDER)}
    table["_protocol_rank"] = table["protocol"].map(protocol_rank).fillna(999)
    return (
        table.sort_values(["_protocol_rank", "comparison"], kind="stable")
        .drop(columns="_protocol_rank")
        .reset_index(drop=True)
    )


def build_external_validation_table(
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> pd.DataFrame:
    """Normalize external-validation summaries into one claim-ready table."""
    columns = [
        "analysis",
        "protocol",
        "feature_view",
        "feature_count",
        "labeled_trial_count",
        "model",
        "comparison",
        "condition",
        "target",
        "metric",
        "seed_count",
        "value_mean",
        "value_sample_std",
        "permuted_value_mean",
        "permuted_value_sample_std",
        "chance_value",
        "ci_low",
        "ci_high",
        "sign_flip_p",
        "evaluation_row_count",
        "subject_count",
        "source_asset",
        "source_sha256",
    ]
    rows: list[dict[str, object]] = []

    def empty_row(asset_name: str, source_hash: str) -> dict[str, object]:
        row = {column: None for column in columns}
        row.update({"source_asset": asset_name, "source_sha256": source_hash})
        return row

    for asset_name, (payload, source_hash) in sources.items():
        results = payload.get("results")
        if isinstance(results, list) and results and isinstance(results[0], Mapping):
            first = results[0]
            if isinstance(first.get("models"), Mapping):
                seed_rows: list[dict[str, object]] = []
                for result in results:
                    if not isinstance(result, Mapping):
                        continue
                    models = result.get("models")
                    if not isinstance(models, Mapping):
                        continue
                    for model, targets in models.items():
                        if not isinstance(targets, Mapping):
                            continue
                        for target, metrics in targets.items():
                            if not isinstance(metrics, Mapping):
                                continue
                            for metric, value in metrics.items():
                                seed_rows.append(
                                    {
                                        "protocol": str(
                                            result.get("protocol", "subject_holdout")
                                        ),
                                        "feature_view": str(
                                            result.get(
                                                "representation",
                                                payload.get("feature_view", ""),
                                            )
                                        ),
                                        "feature_count": int(
                                            result.get(
                                                "feature_count",
                                                payload.get("feature_count", 0),
                                            )
                                        ),
                                        "model": str(model),
                                        "target": str(target),
                                        "metric": str(metric),
                                        "seed": int(result["seed"]),
                                        "value": float(value),
                                    }
                                )
                seed_table = pd.DataFrame(seed_rows)
                if not seed_table.empty:
                    grouped = seed_table.groupby(
                        [
                            "protocol",
                            "feature_view",
                            "feature_count",
                            "model",
                            "target",
                            "metric",
                        ],
                        sort=True,
                        dropna=False,
                    )
                    for key, values in grouped:
                        protocol, feature_view, feature_count, model, target, metric = key
                        row = empty_row(asset_name, source_hash)
                        row.update(
                            {
                                "analysis": "affect_prediction",
                                "protocol": protocol,
                                "feature_view": feature_view,
                                "feature_count": feature_count,
                                "labeled_trial_count": int(
                                    payload.get(
                                        "available_labeled_feature_trials",
                                        payload.get("trial_count", 0),
                                    )
                                ),
                                "model": model,
                                "target": target,
                                "metric": metric,
                                "seed_count": int(values["seed"].nunique()),
                                "value_mean": float(values["value"].mean()),
                                "value_sample_std": float(values["value"].std()),
                            }
                        )
                        rows.append(row)

            if "uniform_chance_accuracy" in payload and "metrics" in first:
                observed = np.asarray(
                    [
                        float(result["metrics"]["balanced_accuracy"])
                        for result in results
                        if isinstance(result, Mapping)
                    ]
                )
                permuted = np.asarray(
                    [
                        float(
                            result["permuted_label_metrics"]["balanced_accuracy"]
                        )
                        for result in results
                        if isinstance(result, Mapping)
                    ]
                )
                row = empty_row(asset_name, source_hash)
                row.update(
                    {
                        "analysis": "condition_probe",
                        "protocol": "subject_holdout",
                        "feature_view": str(payload.get("feature_view", "")),
                        "feature_count": int(payload.get("feature_count", 0)),
                        "metric": "balanced_accuracy",
                        "seed_count": len(observed),
                        "value_mean": float(observed.mean()),
                        "value_sample_std": float(observed.std(ddof=1)),
                        "permuted_value_mean": float(permuted.mean()),
                        "permuted_value_sample_std": float(permuted.std(ddof=1)),
                        "chance_value": float(payload["uniform_chance_accuracy"]),
                    }
                )
                rows.append(row)
                continue

            if isinstance(first.get("comparisons"), Mapping):
                for result in results:
                    if not isinstance(result, Mapping):
                        continue
                    comparisons = result.get("comparisons")
                    if not isinstance(comparisons, Mapping):
                        continue
                    for comparison, statistics in comparisons.items():
                        if not isinstance(statistics, Mapping):
                            continue
                        for metric_prefix in ("ccc", "mae"):
                            interval = statistics[
                                f"{metric_prefix}_delta_cluster_bootstrap_95ci"
                            ]
                            row = empty_row(asset_name, source_hash)
                            row.update(
                                {
                                    "analysis": "paired_comparison",
                                    "protocol": str(result.get("protocol", "")),
                                    "comparison": str(comparison),
                                    "metric": f"{metric_prefix}_delta",
                                    "seed_count": int(result.get("seed_count", 0)),
                                    "value_mean": float(
                                        statistics[
                                            f"{metric_prefix}_delta_mean_over_seeds"
                                        ]
                                    ),
                                    "ci_low": float(interval[0]),
                                    "ci_high": float(interval[1]),
                                    "sign_flip_p": float(
                                        statistics[
                                            f"{metric_prefix}_exact_seed_sign_flip_p"
                                        ]
                                    ),
                                    "evaluation_row_count": int(
                                        result.get("row_count", 0)
                                    ),
                                }
                            )
                            rows.append(row)
                continue

        identity_probes = payload.get("identity_probes")
        if isinstance(identity_probes, list):
            feature_counts = {
                (
                    str(result.get("protocol", "")),
                    int(result.get("seed", -1)),
                    str(result.get("representation", "")),
                ): int(result.get("feature_count", 0))
                for result in payload.get("results", [])
                if isinstance(result, Mapping)
            }
            probe_table = pd.DataFrame(
                [
                    {
                        "protocol": str(result.get("protocol", "")),
                        "seed": int(result["seed"]),
                        "feature_view": str(result.get("representation", "")),
                        "model": str(result.get("probe", "identity_probe")),
                        "value": float(result["balanced_accuracy"]),
                        "permuted": float(result["permuted_balanced_accuracy"]),
                        "chance": float(result["uniform_chance_accuracy"]),
                    }
                    for result in identity_probes
                    if isinstance(result, Mapping)
                ]
            )
            if not probe_table.empty:
                grouped = probe_table.groupby(
                    ["protocol", "feature_view", "model"],
                    sort=True,
                    dropna=False,
                )
                for key, values in grouped:
                    protocol, feature_view, probe = key
                    first_seed = int(values["seed"].iloc[0])
                    row = empty_row(asset_name, source_hash)
                    row.update(
                        {
                            "analysis": "identity_probe",
                            "protocol": protocol,
                            "feature_view": feature_view,
                            "feature_count": feature_counts.get(
                                (protocol, first_seed, feature_view), 0
                            ),
                            "labeled_trial_count": int(payload.get("trial_count", 0)),
                            "model": probe,
                            "metric": "balanced_accuracy",
                            "seed_count": int(values["seed"].nunique()),
                            "value_mean": float(values["value"].mean()),
                            "value_sample_std": float(values["value"].std()),
                            "permuted_value_mean": float(values["permuted"].mean()),
                            "permuted_value_sample_std": float(
                                values["permuted"].std()
                            ),
                            "chance_value": float(values["chance"].mean()),
                        }
                    )
                    rows.append(row)

        conditions = payload.get("conditions")
        if isinstance(conditions, Mapping):
            for condition, details in conditions.items():
                if not isinstance(details, Mapping):
                    continue
                comparisons = details.get("comparisons")
                if not isinstance(comparisons, Mapping):
                    continue
                for comparison, statistics in comparisons.items():
                    if not isinstance(statistics, Mapping):
                        continue
                    interval = statistics["cluster_bootstrap_95ci"]
                    row = empty_row(asset_name, source_hash)
                    row.update(
                        {
                            "analysis": "condition_effect",
                            "protocol": "subject_holdout",
                            "comparison": str(comparison),
                            "condition": str(condition),
                            "metric": "mae_delta",
                            "seed_count": int(payload.get("seed_count", 0)),
                            "value_mean": float(
                                statistics["mae_delta_candidate_minus_baseline"]
                            ),
                            "ci_low": float(interval[0]),
                            "ci_high": float(interval[1]),
                            "sign_flip_p": float(
                                statistics["exact_seed_sign_flip_p"]
                            ),
                            "evaluation_row_count": int(
                                details.get("row_count", 0)
                            ),
                            "subject_count": int(details.get("subject_count", 0)),
                        }
                    )
                    rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            [
                "analysis",
                "feature_view",
                "model",
                "comparison",
                "condition",
                "target",
                "metric",
            ],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_benchmark_scope_table(
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> pd.DataFrame:
    """Flatten the trial-derived benchmark scope summary for tables and claims."""
    columns = [
        "scope_type",
        "dataset_id",
        "name",
        "tier",
        "dataset_count",
        "participant_count",
        "trial_count",
        "stimulus_count",
        "labeled_trial_count",
        "stimulus_identity_status",
        "trial_table_path",
        "trial_table_sha256",
        "source_asset",
        "source_sha256",
    ]
    rows: list[dict[str, object]] = []
    for asset_name, (payload, source_hash) in sources.items():
        datasets = payload.get("datasets")
        if not isinstance(datasets, list):
            raise ManuscriptAssetError(f"Scope asset {asset_name} has no datasets")
        for dataset in datasets:
            if not isinstance(dataset, Mapping):
                continue
            rows.append(
                {
                    "scope_type": "dataset",
                    "dataset_id": str(dataset["dataset_id"]),
                    "name": str(dataset.get("name", "")),
                    "tier": str(dataset.get("tier", "")),
                    "dataset_count": 1,
                    "participant_count": int(dataset["participant_count"]),
                    "trial_count": int(dataset["trial_count"]),
                    "stimulus_count": dataset.get("stimulus_count"),
                    "labeled_trial_count": int(dataset["labeled_trial_count"]),
                    "stimulus_identity_status": str(
                        dataset.get("stimulus_identity_status", "")
                    ),
                    "trial_table_path": str(dataset.get("source_path", "")),
                    "trial_table_sha256": str(dataset.get("source_sha256", "")),
                    "source_asset": asset_name,
                    "source_sha256": source_hash,
                }
            )
        groups = payload.get("groups")
        if not isinstance(groups, Mapping):
            raise ManuscriptAssetError(f"Scope asset {asset_name} has no groups")
        for group_name, group in sorted(groups.items()):
            if not isinstance(group, Mapping):
                continue
            rows.append(
                {
                    "scope_type": "group",
                    "dataset_id": str(group_name),
                    "name": str(group_name),
                    "tier": str(group_name),
                    "dataset_count": int(group["dataset_count"]),
                    "participant_count": int(group["participant_count"]),
                    "trial_count": int(group["trial_count"]),
                    "stimulus_count": group.get("stimulus_count"),
                    "labeled_trial_count": int(group["labeled_trial_count"]),
                    "stimulus_identity_status": "",
                    "trial_table_path": "",
                    "trial_table_sha256": "",
                    "source_asset": asset_name,
                    "source_sha256": source_hash,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["scope_type", "dataset_id"], kind="stable"
    ).reset_index(drop=True)


def resolve_claims(
    claims: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    raw_claims = claims.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ManuscriptAssetError("Claim configuration requires a non-empty list")
    rows: list[dict[str, object]] = []
    for raw in raw_claims:
        if not isinstance(raw, Mapping):
            raise ManuscriptAssetError("Each claim must be a mapping")
        claim_id = str(raw.get("claim_id", ""))
        table_name = str(raw.get("table", ""))
        if table_name not in tables:
            raise ManuscriptAssetError(f"Claim {claim_id} references unknown table")
        selected = tables[table_name]
        where = raw.get("where", {})
        if not isinstance(where, Mapping):
            raise ManuscriptAssetError(f"Claim {claim_id} selector must be a mapping")
        for column, value in where.items():
            if column not in selected.columns:
                raise ManuscriptAssetError(
                    f"Claim {claim_id} selector references unknown column {column}"
                )
            selected = selected.loc[selected[str(column)].eq(value)]
        if selected.empty:
            raise ManuscriptAssetError(f"Claim {claim_id} selector matched no rows")
        value_column = str(raw.get("value_column", ""))
        if value_column not in selected.columns:
            raise ManuscriptAssetError(
                f"Claim {claim_id} references unknown value column {value_column}"
            )
        values = pd.to_numeric(selected[value_column], errors="raise")
        reduction = str(raw.get("reduction", "first"))
        if reduction == "first":
            if len(values) != 1:
                raise ManuscriptAssetError(
                    f"Claim {claim_id} expected one row, matched {len(values)}"
                )
            value = float(values.iloc[0])
        elif reduction == "mean":
            value = float(values.mean())
        else:
            raise ManuscriptAssetError(
                f"Claim {claim_id} has unknown reduction: {reduction}"
            )
        if not math.isfinite(value):
            raise ManuscriptAssetError(f"Claim {claim_id} resolved non-finite value")
        expected = float(raw["expected"])
        tolerance = float(raw.get("tolerance", 1e-12))
        if abs(value - expected) > tolerance:
            raise ManuscriptAssetError(
                f"Claim {claim_id} numeric drift: expected {expected}, observed {value}"
            )
        decimals = int(raw.get("decimals", 3))
        source_assets = sorted(
            set(selected.get("source_asset", pd.Series(dtype=str)).astype(str))
        )
        source_hashes = sorted(
            set(selected.get("source_sha256", pd.Series(dtype=str)).astype(str))
        )
        rows.append(
            {
                "claim_id": claim_id,
                "claim": str(raw.get("claim", "")),
                "numeric_value": value,
                "display_value": f"{value:.{decimals}f}",
                "table": table_name,
                "selector": "; ".join(f"{key}={where[key]}" for key in sorted(where)),
                "source_assets": ";".join(source_assets),
                "source_sha256": ";".join(source_hashes),
                "allowed_wording": str(raw.get("allowed_wording", "")),
            }
        )
    return pd.DataFrame(rows).sort_values("claim_id", kind="stable").reset_index(
        drop=True
    )


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.transparent": False,
        }
    )


def _save_publication_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=300,
        facecolor="white",
        metadata={"Software": "OpenAffect-EEG"},
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        facecolor="white",
        metadata={
            "Creator": "OpenAffect-EEG",
            "Producer": "OpenAffect-EEG",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def plot_benchmark_overview(output_prefix: Path) -> None:
    """Render the benchmark question, data flow, and isolation protocols."""

    _publication_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ink = "#24313A"
    muted = "#5C6B73"
    blue = "#DCECF4"
    blue_dark = "#2A6F97"
    amber = "#F5E6BE"
    amber_dark = "#B7791F"
    green = "#DDECDD"
    green_dark = "#3A7D44"
    rose = "#F2DEDA"
    rose_dark = "#B64B3C"
    paper = "#F7F8F8"

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        face: str,
        edge: str,
        *,
        size: float = 7.0,
        weight: str = "normal",
        align: str = "center",
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.035,rounding_size=0.06",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.75,
        )
        ax.add_patch(patch)
        text_x = x + width / 2 if align == "center" else x + 0.14
        ax.text(
            text_x,
            y + height / 2,
            label,
            ha=align,
            va="center",
            fontsize=size,
            fontweight=weight,
            color=ink,
            linespacing=1.18,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color=muted,
                shrinkA=2,
                shrinkB=2,
            )
        )

    ax.text(1.43, 5.64, "A  Source domains", ha="center", fontsize=8.0, fontweight="bold", color=ink)
    datasets = (
        ("MusicEEG", "music | 31 participants", blue, blue_dark),
        ("DENS", "natural videos | 40", amber, amber_dark),
        ("EmoEEG-MC", "video + imagery | 60", green, green_dark),
        ("ds006866", "image regulation | 148", rose, rose_dark),
    )
    for index, (name, detail, face, edge) in enumerate(datasets):
        y = 4.62 - index * 0.92
        box(0.15, y, 2.55, 0.67, f"{name}\n{detail}", face, edge, size=5.5, align="left")

    arrow(2.78, 3.13, 3.47, 3.13)
    ax.text(5.03, 5.64, "B  Trial contract", ha="center", fontsize=8.0, fontweight="bold", color=ink)
    box(3.47, 1.12, 3.13, 4.15, "", paper, "#9AA7AD")
    ax.text(3.68, 4.82, "Identity", fontsize=6.8, fontweight="bold", color=ink)
    ax.text(3.68, 4.47, "subject_uid\nstimulus_uid\ndataset_id\ncontext", fontsize=5.8, color=muted, va="top", linespacing=1.35)
    ax.plot([5.00, 5.00], [1.42, 4.82], color="#C9D0D4", linewidth=0.7)
    ax.text(5.20, 4.82, "Affect + provenance", fontsize=5.8, fontweight="bold", color=ink)
    ax.text(5.20, 4.47, "valence / arousal\nsource scale\nfeature mask\nsource hashes", fontsize=5.8, color=muted, va="top", linespacing=1.35)
    box(3.70, 1.40, 2.66, 0.64, "Split trials\nbefore windowing", "#FFFFFF", ink, size=5.4, weight="bold")

    arrow(6.70, 3.13, 7.37, 3.13)
    ax.text(8.65, 5.64, "C  Isolation protocols", ha="center", fontsize=8.0, fontweight="bold", color=ink)
    protocols = (
        ("Unseen participant", "subject holdout", blue, blue_dark),
        ("Unseen elicitor", "stimulus holdout", amber, amber_dark),
        ("Both unseen", "joint cold start", green, green_dark),
        ("Unseen domain", "context / dataset", rose, rose_dark),
    )
    for index, (claim, protocol, face, edge) in enumerate(protocols):
        y = 4.62 - index * 0.92
        box(7.37, y, 2.55, 0.67, f"{claim}\n{protocol}", face, edge, size=5.9, align="left")

    arrow(10.02, 3.13, 10.70, 3.13)
    ax.text(12.25, 5.64, "D  Audit targets", ha="center", fontsize=8.0, fontweight="bold", color=ink)
    box(10.70, 4.22, 3.10, 0.95, "Stimulus semantics\ntraining-only prior", amber, amber_dark, size=6.4, weight="bold")
    box(10.70, 2.90, 3.10, 0.95, "Experienced-affect residual\nEEG-only test branch", blue, blue_dark, size=5.8, weight="bold")
    box(10.70, 1.58, 3.10, 0.95, "Identity shortcuts\nsubject / stimulus\n/ dataset", rose, rose_dark, size=5.8, weight="bold")
    ax.text(
        12.25,
        0.68,
        r"$\hat{y}_{i,s}=f_{\mathrm{stim}}(s)+f_{\mathrm{EEG}}(e_{i,s})$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=ink,
    )

    _save_publication_figure(fig, output_prefix)


def plot_protocol_audit(
    table: pd.DataFrame,
    output_prefix: Path,
    comparisons: pd.DataFrame | None = None,
    benchmark_metrics: pd.DataFrame | None = None,
) -> None:
    required = {
        "protocol",
        "representation",
        "ccc_mean",
        "ccc_sample_std",
    }
    missing = required - set(table.columns)
    if missing:
        raise ManuscriptAssetError(f"Protocol plot is missing columns: {sorted(missing)}")
    _publication_style()
    protocols = [item for item in PROTOCOL_ORDER if item in set(table["protocol"])]
    short_protocol_labels = {
        "trial_random_holdout": "Trial\nrandom",
        "stimulus_holdout": "Unseen\nstim.",
        "subject_holdout": "Unseen\nsubj.",
        "subject_stimulus_holdout": "Joint\nunseen",
        "imagery_to_video": "Imagery\nto video",
        "video_to_imagery": "Video to\nimagery",
    }
    heatmap_protocol_labels = {
        "trial_random_holdout": "Random",
        "stimulus_holdout": "Stim.",
        "subject_holdout": "Subj.",
        "subject_stimulus_holdout": "Joint",
        "imagery_to_video": "I->V",
        "video_to_imagery": "V->I",
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 3.0),
        gridspec_kw={"width_ratios": [1.18, 1.08, 1.0]},
        constrained_layout=True,
    )

    axis = axes[0]
    decomposition_protocols = [
        item
        for item in (
            "trial_random_holdout",
            "stimulus_holdout",
            "subject_holdout",
            "subject_stimulus_holdout",
        )
        if item in protocols
    ]
    model_specs = (
        ("direct_eeg", "EEG only", "#6F777B", "o"),
        ("semantic_stimulus_prior", "Stimulus prior", "#D99A2B", "s"),
        (
            "disentangled_prior_plus_eeg_residual",
            "Prior + EEG residual",
            "#3A7D44",
            "^",
        ),
    )
    if benchmark_metrics is not None:
        selected = benchmark_metrics.loc[
            benchmark_metrics["source_asset"].eq("de_core_summary")
            & benchmark_metrics["protocol"].isin(decomposition_protocols)
            & benchmark_metrics["model"].isin([item[0] for item in model_specs])
            & benchmark_metrics["context"].eq("all")
            & benchmark_metrics["target"].eq("macro")
            & benchmark_metrics["metric"].eq("ccc")
        ].copy()
    else:
        selected = pd.DataFrame()
    x = np.arange(len(decomposition_protocols), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(model_specs))
    for offset, (model, label, color, marker) in zip(offsets, model_specs):
        subset = selected.loc[selected["model"].eq(model)].set_index("protocol")
        if not set(decomposition_protocols).issubset(subset.index):
            continue
        values = subset.loc[decomposition_protocols, "value_mean"].to_numpy(float)
        errors = subset.loc[
            decomposition_protocols, "value_sample_std"
        ].to_numpy(float)
        axis.errorbar(
            x + offset,
            values,
            yerr=errors,
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=4.2,
            elinewidth=0.8,
            capsize=2,
            label=label,
            zorder=3,
        )
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.set_xticks(
        x, [short_protocol_labels[item] for item in decomposition_protocols]
    )
    axis.set_ylim(-0.04, 0.72)
    axis.set_ylabel("Macro CCC", fontsize=6.5)
    axis.set_title(
        "A  Signal decomposition", loc="left", fontweight="bold", fontsize=8.2
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles,
            labels,
            frameon=False,
            fontsize=5.7,
            loc="upper right",
            handletextpad=0.35,
        )

    axis = axes[1]
    matrix = (
        table.pivot(index="representation", columns="protocol", values="ccc_mean")
        .reindex(index=REPRESENTATION_ORDER, columns=protocols)
        .to_numpy(float)
    )
    image = axis.imshow(matrix, cmap="cividis", vmin=0.28, vmax=0.65, aspect="auto")
    axis.set_xticks(
        np.arange(len(protocols)),
        [heatmap_protocol_labels[item] for item in protocols],
        fontsize=5.6,
        rotation=35,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(REPRESENTATION_ORDER)),
        ("DE", "CBraMod", "LaBraM pre.", "LaBraM rand."),
        fontsize=5.8,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.2,
                color="white" if value < 0.39 or value > 0.57 else "#182027",
            )
    axis.set_title(
        "B  Protocol sensitivity", loc="left", fontweight="bold", fontsize=8.2
    )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    colorbar.set_label("Macro CCC", fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5, length=2)

    axis = axes[2]
    if comparisons is None:
        forest = pd.DataFrame()
    else:
        forest = comparisons.loc[
            comparisons["comparison"].eq(
                "labram_pretrained_vs_labram_random"
            )
            & comparisons["protocol"].isin(protocols)
        ].set_index("protocol")
    forest_protocols = [item for item in protocols if item in forest.index]
    positions = np.arange(len(forest_protocols), dtype=float)
    if forest_protocols:
        centers = forest.loc[forest_protocols, "ccc_delta_mean"].to_numpy(float)
        lows = forest.loc[forest_protocols, "ccc_ci_low"].to_numpy(float)
        highs = forest.loc[forest_protocols, "ccc_ci_high"].to_numpy(float)
        axis.errorbar(
            centers,
            positions,
            xerr=[centers - lows, highs - centers],
            fmt="o",
            color="#3A7D44",
            markerfacecolor="#3A7D44",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=4.2,
            elinewidth=1.0,
            capsize=2.2,
            zorder=3,
        )
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axis.set_yticks(
        positions,
        [short_protocol_labels[item].replace("\n", " ") for item in forest_protocols],
        fontsize=5.8,
    )
    axis.invert_yaxis()
    axis.set_xlabel("CCC delta (pretrained - random)", fontsize=6.3)
    axis.set_title(
        "C  Pretraining gain", loc="left", fontweight="bold", fontsize=8.2
    )
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.5, zorder=0)

    _save_publication_figure(fig, output_prefix)


def plot_identity_audit(
    table: pd.DataFrame,
    output_prefix: Path,
    seed_table: pd.DataFrame | None = None,
) -> None:
    required = {
        "representation",
        "probe",
        "dataset_id",
        "balanced_accuracy_mean",
        "balanced_accuracy_sample_std",
        "permuted_balanced_accuracy_mean",
        "chance_accuracy",
    }
    missing = required - set(table.columns)
    if missing:
        raise ManuscriptAssetError(f"Identity plot is missing columns: {sorted(missing)}")
    if seed_table is None:
        seed_table = table.rename(
            columns={
                "balanced_accuracy_mean": "balanced_accuracy",
                "permuted_balanced_accuracy_mean": "permuted_balanced_accuracy",
            }
        ).copy()
        seed_table["seed"] = 0
        seed_table["identity_lift"] = (
            seed_table["balanced_accuracy"]
            - seed_table["permuted_balanced_accuracy"]
        )
    seed_required = {
        "representation",
        "probe",
        "dataset_id",
        "seed",
        "balanced_accuracy",
        "permuted_balanced_accuracy",
        "chance_accuracy",
        "identity_lift",
    }
    seed_missing = seed_required - set(seed_table.columns)
    if seed_missing:
        raise ManuscriptAssetError(
            f"Identity seed plot is missing columns: {sorted(seed_missing)}"
        )

    seed_table = seed_table.loc[
        (
            seed_table["probe"].eq("dataset")
            & seed_table["dataset_id"].eq("all")
        )
        | (
            seed_table["probe"].isin(("subject", "stimulus"))
            & seed_table["dataset_id"].eq("ds005540")
        )
    ].copy()
    row_order = (
        ("dataset", "DE"),
        ("subject", "LaBraM pretrained"),
        ("subject", "LaBraM random"),
        ("stimulus", "DE"),
        ("stimulus", "CBraMod pretrained"),
        ("stimulus", "CBraMod random"),
        ("stimulus", "LaBraM pretrained"),
        ("stimulus", "LaBraM random"),
    )
    available_rows = {
        (str(row.probe), str(row.representation))
        for row in seed_table[["probe", "representation"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    row_order = tuple(item for item in row_order if item in available_rows)

    _publication_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.0),
        gridspec_kw={"width_ratios": [1.22, 1.0]},
        constrained_layout=True,
    )
    short_representation = {
        "DE": "DE",
        "CBraMod pretrained": "CBraMod pre.",
        "CBraMod random": "CBraMod rand.",
        "LaBraM pretrained": "LaBraM pre.",
        "LaBraM random": "LaBraM rand.",
    }

    group_colors = {
        "dataset": "#D55E00",
        "subject": "#0072B2",
        "stimulus": "#009E73",
    }
    group_labels = {
        "dataset": "Dataset",
        "subject": "Participant",
        "stimulus": "Stimulus",
    }
    group_positions = {"dataset": 0.0, "subject": 1.45, "stimulus": 3.85}
    group_counts = {"dataset": 0, "subject": 0, "stimulus": 0}
    positions: list[float] = []
    labels: list[str] = []
    for probe, representation in row_order:
        position = group_positions[probe] + group_counts[probe]
        group_counts[probe] += 1
        positions.append(position)
        labels.append(
            f"{group_labels[probe]}  |  "
            f"{short_representation.get(representation, representation)}"
        )
        selected = seed_table.loc[
            seed_table["probe"].eq(probe)
            & seed_table["representation"].eq(representation)
        ].sort_values("seed", kind="stable")
        offsets = np.linspace(-0.11, 0.11, len(selected))
        color = group_colors[probe]
        axes[0].scatter(
            selected["identity_lift"],
            position + offsets,
            s=11,
            color=color,
            alpha=0.58,
            linewidth=0,
            zorder=3,
        )
        center = float(selected["identity_lift"].mean())
        spread = float(selected["identity_lift"].std(ddof=1))
        if math.isnan(spread):
            spread = 0.0
        axes[0].errorbar(
            center,
            position,
            xerr=spread,
            fmt="D",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=4.4,
            elinewidth=1.0,
            capsize=2.2,
            zorder=4,
        )

    axes[0].axvline(0.0, color="#4D565A", linestyle="--", linewidth=0.8)
    axes[0].set_xlim(-0.065, 1.02)
    axes[0].set_yticks(positions, labels, fontsize=5.7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Identity lift (observed - permuted BA)", fontsize=6.5)
    axes[0].set_title(
        "A  Shortcut effect size", loc="left", fontweight="bold", fontsize=8.2
    )
    axes[0].text(
        0.0,
        1.005,
        "individual seeds; diamond = mean +/- SD",
        transform=axes[0].transAxes,
        fontsize=5.7,
        color="#5C6B73",
        va="bottom",
    )
    axes[0].grid(axis="x", color="#D9D9D9", linewidth=0.5, zorder=0)
    axes[0].tick_params(axis="x", labelsize=6)

    stimulus_order = [
        representation
        for probe, representation in row_order
        if probe == "stimulus"
    ]
    stimulus_positions = np.arange(len(stimulus_order), dtype=float)
    max_raw = 0.06
    for position, representation in zip(stimulus_positions, stimulus_order):
        selected = seed_table.loc[
            seed_table["probe"].eq("stimulus")
            & seed_table["representation"].eq(representation)
        ].sort_values("seed", kind="stable")
        offsets = np.linspace(-0.12, 0.12, len(selected))
        y_values = position + offsets
        observed = selected["balanced_accuracy"].to_numpy(float)
        permuted = selected["permuted_balanced_accuracy"].to_numpy(float)
        max_raw = max(max_raw, float(np.max(np.r_[observed, permuted])))
        for y_value, observed_value, permuted_value in zip(
            y_values, observed, permuted
        ):
            axes[1].plot(
                [permuted_value, observed_value],
                [y_value, y_value],
                color="#AAB4B8",
                linewidth=0.55,
                alpha=0.7,
                zorder=1,
            )
        axes[1].scatter(
            observed,
            y_values,
            s=12,
            color="#0072B2",
            linewidth=0,
            alpha=0.72,
            zorder=3,
        )
        axes[1].scatter(
            permuted,
            y_values,
            s=15,
            facecolor="white",
            edgecolor="#D55E00",
            linewidth=0.75,
            alpha=0.88,
            zorder=3,
        )
        axes[1].scatter(
            [float(np.mean(observed))],
            [position],
            marker="D",
            s=24,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axes[1].scatter(
            [float(np.mean(permuted))],
            [position],
            marker="D",
            s=24,
            facecolor="white",
            edgecolor="#D55E00",
            linewidth=0.9,
            zorder=4,
        )
    chance = float(
        seed_table.loc[
            seed_table["probe"].eq("stimulus"), "chance_accuracy"
        ].mean()
    )
    axes[1].axvline(
        chance,
        color="#4D565A",
        linestyle=":",
        linewidth=1.0,
        zorder=2,
    )
    axes[1].set_xlim(0.0, max(0.065, max_raw + 0.004))
    axes[1].set_yticks(
        stimulus_positions,
        [short_representation.get(item, item) for item in stimulus_order],
        fontsize=5.8,
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Balanced accuracy", fontsize=6.5)
    axes[1].set_title(
        "B  Stimulus identity (zoom)",
        loc="left",
        fontweight="bold",
        fontsize=8.2,
    )
    axes[1].text(
        0.0,
        1.005,
        "raw scale; unseen participants",
        transform=axes[1].transAxes,
        fontsize=5.7,
        color="#5C6B73",
        va="bottom",
    )
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.5, zorder=0)
    axes[1].tick_params(axis="x", labelsize=6)
    axes[1].legend(
        handles=(
            Line2D(
                [0], [0], marker="o", linestyle="none", color="#0072B2",
                markerfacecolor="#0072B2", markersize=3.8, label="Observed",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", color="#D55E00",
                markerfacecolor="white", markersize=3.8, label="Permuted",
            ),
            Line2D(
                [0], [0], linestyle=":", color="#4D565A", linewidth=1.0,
                label="Chance",
            ),
        ),
        frameon=False,
        fontsize=5.5,
        loc="lower right",
        handletextpad=0.35,
    )

    _save_publication_figure(fig, output_prefix)


def _load_summary_sources(
    registry: ResultRegistry,
    names: set[str],
) -> dict[str, tuple[Mapping[str, object], str]]:
    sources: dict[str, tuple[Mapping[str, object], str]] = {}
    for asset in registry.assets:
        if asset.name not in names:
            continue
        payload = json.loads(asset.path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ManuscriptAssetError(f"Summary {asset.name} is not a JSON object")
        sources[asset.name] = (payload, asset.sha256)
    return sources


def _benchmark_metric_summary(
    benchmark_metrics: Path,
    registry: ResultRegistry,
) -> pd.DataFrame:
    table = pd.read_csv(benchmark_metrics)
    columns = ["asset", "protocol", "model", "context", "target", "metric"]
    output_columns = [
        "source_asset",
        "protocol",
        "model",
        "context",
        "target",
        "metric",
        "seed_count",
        "value_mean",
        "value_sample_std",
        "source_sha256",
    ]
    if table.empty:
        return pd.DataFrame(columns=output_columns)
    summary = (
        table.groupby(columns, sort=True, dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            value_mean=("value", "mean"),
            value_sample_std=("value", "std"),
        )
        .reset_index()
        .rename(columns={"asset": "source_asset"})
    )
    hashes = {asset.name: asset.sha256 for asset in registry.assets}
    summary["source_sha256"] = summary["source_asset"].map(hashes)
    return summary[output_columns]


def _write_table(table: pd.DataFrame, path: Path, separator: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path,
        index=False,
        sep=separator,
        lineterminator="\n",
        float_format="%.12g",
    )


def write_claim_value_macros(ledger: pd.DataFrame, path: Path) -> None:
    """Write deterministic LaTeX accessors for hash-verified claim values."""
    required = {"claim_id", "display_value"}
    if not required.issubset(ledger.columns):
        missing = ", ".join(sorted(required - set(ledger.columns)))
        raise ManuscriptAssetError(f"Claim ledger lacks LaTeX macro columns: {missing}")

    lines = [
        "% Generated from claim_ledger.tsv; do not edit by hand.",
        r"\providecommand{\claimvalue}[1]{%",
        r"  \ifcsname OAEClaim#1\endcsname%",
        r"    \csname OAEClaim#1\endcsname%",
        r"  \else%",
        r"    \PackageError{openaffect-claims}{Unknown claim #1}%",
        r"      {Rebuild manuscript assets from the frozen claim ledger.}%",
        r"  \fi%",
        r"}",
    ]
    for row in ledger.itertuples(index=False):
        claim_id = str(row.claim_id).strip()
        if not re.fullmatch(r"C\d+", claim_id):
            raise ManuscriptAssetError(
                f"Claim ID {claim_id!r} cannot be represented in LaTeX"
            )
        display_value = str(row.display_value).strip()
        if re.fullmatch(r"-?\d+", display_value) and abs(int(display_value)) >= 1000:
            display_value = f"{int(display_value):,}"
        lines.append(
            rf"\expandafter\def\csname OAEClaim{claim_id}\endcsname"
            rf"{{{display_value}}}% \claimvalue{{{claim_id}}}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_manuscript_assets(
    registry: ResultRegistry,
    claims: Mapping[str, object],
    output: Path,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    registry_output = output / "registry"
    compile_artifacts(registry, registry_output)

    representation_names = {
        "representation_statistics_core",
        "representation_statistics_context",
    }
    identity_names = set(IDENTITY_LABELS)
    representation_sources = _load_summary_sources(registry, representation_names)
    identity_sources = _load_summary_sources(registry, identity_names)
    external_sources = _load_summary_sources(registry, EXTERNAL_ASSET_NAMES)
    scope_sources = _load_summary_sources(registry, SCOPE_ASSET_NAMES)
    reviewer_control_sources = _load_summary_sources(
        registry, REVIEWER_CONTROL_ASSET_NAMES
    )
    data_quality_sources = _load_summary_sources(
        registry, DATA_QUALITY_ASSET_NAMES
    )
    if not representation_sources:
        raise ManuscriptAssetError("Registry has no representation statistics")
    if not identity_sources:
        raise ManuscriptAssetError("Registry has no identity-probe summaries")

    identity_seed_table = build_identity_probe_seed_table(identity_sources)
    tables = {
        "protocol_models": build_protocol_model_table(representation_sources),
        "representation_comparisons": build_representation_comparison_table(
            representation_sources
        ),
        "identity_probes": build_identity_probe_table(identity_sources),
        "identity_probe_seeds": identity_seed_table,
        "benchmark_metrics": _benchmark_metric_summary(
            registry_output / "benchmark_metrics.csv", registry
        ),
        "external_validation": build_external_validation_table(external_sources),
        "benchmark_scope": build_benchmark_scope_table(scope_sources),
        "reviewer_controls": build_reviewer_control_table(
            reviewer_control_sources
        ),
        "data_quality": build_data_quality_table(data_quality_sources),
    }
    table_files = {
        "protocol_models": "table_protocol_models.csv",
        "representation_comparisons": "table_representation_comparisons.csv",
        "identity_probes": "table_identity_probes.csv",
        "identity_probe_seeds": "table_identity_probe_seeds.csv",
        "benchmark_metrics": "table_benchmark_metrics_summary.csv",
        "external_validation": "table_external_validation.csv",
        "benchmark_scope": "table_benchmark_scope.csv",
        "reviewer_controls": "table_reviewer_controls.csv",
        "data_quality": "table_data_quality.csv",
    }
    for name, filename in table_files.items():
        _write_table(tables[name], output / filename)

    ledger = resolve_claims(claims, tables)
    _write_table(ledger, output / "claim_ledger.tsv", separator="\t")
    write_claim_value_macros(ledger, output / "claim_values.tex")
    plot_benchmark_overview(output / "figure_benchmark_overview")
    plot_protocol_audit(
        tables["protocol_models"],
        output / "figure_protocol_audit",
        tables["representation_comparisons"],
        tables["benchmark_metrics"],
    )
    plot_identity_audit(
        tables["identity_probes"],
        output / "figure_identity_audit",
        tables["identity_probe_seeds"],
    )

    generated = [
        *table_files.values(),
        "claim_ledger.tsv",
        "claim_values.tex",
        "figure_benchmark_overview.png",
        "figure_benchmark_overview.pdf",
        "figure_protocol_audit.png",
        "figure_protocol_audit.pdf",
        "figure_identity_audit.png",
        "figure_identity_audit.pdf",
    ]
    outputs = {
        filename: {
            "sha256": sha256_file(output / filename),
            "size_bytes": (output / filename).stat().st_size,
        }
        for filename in sorted(generated)
    }
    manifest: dict[str, object] = {
        "registry_version": registry.version,
        "registry_sha256": registry.source_sha256,
        "claim_count": len(ledger),
        "outputs": outputs,
    }
    (output / "manuscript_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
