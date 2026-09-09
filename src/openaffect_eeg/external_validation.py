"""Condition-stratified inference for the large external validation dataset."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openaffect_eeg.baselines import regression_metrics
from openaffect_eeg.probes import classification_metrics
from openaffect_eeg.statistics import (
    cluster_bootstrap_indices,
    exact_sign_flip_pvalue,
    percentile_interval,
)

TARGET_COLUMNS = ["target_valence", "target_arousal"]


def _condition_probe_model(alpha: float):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=alpha, class_weight="balanced"),
    )


def fit_condition_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alphas: tuple[float, ...],
    seed: int,
) -> dict[str, object]:
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("Probe alphas must be positive")
    validation_scores = {}
    for alpha in alphas:
        model = _condition_probe_model(alpha).fit(train_x, train_y)
        validation_scores[alpha] = classification_metrics(
            validation_y,
            model.predict(validation_x),
        )["balanced_accuracy"]
    selected_alpha = max(
        alphas,
        key=lambda alpha: (validation_scores[alpha], -alpha),
    )
    fit_x = np.concatenate([train_x, validation_x])
    fit_y = np.concatenate([train_y, validation_y])
    model = _condition_probe_model(selected_alpha).fit(fit_x, fit_y)
    metrics = classification_metrics(test_y, model.predict(test_x))
    rng = np.random.default_rng(seed)
    permuted = _condition_probe_model(selected_alpha).fit(
        fit_x,
        rng.permutation(fit_y),
    )
    permutation_metrics = classification_metrics(
        test_y,
        permuted.predict(test_x),
    )
    return {
        "selected_alpha": selected_alpha,
        "validation_balanced_accuracy": {
            str(alpha): validation_scores[alpha] for alpha in alphas
        },
        "metrics": metrics,
        "permuted_label_metrics": permutation_metrics,
    }


def prediction_columns(model: str) -> list[str]:
    return [f"{model}_valence", f"{model}_arousal"]


def trial_mae_delta(
    table: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
) -> np.ndarray:
    required = {
        *TARGET_COLUMNS,
        *prediction_columns(candidate),
        *prediction_columns(baseline),
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    target = table[TARGET_COLUMNS].to_numpy(dtype=float)
    candidate_prediction = table[prediction_columns(candidate)].to_numpy(
        dtype=float
    )
    baseline_prediction = table[prediction_columns(baseline)].to_numpy(dtype=float)
    return np.abs(candidate_prediction - target).mean(axis=1) - np.abs(
        baseline_prediction - target
    ).mean(axis=1)


def _model_metrics(
    table: pd.DataFrame,
    models: list[str],
) -> dict[str, dict[str, float]]:
    target = table[TARGET_COLUMNS].to_numpy(dtype=float)
    return {
        model: regression_metrics(
            target,
            table[prediction_columns(model)].to_numpy(dtype=float),
        )["macro"]
        for model in models
    }


def condition_effect_analysis(
    table: pd.DataFrame,
    *,
    baseline: str,
    candidates: list[str],
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    required = {"seed", "subject_id", "context", *TARGET_COLUMNS}
    for model in [baseline, *candidates]:
        required.update(prediction_columns(model))
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    if table[[*TARGET_COLUMNS, *itertools.chain.from_iterable(
        prediction_columns(model) for model in [baseline, *candidates]
    )]].isna().any().any():
        raise ValueError("Prediction table contains missing targets or predictions")

    seeds = sorted(int(seed) for seed in table["seed"].unique())
    conditions = sorted(str(value) for value in table["context"].unique())
    models = [baseline, *candidates]
    condition_results: dict[str, object] = {}
    for condition in conditions:
        condition_table = table.loc[table["context"].astype(str).eq(condition)]
        seed_metrics = {
            seed: _model_metrics(
                condition_table.loc[condition_table["seed"].eq(seed)], models
            )
            for seed in seeds
        }
        comparisons = {}
        for candidate in candidates:
            seed_delta = np.asarray(
                [
                    trial_mae_delta(
                        condition_table.loc[condition_table["seed"].eq(seed)],
                        candidate=candidate,
                        baseline=baseline,
                    ).mean()
                    for seed in seeds
                ]
            )
            bootstrap = np.empty(iterations)
            for iteration in range(iterations):
                sampled_seed_deltas = []
                for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
                    seed_table = condition_table.loc[
                        condition_table["seed"].eq(sampled_seed)
                    ].reset_index(drop=True)
                    indices = cluster_bootstrap_indices(
                        seed_table["subject_id"].astype(str).to_numpy(), rng
                    )
                    sampled_seed_deltas.append(
                        trial_mae_delta(
                            seed_table.iloc[indices],
                            candidate=candidate,
                            baseline=baseline,
                        ).mean()
                    )
                bootstrap[iteration] = np.mean(sampled_seed_deltas)
            low, high = percentile_interval(bootstrap)
            comparisons[candidate] = {
                "mae_delta_candidate_minus_baseline": float(seed_delta.mean()),
                "cluster_bootstrap_95ci": [low, high],
                "exact_seed_sign_flip_p": exact_sign_flip_pvalue(seed_delta),
                "interpretation": "Negative values favor the candidate",
            }
        condition_results[condition] = {
            "row_count": len(condition_table),
            "subject_count": int(condition_table["subject_id"].nunique()),
            "model_metrics_mean_over_seeds": {
                model: {
                    metric: float(
                        np.mean(
                            [seed_metrics[seed][model][metric] for seed in seeds]
                        )
                    )
                    for metric in ("ccc", "mae", "rmse")
                }
                for model in models
            },
            "comparisons": comparisons,
        }

    contrasts = {}
    for domain in ("nonsocial", "social"):
        reappraise = f"{domain}_negative_reappraise"
        watch = f"{domain}_negative_watch"
        if not {reappraise, watch}.issubset(conditions):
            continue
        domain_table = table.loc[
            table["context"].astype(str).isin([reappraise, watch])
        ]
        for candidate in candidates:
            seed_contrast = []
            for seed in seeds:
                seed_table = domain_table.loc[domain_table["seed"].eq(seed)]
                deltas = trial_mae_delta(
                    seed_table,
                    candidate=candidate,
                    baseline=baseline,
                )
                contexts = seed_table["context"].astype(str).to_numpy()
                seed_contrast.append(
                    deltas[contexts == reappraise].mean()
                    - deltas[contexts == watch].mean()
                )
            seed_contrast_array = np.asarray(seed_contrast)
            bootstrap = np.empty(iterations)
            for iteration in range(iterations):
                sampled_seed_contrasts = []
                for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
                    seed_table = domain_table.loc[
                        domain_table["seed"].eq(sampled_seed)
                    ].reset_index(drop=True)
                    indices = cluster_bootstrap_indices(
                        seed_table["subject_id"].astype(str).to_numpy(), rng
                    )
                    sampled = seed_table.iloc[indices]
                    deltas = trial_mae_delta(
                        sampled,
                        candidate=candidate,
                        baseline=baseline,
                    )
                    contexts = sampled["context"].astype(str).to_numpy()
                    sampled_seed_contrasts.append(
                        deltas[contexts == reappraise].mean()
                        - deltas[contexts == watch].mean()
                    )
                bootstrap[iteration] = np.mean(sampled_seed_contrasts)
            low, high = percentile_interval(bootstrap)
            contrasts[f"{domain}:{candidate}"] = {
                "contrast": "reappraise MAE delta minus watch-negative MAE delta",
                "mean_over_seeds": float(seed_contrast_array.mean()),
                "cluster_bootstrap_95ci": [low, high],
                "exact_seed_sign_flip_p": exact_sign_flip_pvalue(
                    seed_contrast_array
                ),
                "interpretation": (
                    "Negative values indicate a larger candidate benefit during "
                    "reappraisal"
                ),
            }
    return {
        "seed_count": len(seeds),
        "bootstrap_iterations": iterations,
        "baseline": baseline,
        "candidates": candidates,
        "conditions": condition_results,
        "strategy_contrasts": contrasts,
    }
