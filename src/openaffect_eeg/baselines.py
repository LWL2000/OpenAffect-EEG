"""Transparent affect-regression baselines and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_NAMES = ("valence", "arousal")
TARGET_SCALES = {
    "ds002721": (1.0, 9.0, "1-9"),
    "ds003751": (1.0, 9.0, "1-9"),
    "ds005540": (0.0, 7.0, "0-7"),
    "ds006866": (1.0, 9.0, "1-9"),
}


def _numeric_column(table: pd.DataFrame, name: str) -> pd.Series:
    if name not in table.columns:
        return pd.Series(np.nan, index=table.index, dtype=float)
    return pd.to_numeric(table[name], errors="coerce")


def harmonize_affect_targets(table: pd.DataFrame) -> pd.DataFrame:
    if "dataset_id" not in table.columns:
        raise ValueError("Trial table has no dataset_id column")
    result = table.copy()
    result["target_valence_raw"] = np.nan
    result["target_arousal_raw"] = np.nan
    result["stimulus_prior_valence_raw"] = np.nan
    result["stimulus_prior_arousal_raw"] = np.nan

    music = result["dataset_id"].eq("ds002721")
    dens = result["dataset_id"].eq("ds003751")
    emo_eeg = result["dataset_id"].eq("ds005540")
    emotion_regulation = result["dataset_id"].eq("ds006866")
    result.loc[music, "target_valence_raw"] = _numeric_column(
        result, "self_pleasant"
    ).loc[music]
    result.loc[music, "target_arousal_raw"] = _numeric_column(
        result, "self_energetic"
    ).loc[music]
    result.loc[music, "stimulus_prior_valence_raw"] = _numeric_column(
        result, "normative_pleasant"
    ).loc[music]
    result.loc[music, "stimulus_prior_arousal_raw"] = _numeric_column(
        result, "normative_energetic"
    ).loc[music]
    result.loc[dens, "target_valence_raw"] = _numeric_column(
        result, "self_valence"
    ).loc[dens]
    result.loc[dens, "target_arousal_raw"] = _numeric_column(
        result, "self_arousal"
    ).loc[dens]
    result.loc[emo_eeg, "target_valence_raw"] = _numeric_column(
        result, "self_valence"
    ).loc[emo_eeg]
    result.loc[emo_eeg, "target_arousal_raw"] = _numeric_column(
        result, "self_arousal"
    ).loc[emo_eeg]
    result.loc[emotion_regulation, "target_valence_raw"] = _numeric_column(
        result, "valence"
    ).loc[emotion_regulation]
    result.loc[emotion_regulation, "target_arousal_raw"] = _numeric_column(
        result, "arousal"
    ).loc[emotion_regulation]

    scale_min = result["dataset_id"].map(
        {dataset_id: bounds[0] for dataset_id, bounds in TARGET_SCALES.items()}
    )
    scale_max = result["dataset_id"].map(
        {dataset_id: bounds[1] for dataset_id, bounds in TARGET_SCALES.items()}
    )
    result["target_scale"] = result["dataset_id"].map(
        {dataset_id: bounds[2] for dataset_id, bounds in TARGET_SCALES.items()}
    )
    scale_width = scale_max - scale_min
    for name in TARGET_NAMES:
        result[f"target_{name}"] = (
            result[f"target_{name}_raw"] - scale_min
        ) / scale_width
        result[f"stimulus_prior_{name}"] = (
            result[f"stimulus_prior_{name}_raw"] - scale_min
        ) / scale_width

    if "label_available" in result.columns:
        reported_available = result["label_available"].fillna(False).astype(bool)
    else:
        reported_available = pd.Series(True, index=result.index)
    result["target_available"] = (
        reported_available
        & result["target_valence"].between(0, 1, inclusive="both")
        & result["target_arousal"].between(0, 1, inclusive="both")
    )
    return result


def concordance_correlation_coefficient(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    observed = np.asarray(truth, dtype=float)
    estimated = np.asarray(prediction, dtype=float)
    if observed.shape != estimated.shape or observed.ndim != 1:
        raise ValueError("CCC inputs must be one-dimensional arrays of equal shape")
    observed_mean = observed.mean()
    estimated_mean = estimated.mean()
    covariance = np.mean(
        (observed - observed_mean) * (estimated - estimated_mean)
    )
    denominator = (
        observed.var()
        + estimated.var()
        + (observed_mean - estimated_mean) ** 2
    )
    if denominator == 0:
        return float("nan")
    return float(2 * covariance / denominator)


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    target_names: tuple[str, ...] = TARGET_NAMES,
) -> dict[str, dict[str, float]]:
    observed = np.asarray(truth, dtype=float)
    estimated = np.asarray(prediction, dtype=float)
    if observed.shape != estimated.shape or observed.ndim != 2:
        raise ValueError("Regression targets must be equal two-dimensional arrays")
    names = tuple(str(name) for name in target_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Target names must be unique and non-empty")
    if observed.shape[1] != len(names):
        raise ValueError(f"Expected {len(names)} affect dimensions")

    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        error = estimated[:, index] - observed[:, index]
        metrics[name] = {
            "ccc": concordance_correlation_coefficient(
                observed[:, index], estimated[:, index]
            ),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
        }
    macro: dict[str, float] = {}
    for metric in ("ccc", "mae", "rmse"):
        values = np.asarray([metrics[name][metric] for name in names])
        finite = values[np.isfinite(values)]
        macro[metric] = float(finite.mean()) if len(finite) else float("nan")
    metrics["macro"] = macro
    return metrics


def predict_group_mean(
    train_targets: np.ndarray,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
) -> np.ndarray:
    targets = np.asarray(train_targets, dtype=float)
    groups = np.asarray(train_groups)
    requested = np.asarray(test_groups)
    if targets.ndim != 2 or len(targets) != len(groups):
        raise ValueError("Training targets and groups have incompatible shapes")
    global_mean = targets.mean(axis=0)
    means = {
        group: targets[groups == group].mean(axis=0) for group in np.unique(groups)
    }
    return np.stack([means.get(group, global_mean) for group in requested])


def condition_residual_targets(
    targets: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    identities = np.asarray(groups)
    return values - predict_group_mean(values, identities, identities)
