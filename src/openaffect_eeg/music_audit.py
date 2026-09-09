"""Leakage-resistant MusicEEG audit utilities.

The helpers in this module operate on whole-trial assignments.  They are kept
separate from model code so the split contract, feature alignment, and
self-report diagnostics can be tested and reused by the public release.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openaffect_eeg.baselines import (
    concordance_correlation_coefficient,
    regression_metrics,
)

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
TARGET_COLUMNS = ("target_valence", "target_arousal")
SPLITS = ("train", "validation", "test")
EXCLUDED_SPLIT = "excluded"


def sha256_file(path: str | Path) -> str:
    """Return a content hash for a raw input or derived feature artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bandpower_feature_table(
    path: str | Path,
    feature_view: str,
) -> pd.DataFrame:
    """Load a validated MusicEEG band-power view in trial-table form."""
    if feature_view not in {"channel", "global"}:
        raise ValueError("feature_view must be 'channel' or 'global'")
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            trial_uids = np.asarray(archive["trial_uid"]).astype(str)
            values = np.asarray(
                archive[f"{feature_view}_log_bandpower"], dtype=np.float64
            )
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"Cannot load {feature_view} band-power features: {source}") from error
    if trial_uids.ndim != 1 or len(trial_uids) != len(values):
        raise ValueError("Band-power trial IDs and feature rows are incompatible")
    if len(np.unique(trial_uids)) != len(trial_uids):
        raise ValueError("Band-power feature trial IDs are not unique")
    values = values.reshape(len(values), -1)
    if not np.isfinite(values).all():
        raise ValueError("Band-power features contain non-finite values")
    table = pd.DataFrame(
        values,
        columns=[f"eeg_feature_{index:04d}" for index in range(values.shape[1])],
    )
    table.insert(0, "trial_uid", trial_uids)
    return table


def strict_partitions(
    trials: pd.DataFrame,
    assignment: pd.DataFrame,
    protocol: str,
) -> dict[str, pd.DataFrame]:
    """Validate and return train, validation, and test whole-trial partitions."""
    required_trial_columns = {"trial_uid", "subject_uid", "stimulus_uid"}
    missing_trials = sorted(required_trial_columns.difference(trials.columns))
    if missing_trials:
        raise ValueError(f"Trial table is missing columns: {missing_trials}")
    required_assignment_columns = {"trial_uid", "split", "seed"}
    missing_assignments = sorted(
        required_assignment_columns.difference(assignment.columns)
    )
    if missing_assignments:
        raise ValueError(f"Split assignment is missing columns: {missing_assignments}")
    if assignment["trial_uid"].duplicated().any():
        raise ValueError("Split assignment repeats trial identifiers")
    if not set(assignment["split"]).issubset({*SPLITS, EXCLUDED_SPLIT}):
        raise ValueError("Split assignment has an unknown split name")
    if assignment["seed"].nunique() != 1:
        raise ValueError("Each split assignment must have exactly one seed")
    merged = trials.merge(
        assignment[["trial_uid", "split", "seed"]],
        on="trial_uid",
        how="inner",
        validate="1:1",
    )
    if len(merged) != len(trials) or len(merged) != len(assignment):
        raise ValueError("Trial table and split assignment do not contain identical trials")
    partitions = {
        name: merged.loc[merged["split"].eq(name)].copy() for name in SPLITS
    }
    if any(len(partition) < 2 for partition in partitions.values()):
        raise ValueError(f"Insufficient rows under {protocol}")
    fit = pd.concat(
        [partitions["train"], partitions["validation"]], ignore_index=True
    )
    if protocol in {"subject_holdout", "subject_stimulus_holdout"}:
        repeated = set(fit["subject_uid"]).intersection(
            partitions["test"]["subject_uid"]
        )
        if repeated:
            raise ValueError(f"{protocol} repeats a participant in fit and test")
    if protocol in {"stimulus_holdout", "subject_stimulus_holdout"}:
        repeated = set(fit["stimulus_uid"]).intersection(
            partitions["test"]["stimulus_uid"]
        )
        if repeated:
            raise ValueError(f"{protocol} repeats a stimulus in fit and test")
    return partitions


def _feature_columns(table: pd.DataFrame) -> list[str]:
    columns = [column for column in table if column.startswith("eeg_feature_")]
    if not columns:
        raise ValueError("No EEG feature columns are available")
    return columns


def _ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))


def select_ridge_prediction(
    partitions: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, float, dict[float, float]]:
    """Tune a transparent Ridge model on validation MAE, then test once."""
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    feature_columns = _feature_columns(train)
    missing_targets = set(TARGET_COLUMNS).difference(train.columns)
    if missing_targets:
        raise ValueError(f"Missing affect targets: {sorted(missing_targets)}")
    scores: dict[float, float] = {}
    for alpha in ALPHAS:
        model = _ridge(alpha).fit(
            train[feature_columns], train[list(TARGET_COLUMNS)]
        )
        prediction = model.predict(validation[feature_columns])
        scores[alpha] = float(
            np.mean(
                np.abs(prediction - validation[list(TARGET_COLUMNS)].to_numpy())
            )
        )
    selected_alpha = min(scores, key=scores.get)
    fit = pd.concat([train, validation], ignore_index=True)
    model = _ridge(selected_alpha).fit(
        fit[feature_columns], fit[list(TARGET_COLUMNS)]
    )
    return model.predict(test[feature_columns]), float(selected_alpha), scores


def evaluate_regression_seed(
    partitions: dict[str, pd.DataFrame],
    features: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate a feature table under an already validated split assignment."""
    merged = {
        name: partition.merge(features, on="trial_uid", how="inner", validate="1:1")
        for name, partition in partitions.items()
    }
    if any(len(merged[name]) != len(partitions[name]) for name in SPLITS):
        raise ValueError("Feature archive does not cover every assigned trial")
    prediction, alpha, validation_scores = select_ridge_prediction(merged)
    return {
        "selected_alpha": alpha,
        "feature_count": len(_feature_columns(merged["train"])),
        "validation_macro_mae_by_alpha": {
            str(key): value for key, value in validation_scores.items()
        },
        "metrics": regression_metrics(
            merged["test"][list(TARGET_COLUMNS)].to_numpy(), prediction
        ),
    }


def evaluate_identity_probe(
    partitions: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    *,
    label_column: str,
    seed: int,
) -> dict[str, float | int]:
    """Probe a seen identity only under a split that holds out the other factor."""
    if label_column not in {"subject_uid", "stimulus_uid"}:
        raise ValueError("Identity probes support subject_uid or stimulus_uid")
    merged = {
        name: partition.merge(features, on="trial_uid", how="inner", validate="1:1")
        for name, partition in partitions.items()
    }
    if any(len(merged[name]) != len(partitions[name]) for name in SPLITS):
        raise ValueError("Feature archive does not cover every assigned trial")
    fit = pd.concat([merged["train"], merged["validation"]], ignore_index=True)
    test = merged["test"]
    covered_test = test.loc[test[label_column].isin(set(fit[label_column]))].copy()
    feature_columns = _feature_columns(fit)
    if fit[label_column].nunique() < 2:
        raise ValueError("Identity probe needs at least two fitted classes")
    if len(covered_test) < 2:
        raise ValueError("Identity probe has fewer than two seen test rows")
    model = make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=1.0, class_weight="balanced")
    ).fit(fit[feature_columns], fit[label_column])
    observed = balanced_accuracy_score(
        covered_test[label_column], model.predict(covered_test[feature_columns])
    )
    rng = np.random.default_rng(seed)
    permuted = make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=1.0, class_weight="balanced")
    ).fit(fit[feature_columns], rng.permutation(fit[label_column].to_numpy()))
    permutation = balanced_accuracy_score(
        covered_test[label_column], permuted.predict(covered_test[feature_columns])
    )
    return {
        "balanced_accuracy": float(observed),
        "permuted_balanced_accuracy": float(permutation),
        "uniform_chance_accuracy": 1.0 / int(fit[label_column].nunique()),
        "fit_class_count": int(fit[label_column].nunique()),
        "fit_trial_count": len(fit),
        "test_trial_count": len(test),
        "covered_test_trial_count": len(covered_test),
        "seen_test_fraction": float(len(covered_test) / len(test)),
    }


def seed_summary(values: dict[int, float]) -> dict[str, float | int]:
    """Summarize the prespecified frozen split seeds without pooling trials."""
    ordered = np.asarray([values[seed] for seed in sorted(values)], dtype=float)
    return {
        "mean": float(ordered.mean()),
        "sample_sd": float(ordered.std(ddof=1)) if len(ordered) > 1 else 0.0,
        "seed_count": len(ordered),
    }


def paired_seed_summary(
    left: dict[int, float],
    right: dict[int, float],
) -> dict[str, float | int]:
    """Summarize a fixed-seed difference after verifying paired seed coverage."""
    if set(left) != set(right):
        raise ValueError("Paired summaries require identical seed sets")
    return seed_summary({seed: left[seed] - right[seed] for seed in left})


def leave_one_out_stimulus_consensus(
    table: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Predict each rating from other ratings of the same stimulus only.

    A singleton stimulus has no valid consensus prediction and remains missing.
    This is a rating-agreement diagnostic, not an upper bound on EEG prediction.
    """
    required = {"trial_uid", "stimulus_uid", target_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Consensus table is missing columns: {missing}")
    values = pd.to_numeric(table[target_column], errors="coerce")
    valid = table.loc[values.notna(), ["trial_uid", "stimulus_uid"]].copy()
    valid["target"] = values.loc[valid.index].to_numpy(dtype=float)
    grouped = valid.groupby("stimulus_uid", sort=False)["target"]
    count = grouped.transform("count")
    total = grouped.transform("sum")
    valid["rating_count"] = count.astype(int)
    valid["prediction"] = np.where(count > 1, (total - valid["target"]) / (count - 1), np.nan)
    result = table[["trial_uid"]].merge(
        valid[["trial_uid", "rating_count", "prediction"]],
        on="trial_uid",
        how="left",
        validate="1:1",
    )
    result["rating_count"] = result["rating_count"].fillna(0).astype(int)
    return result


def summarize_stimulus_consensus(
    table: pd.DataFrame,
    target_column: str,
) -> dict[str, float | int]:
    """Report leave-one-rating-out agreement for one self-report dimension."""
    consensus = leave_one_out_stimulus_consensus(table, target_column)
    observed = pd.to_numeric(table[target_column], errors="coerce")
    usable = observed.notna() & consensus["prediction"].notna()
    truth = observed.loc[usable].to_numpy(dtype=float)
    prediction = consensus.loc[usable, "prediction"].to_numpy(dtype=float)
    if len(truth) < 2:
        raise ValueError("Stimulus consensus needs at least two supported ratings")
    return {
        "supported_rating_count": len(truth),
        "supported_stimulus_count": int(
            table.loc[usable, "stimulus_uid"].nunique()
        ),
        "ccc": concordance_correlation_coefficient(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
    }


def _design_matrix(table: pd.DataFrame, factors: tuple[str, ...]) -> np.ndarray:
    blocks = [np.ones((len(table), 1), dtype=float)]
    for factor in factors:
        categories = pd.get_dummies(table[factor].astype(str), dtype=float)
        if categories.shape[1] > 1:
            blocks.append(categories.iloc[:, 1:].to_numpy(dtype=float))
    return np.concatenate(blocks, axis=1)


def _in_sample_r2(target: np.ndarray, design: np.ndarray) -> float:
    fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    total_sum_squares = float(np.sum((target - target.mean()) ** 2))
    if total_sum_squares == 0:
        return float("nan")
    return float(1.0 - np.sum((target - fitted) ** 2) / total_sum_squares)


def summarize_additive_structure(
    table: pd.DataFrame,
    target_column: str,
) -> dict[str, object]:
    """Describe, rather than causally attribute, observed rating structure."""
    required = {"subject_uid", "stimulus_uid", target_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Additive summary table is missing columns: {missing}")
    target = pd.to_numeric(table[target_column], errors="coerce")
    available = table.loc[target.notna(), ["subject_uid", "stimulus_uid"]].copy()
    values = target.loc[available.index].to_numpy(dtype=float)
    if len(available) < 2:
        raise ValueError("Additive summary needs at least two ratings")
    subject_r2 = _in_sample_r2(values, _design_matrix(available, ("subject_uid",)))
    stimulus_r2 = _in_sample_r2(values, _design_matrix(available, ("stimulus_uid",)))
    both_r2 = _in_sample_r2(
        values, _design_matrix(available, ("subject_uid", "stimulus_uid"))
    )
    return {
        "observation_count": len(available),
        "subject_count": int(available["subject_uid"].nunique()),
        "stimulus_count": int(available["stimulus_uid"].nunique()),
        "target_sample_variance": float(np.var(values, ddof=1)),
        "descriptive_in_sample_r2": {
            "subject_only": subject_r2,
            "stimulus_only": stimulus_r2,
            "subject_plus_stimulus": both_r2,
        },
        "incremental_descriptive_r2": {
            "stimulus_given_subject": both_r2 - subject_r2,
            "subject_given_stimulus": both_r2 - stimulus_r2,
        },
    }
