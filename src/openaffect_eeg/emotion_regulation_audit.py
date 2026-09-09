"""Protocol-matched audits for the ds006866 emotion-regulation dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openaffect_eeg.baselines import (
    condition_residual_targets,
    predict_group_mean,
    regression_metrics,
)

TARGET_COLUMNS = ("target_valence", "target_arousal")
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


class EmotionRegulationAuditError(ValueError):
    """Raised when an external-audit protocol is not identifiable."""


def runtime_eeg_availability(
    table: pd.DataFrame, dataset_root: Path
) -> pd.Series:
    """Resolve EEG availability from current files rather than stale metadata."""

    if "eeg_path" not in table:
        raise EmotionRegulationAuditError("Trial table is missing eeg_path")
    root = dataset_root.resolve()

    def is_available(relative_path: object) -> bool:
        if pd.isna(relative_path):
            return False
        candidate = (root / str(relative_path)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        return candidate.is_file()

    return table["eeg_path"].map(is_available).astype(bool)


def align_feature_support(
    feature_table: pd.DataFrame, expected_trial_uids: set[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Require complete labelled support and discard explicitly counted extras."""

    if "trial_uid" not in feature_table:
        raise EmotionRegulationAuditError("Feature table is missing trial_uid")
    if feature_table["trial_uid"].duplicated().any():
        raise EmotionRegulationAuditError("Feature table contains duplicate trials")
    observed = set(feature_table["trial_uid"].astype(str))
    missing = expected_trial_uids.difference(observed)
    extra = observed.difference(expected_trial_uids)
    if missing:
        raise EmotionRegulationAuditError(
            f"Feature archive misses {len(missing)} labelled trials"
        )
    aligned = feature_table.loc[
        feature_table["trial_uid"].astype(str).isin(expected_trial_uids)
    ].copy()
    return aligned, {
        "archive_trial_count": len(feature_table),
        "used_labelled_trial_count": len(aligned),
        "ignored_unlabelled_trial_count": len(extra),
    }


def condition_family(table: pd.DataFrame) -> pd.Series:
    """Return the content-invariant valence/regulation condition family."""

    required = {"nominal_valence", "regulation"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise EmotionRegulationAuditError(
            f"Condition-family columns are missing: {missing}"
        )
    return (
        table["nominal_valence"].astype(str)
        + "_"
        + table["regulation"].astype(str)
    )


def build_context_only_transfer_split(
    table: pd.DataFrame,
    *,
    subject_column: str,
    context_column: str,
    source_context: str,
    target_context: str,
    seed: int,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    """Train in one context and test in another for the same participants."""

    required = {
        "trial_uid",
        subject_column,
        context_column,
        "nominal_valence",
        "regulation",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise EmotionRegulationAuditError(f"Split columns are missing: {missing}")
    if table["trial_uid"].duplicated().any():
        raise EmotionRegulationAuditError("Trial identifiers must be unique")
    if source_context == target_context:
        raise EmotionRegulationAuditError("Context transfer needs two contexts")
    if not 0 < validation_fraction < 0.5:
        raise EmotionRegulationAuditError(
            "Validation fraction must be between zero and one half"
        )
    observed = set(table[context_column].astype(str))
    if not {source_context, target_context}.issubset(observed):
        raise EmotionRegulationAuditError("Requested contexts are unavailable")

    frame = table.copy()
    frame["condition_family"] = condition_family(frame)
    split = pd.Series("excluded", index=frame.index, dtype=object)
    source = frame[context_column].astype(str).eq(source_context)
    target = frame[context_column].astype(str).eq(target_context)
    split.loc[source] = "train"
    split.loc[target] = "test"

    rng = np.random.default_rng(seed)
    grouped = frame.loc[source].groupby(
        [subject_column, "condition_family"], sort=True
    )
    for _, rows in grouped:
        indices = rows.index.to_numpy()
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_count = min(validation_count, len(indices) - 1)
        if validation_count < 1:
            raise EmotionRegulationAuditError(
                "Each participant-condition family needs at least two source trials"
            )
        validation_indices = rng.choice(
            indices, size=validation_count, replace=False
        )
        split.loc[validation_indices] = "validation"

    return pd.DataFrame(
        {
            "trial_uid": frame["trial_uid"],
            "split": split,
            "protocol": f"{source_context}_to_{target_context}_context_only",
            "seed": seed,
        }
    ).sort_values("trial_uid").reset_index(drop=True)


def strict_partitions(
    trials: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    participant_disjoint: bool,
    context_disjoint: bool,
) -> dict[str, pd.DataFrame]:
    """Validate a complete-trial assignment and return model partitions."""

    required_trials = {"trial_uid", "subject_uid", "content", *TARGET_COLUMNS}
    missing_trials = sorted(required_trials.difference(trials.columns))
    if missing_trials:
        raise EmotionRegulationAuditError(
            f"Trial table is missing audit columns: {missing_trials}"
        )
    required_assignments = {"trial_uid", "split", "seed", "protocol"}
    missing_assignments = sorted(required_assignments.difference(assignments.columns))
    if missing_assignments:
        raise EmotionRegulationAuditError(
            f"Assignment is missing columns: {missing_assignments}"
        )
    if assignments["trial_uid"].duplicated().any():
        raise EmotionRegulationAuditError("Assignment contains duplicate trials")
    if not set(trials["trial_uid"]).issubset(set(assignments["trial_uid"])):
        raise EmotionRegulationAuditError("Assignment does not cover all model trials")

    merged = trials.merge(
        assignments[["trial_uid", "split", "seed", "protocol"]],
        on="trial_uid",
        how="left",
        validate="1:1",
    )
    partitions = {
        name: merged.loc[merged["split"].eq(name)].copy()
        for name in ("train", "validation", "test")
    }
    if any(len(partition) < 2 for partition in partitions.values()):
        raise EmotionRegulationAuditError("Every model partition needs two trials")
    fit = pd.concat([partitions["train"], partitions["validation"]])
    if participant_disjoint and set(fit["subject_uid"]).intersection(
        partitions["test"]["subject_uid"]
    ):
        raise EmotionRegulationAuditError("Participant identities repeat in test")
    if context_disjoint and set(fit["content"]).intersection(
        partitions["test"]["content"]
    ):
        raise EmotionRegulationAuditError("Contexts repeat in test")
    return partitions


def _ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))


def fit_affect_models(
    partitions: dict[str, pd.DataFrame],
    feature_columns: list[str],
    *,
    prior_column: str,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict[str, object]:
    """Fit global, semantic-prior, EEG, and prior-plus-residual predictors."""

    if not feature_columns:
        raise EmotionRegulationAuditError("No EEG features were selected")
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise EmotionRegulationAuditError("Ridge alphas must be positive")
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    for frame in (train, validation, test):
        required = {prior_column, *feature_columns, *TARGET_COLUMNS}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise EmotionRegulationAuditError(
                f"Model partition is missing columns: {missing}"
            )

    train_x = train[feature_columns].to_numpy(dtype=float)
    validation_x = validation[feature_columns].to_numpy(dtype=float)
    train_y = train[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    validation_y = validation[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    train_group = train[prior_column].astype(str).to_numpy()
    validation_group = validation[prior_column].astype(str).to_numpy()
    train_residual = condition_residual_targets(train_y, train_group)
    validation_prior = predict_group_mean(
        train_y, train_group, validation_group
    )

    direct_validation: dict[float, float] = {}
    residual_validation: dict[float, float] = {}
    for alpha in alphas:
        direct_prediction = _ridge(alpha).fit(train_x, train_y).predict(
            validation_x
        )
        direct_validation[alpha] = regression_metrics(
            validation_y, direct_prediction
        )["macro"]["mae"]
        residual_prediction = _ridge(alpha).fit(
            train_x, train_residual
        ).predict(validation_x)
        residual_validation[alpha] = regression_metrics(
            validation_y, validation_prior + residual_prediction
        )["macro"]["mae"]
    direct_alpha = min(direct_validation, key=direct_validation.get)
    residual_alpha = min(residual_validation, key=residual_validation.get)

    fit = pd.concat([train, validation], ignore_index=True)
    fit_x = fit[feature_columns].to_numpy(dtype=float)
    fit_y = fit[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    fit_group = fit[prior_column].astype(str).to_numpy()
    test_x = test[feature_columns].to_numpy(dtype=float)
    test_y = test[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    test_group = test[prior_column].astype(str).to_numpy()
    global_prediction = np.repeat(fit_y.mean(axis=0)[None, :], len(test), axis=0)
    prior_prediction = predict_group_mean(fit_y, fit_group, test_group)
    direct_prediction = _ridge(direct_alpha).fit(fit_x, fit_y).predict(test_x)
    fit_residual = condition_residual_targets(fit_y, fit_group)
    residual_prediction = _ridge(residual_alpha).fit(
        fit_x, fit_residual
    ).predict(test_x)
    predictions = {
        "global_mean": global_prediction,
        "semantic_prior": prior_prediction,
        "eeg_only": direct_prediction,
        "prior_plus_eeg_residual": prior_prediction + residual_prediction,
    }
    return {
        "selected_alpha": {
            "eeg_only": float(direct_alpha),
            "prior_plus_eeg_residual": float(residual_alpha),
        },
        "models": {
            name: regression_metrics(test_y, prediction)
            for name, prediction in predictions.items()
        },
        "predictions": predictions,
    }


def fit_fixed_linear_probe(
    partitions: dict[str, pd.DataFrame],
    feature_columns: list[str],
    *,
    label_column: str,
    seed: int,
    alpha: float = 1.0,
    device: str = "cpu",
) -> dict[str, object]:
    """Audit linearly decodable labels with a matched permutation control."""

    fit = pd.concat(
        [partitions["train"], partitions["validation"]], ignore_index=True
    )
    test = partitions["test"].copy()
    fit_labels = fit[label_column].astype(str)
    test_labels = test[label_column].astype(str)
    seen = test_labels.isin(set(fit_labels))
    covered = test.loc[seen].copy()
    if covered.empty:
        raise EmotionRegulationAuditError("Probe has no seen-class test trials")
    rng = np.random.default_rng(seed)
    permuted_labels = rng.permutation(fit_labels.to_numpy())
    if device == "cpu":
        observed_prediction = _fit_sklearn_ridge_probe(
            fit[feature_columns],
            fit_labels,
            covered[feature_columns],
            alpha=alpha,
        )
        permuted_prediction = _fit_sklearn_ridge_probe(
            fit[feature_columns],
            permuted_labels,
            covered[feature_columns],
            alpha=alpha,
        )
        backend = "sklearn_svd_cpu"
    elif device == "cuda":
        observed_prediction = _fit_torch_ridge_probe(
            fit[feature_columns],
            fit_labels.to_numpy(),
            covered[feature_columns],
            alpha=alpha,
        )
        permuted_prediction = _fit_torch_ridge_probe(
            fit[feature_columns],
            permuted_labels,
            covered[feature_columns],
            alpha=alpha,
        )
        backend = "torch_closed_form_cuda_float64"
    else:
        raise EmotionRegulationAuditError(
            f"Probe device must be 'cpu' or 'cuda', got {device!r}"
        )

    observed = balanced_accuracy_score(
        covered[label_column].astype(str), observed_prediction
    )
    permutation = balanced_accuracy_score(
        covered[label_column].astype(str), permuted_prediction
    )
    class_count = int(fit_labels.nunique())
    return {
        "balanced_accuracy": float(observed),
        "permuted_balanced_accuracy": float(permutation),
        "uniform_chance_accuracy": 1.0 / class_count,
        "fit_class_count": class_count,
        "test_trial_count": len(test),
        "seen_test_count": len(covered),
        "seen_test_fraction": float(seen.mean()),
        "backend": backend,
    }


def _fit_sklearn_ridge_probe(
    fit_features: pd.DataFrame,
    fit_labels: pd.Series | np.ndarray,
    test_features: pd.DataFrame,
    *,
    alpha: float,
) -> np.ndarray:
    classifier = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=alpha, class_weight="balanced", solver="svd"),
    )
    classifier.fit(fit_features, fit_labels)
    return classifier.predict(test_features)


def _fit_torch_ridge_probe(
    fit_features: pd.DataFrame,
    fit_labels: np.ndarray,
    test_features: pd.DataFrame,
    *,
    alpha: float,
) -> np.ndarray:
    """Fit a balanced one-vs-rest ridge probe with CUDA linear algebra."""

    try:
        import torch
    except ImportError as exc:
        raise EmotionRegulationAuditError(
            "CUDA probes require the foundation extra with PyTorch"
        ) from exc
    if not torch.cuda.is_available():
        raise EmotionRegulationAuditError("CUDA probe requested but CUDA is unavailable")

    classes, encoded = np.unique(np.asarray(fit_labels, dtype=str), return_inverse=True)
    if len(classes) < 2:
        raise EmotionRegulationAuditError("Probe requires at least two fit classes")
    counts = np.bincount(encoded, minlength=len(classes))
    sample_weight = len(encoded) / (len(classes) * counts[encoded])

    x_fit = torch.as_tensor(
        fit_features.to_numpy(dtype=np.float64).copy(),
        device="cuda",
        dtype=torch.float64,
    )
    x_test = torch.as_tensor(
        test_features.to_numpy(dtype=np.float64).copy(),
        device="cuda",
        dtype=torch.float64,
    )
    mean = x_fit.mean(dim=0)
    scale = x_fit.std(dim=0, correction=0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_fit = (x_fit - mean) / scale
    x_test = (x_test - mean) / scale
    x_fit = torch.cat(
        [x_fit, torch.ones((len(x_fit), 1), device="cuda", dtype=torch.float64)],
        dim=1,
    )
    x_test = torch.cat(
        [x_test, torch.ones((len(x_test), 1), device="cuda", dtype=torch.float64)],
        dim=1,
    )

    targets = torch.full(
        (len(encoded), len(classes)),
        -1.0,
        device="cuda",
        dtype=torch.float64,
    )
    encoded_tensor = torch.as_tensor(encoded, device="cuda", dtype=torch.long)
    targets[torch.arange(len(encoded), device="cuda"), encoded_tensor] = 1.0
    weights = torch.as_tensor(sample_weight, device="cuda", dtype=torch.float64)
    weighted_fit = x_fit * weights[:, None]
    gram = x_fit.T @ weighted_fit
    penalty = torch.full(
        (gram.shape[0],), alpha, device="cuda", dtype=torch.float64
    )
    penalty[-1] = 0.0
    gram.diagonal().add_(penalty)
    coefficients = torch.linalg.solve(gram, x_fit.T @ (targets * weights[:, None]))
    prediction = torch.argmax(x_test @ coefficients, dim=1).cpu().numpy()
    return classes[prediction]


def summarize_seed_values(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise EmotionRegulationAuditError("Seed summary cannot be empty")
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "seed_count": len(array),
    }
