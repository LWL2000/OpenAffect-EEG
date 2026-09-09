"""Metrics and safeguards for representation identity probes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


def classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    observed = np.asarray(truth)
    estimated = np.asarray(prediction)
    if observed.shape != estimated.shape or observed.ndim != 1:
        raise ValueError("Classification inputs must be equal one-dimensional arrays")
    return {
        "accuracy": float(accuracy_score(observed, estimated)),
        "balanced_accuracy": float(
            recall_score(
                observed,
                estimated,
                labels=np.unique(observed),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(f1_score(observed, estimated, average="macro")),
    }


def seen_class_mask(training_labels: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
    training = np.asarray(training_labels)
    test = np.asarray(test_labels)
    if training.ndim != 1 or test.ndim != 1:
        raise ValueError("Probe labels must be one-dimensional")
    return np.isin(test, np.unique(training))


def covered_probe_partitions(
    table: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return fit rows and test rows whose identity class was observed in fit."""
    required = {"split", label_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Probe table is missing columns: {sorted(missing)}")
    fit = table.loc[table["split"].isin(["train", "validation"])].copy()
    raw_test = table.loc[table["split"].eq("test")].copy()
    if fit.empty or raw_test.empty:
        raise ValueError("Probe requires non-empty fit and test partitions")
    mask = seen_class_mask(
        fit[label_column].to_numpy(),
        raw_test[label_column].to_numpy(),
    )
    return fit, raw_test.loc[mask].copy(), float(mask.mean())
