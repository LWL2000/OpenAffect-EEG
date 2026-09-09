"""Training-only empirical shrinkage controls, not a latent-effects model."""
from __future__ import annotations

import numpy as np


def estimate_shrinkage(residuals, participants):
    """Estimate a bounded offset rule from population-training residuals only.

    A method-of-moments correction subtracts sampling noise from the variance
    of participant means. Correlated stimulus residuals make this a pragmatic
    prediction control, not an identified random-effects variance estimator.
    """
    values = np.asarray(residuals, dtype=float)
    ids = np.asarray(participants).astype(str)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) != len(ids):
        raise ValueError("Expected two-dimensional affect residuals and participant IDs")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite training residuals")
    groups = [values[ids == identity] for identity in np.unique(ids)]
    if len(groups) < 2 or len(values) <= len(groups):
        raise ValueError("Shrinkage requires multiple participants and within-participant replication")
    counts = np.array([len(g) for g in groups], dtype=float)
    means = np.stack([g.mean(0) for g in groups])
    within = sum(((g - g.mean(0))**2).sum(0) for g in groups) / (len(values)-len(groups))
    between = np.maximum(means.var(0, ddof=1) - within * np.mean(1/counts), 0)
    return {"within_variance": within.tolist(), "between_variance": between.tolist(),
            "training_participants": len(groups), "training_trials": len(values),
            "method": "training_only_residual_moments_not_identified_variance_components"}


def shrinkage_weights(estimate, counts):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Calibration counts must be finite nonnegative values")
    within = np.asarray(estimate["within_variance"], dtype=float)
    between = np.asarray(estimate["between_variance"], dtype=float)
    if within.shape != (2,) or between.shape != (2,) or not np.isfinite([within, between]).all() or np.any(within < 0) or np.any(between < 0):
        raise ValueError("Invalid training variance estimates")
    numerator = counts[:, None] * between
    denominator = numerator + within
    weights = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    weights[counts == 0] = 0
    return weights


def shrink_predictions(table, weights):
    """Apply identical weights to EEG-free and EEG calibration offsets."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(table), 2) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Offset weights must be N x 2 in [0, 1]")
    result = table.copy()
    for base, calibrated in (("prior", "prior_personalized"),
                             ("combined_population", "combined_personalized"),
                             ("eeg_population", "eeg_personalized")):
        b = [f"{base}_{target}" for target in ("valence", "arousal")]
        c = [f"{calibrated}_{target}" for target in ("valence", "arousal")]
        result[c] = table[b].to_numpy() + weights * (table[c].to_numpy() - table[b].to_numpy())
    return result
