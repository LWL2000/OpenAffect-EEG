"""Small dependency-free statistical helpers for paired EEG evaluations."""

from __future__ import annotations

from itertools import product

import numpy as np


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Return the exact two-sided sign-flip p-value for paired differences."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Paired differences must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("Paired differences contain non-finite values")
    if len(values) > 20:
        raise ValueError("Exact sign-flip enumeration supports at most 20 pairs")
    observed = abs(float(values.mean()))
    exceedances = 0
    assignment_count = 2 ** len(values)
    for signs in product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        if statistic >= observed - 1e-12:
            exceedances += 1
    return exceedances / assignment_count


def cluster_bootstrap_indices(
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample complete clusters with replacement and return their row indices."""
    identities = np.asarray(groups)
    if identities.ndim != 1 or not len(identities):
        raise ValueError("Bootstrap groups must be a non-empty vector")
    unique = np.unique(identities)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(identities == group) for group in sampled])


def crossed_cluster_bootstrap_indices(
    first_groups: np.ndarray,
    second_groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample two crossed clustering factors and return weighted row indices."""
    first = np.asarray(first_groups)
    second = np.asarray(second_groups)
    if first.ndim != 1 or second.ndim != 1 or not len(first):
        raise ValueError("Crossed bootstrap groups must be non-empty vectors")
    if len(first) != len(second):
        raise ValueError("Crossed bootstrap group vectors must have equal length")
    first_unique = np.unique(first)
    second_unique = np.unique(second)
    sampled_first = rng.choice(first_unique, size=len(first_unique), replace=True)
    sampled_second = rng.choice(second_unique, size=len(second_unique), replace=True)
    first_counts = {group: int(np.count_nonzero(sampled_first == group)) for group in first_unique}
    second_counts = {
        group: int(np.count_nonzero(sampled_second == group)) for group in second_unique
    }
    weights = np.asarray(
        [first_counts[first_group] * second_counts[second_group]
         for first_group, second_group in zip(first, second, strict=True)],
        dtype=int,
    )
    return np.repeat(np.arange(len(first)), weights)


def group_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Return the multivariate variance fraction explained by group means."""
    samples = np.asarray(values, dtype=float)
    identities = np.asarray(groups)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or identities.ndim != 1 or len(samples) != len(identities):
        raise ValueError("Values and groups have incompatible shapes")
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Values must be non-empty and finite")
    grand_mean = samples.mean(axis=0)
    total = float(np.square(samples - grand_mean).sum())
    if total < 1e-15:
        return 0.0
    between = 0.0
    for group in np.unique(identities):
        selected = identities == group
        difference = samples[selected].mean(axis=0) - grand_mean
        between += float(selected.sum() * np.square(difference).sum())
    return float(np.clip(between / total, 0.0, 1.0))


def percentile_interval(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Interval samples must be a non-empty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between zero and one")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(low), float(high)
