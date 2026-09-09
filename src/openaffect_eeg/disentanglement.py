"""Leakage-resistant stimulus-prior and experienced-affect decomposition."""

from __future__ import annotations

import numpy as np


class ConditionalNuisanceRidge:
    """Ridge regression that penalizes stimulus-linked prediction covariance."""

    def __init__(self, *, alpha: float, penalty: float):
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("Ridge alpha must be finite and non-negative")
        if not np.isfinite(penalty) or penalty < 0.0:
            raise ValueError("Nuisance penalty must be finite and non-negative")
        self.alpha = float(alpha)
        self.penalty = float(penalty)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        nuisance_groups: np.ndarray,
    ) -> ConditionalNuisanceRidge:
        values = np.asarray(features, dtype=float)
        responses, groups = _validate_targets_and_groups(targets, nuisance_groups)
        if values.ndim != 2 or values.shape[0] != responses.shape[0]:
            raise ValueError("Features and targets have incompatible shapes")
        if not np.isfinite(values).all():
            raise ValueError("Features contain non-finite values")

        self.feature_mean_ = values.mean(axis=0)
        self.feature_scale_ = values.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-12] = 1.0
        standardized = (values - self.feature_mean_) / self.feature_scale_
        self.target_mean_ = responses.mean(axis=0)
        centered_targets = responses - self.target_mean_

        identities = np.unique(groups)
        indicators = np.column_stack([groups == group for group in identities]).astype(
            float
        )
        indicators -= indicators.mean(axis=0)
        indicator_scale = indicators.std(axis=0)
        indicator_scale[indicator_scale < 1e-12] = 1.0
        indicators /= indicator_scale

        gram = standardized.T @ standardized
        cross_nuisance = standardized.T @ indicators
        nuisance_gram = cross_nuisance @ cross_nuisance.T / len(standardized)
        regularized = (
            gram
            + self.alpha * np.eye(standardized.shape[1])
            + self.penalty * nuisance_gram
        )
        self.coef_ = np.linalg.solve(regularized, standardized.T @ centered_targets)
        self.nuisance_group_count_ = len(identities)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise ValueError("Conditional nuisance Ridge is not fitted")
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean_):
            raise ValueError("Requested features have an incompatible shape")
        if not np.isfinite(values).all():
            raise ValueError("Requested features contain non-finite values")
        standardized = (values - self.feature_mean_) / self.feature_scale_
        return self.target_mean_ + standardized @ self.coef_


class ResponsePreservingStimulusProjector:
    """Remove stimulus directions orthogonal to affect-predictive covariance."""

    def __init__(self, *, n_components: int):
        if not isinstance(n_components, int) or n_components < 0:
            raise ValueError("Projection component count must be a non-negative integer")
        self.n_components = n_components

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        nuisance_groups: np.ndarray,
    ) -> ResponsePreservingStimulusProjector:
        values = np.asarray(features, dtype=float)
        responses, groups = _validate_targets_and_groups(targets, nuisance_groups)
        if values.ndim != 2 or values.shape[0] != responses.shape[0]:
            raise ValueError("Features and targets have incompatible shapes")
        if not np.isfinite(values).all():
            raise ValueError("Features contain non-finite values")

        self.feature_mean_ = values.mean(axis=0)
        self.feature_scale_ = values.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-12] = 1.0
        standardized = (values - self.feature_mean_) / self.feature_scale_
        centered_targets = responses - responses.mean(axis=0)
        affect_cross = standardized.T @ centered_targets
        affect_basis = self._row_space_basis(affect_cross.T).T

        centroid_rows = []
        for group in np.unique(groups):
            selected = groups == group
            centroid_rows.append(
                np.sqrt(selected.sum()) * standardized[selected].mean(axis=0)
            )
        stimulus_centroids = np.asarray(centroid_rows)
        if affect_basis.shape[1]:
            stimulus_centroids -= (
                stimulus_centroids @ affect_basis
            ) @ affect_basis.T
        stimulus_basis = self._row_space_basis(stimulus_centroids)
        component_count = min(self.n_components, len(stimulus_basis))
        self.directions_ = stimulus_basis[:component_count].T
        self.available_component_count_ = len(stimulus_basis)
        return self

    @staticmethod
    def _row_space_basis(matrix: np.ndarray) -> np.ndarray:
        _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        if not len(singular_values) or singular_values[0] < 1e-12:
            return np.empty((0, matrix.shape[1]))
        tolerance = (
            max(matrix.shape) * np.finfo(float).eps * singular_values[0]
        )
        return right[singular_values > tolerance]

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "directions_"):
            raise ValueError("Response-preserving projector is not fitted")
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean_):
            raise ValueError("Requested features have an incompatible shape")
        if not np.isfinite(values).all():
            raise ValueError("Requested features contain non-finite values")
        standardized = (values - self.feature_mean_) / self.feature_scale_
        if not self.directions_.shape[1]:
            return standardized
        return standardized - (standardized @ self.directions_) @ self.directions_.T

    def fit_transform(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        nuisance_groups: np.ndarray,
    ) -> np.ndarray:
        return self.fit(features, targets, nuisance_groups).transform(features)


def _validate_targets_and_groups(
    targets: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(targets, dtype=float)
    identities = np.asarray(groups)
    if values.ndim != 2 or len(values) != len(identities):
        raise ValueError("Targets and groups have incompatible shapes")
    if len(values) < 2:
        raise ValueError("At least two target rows are required")
    if not np.isfinite(values).all():
        raise ValueError("Targets contain non-finite values")
    return values, identities


def leave_one_out_group_prior(
    targets: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Estimate each row's group prior without using that row's target."""

    values, identities = _validate_targets_and_groups(targets, groups)
    total_sum = values.sum(axis=0)
    result = np.empty_like(values)
    for group in np.unique(identities):
        selected = identities == group
        group_count = int(selected.sum())
        if group_count > 1:
            group_sum = values[selected].sum(axis=0)
            result[selected] = (group_sum - values[selected]) / (group_count - 1)
        else:
            row_index = int(np.flatnonzero(selected)[0])
            result[row_index] = (total_sum - values[row_index]) / (len(values) - 1)
    return result


def predict_seen_group_prior(
    train_targets: np.ndarray,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    semantic_fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use train-only group means when available and semantics otherwise."""

    values, identities = _validate_targets_and_groups(train_targets, train_groups)
    requested = np.asarray(test_groups)
    fallback = np.asarray(semantic_fallback, dtype=float)
    expected_shape = (len(requested), values.shape[1])
    if fallback.shape != expected_shape:
        raise ValueError(
            f"Semantic fallback has shape {fallback.shape}; expected {expected_shape}"
        )
    if not np.isfinite(fallback).all():
        raise ValueError("Semantic fallback contains non-finite values")

    means = {
        group: values[identities == group].mean(axis=0)
        for group in np.unique(identities)
    }
    seen = np.asarray([group in means for group in requested], dtype=bool)
    prediction = fallback.copy()
    for index, group in enumerate(requested):
        if seen[index]:
            prediction[index] = means[group]
    return prediction, seen


def remove_group_nuisance(
    train_features: np.ndarray,
    train_groups: np.ndarray,
    requested_features: np.ndarray,
    requested_groups: np.ndarray,
    semantic_fallback: np.ndarray,
    *,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove train-only stimulus centroids or semantic estimates from features."""
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Nuisance-removal strength must be between zero and one")
    train_values, identities = _validate_targets_and_groups(
        train_features, train_groups
    )
    requested_values = np.asarray(requested_features, dtype=float)
    if requested_values.ndim != 2 or requested_values.shape[1] != train_values.shape[1]:
        raise ValueError("Requested features and train features have incompatible shapes")
    if not np.isfinite(requested_values).all():
        raise ValueError("Requested features contain non-finite values")

    train_nuisance = leave_one_out_group_prior(train_values, identities)
    requested_nuisance, seen = predict_seen_group_prior(
        train_values,
        identities,
        requested_groups,
        semantic_fallback,
    )
    return (
        train_values - strength * train_nuisance,
        requested_values - strength * requested_nuisance,
        seen,
    )
