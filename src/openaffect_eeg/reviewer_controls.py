"""Reviewer-driven controls for protocol support and identity reliance."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


class ReviewerControlError(ValueError):
    """Raised when a reviewer control cannot satisfy its audit contract."""


def _seeded_rank(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def build_support_matched_overlap_split(
    trials: pd.DataFrame,
    reference_assignment: pd.DataFrame,
    *,
    subject_column: str,
    stimulus_column: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Match a joint split's support while deliberately restoring identity overlap.

    Test trials are copied exactly from the reference assignment. Training and
    validation counts are also copied, but their trials are selected from all
    remaining eligible rows. The training subset is seeded with a deterministic
    cover so every test subject and stimulus is represented whenever possible.
    """

    required_trial_columns = {"trial_uid", subject_column, stimulus_column}
    missing_trials = sorted(required_trial_columns - set(trials.columns))
    if missing_trials:
        raise ReviewerControlError(
            f"Trial table is missing columns: {', '.join(missing_trials)}"
        )
    missing_reference = sorted(
        {"trial_uid", "split"} - set(reference_assignment.columns)
    )
    if missing_reference:
        raise ReviewerControlError(
            f"Reference assignment is missing columns: {', '.join(missing_reference)}"
        )
    if trials["trial_uid"].duplicated().any():
        raise ReviewerControlError("Eligible trial identifiers must be unique")
    if reference_assignment["trial_uid"].duplicated().any():
        raise ReviewerControlError("Reference trial identifiers must be unique")

    eligible = trials[list(required_trial_columns)].copy()
    if eligible[[subject_column, stimulus_column]].isna().any().any():
        raise ReviewerControlError("Eligible identity columns cannot contain missing values")
    eligible["trial_uid"] = eligible["trial_uid"].astype(str)
    eligible = eligible.sort_values("trial_uid").reset_index(drop=True)

    reference = reference_assignment[["trial_uid", "split"]].copy()
    reference["trial_uid"] = reference["trial_uid"].astype(str)
    joined = eligible.merge(reference, on="trial_uid", how="inner", validate="1:1")
    counts = {
        split: int(joined["split"].eq(split).sum())
        for split in ("train", "validation", "test")
    }
    if any(counts[split] < 1 for split in counts):
        raise ReviewerControlError("Reference assignment needs nonempty train, validation, and test")

    test = joined.loc[joined["split"].eq("test")].copy()
    test_ids = set(test["trial_uid"])
    pool = eligible.loc[~eligible["trial_uid"].isin(test_ids)].copy()
    required_pool_count = counts["train"] + counts["validation"]
    if len(pool) < required_pool_count:
        raise ReviewerControlError(
            "Eligible pool is too small to match reference train and validation counts"
        )

    test_subjects = set(test[subject_column])
    test_stimuli = set(test[stimulus_column])
    uncovered = {("subject", value) for value in test_subjects} | {
        ("stimulus", value) for value in test_stimuli
    }

    coverage: dict[str, set[tuple[str, object]]] = {}
    for row in pool.itertuples(index=False):
        row_values = row._asdict()
        covered: set[tuple[str, object]] = set()
        if row_values[subject_column] in test_subjects:
            covered.add(("subject", row_values[subject_column]))
        if row_values[stimulus_column] in test_stimuli:
            covered.add(("stimulus", row_values[stimulus_column]))
        coverage[str(row_values["trial_uid"])] = covered

    selected_train: list[str] = []
    available_ids = set(pool["trial_uid"].astype(str))
    while uncovered:
        ranked = sorted(
            available_ids,
            key=lambda trial_uid: (
                -len(coverage[trial_uid] & uncovered),
                _seeded_rank(seed, trial_uid),
                trial_uid,
            ),
        )
        best = ranked[0]
        newly_covered = coverage[best] & uncovered
        if not newly_covered:
            missing = ", ".join(
                f"{kind}={value}" for kind, value in sorted(uncovered, key=str)
            )
            raise ReviewerControlError(
                f"Test identities cannot be represented outside test: {missing}"
            )
        selected_train.append(best)
        available_ids.remove(best)
        uncovered -= newly_covered

    if len(selected_train) > counts["train"]:
        raise ReviewerControlError(
            "Reference training count is too small to cover all test identities"
        )

    remaining_ranked = sorted(
        available_ids, key=lambda trial_uid: (_seeded_rank(seed, trial_uid), trial_uid)
    )
    selected_train.extend(
        remaining_ranked[: counts["train"] - len(selected_train)]
    )
    remaining_after_train = [
        trial_uid for trial_uid in remaining_ranked if trial_uid not in selected_train
    ]
    selected_validation = remaining_after_train[: counts["validation"]]

    split_by_trial = {
        **dict.fromkeys(selected_train, "train"),
        **dict.fromkeys(selected_validation, "validation"),
        **dict.fromkeys(test_ids, "test"),
    }
    assignment = pd.DataFrame(
        {
            "trial_uid": list(split_by_trial),
            "split": list(split_by_trial.values()),
            "protocol": "support_matched_identity_overlap",
            "seed": int(seed),
        }
    ).sort_values("trial_uid", ignore_index=True)

    assigned = eligible.merge(assignment, on="trial_uid", validate="1:1")
    train = assigned.loc[assigned["split"].eq("train")]
    matched_test = assigned.loc[assigned["split"].eq("test")]
    train_subjects = set(train[subject_column])
    train_stimuli = set(train[stimulus_column])
    audit = {
        "seed": int(seed),
        "protocol": "support_matched_identity_overlap",
        "reference_protocol": str(
            reference_assignment.get("protocol", pd.Series(["unknown"])).iloc[0]
        ),
        "reference_counts": counts,
        "matched_counts": {
            split: int(assignment["split"].eq(split).sum()) for split in counts
        },
        "same_test_trials": set(matched_test["trial_uid"]) == test_ids,
        "test_subject_count": len(test_subjects),
        "test_stimulus_count": len(test_stimuli),
        "subject_overlap_fraction": len(test_subjects & train_subjects)
        / len(test_subjects),
        "stimulus_overlap_fraction": len(test_stimuli & train_stimuli)
        / len(test_stimuli),
    }
    return assignment, audit


class GroupIdentityProjector:
    """Remove leading between-group centroid directions learned on training data."""

    def __init__(self, *, n_components: int):
        if not isinstance(n_components, int) or n_components < 0:
            raise ReviewerControlError(
                "Identity component count must be a non-negative integer"
            )
        self.n_components = n_components

    def fit(self, features: np.ndarray, groups: np.ndarray) -> GroupIdentityProjector:
        values = np.asarray(features, dtype=float)
        identities = np.asarray(groups)
        if values.ndim != 2 or len(values) != len(identities):
            raise ReviewerControlError("Features and identity labels have incompatible shapes")
        if len(values) < 2 or not np.isfinite(values).all():
            raise ReviewerControlError("Identity projection requires finite feature rows")

        self.feature_mean_ = values.mean(axis=0)
        self.feature_scale_ = values.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-12] = 1.0
        standardized = (values - self.feature_mean_) / self.feature_scale_

        centroid_rows = []
        for group in np.unique(identities):
            selected = identities == group
            centroid_rows.append(
                np.sqrt(selected.sum()) * standardized[selected].mean(axis=0)
            )
        centroids = np.asarray(centroid_rows)
        _, singular_values, right = np.linalg.svd(centroids, full_matrices=False)
        if not len(singular_values) or singular_values[0] < 1e-12:
            basis = np.empty((0, values.shape[1]))
        else:
            tolerance = max(centroids.shape) * np.finfo(float).eps * singular_values[0]
            basis = right[singular_values > tolerance]
        self.available_component_count_ = len(basis)
        component_count = min(self.n_components, self.available_component_count_)
        self.directions_ = basis[:component_count].T
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "directions_"):
            raise ReviewerControlError("Identity projector is not fitted")
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean_):
            raise ReviewerControlError("Requested features have an incompatible shape")
        if not np.isfinite(values).all():
            raise ReviewerControlError("Requested features contain non-finite values")
        standardized = (values - self.feature_mean_) / self.feature_scale_
        if not self.directions_.shape[1]:
            return standardized
        return standardized - (standardized @ self.directions_) @ self.directions_.T

    def fit_transform(self, features: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return self.fit(features, groups).transform(features)


class FeatureSubspaceProjector:
    """Remove an equal-rank PCA or seeded random subspace after standardization."""

    def __init__(
        self,
        *,
        n_components: int,
        method: str,
        random_state: int = 0,
    ):
        if not isinstance(n_components, int) or n_components < 0:
            raise ReviewerControlError(
                "Subspace component count must be a non-negative integer"
            )
        if method not in {"pca", "random"}:
            raise ReviewerControlError(f"Unknown feature subspace method: {method}")
        self.n_components = n_components
        self.method = method
        self.random_state = int(random_state)

    def fit(self, features: np.ndarray) -> FeatureSubspaceProjector:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ReviewerControlError(
                "Feature subspace projection requires finite feature rows"
            )
        self.feature_mean_ = values.mean(axis=0)
        self.feature_scale_ = values.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-12] = 1.0
        standardized = (values - self.feature_mean_) / self.feature_scale_
        available = min(values.shape[0] - 1, values.shape[1])
        component_count = min(self.n_components, available)
        self.available_component_count_ = available
        if component_count == 0:
            self.directions_ = np.empty((values.shape[1], 0))
            return self
        if self.method == "pca":
            _, _, right = np.linalg.svd(standardized, full_matrices=False)
            directions = right[:component_count].T
        else:
            rng = np.random.default_rng(self.random_state)
            random_matrix = rng.normal(size=(values.shape[1], component_count))
            directions, _ = np.linalg.qr(random_matrix, mode="reduced")
        self.directions_ = directions
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "directions_"):
            raise ReviewerControlError("Feature subspace projector is not fitted")
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean_):
            raise ReviewerControlError("Requested features have an incompatible shape")
        if not np.isfinite(values).all():
            raise ReviewerControlError("Requested features contain non-finite values")
        standardized = (values - self.feature_mean_) / self.feature_scale_
        if not self.directions_.shape[1]:
            return standardized
        return standardized - (standardized @ self.directions_) @ self.directions_.T

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
