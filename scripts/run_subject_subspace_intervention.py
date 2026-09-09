"""Intervene on subject-identity directions and re-evaluate affect prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from openaffect_eeg.baselines import harmonize_affect_targets, regression_metrics
from openaffect_eeg.disentanglement import (
    leave_one_out_group_prior,
    predict_seen_group_prior,
)
from openaffect_eeg.foundation import load_foundation_archive
from openaffect_eeg.probes import classification_metrics
from openaffect_eeg.reviewer_controls import (
    FeatureSubspaceProjector,
    GroupIdentityProjector,
)

TARGET_COLUMNS = ["target_valence", "target_arousal"]
SEMANTIC_COLUMNS = [
    "stimulus_uid",
    "stimulus_description",
    "nominal_category",
    "context_code",
    "stimulus_intensity_code",
]
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_COMPONENTS = (0, 1, 2, 4, 8, 16, 24, 29)
DEFAULT_PROJECTION_KINDS = ("subject_centroid", "pca", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_table", type=Path)
    parser.add_argument("split_root", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--components", default=",".join(map(str, DEFAULT_COMPONENTS)))
    parser.add_argument("--projection-kind", action="append")
    parser.add_argument("--permutation-repeats", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_model(alpha: float) -> object:
    features = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["nominal_category", "context_code", "stimulus_intensity_code"],
            ),
            (
                "description",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=1024,
                    sublinear_tf=True,
                ),
                "stimulus_description",
            ),
        ]
    )
    return make_pipeline(features, Ridge(alpha=alpha, solver="lsqr"))


def _aggregate_stimuli(table: pd.DataFrame) -> pd.DataFrame:
    semantics = table.groupby("stimulus_uid", sort=True)[SEMANTIC_COLUMNS[1:]].first()
    targets = table.groupby("stimulus_uid", sort=True)[TARGET_COLUMNS].mean()
    return semantics.join(targets).reset_index()


def _select_semantic_alpha(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    alphas: tuple[float, ...],
) -> tuple[float, dict[str, float]]:
    train_stimuli = _aggregate_stimuli(train)
    validation_y = validation[TARGET_COLUMNS].to_numpy()
    scores: dict[str, float] = {}
    for alpha in alphas:
        model = _semantic_model(alpha)
        model.fit(train_stimuli, train_stimuli[TARGET_COLUMNS].to_numpy())
        scores[f"{alpha:g}"] = float(
            np.mean(np.abs(model.predict(validation) - validation_y))
        )
    selected = min(scores, key=scores.get)
    return float(selected), scores


def _ridge(alpha: float) -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _select_affect_alphas(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    train_y: np.ndarray,
    train_residual_y: np.ndarray,
    validation_y: np.ndarray,
    validation_prior: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, float, dict[str, dict[str, float]]]:
    direct_scores: dict[str, float] = {}
    combined_scores: dict[str, float] = {}
    for alpha in alphas:
        direct = _ridge(alpha)
        direct.fit(train_x, train_y)
        direct_scores[f"{alpha:g}"] = float(
            np.mean(np.abs(direct.predict(validation_x) - validation_y))
        )

        residual = _ridge(alpha)
        residual.fit(train_x, train_residual_y)
        combined = validation_prior + residual.predict(validation_x)
        combined_scores[f"{alpha:g}"] = float(
            np.mean(np.abs(combined - validation_y))
        )
    return (
        float(min(direct_scores, key=direct_scores.get)),
        float(min(combined_scores, key=combined_scores.get)),
        {"direct": direct_scores, "combined": combined_scores},
    )


def _subject_probe(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
    permutation_repeats: int,
) -> dict[str, object]:
    model = RidgeClassifier(alpha=1.0, class_weight="balanced")
    model.fit(fit_x, fit_y)
    observed = classification_metrics(test_y, model.predict(test_x))

    rng = np.random.default_rng(seed)
    permutation_values = []
    for _ in range(permutation_repeats):
        permuted = RidgeClassifier(alpha=1.0, class_weight="balanced")
        permuted.fit(fit_x, rng.permutation(fit_y))
        permutation_values.append(
            classification_metrics(test_y, permuted.predict(test_x))[
                "balanced_accuracy"
            ]
        )
    return {
        **observed,
        "class_count": int(np.unique(fit_y).size),
        "chance_accuracy": 1.0 / np.unique(fit_y).size,
        "permutation_repeats": int(permutation_repeats),
        "permuted_balanced_accuracy_mean": float(np.mean(permutation_values)),
        "permuted_balanced_accuracy_std": float(np.std(permutation_values, ddof=1))
        if len(permutation_values) > 1
        else 0.0,
    }


def evaluate_subject_subspace_assignment(
    available: pd.DataFrame,
    assignment: pd.DataFrame,
    feature_columns: list[str],
    *,
    component_grid: tuple[int, ...] = DEFAULT_COMPONENTS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    permutation_repeats: int = 100,
    projection_kinds: tuple[str, ...] = ("subject_centroid",),
) -> tuple[dict[str, object], pd.DataFrame]:
    experiment = available.merge(
        assignment[["trial_uid", "split", "protocol", "seed"]],
        on="trial_uid",
        how="inner",
        validate="1:1",
    )
    partitions = {
        split: experiment.loc[experiment["split"].eq(split)].copy()
        for split in ("train", "validation", "test")
    }
    if any(len(partitions[split]) < 2 for split in partitions):
        raise ValueError("Subject-subspace intervention needs nonempty partitions")
    train, validation, test = (
        partitions["train"],
        partitions["validation"],
        partitions["test"],
    )
    seed = int(assignment["seed"].iloc[0])
    protocol = str(assignment["protocol"].iloc[0])

    semantic_alpha, semantic_scores = _select_semantic_alpha(train, validation, alphas)
    semantic = _semantic_model(semantic_alpha)
    train_stimuli = _aggregate_stimuli(train)
    semantic.fit(train_stimuli, train_stimuli[TARGET_COLUMNS].to_numpy())
    validation_semantic = semantic.predict(validation)
    validation_prior, _ = predict_seen_group_prior(
        train[TARGET_COLUMNS].to_numpy(),
        train["stimulus_uid"].to_numpy(),
        validation["stimulus_uid"].to_numpy(),
        validation_semantic,
    )
    train_y = train[TARGET_COLUMNS].to_numpy()
    train_residual_y = train_y - leave_one_out_group_prior(
        train_y, train["stimulus_uid"].to_numpy()
    )

    fit = pd.concat([train, validation], ignore_index=True)
    fit_y = fit[TARGET_COLUMNS].to_numpy()
    fit_residual_y = fit_y - leave_one_out_group_prior(
        fit_y, fit["stimulus_uid"].to_numpy()
    )
    final_semantic = _semantic_model(semantic_alpha)
    fit_stimuli = _aggregate_stimuli(fit)
    final_semantic.fit(fit_stimuli, fit_stimuli[TARGET_COLUMNS].to_numpy())
    test_semantic = final_semantic.predict(test)
    test_prior, test_seen_stimulus = predict_seen_group_prior(
        fit_y,
        fit["stimulus_uid"].to_numpy(),
        test["stimulus_uid"].to_numpy(),
        test_semantic,
    )

    curve = []
    prediction_tables = []
    unknown_projection_kinds = set(projection_kinds) - set(DEFAULT_PROJECTION_KINDS)
    if unknown_projection_kinds:
        raise ValueError(f"Unknown projection kinds: {sorted(unknown_projection_kinds)}")
    train_x = train[feature_columns].to_numpy()
    validation_x = validation[feature_columns].to_numpy()
    fit_x = fit[feature_columns].to_numpy()
    test_x = test[feature_columns].to_numpy()
    train_subjects = train["subject_uid"].to_numpy()
    fit_subjects = fit["subject_uid"].to_numpy()
    for projection_kind in projection_kinds:
        for requested_components in component_grid:
            validation_identity = GroupIdentityProjector(
                n_components=requested_components
            ).fit(train_x, train_subjects)
            final_identity = GroupIdentityProjector(
                n_components=requested_components
            ).fit(fit_x, fit_subjects)
            validation_rank = validation_identity.directions_.shape[1]
            final_rank = final_identity.directions_.shape[1]
            if projection_kind == "subject_centroid":
                validation_projector = validation_identity
                final_projector = final_identity
            else:
                validation_projector = FeatureSubspaceProjector(
                    n_components=validation_rank,
                    method=projection_kind,
                    random_state=seed * 1009 + requested_components * 37 + 1,
                ).fit(train_x)
                final_projector = FeatureSubspaceProjector(
                    n_components=final_rank,
                    method=projection_kind,
                    random_state=seed * 1009 + requested_components * 37 + 2,
                ).fit(fit_x)
            projected_train = validation_projector.transform(train_x)
            projected_validation = validation_projector.transform(validation_x)
            direct_alpha, combined_alpha, validation_scores = _select_affect_alphas(
                projected_train,
                projected_validation,
                train_y,
                train_residual_y,
                validation[TARGET_COLUMNS].to_numpy(),
                validation_prior,
                alphas,
            )
            projected_fit = final_projector.transform(fit_x)
            projected_test = final_projector.transform(test_x)

            direct = _ridge(direct_alpha)
            direct.fit(projected_fit, fit_y)
            direct_prediction = direct.predict(projected_test)

            residual = _ridge(combined_alpha)
            residual.fit(projected_fit, fit_residual_y)
            combined_prediction = test_prior + residual.predict(projected_test)

            probe = _subject_probe(
                projected_fit,
                fit_subjects,
                projected_test,
                test["subject_uid"].to_numpy(),
                seed=seed * 1009 + requested_components,
                permutation_repeats=permutation_repeats,
            )
            curve.append(
                {
                    "projection_kind": projection_kind,
                    "requested_components": int(requested_components),
                    "available_components": int(
                        final_identity.available_component_count_
                    ),
                    "removed_components": int(final_projector.directions_.shape[1]),
                    "direct_alpha": direct_alpha,
                    "combined_alpha": combined_alpha,
                    "validation_scores": validation_scores,
                    "direct_metrics": regression_metrics(
                        test[TARGET_COLUMNS].to_numpy(), direct_prediction
                    ),
                    "combined_metrics": regression_metrics(
                        test[TARGET_COLUMNS].to_numpy(), combined_prediction
                    ),
                    "subject_probe": probe,
                }
            )

            predictions = test[
                [
                    "trial_uid",
                    "subject_uid",
                    "stimulus_uid",
                    *TARGET_COLUMNS,
                ]
            ].reset_index(drop=True)
            predictions.insert(
                0, "removed_components", final_projector.directions_.shape[1]
            )
            predictions.insert(0, "requested_components", requested_components)
            predictions.insert(0, "projection_kind", projection_kind)
            predictions.insert(0, "seed", seed)
            predictions.insert(0, "protocol", protocol)
            for index, target in enumerate(("valence", "arousal")):
                predictions[f"semantic_prior_{target}"] = test_prior[:, index]
                predictions[f"direct_eeg_{target}"] = direct_prediction[:, index]
                predictions[f"prior_plus_residual_{target}"] = combined_prediction[:, index]
            prediction_tables.append(predictions)

    result = {
        "protocol": protocol,
        "seed": seed,
        "counts": {split: len(frame) for split, frame in partitions.items()},
        "semantic_alpha": semantic_alpha,
        "semantic_validation_scores": semantic_scores,
        "test_seen_stimulus_fraction": float(test_seen_stimulus.mean()),
        "projection_kinds": list(projection_kinds),
        "curve": curve,
    }
    return result, pd.concat(prediction_tables, ignore_index=True)


def main() -> int:
    args = parse_args()
    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be positive")
    component_grid = tuple(
        sorted({int(value) for value in args.components.split(",") if value.strip()})
    )
    if not component_grid or component_grid[0] < 0:
        raise ValueError("--components must contain non-negative integers")
    projection_kinds = tuple(
        dict.fromkeys(args.projection_kind or DEFAULT_PROJECTION_KINDS)
    )

    trials = harmonize_affect_targets(pd.read_csv(args.trial_table, sep="\t"))
    trials = trials.loc[trials["dataset_id"].eq("ds005540")].copy()
    if "subject_uid" not in trials:
        trials["subject_uid"] = (
            trials["dataset_id"].astype(str) + ":" + trials["subject_id"].astype(str)
        )
    trial_uids, feature_values = load_foundation_archive(args.features)
    features = pd.DataFrame(
        feature_values,
        columns=[f"eeg_feature_{index:04d}" for index in range(feature_values.shape[1])],
    )
    features.insert(0, "trial_uid", trial_uids)
    available = trials.merge(features, on="trial_uid", how="inner", validate="1:1")
    available = available.loc[available["target_available"]].copy()
    feature_columns = [
        column for column in available if column.startswith("eeg_feature_")
    ]

    assignments = sorted((args.split_root / "stimulus_holdout").glob("seed-*.tsv.gz"))
    if not assignments:
        raise ValueError("Stimulus-holdout assignments are required")
    results = []
    prediction_tables = []
    for assignment_path in assignments:
        assignment = pd.read_csv(assignment_path, sep="\t")
        result, predictions = evaluate_subject_subspace_assignment(
            available,
            assignment,
            feature_columns,
            component_grid=component_grid,
            permutation_repeats=args.permutation_repeats,
            projection_kinds=projection_kinds,
        )
        results.append(result)
        prediction_tables.append(predictions)
        identity_curve = [
            row
            for row in result["curve"]
            if row["projection_kind"] == "subject_centroid"
        ]
        baseline = identity_curve[0]
        final = identity_curve[-1]
        print(
            result["seed"],
            "subject BA",
            baseline["subject_probe"]["balanced_accuracy"],
            "->",
            final["subject_probe"]["balanced_accuracy"],
            "combined CCC",
            baseline["combined_metrics"]["macro"]["ccc"],
            "->",
            final["combined_metrics"]["macro"]["ccc"],
            flush=True,
        )

    prediction_artifact = None
    if args.predictions_output is not None:
        predictions = pd.concat(prediction_tables, ignore_index=True)
        identity_columns = [
            "protocol",
            "seed",
            "projection_kind",
            "requested_components",
            "trial_uid",
        ]
        if predictions.duplicated(identity_columns).any():
            raise ValueError("Intervention predictions contain duplicate keys")
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions_output, sep="\t", index=False)
        prediction_artifact = {
            "path": str(args.predictions_output.resolve()),
            "sha256": sha256(args.predictions_output),
            "row_count": len(predictions),
        }

    payload = {
        "analysis": "subject_identity_subspace_intervention",
        "dataset_id": "ds005540",
        "representation": "LaBraM pretrained",
        "trial_table": str(args.trial_table.resolve()),
        "trial_table_sha256": sha256(args.trial_table),
        "features": str(args.features.resolve()),
        "features_sha256": sha256(args.features),
        "component_grid": list(component_grid),
        "projection_kinds": list(projection_kinds),
        "permutation_repeats": args.permutation_repeats,
        "prediction_artifact": prediction_artifact,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
