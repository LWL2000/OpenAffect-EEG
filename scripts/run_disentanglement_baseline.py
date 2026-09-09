"""Run the stimulus-prior plus experienced-affect residual baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from openaffect_eeg.baselines import (
    harmonize_affect_targets,
    regression_metrics,
)
from openaffect_eeg.disentanglement import (
    ConditionalNuisanceRidge,
    ResponsePreservingStimulusProjector,
    leave_one_out_group_prior,
    predict_seen_group_prior,
    remove_group_nuisance,
)
from openaffect_eeg.emo_eeg import pool_video_embedding
from openaffect_eeg.foundation import load_foundation_archive

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
NUISANCE_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONDITIONAL_PENALTIES = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
PROJECTION_COMPONENTS = (0, 1, 2, 4, 8, 16, 32)
TARGET_COLUMNS = ["target_valence", "target_arousal"]
SEMANTIC_COLUMNS = [
    "stimulus_uid",
    "stimulus_description",
    "nominal_category",
    "context_code",
    "stimulus_intensity_code",
]
PROTOCOLS = (
    "trial_random_holdout",
    "support_matched_identity_overlap",
    "subject_holdout",
    "stimulus_holdout",
    "subject_stimulus_holdout",
    "video_to_imagery",
    "imagery_to_video",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_table", type=Path)
    parser.add_argument("split_root", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument(
        "--feature-view",
        choices=("global", "channel", "foundation"),
        default="channel",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--semantic-view",
        choices=("structured_text", "multimodal"),
        default="structured_text",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(path: Path, view: str) -> pd.DataFrame:
    if view == "foundation":
        trial_uids, values = load_foundation_archive(path)
    else:
        with np.load(path, allow_pickle=False) as archive:
            trial_uids = archive["trial_uid"].astype(str)
            if view == "global":
                values = archive["global_log_bandpower"].astype(float)
            else:
                channel = archive["channel_log_bandpower"].astype(float)
                values = channel.reshape(len(channel), -1)
    table = pd.DataFrame(
        values,
        columns=[f"eeg_feature_{index:04d}" for index in range(values.shape[1])],
    )
    table.insert(0, "trial_uid", trial_uids)
    if table["trial_uid"].duplicated().any():
        raise ValueError("Feature trial identifiers are not unique")
    return table


def load_stimulus_embeddings(
    trials: pd.DataFrame,
    dataset_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    stimuli = trials[
        ["stimulus_uid", "context_code", "stimulus_embedding_path"]
    ].drop_duplicates()
    pooled_by_stimulus: dict[str, np.ndarray] = {}
    width: int | None = None
    for row in stimuli.itertuples(index=False):
        if row.context_code != "vid":
            continue
        values = pool_video_embedding(dataset_root / str(row.stimulus_embedding_path))
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError("Pooled video embedding widths differ")
        pooled_by_stimulus[str(row.stimulus_uid)] = values
    if width is None:
        raise ValueError("No video stimulus embeddings were loaded")
    columns = [f"stimulus_embedding_{index:04d}" for index in range(width)]
    records = []
    for row in stimuli.itertuples(index=False):
        values = pooled_by_stimulus.get(str(row.stimulus_uid), np.zeros(width))
        records.append(
            {"stimulus_uid": str(row.stimulus_uid), **dict(zip(columns, values))}
        )
    return pd.DataFrame(records), columns


def semantic_model(alpha: float, numeric_columns: list[str]) -> object:
    transformers: list[tuple[str, object, object]] = [
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
    if numeric_columns:
        transformers.append(
            (
                "published_video_embedding",
                StandardScaler(with_mean=False),
                numeric_columns,
            )
        )
    features = ColumnTransformer(transformers)
    return make_pipeline(features, Ridge(alpha=alpha, solver="lsqr"))


def eeg_model(alpha: float) -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def select_conditional_eeg_model(
    train_x: np.ndarray,
    residual_y: np.ndarray,
    train_stimulus: np.ndarray,
    validation_x: np.ndarray,
    validation_prior: np.ndarray,
    validation_y: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    scores: dict[str, float] = {}
    for penalty in CONDITIONAL_PENALTIES:
        for alpha in ALPHAS:
            model = ConditionalNuisanceRidge(alpha=alpha, penalty=penalty)
            model.fit(train_x, residual_y, train_stimulus)
            prediction = validation_prior + model.predict(validation_x)
            key = f"penalty={penalty:g},alpha={alpha:g}"
            scores[key] = float(np.mean(np.abs(prediction - validation_y)))
    selected = min(scores, key=scores.get)
    selected_parts = dict(
        part.split("=", maxsplit=1) for part in selected.split(",")
    )
    return (
        float(selected_parts["alpha"]),
        float(selected_parts["penalty"]),
        scores,
    )


def select_response_preserving_eeg_model(
    train_x: np.ndarray,
    residual_y: np.ndarray,
    train_stimulus: np.ndarray,
    validation_x: np.ndarray,
    validation_prior: np.ndarray,
    validation_y: np.ndarray,
) -> tuple[float, int, dict[str, float]]:
    scores: dict[str, float] = {}
    for components in PROJECTION_COMPONENTS:
        projector = ResponsePreservingStimulusProjector(n_components=components)
        projected_train = projector.fit_transform(
            train_x, residual_y, train_stimulus
        )
        projected_validation = projector.transform(validation_x)
        for alpha in ALPHAS:
            model = eeg_model(alpha)
            model.fit(projected_train, residual_y)
            prediction = validation_prior + model.predict(projected_validation)
            key = f"components={components},alpha={alpha:g}"
            scores[key] = float(np.mean(np.abs(prediction - validation_y)))
    selected = min(scores, key=scores.get)
    selected_parts = dict(
        part.split("=", maxsplit=1) for part in selected.split(",")
    )
    return (
        float(selected_parts["alpha"]),
        int(selected_parts["components"]),
        scores,
    )


def aggregate_stimuli(
    table: pd.DataFrame,
    numeric_columns: list[str],
    value_columns: list[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    semantics = table.groupby("stimulus_uid", sort=True)[
        [*SEMANTIC_COLUMNS[1:], *numeric_columns]
    ].first()
    values = table.groupby("stimulus_uid", sort=True)[value_columns].mean()
    return semantics.join(values).reset_index()


def fit_semantic_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[float, dict[float, float]]:
    train_stimuli = aggregate_stimuli(train, numeric_columns)
    validation_y = validation[TARGET_COLUMNS].to_numpy()
    scores: dict[float, float] = {}
    for alpha in ALPHAS:
        model = semantic_model(alpha, numeric_columns)
        model.fit(train_stimuli, train_stimuli[TARGET_COLUMNS].to_numpy())
        prediction = model.predict(validation)
        scores[alpha] = float(np.mean(np.abs(prediction - validation_y)))
    selected = min(scores, key=scores.get)
    return selected, scores


def fit_semantic_nuisance_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    numeric_columns: list[str],
    feature_columns: list[str],
) -> tuple[float, dict[float, float]]:
    train_stimuli = aggregate_stimuli(train, numeric_columns, feature_columns)
    validation_x = validation[feature_columns].to_numpy()
    scale = train[feature_columns].std(axis=0).to_numpy(copy=True)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    scores: dict[float, float] = {}
    for alpha in ALPHAS:
        model = semantic_model(alpha, numeric_columns)
        model.fit(train_stimuli, train_stimuli[feature_columns].to_numpy())
        prediction = model.predict(validation)
        scores[alpha] = float(np.mean(np.abs((prediction - validation_x) / scale)))
    selected = min(scores, key=scores.get)
    return selected, scores


def hybrid_prior(
    fit: pd.DataFrame,
    requested: pd.DataFrame,
    semantic_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return predict_seen_group_prior(
        fit[TARGET_COLUMNS].to_numpy(),
        fit["stimulus_uid"].to_numpy(),
        requested["stimulus_uid"].to_numpy(),
        semantic_prediction,
    )


def select_eeg_alphas(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    semantic_alpha: float,
    feature_columns: list[str],
    numeric_columns: list[str],
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    dict[str, dict[object, float]],
]:
    train_x = train[feature_columns].to_numpy()
    train_y = train[TARGET_COLUMNS].to_numpy()
    validation_x = validation[feature_columns].to_numpy()
    validation_y = validation[TARGET_COLUMNS].to_numpy()
    semantic = semantic_model(semantic_alpha, numeric_columns)
    train_stimuli = aggregate_stimuli(train, numeric_columns)
    semantic.fit(train_stimuli, train_stimuli[TARGET_COLUMNS].to_numpy())
    validation_semantic = semantic.predict(validation)
    validation_prior, _ = hybrid_prior(train, validation, validation_semantic)
    train_loo_prior = leave_one_out_group_prior(
        train_y, train["stimulus_uid"].to_numpy()
    )
    residual_y = train_y - train_loo_prior
    conditional_alpha, conditional_penalty, conditional_scores = (
        select_conditional_eeg_model(
            train_x,
            residual_y,
            train["stimulus_uid"].to_numpy(),
            validation_x,
            validation_prior,
            validation_y,
        )
    )
    projected_alpha, projected_components, projected_scores = (
        select_response_preserving_eeg_model(
            train_x,
            residual_y,
            train["stimulus_uid"].to_numpy(),
            validation_x,
            validation_prior,
            validation_y,
        )
    )
    nuisance_alpha, nuisance_scores = fit_semantic_nuisance_model(
        train,
        validation,
        numeric_columns,
        feature_columns,
    )
    semantic_nuisance = semantic_model(nuisance_alpha, numeric_columns)
    train_stimuli = aggregate_stimuli(train, numeric_columns, feature_columns)
    semantic_nuisance.fit(
        train_stimuli, train_stimuli[feature_columns].to_numpy()
    )
    validation_nuisance_fallback = semantic_nuisance.predict(validation)
    direct_scores: dict[float, float] = {}
    residual_scores: dict[float, float] = {}
    debiased_scores: dict[str, float] = {}
    for alpha in ALPHAS:
        direct = eeg_model(alpha)
        direct.fit(train_x, train_y)
        direct_scores[alpha] = float(
            np.mean(np.abs(direct.predict(validation_x) - validation_y))
        )
        residual = eeg_model(alpha)
        residual.fit(train_x, residual_y)
        residual_prediction = validation_prior + residual.predict(validation_x)
        residual_scores[alpha] = float(
            np.mean(np.abs(residual_prediction - validation_y))
        )
    for strength in NUISANCE_STRENGTHS:
        train_debiased_x, validation_debiased_x, _ = remove_group_nuisance(
            train_x,
            train["stimulus_uid"].to_numpy(),
            validation_x,
            validation["stimulus_uid"].to_numpy(),
            validation_nuisance_fallback,
            strength=strength,
        )
        for alpha in ALPHAS:
            debiased = eeg_model(alpha)
            debiased.fit(train_debiased_x, residual_y)
            debiased_prediction = validation_prior + debiased.predict(
                validation_debiased_x
            )
            key = f"strength={strength:g},alpha={alpha:g}"
            debiased_scores[key] = float(
                np.mean(np.abs(debiased_prediction - validation_y))
            )
    selected_debiased = min(debiased_scores, key=debiased_scores.get)
    selected_parts = dict(
        part.split("=", maxsplit=1) for part in selected_debiased.split(",")
    )
    return (
        min(direct_scores, key=direct_scores.get),
        min(residual_scores, key=residual_scores.get),
        float(selected_parts["alpha"]),
        float(selected_parts["strength"]),
        conditional_alpha,
        conditional_penalty,
        projected_alpha,
        projected_components,
        {
            "direct_eeg": direct_scores,
            "disentangled": residual_scores,
            "semantic_eeg_nuisance": nuisance_scores,
            "semantic_nuisance_removed": debiased_scores,
            "conditional_stimulus_invariant": conditional_scores,
            "response_preserving_projection": projected_scores,
        },
    )


def metrics_by_context(
    test: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {
        "all": regression_metrics(test[TARGET_COLUMNS].to_numpy(), prediction)
    }
    for context in sorted(test["context"].unique()):
        selected = test["context"].eq(context).to_numpy()
        if selected.sum() >= 2:
            result[str(context)] = regression_metrics(
                test.loc[selected, TARGET_COLUMNS].to_numpy(), prediction[selected]
            )
    return result


def build_prediction_table(
    trials: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    protocol: str,
    seed: int,
) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in (
            "trial_uid",
            "dataset_id",
            "subject_id",
            "stimulus_uid",
            "context",
            "context_code",
            *TARGET_COLUMNS,
        )
        if column in trials
    ]
    if "trial_uid" not in metadata_columns or not set(TARGET_COLUMNS).issubset(
        metadata_columns
    ):
        raise ValueError("Prediction rows require trial IDs and affect targets")
    table = trials[metadata_columns].reset_index(drop=True).copy()
    if table["trial_uid"].duplicated().any():
        raise ValueError("Prediction trial identifiers must be unique per assignment")
    table.insert(0, "seed", int(seed))
    table.insert(0, "protocol", str(protocol))
    for name, values in predictions.items():
        prediction = np.asarray(values, dtype=float)
        if prediction.shape != (len(table), len(TARGET_COLUMNS)):
            raise ValueError(
                f"Prediction {name} has shape {prediction.shape}; "
                f"expected {(len(table), len(TARGET_COLUMNS))}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"Prediction {name} contains non-finite values")
        for index, target in enumerate(("valence", "arousal")):
            table[f"{name}_{target}"] = prediction[:, index]
    return table


def run_assignment(
    available: pd.DataFrame,
    assignment_path: Path,
    feature_columns: list[str],
    numeric_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    assignment = pd.read_csv(assignment_path, sep="\t")
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
        raise ValueError(f"Insufficient EmoEEG-MC rows for {assignment_path}")
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]

    semantic_alpha, semantic_scores = fit_semantic_model(
        train, validation, numeric_columns
    )
    (
        direct_alpha,
        residual_alpha,
        debiased_alpha,
        debiased_strength,
        conditional_alpha,
        conditional_penalty,
        projected_alpha,
        projected_components,
        eeg_scores,
    ) = select_eeg_alphas(
        train, validation, semantic_alpha, feature_columns, numeric_columns
    )
    nuisance_alpha = min(
        eeg_scores["semantic_eeg_nuisance"],
        key=eeg_scores["semantic_eeg_nuisance"].get,
    )
    fit = pd.concat([train, validation], ignore_index=True)
    fit_x = fit[feature_columns].to_numpy()
    fit_y = fit[TARGET_COLUMNS].to_numpy()
    test_x = test[feature_columns].to_numpy()

    semantic = semantic_model(semantic_alpha, numeric_columns)
    fit_stimuli = aggregate_stimuli(fit, numeric_columns)
    semantic.fit(fit_stimuli, fit_stimuli[TARGET_COLUMNS].to_numpy())
    semantic_prediction = semantic.predict(test)
    prior_prediction, seen = hybrid_prior(fit, test, semantic_prediction)

    direct = eeg_model(direct_alpha)
    direct.fit(fit_x, fit_y)
    direct_prediction = direct.predict(test_x)

    fit_prior = leave_one_out_group_prior(
        fit_y, fit["stimulus_uid"].to_numpy()
    )
    residual = eeg_model(residual_alpha)
    residual.fit(fit_x, fit_y - fit_prior)
    residual_component = residual.predict(test_x)
    disentangled_prediction = prior_prediction + residual_component

    conditional = ConditionalNuisanceRidge(
        alpha=conditional_alpha,
        penalty=conditional_penalty,
    )
    conditional.fit(
        fit_x,
        fit_y - fit_prior,
        fit["stimulus_uid"].to_numpy(),
    )
    conditional_component = conditional.predict(test_x)
    conditional_prediction = prior_prediction + conditional_component

    projector = ResponsePreservingStimulusProjector(
        n_components=projected_components
    )
    projected_fit_x = projector.fit_transform(
        fit_x,
        fit_y - fit_prior,
        fit["stimulus_uid"].to_numpy(),
    )
    projected_test_x = projector.transform(test_x)
    projected = eeg_model(projected_alpha)
    projected.fit(projected_fit_x, fit_y - fit_prior)
    projected_component = projected.predict(projected_test_x)
    projected_prediction = prior_prediction + projected_component

    semantic_nuisance = semantic_model(nuisance_alpha, numeric_columns)
    nuisance_stimuli = aggregate_stimuli(fit, numeric_columns, feature_columns)
    semantic_nuisance.fit(
        nuisance_stimuli, nuisance_stimuli[feature_columns].to_numpy()
    )
    test_nuisance_fallback = semantic_nuisance.predict(test)
    fit_debiased_x, test_debiased_x, nuisance_seen = remove_group_nuisance(
        fit_x,
        fit["stimulus_uid"].to_numpy(),
        test_x,
        test["stimulus_uid"].to_numpy(),
        test_nuisance_fallback,
        strength=debiased_strength,
    )
    debiased = eeg_model(debiased_alpha)
    debiased.fit(fit_debiased_x, fit_y - fit_prior)
    debiased_component = debiased.predict(test_debiased_x)
    debiased_prediction = prior_prediction + debiased_component
    global_prediction = np.repeat(fit_y.mean(axis=0, keepdims=True), len(test), axis=0)

    models = {
        "global_mean": global_prediction,
        "semantic_stimulus_prior": semantic_prediction,
        "hybrid_stimulus_prior": prior_prediction,
        "direct_eeg": direct_prediction,
        "disentangled_prior_plus_eeg_residual": disentangled_prediction,
        "conditional_stimulus_invariant_eeg_residual": conditional_prediction,
        "response_preserving_projected_eeg_residual": projected_prediction,
        "semantic_nuisance_removed_eeg_residual": debiased_prediction,
    }
    protocol = str(assignment["protocol"].iloc[0])
    seed = int(assignment["seed"].iloc[0])
    result = {
        "assignment": str(assignment_path.resolve()),
        "protocol": protocol,
        "seed": seed,
        "counts": {split: len(frame) for split, frame in partitions.items()},
        "selected_alpha": {
            "semantic": semantic_alpha,
            "direct_eeg": direct_alpha,
            "eeg_residual": residual_alpha,
            "semantic_eeg_nuisance": nuisance_alpha,
            "debiased_eeg_residual": debiased_alpha,
            "nuisance_removal_strength": debiased_strength,
            "conditional_eeg_residual": conditional_alpha,
            "conditional_stimulus_penalty": conditional_penalty,
            "projected_eeg_residual": projected_alpha,
            "response_preserving_projection_components": projected_components,
        },
        "validation_mae": {
            "semantic": semantic_scores,
            **eeg_scores,
        },
        "test_seen_stimulus_fraction": float(seen.mean()),
        "test_seen_eeg_nuisance_fraction": float(nuisance_seen.mean()),
        "test_context_counts": test["context"].value_counts().sort_index().to_dict(),
        "residual_component_std": residual_component.std(axis=0).tolist(),
        "conditional_residual_component_std": conditional_component.std(
            axis=0
        ).tolist(),
        "projected_residual_component_std": projected_component.std(axis=0).tolist(),
        "debiased_residual_component_std": debiased_component.std(axis=0).tolist(),
        "models": {
            name: metrics_by_context(test, prediction)
            for name, prediction in models.items()
        },
    }
    prediction_table = build_prediction_table(
        test,
        models,
        protocol=protocol,
        seed=seed,
    )
    return result, prediction_table


def main() -> int:
    args = parse_args()
    trials = harmonize_affect_targets(pd.read_csv(args.trial_table, sep="\t"))
    trials = trials.loc[trials["dataset_id"].eq("ds005540")].copy()
    numeric_columns: list[str] = []
    if args.semantic_view == "multimodal":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for multimodal semantics")
        stimulus_embeddings, numeric_columns = load_stimulus_embeddings(
            trials, args.dataset_root
        )
        trials = trials.merge(
            stimulus_embeddings, on="stimulus_uid", how="left", validate="m:1"
        )
    features = load_features(args.features, args.feature_view)
    available = trials.merge(features, on="trial_uid", how="inner", validate="1:1")
    available = available.loc[available["target_available"]].copy()
    feature_columns = [
        column for column in available if column.startswith("eeg_feature_")
    ]
    assignments = [
        path
        for protocol in PROTOCOLS
        for path in sorted((args.split_root / protocol).glob("seed-*.tsv.gz"))
    ]
    if not assignments:
        raise ValueError(f"No supported assignments found under {args.split_root}")

    results = []
    prediction_tables = []
    for path in assignments:
        result, prediction_table = run_assignment(
            available, path, feature_columns, numeric_columns
        )
        results.append(result)
        prediction_tables.append(prediction_table)
        metrics = result["models"]["disentangled_prior_plus_eeg_residual"]["all"][
            "macro"
        ]
        print(
            result["protocol"],
            result["seed"],
            "seen",
            result["test_seen_stimulus_fraction"],
            "CCC",
            metrics["ccc"],
            "MAE",
            metrics["mae"],
            flush=True,
        )

    prediction_artifact: dict[str, object] | None = None
    if args.predictions_output is not None:
        predictions = pd.concat(prediction_tables, ignore_index=True)
        identity_columns = ["protocol", "seed", "trial_uid"]
        if predictions.duplicated(identity_columns).any():
            raise ValueError("Prediction archive contains duplicate assignment rows")
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions_output, sep="\t", index=False)
        prediction_artifact = {
            "path": str(args.predictions_output.resolve()),
            "sha256": sha256(args.predictions_output),
            "row_count": len(predictions),
            "column_count": len(predictions.columns),
        }
        print(f"Predictions: {args.predictions_output}")

    payload = {
        "dataset_id": "ds005540",
        "objective": (
            "y_subject_stimulus = train_only_stimulus_prior + EEG_experienced_"
            "affect_residual"
        ),
        "target_normalization": "MusicEEG/DENS (x-1)/8; EmoEEG-MC x/7",
        "semantic_features": (
            "English stimulus description TF-IDF plus nominal category, context, "
            "and exemplar intensity code; multimodal mode additionally includes "
            "mean-pooled published 768-D visual ViT and 768-D audio DeiT features"
        ),
        "semantic_view": args.semantic_view,
        "numeric_semantic_feature_count": len(numeric_columns),
        "train_residual_policy": (
            "Leave-one-trial-out stimulus means; singleton stimuli fall back to "
            "the leave-one-trial-out global mean"
        ),
        "feature_view": args.feature_view,
        "feature_count": len(feature_columns),
        "available_trial_count": len(available),
        "trial_table": str(args.trial_table.resolve()),
        "trial_table_sha256": sha256(args.trial_table),
        "features": str(args.features.resolve()),
        "features_sha256": sha256(args.features),
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
