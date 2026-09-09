#!/usr/bin/env python3
"""Run nonlinear stimulus-adversarial experienced-affect residual models."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from run_disentanglement_baseline import (
    PROTOCOLS,
    TARGET_COLUMNS,
    aggregate_stimuli,
    build_prediction_table,
    fit_semantic_model,
    hybrid_prior,
    load_features,
    load_stimulus_embeddings,
    metrics_by_context,
    semantic_model,
    sha256,
)

from openaffect_eeg.adversarial import (
    AdversarialTrainingConfig,
    fit_adversarial_residual_model,
    fit_adversarial_residual_model_fixed_epochs,
)
from openaffect_eeg.baselines import harmonize_affect_targets
from openaffect_eeg.disentanglement import leave_one_out_group_prior
from openaffect_eeg.statistics import group_eta_squared

DEFAULT_STRENGTHS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5)
ZERO_MODEL = "neural_zero_adversary_prior_plus_eeg_residual"
SELECTED_MODEL = "neural_selected_adversary_prior_plus_eeg_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--strength", type=float, action="append")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--assignment-seed", type=int, action="append")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def select_adversary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: list[str],
    numeric_columns: list[str],
    semantic_alpha: float,
    strengths: tuple[float, ...],
    seed: int,
    config: AdversarialTrainingConfig,
) -> tuple[float, dict[float, dict[str, float | int]]]:
    train_x = train[feature_columns].to_numpy(dtype=float)
    train_y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    validation_x = validation[feature_columns].to_numpy(dtype=float)
    validation_y = validation[TARGET_COLUMNS].to_numpy(dtype=float)
    train_prior = leave_one_out_group_prior(
        train_y, train["stimulus_uid"].astype(str).to_numpy()
    )

    semantic = semantic_model(semantic_alpha, numeric_columns)
    train_stimuli = aggregate_stimuli(train, numeric_columns)
    semantic.fit(train_stimuli, train_stimuli[TARGET_COLUMNS].to_numpy())
    validation_semantic = semantic.predict(validation)
    validation_prior, _ = hybrid_prior(train, validation, validation_semantic)
    validation_residual = validation_y - validation_prior

    scores: dict[float, dict[str, float | int]] = {}
    for strength in strengths:
        fitted = fit_adversarial_residual_model(
            train_x,
            train_y - train_prior,
            train["stimulus_uid"].astype(str).to_numpy(),
            validation_x,
            validation_residual,
            adversary_strength=strength,
            seed=seed,
            config=config,
        )
        scores[strength] = {
            "validation_mae": fitted.validation_mae,
            "best_epoch": fitted.best_epoch,
        }
    selected = min(strengths, key=lambda strength: scores[strength]["validation_mae"])
    return selected, scores


def latent_identity_audit(
    latent: np.ndarray,
    trials: pd.DataFrame,
) -> dict[str, float]:
    return {
        "stimulus_eta_squared": group_eta_squared(
            latent, trials["stimulus_uid"].astype(str).to_numpy()
        ),
        "subject_eta_squared": group_eta_squared(
            latent, trials["subject_id"].astype(str).to_numpy()
        ),
    }


def run_assignment(
    available: pd.DataFrame,
    assignment_path: Path,
    *,
    feature_columns: list[str],
    numeric_columns: list[str],
    strengths: tuple[float, ...],
    config: AdversarialTrainingConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
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
    protocol = str(assignment["protocol"].iloc[0])
    seed = int(assignment["seed"].iloc[0])

    semantic_alpha, semantic_scores = fit_semantic_model(
        train, validation, numeric_columns
    )
    selected_strength, adversary_scores = select_adversary(
        train,
        validation,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        semantic_alpha=semantic_alpha,
        strengths=strengths,
        seed=seed,
        config=config,
    )

    fit = pd.concat([train, validation], ignore_index=True)
    fit_x = fit[feature_columns].to_numpy(dtype=float)
    fit_y = fit[TARGET_COLUMNS].to_numpy(dtype=float)
    test_x = test[feature_columns].to_numpy(dtype=float)
    fit_prior = leave_one_out_group_prior(
        fit_y, fit["stimulus_uid"].astype(str).to_numpy()
    )
    fit_residual = fit_y - fit_prior

    semantic = semantic_model(semantic_alpha, numeric_columns)
    fit_stimuli = aggregate_stimuli(fit, numeric_columns)
    semantic.fit(fit_stimuli, fit_stimuli[TARGET_COLUMNS].to_numpy())
    semantic_prediction = semantic.predict(test)
    prior_prediction, seen = hybrid_prior(fit, test, semantic_prediction)

    zero_epoch_count = int(adversary_scores[0.0]["best_epoch"]) + 1
    zero = fit_adversarial_residual_model_fixed_epochs(
        fit_x,
        fit_residual,
        fit["stimulus_uid"].astype(str).to_numpy(),
        adversary_strength=0.0,
        epoch_count=zero_epoch_count,
        seed=seed,
        config=config,
    )
    if selected_strength == 0.0:
        selected = zero
    else:
        selected_epoch_count = (
            int(adversary_scores[selected_strength]["best_epoch"]) + 1
        )
        selected = fit_adversarial_residual_model_fixed_epochs(
            fit_x,
            fit_residual,
            fit["stimulus_uid"].astype(str).to_numpy(),
            adversary_strength=selected_strength,
            epoch_count=selected_epoch_count,
            seed=seed,
            config=config,
        )

    zero_residual = zero.predict(test_x)
    selected_residual = selected.predict(test_x)
    models = {
        "semantic_stimulus_prior": semantic_prediction,
        "hybrid_stimulus_prior": prior_prediction,
        ZERO_MODEL: prior_prediction + zero_residual,
        SELECTED_MODEL: prior_prediction + selected_residual,
    }
    result: dict[str, object] = {
        "assignment": str(assignment_path.resolve()),
        "protocol": protocol,
        "seed": seed,
        "counts": {split: len(frame) for split, frame in partitions.items()},
        "semantic_alpha": semantic_alpha,
        "semantic_validation_mae": semantic_scores,
        "adversary_validation": {
            f"{strength:g}": values
            for strength, values in adversary_scores.items()
        },
        "selected_adversary_strength": selected_strength,
        "zero_adversary_epoch_count": zero_epoch_count,
        "selected_adversary_epoch_count": selected.best_epoch + 1,
        "test_seen_stimulus_fraction": float(seen.mean()),
        "test_context_counts": test["context"].value_counts().sort_index().to_dict(),
        "latent_identity_audit": {
            "zero_adversary": latent_identity_audit(zero.transform(test_x), test),
            "selected_adversary": latent_identity_audit(
                selected.transform(test_x), test
            ),
        },
        "residual_identity_audit": {
            "zero_adversary": latent_identity_audit(zero_residual, test),
            "selected_adversary": latent_identity_audit(selected_residual, test),
        },
        "models": {
            name: metrics_by_context(test, prediction)
            for name, prediction in models.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    predictions = build_prediction_table(
        test,
        models,
        protocol=protocol,
        seed=seed,
    )
    return result, predictions


def main() -> int:
    args = parse_args()
    strengths = tuple(args.strength) if args.strength else DEFAULT_STRENGTHS
    if 0.0 not in strengths:
        raise ValueError("Strength grid must include the nested zero-adversary baseline")
    if any(not np.isfinite(value) or value < 0.0 for value in strengths):
        raise ValueError("Adversary strengths must be finite and non-negative")
    config = AdversarialTrainingConfig(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        thread_count=args.threads,
    )

    trials = harmonize_affect_targets(pd.read_csv(args.trial_table, sep="\t"))
    trials = trials.loc[trials["dataset_id"].eq("ds005540")].copy()
    numeric_columns: list[str] = []
    if args.semantic_view == "multimodal":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for multimodal semantics")
        embeddings, numeric_columns = load_stimulus_embeddings(
            trials, args.dataset_root
        )
        trials = trials.merge(embeddings, on="stimulus_uid", how="left", validate="m:1")
    features = load_features(args.features, args.feature_view)
    available = trials.merge(features, on="trial_uid", how="inner", validate="1:1")
    available = available.loc[available["target_available"]].copy()
    feature_columns = [
        column for column in available if column.startswith("eeg_feature_")
    ]

    requested_protocols = tuple(args.protocol) if args.protocol else PROTOCOLS
    requested_seeds = set(args.assignment_seed) if args.assignment_seed else None
    assignments = []
    for protocol in requested_protocols:
        for path in sorted((args.split_root / protocol).glob("seed-*.tsv.gz")):
            if requested_seeds is not None:
                seed = int(path.name.removeprefix("seed-").removesuffix(".tsv.gz"))
                if seed not in requested_seeds:
                    continue
            assignments.append(path)
    if not assignments:
        raise ValueError(f"No supported assignments found under {args.split_root}")

    results = []
    prediction_tables = []
    for path in assignments:
        result, predictions = run_assignment(
            available,
            path,
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            strengths=strengths,
            config=config,
        )
        results.append(result)
        prediction_tables.append(predictions)
        zero_metrics = result["models"][ZERO_MODEL]["all"]["macro"]
        selected_metrics = result["models"][SELECTED_MODEL]["all"]["macro"]
        print(
            result["protocol"],
            result["seed"],
            "strength",
            result["selected_adversary_strength"],
            "zero MAE",
            zero_metrics["mae"],
            "selected MAE",
            selected_metrics["mae"],
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

    payload = {
        "dataset_id": "ds005540",
        "objective": (
            "train_only_stimulus_prior plus nonlinear EEG experienced-affect "
            "residual with gradient-reversal stimulus invariance"
        ),
        "device": "cpu",
        "feature_view": args.feature_view,
        "feature_count": len(feature_columns),
        "semantic_view": args.semantic_view,
        "numeric_semantic_feature_count": len(numeric_columns),
        "adversary_strengths": strengths,
        "training_config": asdict(config),
        "selection_policy": (
            "Validation residual MAE selects strength and epoch; test is read once "
            "after fixed-epoch refit on train plus validation"
        ),
        "zero_adversary_policy": (
            "Identical architecture, optimizer, stimulus head, and selection; "
            "gradient-reversal strength fixed to zero"
        ),
        "trial_table": str(args.trial_table.resolve()),
        "trial_table_sha256": sha256(args.trial_table),
        "features": str(args.features.resolve()),
        "features_sha256": sha256(args.features),
        "available_trial_count": len(available),
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
