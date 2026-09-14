"""Joint identity and training-seed uncertainty for matched resource surfaces."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openaffect_eeg.exposure_statistics import weighted_scores


def _identity_weights(
    table: pd.DataFrame,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    participants, participant_index = np.unique(
        table["subject_uid"].astype(str), return_inverse=True
    )
    stimuli, stimulus_index = np.unique(
        table["stimulus_uid"].astype(str), return_inverse=True
    )
    participant_weights = rng.multinomial(
        len(participants),
        np.full(len(participants), 1.0 / len(participants)),
        size=iterations,
    )
    stimulus_weights = rng.multinomial(
        len(stimuli),
        np.full(len(stimuli), 1.0 / len(stimuli)),
        size=iterations,
    )
    bootstrap = (
        participant_weights[:, participant_index]
        * stimulus_weights[:, stimulus_index]
    )
    return np.concatenate([np.ones((1, len(table))), bootstrap], axis=0)


def analyze_training_uncertainty(
    predictions: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 2026091400,
    equivalence_margin: float = 0.05,
    strict_margin: float = 0.025,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Analyze a complete grid with shared identity and seed resampling.

    The input must contain out-of-fold predictions for every training seed and
    resource cell. Assignment folds are concatenated into one fixed support per
    seed. This function includes stochastic training-seed uncertainty but does
    not invent unexecuted fold rotations or arbitrary hyperparameter searches.
    """
    if iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required")
    if not 0 < strict_margin <= equivalence_margin:
        raise ValueError("Equivalence margins must be positive and ordered")
    required = {
        "assignment_seed",
        "training_seed",
        "participant_dose",
        "stimulus_dose",
        "trial_uid",
        "subject_uid",
        "stimulus_uid",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    targets = tuple(
        name
        for name in ("valence", "arousal")
        if {
            f"target_{name}",
            f"prior_personalized_{name}",
            f"combined_personalized_{name}",
        }.issubset(predictions.columns)
    )
    if not targets:
        raise ValueError("No complete target prediction triplet was found")

    training_seeds = sorted(int(value) for value in predictions.training_seed.unique())
    if len(training_seeds) < 2:
        raise ValueError("At least two independently trained seeds are required")
    participant_doses = sorted(int(value) for value in predictions.participant_dose.unique())
    stimulus_doses = sorted(int(value) for value in predictions.stimulus_dose.unique())
    grid = [(p, s) for p in participant_doses for s in stimulus_doses]
    if len(grid) != len(participant_doses) * len(stimulus_doses):
        raise ValueError("Malformed resource grid")

    key = ["assignment_seed", "trial_uid"]
    metadata_columns = [
        *key,
        "subject_uid",
        "stimulus_uid",
        *[f"target_{name}" for name in targets],
    ]
    reference: pd.DataFrame | None = None
    arrays: list[np.ndarray] = []
    for training_seed in training_seeds:
        for participant_dose, stimulus_dose in grid:
            cell = predictions.loc[
                predictions.training_seed.eq(training_seed)
                & predictions.participant_dose.eq(participant_dose)
                & predictions.stimulus_dose.eq(stimulus_dose)
            ].sort_values(key)
            if cell.empty or cell.duplicated(key).any():
                raise ValueError("Missing or duplicate seed/grid cell")
            metadata = cell[metadata_columns].reset_index(drop=True)
            if reference is None:
                reference = metadata
            elif not reference.equals(metadata):
                raise ValueError("Training seeds and grid cells lack fixed test support")
            arrays.extend(
                cell[[f"{predictor}_{name}" for name in targets]].to_numpy(float)
                for predictor in ("prior_personalized", "combined_personalized")
            )
    assert reference is not None
    estimates = np.stack(arrays, axis=1)
    truth = reference[[f"target_{name}" for name in targets]].to_numpy(float)
    if not np.isfinite(truth).all() or not np.isfinite(estimates).all():
        raise ValueError("Truth and predictions must be finite")

    rng = np.random.default_rng(seed)
    weights = _identity_weights(reference, iterations=iterations, rng=rng)
    ccc, mae = weighted_scores(truth, estimates, weights)
    shape = (iterations + 1, len(training_seeds), len(grid), 2)
    ccc = ccc.reshape(shape)
    mae = mae.reshape(shape)
    ccc_delta = ccc[..., 1] - ccc[..., 0]
    mae_delta = mae[..., 1] - mae[..., 0]

    seed_draws = rng.integers(
        0,
        len(training_seeds),
        size=(iterations, len(training_seeds)),
    )
    point_cells = ccc_delta[0].mean(axis=0)
    point_mae_cells = mae_delta[0].mean(axis=0)
    bootstrap_cells = np.empty((iterations, len(grid)))
    bootstrap_mae_cells = np.empty((iterations, len(grid)))
    for draw in range(iterations):
        bootstrap_cells[draw] = ccc_delta[draw + 1, seed_draws[draw]].mean(axis=0)
        bootstrap_mae_cells[draw] = mae_delta[
            draw + 1, seed_draws[draw]
        ].mean(axis=0)
    valid = np.isfinite(bootstrap_cells).all(axis=1)
    if valid.sum() < max(100, int(iterations * 0.9)):
        raise RuntimeError("Too many degenerate crossed-bootstrap draws")
    bootstrap_cells = bootstrap_cells[valid]
    bootstrap_mae_cells = bootstrap_mae_cells[valid]

    errors = np.max(np.abs(bootstrap_cells - point_cells), axis=1)
    simultaneous_radius = float(np.quantile(errors, 0.95))
    rows = []
    for index, (participant_dose, stimulus_dose) in enumerate(grid):
        low, high = np.quantile(bootstrap_cells[:, index], [0.025, 0.975])
        mae_low, mae_high = np.quantile(
            bootstrap_mae_cells[:, index], [0.025, 0.975]
        )
        rows.append(
            {
                "participant_dose": participant_dose,
                "stimulus_dose": stimulus_dose,
                "ccc_delta": float(point_cells[index]),
                "ccc_ci_low": float(low),
                "ccc_ci_high": float(high),
                "ccc_simultaneous_low": float(
                    point_cells[index] - simultaneous_radius
                ),
                "ccc_simultaneous_high": float(
                    point_cells[index] + simultaneous_radius
                ),
                "mae_delta": float(point_mae_cells[index]),
                "mae_ci_low": float(mae_low),
                "mae_ci_high": float(mae_high),
            }
        )

    primary_point = float(point_cells.mean())
    primary_draws = bootstrap_cells.mean(axis=1)
    primary_low, primary_high = np.quantile(primary_draws, [0.025, 0.975])
    equivalence_low, equivalence_high = np.quantile(primary_draws, [0.05, 0.95])

    def equivalence(margin: float) -> dict[str, object]:
        established = bool(equivalence_low > -margin and equivalence_high < margin)
        return {
            "margin": margin,
            "alpha_each_side": 0.05,
            "decision_interval_90": [
                float(equivalence_low), float(equivalence_high)
            ],
            "established": established,
            "decision": "equivalent" if established else "inconclusive",
        }

    report: dict[str, object] = {
        "status": "complete",
        "targets": list(targets),
        "training_seeds": training_seeds,
        "training_seed_count": len(training_seeds),
        "assignment_seeds": sorted(
            int(value) for value in reference.assignment_seed.unique()
        ),
        "grid": [list(item) for item in grid],
        "test_rows_per_cell": len(reference),
        "participants": int(reference.subject_uid.nunique()),
        "stimuli": int(reference.stimulus_uid.nunique()),
        "bootstrap_iterations_requested": iterations,
        "bootstrap_iterations_valid": int(valid.sum()),
        "bootstrap_seed": seed,
        "resampling": (
            "Shared participant-by-stimulus multiplicities and resampled "
            "executed training seeds."
        ),
        "primary": {
            "estimand": "uniform mean matched EEG CCC increment over resource cells",
            "estimate": primary_point,
            "ci_95": [float(primary_low), float(primary_high)],
            "equivalence_primary": equivalence(equivalence_margin),
            "equivalence_strict_sensitivity": equivalence(strict_margin),
        },
        "simultaneous_cell_band": {
            "method": "bootstrap max absolute deviation",
            "confidence": 0.95,
            "radius": simultaneous_radius,
        },
        "boundary": (
            "Includes identity and executed training-seed uncertainty. Fold "
            "rotations are fixed unless distinct rotation outputs are supplied; "
            "arbitrary architecture and tuning-policy uncertainty is not covered."
        ),
    }
    return pd.DataFrame(rows), report
