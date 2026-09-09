"""Validation-only selector freezing and paired independent-test evaluation."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from openaffect_eeg.exposure_statistics import weighted_scores, _interval

TARGETS = ["target_valence", "target_arousal"]
PREDS = ["prediction_valence", "prediction_arousal"]
KEYS = ["seed", "trial_uid"]
META = KEYS + ["subject_uid", "stimulus_uid"] + TARGETS


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _aligned(table, split):
    required = set(META + PREDS + ["split", "candidate", "participant_dose", "stimulus_dose"])
    if not required.issubset(table.columns) or table.empty or set(table["split"]) != {split}:
        raise ValueError(f"Only a nonempty {split} prediction table with complete columns is accepted")
    if table[list(required)].isna().any().any() or not np.isfinite(table[TARGETS + PREDS]).all().all():
        raise ValueError("Missing metadata or non-finite predictions")
    for column in ("trial_uid", "subject_uid", "stimulus_uid", "candidate"):
        if not table[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError("Identity and candidate fields must be nonempty strings")
    doses = table[["participant_dose", "stimulus_dose"]].to_numpy(float)
    if not np.isfinite(doses).all() or np.any(doses < 0) or np.any(doses != np.floor(doses)):
        raise ValueError("Exposure doses must be nonnegative integers")
    models = sorted(table["candidate"].unique())
    grid = [(int(p), int(s)) for p in sorted(table["participant_dose"].unique())
            for s in sorted(table["stimulus_dose"].unique())]
    arrays, reference = [], None
    for p, s in grid:
        for model in models:
            cell = table.loc[table["candidate"].eq(model) & table["participant_dose"].eq(p)
                             & table["stimulus_dose"].eq(s)].sort_values(KEYS)
            if cell.empty or cell.duplicated(KEYS).any():
                raise ValueError("Missing candidate cell or duplicate trial prediction")
            meta = cell[META].reset_index(drop=True)
            if reference is None:
                reference = meta
            elif not reference.equals(meta):
                raise ValueError("Candidates and cells must have identical truth and test support")
            arrays.append(cell[PREDS].to_numpy(float))
    return reference, np.stack(arrays, axis=1), models, grid


def fit_selector(validation):
    reference, predictions, models, grid = _aligned(validation, "validation")
    truth = reference[TARGETS].to_numpy(float)
    scores = []
    for seed in sorted(reference["seed"].unique()):
        mask = reference["seed"].eq(seed).to_numpy()
        ccc, _ = weighted_scores(truth[mask], predictions[mask], np.ones((1, mask.sum())))
        scores.append(ccc[0])
    mean = np.mean(scores, axis=0).reshape(len(grid), len(models))
    if not np.isfinite(mean).all():
        raise ValueError("Undefined validation score; do not choose candidates from incomplete support")
    # Sorted candidates make exact ties deterministic. Both rules use these same scores.
    global_index = int(np.argmax(mean.mean(0)))
    choices = [int(np.argmax(row)) for row in mean]
    result = {
        "schema_version": 1, "candidate_names": models, "grid": [list(x) for x in grid],
        "global_choice": models[global_index], "cell_choices": [models[i] for i in choices],
        "validation_scores": mean.tolist(), "validation_input_sha256": hashlib.sha256(
            validation.sort_values(["candidate", "participant_dose", "stimulus_dose", *KEYS]).to_csv(index=False).encode()).hexdigest(),
        "validation_trial_uids": sorted(reference["trial_uid"].astype(str).unique()),
        "validation_subject_uids": sorted(reference["subject_uid"].astype(str).unique()),
        "validation_stimulus_uids": sorted(reference["stimulus_uid"].astype(str).unique()),
        "selection_rule": "global uniform validation-cell mean versus declared cell; macro CCC; lexicographic exact ties",
        "scope": "Model selection only; no evidence of independent test benefit until evaluated"}
    result["decision_sha256"] = _digest(result)
    return result


def evaluate_selector(frozen, test, *, iterations=2000, seed=20260902):
    if iterations < 100:
        raise ValueError("At least 100 paired bootstrap draws are required")
    body = {k: v for k, v in frozen.items() if k != "decision_sha256"}
    if _digest(body) != frozen.get("decision_sha256"):
        raise ValueError("Frozen selector hash mismatch")
    reference, predictions, models, grid = _aligned(test, "test")
    if models != frozen["candidate_names"] or [list(x) for x in grid] != frozen["grid"]:
        raise ValueError("Candidate set or resource grid differs from the frozen selector")
    for axis in ("trial", "subject", "stimulus"):
        if set(reference[f"{axis}_uid"].astype(str)) & set(frozen[f"validation_{axis}_uids"]):
            raise ValueError(f"Independent selector test overlaps validation {axis} identities")
    selected = []
    for gi, choice in enumerate(frozen["cell_choices"]):
        for candidate in (frozen["global_choice"], choice):
            selected.append(predictions[:, gi*len(models)+models.index(candidate), :])
    estimates = np.stack(selected, axis=1)
    truth = reference[TARGETS].to_numpy(float)
    subjects, si = np.unique(reference["subject_uid"].astype(str), return_inverse=True)
    stimuli, ti = np.unique(reference["stimulus_uid"].astype(str), return_inverse=True)
    if len(subjects) < 2 or len(stimuli) < 2:
        raise ValueError("Independent test requires replicated participants and stimuli")
    rng = np.random.default_rng(seed)
    sw = rng.multinomial(len(subjects), np.full(len(subjects), 1/len(subjects)), size=iterations)
    tw = rng.multinomial(len(stimuli), np.full(len(stimuli), 1/len(stimuli)), size=iterations)
    weights = np.concatenate([np.ones((1, len(reference))), sw[:, si]*tw[:, ti]])
    cccs, maes = [], []
    for split_seed in sorted(reference["seed"].unique()):
        mask = reference["seed"].eq(split_seed).to_numpy()
        ccc, mae = weighted_scores(truth[mask], estimates[mask], weights[:, mask])
        cccs.append(ccc)
        maes.append(mae)
    ccc = np.mean(cccs, axis=0).reshape(iterations+1, len(grid), 2)
    mae = np.mean(maes, axis=0).reshape(iterations+1, len(grid), 2)
    result = {"decision_sha256": frozen["decision_sha256"], "bootstrap_draws": iterations,
        "resampling": "shared participant x stimulus multiplicities across policies, cells and fixed split seeds",
        "boundary": "Conditional on frozen fits and validation selection; no training-population or selector-training uncertainty",
        "cells": [], "surface": {}}
    for metric, values in (("ccc", ccc), ("mae", mae)):
        delta = values[:, :, 1]-values[:, :, 0]
        lo, hi, count = _interval(delta[1:].mean(1))
        result["surface"][metric] = {"conditional_minus_global": float(delta[0].mean()),
            "ci_low": lo, "ci_high": hi, "valid_bootstrap": count,
            "global": float(values[0, :, 0].mean()), "conditional": float(values[0, :, 1].mean())}
        finite = np.isfinite(delta[1:]).all(1)
        radius = float(np.quantile(np.max(np.abs(delta[1:][finite]-delta[0]), axis=1), .95)) if finite.any() else np.nan
        for i, (p, s) in enumerate(grid):
            lo, hi, count = _interval(delta[1:, i])
            result["cells"].append({"participant_dose": p, "stimulus_dose": s, "metric": metric,
                "conditional_minus_global": float(delta[0, i]), "ci_low": lo, "ci_high": hi,
                "valid_bootstrap": count, "simultaneous_low": float(delta[0, i]-radius),
                "simultaneous_high": float(delta[0, i]+radius)})
    return result
