"""Disjoint identity roles for validation-frozen deployment policy comparisons."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from openaffect_eeg.identity_exposure import (
    TARGET_COLUMNS, compile_exposure_cell, evaluate_exposure_cell,
)


def identity_roles(trials, *, seed, fractions):
    if set(fractions) != {"validation", "test", "tuning"}:
        raise ValueError("Explicit validation, test and tuning fractions required")
    if any(not 0 < value < 1 for value in fractions.values()) or sum(fractions.values()) >= 1:
        raise ValueError("Fractions must leave population-fitting identities")
    if trials.trial_uid.duplicated().any():
        raise ValueError("Duplicate trial identifiers")
    result = trials[["trial_uid", "subject_uid", "stimulus_uid"]].copy()
    for axis in ("subject", "stimulus"):
        column = f"{axis}_uid"
        if trials[column].isna().any() or trials[column].astype(str).str.strip().eq("").any():
            raise ValueError("Missing identity")
        ids = sorted(trials[column].astype(str).unique(), key=lambda x: hashlib.sha256(
            f"{seed}:{axis}:{x}".encode()).hexdigest())
        mapping, start = {}, 0
        for role in ("validation", "test", "tuning"):
            count = max(2, int(len(ids)*fractions[role]))
            mapping.update({x: role for x in ids[start:start+count]})
            start += count
        if len(ids)-start < 2:
            raise ValueError("Too few identities for disjoint fit/tune/validation/test roles")
        mapping.update({x: "fit" for x in ids[start:]})
        result[f"{axis}_role"] = result[column].astype(str).map(mapping)
    return result


def compile_phase(trials, roles, *, phase, p, s, pmax, smax, seed):
    if phase not in {"validation", "test"}:
        raise ValueError("Unknown selector phase")
    columns = ["trial_uid", "subject_uid", "stimulus_uid"]
    reference = trials[columns].sort_values("trial_uid").reset_index(drop=True)
    if not reference.equals(roles[columns].sort_values("trial_uid").reset_index(drop=True)):
        raise ValueError("Role table identities differ from trial metadata")
    table = trials.merge(roles[["trial_uid", "subject_role", "stimulus_role"]],
                         on="trial_uid", validate="1:1")
    forbidden = "test" if phase == "validation" else "validation"
    table = table.loc[~table.subject_role.eq(forbidden) & ~table.stimulus_role.eq(forbidden)].copy()
    strict = table[["trial_uid"]].copy()
    strict["split"] = "excluded"
    for role, split in (("fit", "train"), ("tuning", "validation"), (phase, "test")):
        mask = table.subject_role.eq(role) & table.stimulus_role.eq(role)
        strict.loc[mask, "split"] = split
    # Only target-block outcomes are blinded. Permitted resource labels stay intact.
    blinded = table.copy()
    blinded.loc[strict.split.eq("test"), list(TARGET_COLUMNS)] = 0.0
    assignment, audit = compile_exposure_cell(
        blinded, strict, participant_dose=p, stimulus_dose=s,
        maximum_participant_dose=pmax, maximum_stimulus_dose=smax, seed=seed)
    target = table.loc[table.trial_uid.isin(assignment.loc[assignment.split.eq("test"), "trial_uid"])]
    if any(set(target[f"{axis}_uid"]) & set(roles.loc[
            roles[f"{axis}_role"].eq(forbidden), f"{axis}_uid"]) for axis in ("subject", "stimulus")):
        raise ValueError("Cross-phase target identity overlap")
    audit["selector_phase"] = phase
    audit["quarantined_role"] = forbidden
    audit["target_labels_used_for_assignment"] = False
    return table.drop(columns=["subject_role", "stimulus_role"]), assignment, audit


def phase_predictions(trials, assignment, features, *, phase, alphas, backend):
    """Keep one resource-matched prior; all candidates share exact target support."""
    tables, diagnostics, reference = [], {}, None
    for name, feature_table in features.items():
        result, pred = evaluate_exposure_cell(trials, assignment, feature_table,
                                              alphas=tuple(alphas), backend=backend)
        pred = pred.sort_values("trial_uid").reset_index(drop=True)
        common = ["trial_uid", "subject_uid", "stimulus_uid", *TARGET_COLUMNS]
        prior_columns = ["prior_personalized_valence", "prior_personalized_arousal"]
        if reference is None:
            reference = pred[common + prior_columns]
            predictors = [("prior_personalized", "prior_personalized")]
        else:
            if not reference[common].equals(pred[common]):
                raise ValueError("Candidate test support or truth differs")
            np.testing.assert_allclose(reference[prior_columns], pred[prior_columns], atol=1e-12, rtol=0)
            predictors = []
        predictors.extend([(f"{name}:direct", "eeg_personalized"),
                           (f"{name}:combined", "combined_personalized")])
        for candidate, predictor in predictors:
            output = pred[common + ["seed", "participant_dose", "stimulus_dose"]].copy()
            output["split"] = phase
            output["candidate"] = candidate
            for target in ("valence", "arousal"):
                output[f"prediction_{target}"] = pred[f"{predictor}_{target}"]
            tables.append(output)
        diagnostics[name] = {"selected_alpha": result["selected_alpha"],
                             "validation_mae": result["validation_mae"]}
    if not tables:
        raise ValueError("No candidate features supplied")
    return pd.concat(tables, ignore_index=True), diagnostics
