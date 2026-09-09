"""Prespecified crossed-bootstrap interval diagnostics, separate from production."""
from __future__ import annotations

import numpy as np
from scipy.stats import t

from openaffect_eeg.exposure_statistics import weighted_scores

METHODS = ("percentile", "basic", "normal_t")


def intervals(point, draws, cluster_df):
    """Three conventional candidates; t correction is heuristic, not exact."""
    samples = np.asarray(draws, float)
    if samples.ndim != 2 or not np.isfinite(samples).all() or cluster_df < 1:
        raise ValueError("Invalid draws or cluster degrees of freedom")
    point = np.asarray(point, float)
    lo, hi = np.quantile(samples, [.025, .975], axis=0)
    radius = t.ppf(.975, cluster_df)*samples.std(axis=0, ddof=1)
    return {"percentile": (lo, hi), "basic": (2*point-hi, 2*point-lo),
            "normal_t": (point-radius, point+radius)}


def paired_primary_draws(table, *, iterations, seed):
    """Shared identity weights; same oracle support and two-target macro CCC."""
    estimates, ref, grid = [], None, []
    columns = ["trial_uid", "subject_uid", "stimulus_uid", "target_valence", "target_arousal"]
    for (p, s), cell in table.groupby(["participant_dose", "stimulus_dose"], sort=True):
        cell = cell.sort_values("trial_uid").reset_index(drop=True)
        if ref is None:
            ref = cell[columns]
        elif not ref.equals(cell[columns]):
            raise ValueError("Different fixed test support")
        grid.append((int(p), int(s)))
        for name in ("prior_personalized", "combined_personalized"):
            estimates.append(cell[[name+"_valence", name+"_arousal"]].to_numpy(float))
    _, pi = np.unique(ref.subject_uid, return_inverse=True)
    _, si = np.unique(ref.stimulus_uid, return_inverse=True)
    npart, nstim = pi.max()+1, si.max()+1
    rng = np.random.default_rng(seed)
    pw = rng.multinomial(npart, np.full(npart, 1/npart), size=iterations)
    sw = rng.multinomial(nstim, np.full(nstim, 1/nstim), size=iterations)
    w = np.concatenate([np.ones((1, len(ref))), pw[:, pi]*sw[:, si]])
    scores, _ = weighted_scores(ref[["target_valence", "target_arousal"]].to_numpy(),
                                np.stack(estimates, axis=1), w)
    delta = scores[:, 1::2]-scores[:, ::2]
    valid = np.isfinite(delta[1:]).all(axis=1)
    return delta[0], delta[1:][valid], grid, {
        "n_participants": int(npart), "n_stimuli": int(nstim), "n_trials": len(ref),
        "valid_bootstrap": int(valid.sum()), "invalid_bootstrap": int((~valid).sum())}
