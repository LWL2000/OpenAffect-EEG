"""Known-estimand simulation and production-pipeline validation for EEG audits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openaffect_eeg.exposure_statistics import CONTRASTS, PREDICTORS


SCENARIOS = {
    "labels_only": (0.0, 0.0, 0.0),
    "trial_signal": (0.5, 0.0, 0.0),
    "participant_constant": (0.0, 1.0, 0.0),
    "independent_noise": (0.0, 0.0, 0.6),
}
A_SD, B_SD, BETA = 0.7, 0.8, 0.5


def population_scores(scenario, noise_sd, p, s):
    """Exact marginal CCC from independent Gaussian component coefficients."""
    k, lam, nu = SCENARIOS[scenario]
    # A, B, Z, eps, prior error, calibration B/Z/eps/eta, test eta.
    target = np.zeros(10)
    target[:4] = A_SD, B_SD, BETA, noise_sd
    prior = np.zeros(10)
    if s:
        prior[1] = B_SD
        prior[4] = np.sqrt((A_SD**2 + BETA**2 + noise_sd**2) / s)
    eeg = np.zeros(10)
    eeg[[0, 2, 9]] = lam*A_SD, k, nu
    offset = np.zeros(10)
    eeg_cal = np.zeros(10)
    if p:
        offset[[0, 5, 6, 7]] = A_SD, B_SD/np.sqrt(p), BETA/np.sqrt(p), noise_sd/np.sqrt(p)
        eeg_cal[[0, 6, 8]] = lam*A_SD, k/np.sqrt(p), nu/np.sqrt(p)
    vectors = {
        "prior": prior, "prior_personalized": prior+offset,
        "eeg_population": eeg, "eeg_personalized": eeg+offset-eeg_cal,
        "combined_population": prior+eeg,
        "combined_personalized": prior+eeg+offset-eeg_cal,
    }
    return {name: float(2*np.dot(target, v)/(np.dot(target, target)+np.dot(v, v)))
            for name, v in vectors.items()}


def simulate_oracle(*, seed, participants, stimuli, density, noise_sd, scenario,
                    participant_doses, stimulus_doses):
    """Sample nested resource sets and exact-predictor test tables."""
    if not 0 < density <= 1 or min(participants, stimuli) < 2 or noise_sd <= 0:
        raise ValueError("Invalid simulation support/noise")
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(participants, 1, 2))*A_SD
    b = rng.normal(size=(1, stimuli, 2))*B_SD
    z = rng.normal(size=(participants, stimuli, 2))
    eps = rng.normal(size=z.shape)*noise_sd
    eta = rng.normal(size=z.shape)
    pmax, smax = max(participant_doses), max(stimulus_doses)
    cb = rng.normal(size=(participants, pmax, 2))*B_SD
    cz = rng.normal(size=cb.shape)
    ce = rng.normal(size=cb.shape)*noise_sd
    cn = rng.normal(size=cb.shape)
    prior_errors = rng.normal(size=(stimuli, smax, 2))*np.sqrt(A_SD**2+BETA**2+noise_sd**2)
    mask = rng.random((participants, stimuli)) < density
    ii, jj = np.where(mask)
    if len(ii) < 4 or len(np.unique(ii)) < 2 or len(np.unique(jj)) < 2:
        raise ValueError("Insufficient sampled test support")
    k, lam, nu = SCENARIOS[scenario]
    y = a+b+BETA*z+eps
    g = k*z+lam*a+nu*eta
    records, truth_rows = [], []
    for p in participant_doses:
        if p:
            off = a[:, 0] + (cb[:, :p]+BETA*cz[:, :p]+ce[:, :p]).mean(1)
            gcal = lam*a[:, 0] + (k*cz[:, :p]+nu*cn[:, :p]).mean(1)
        else:
            off = gcal = np.zeros((participants, 2))
        for s in stimulus_doses:
            prior = b[0]+prior_errors[:, :s].mean(1) if s else np.zeros((stimuli, 2))
            values = {
                "prior": prior[jj], "prior_personalized": prior[jj]+off[ii],
                "eeg_population": g[ii, jj], "eeg_personalized": g[ii, jj]+off[ii]-gcal[ii],
                "combined_population": prior[jj]+g[ii, jj],
                "combined_personalized": prior[jj]+g[ii, jj]+off[ii]-gcal[ii],
            }
            table = pd.DataFrame({
                "seed": seed, "representation": scenario,
                "trial_uid": [f"{i}:{j}" for i, j in zip(ii, jj)],
                "subject_uid": ii.astype(str), "stimulus_uid": jj.astype(str),
                "participant_dose": p, "stimulus_dose": s,
                "target_valence": y[ii, jj, 0], "target_arousal": y[ii, jj, 1],
            })
            for name in PREDICTORS:
                for q, target_name in enumerate(("valence", "arousal")):
                    table[f"{name}_{target_name}"] = values[name][:, q]
            records.append(table)
            exact = population_scores(scenario, noise_sd, p, s)
            for contrast, (right, left) in CONTRASTS.items():
                truth_rows.append({"participant_dose": p, "stimulus_dose": s,
                                   "contrast": contrast, "true_delta": exact[right]-exact[left]})
    return pd.concat(records, ignore_index=True), pd.DataFrame(truth_rows)


def simulate_trials(*, seed, participants, stimuli, density, noise_sd, scenario):
    """Independent production test with observed features, not oracle predictions."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(participants, 2))*A_SD
    b = rng.normal(size=(stimuli, 2))*B_SD
    z = rng.normal(size=(participants, stimuli, 2))
    y = a[:, None]+b[None]+BETA*z+rng.normal(size=z.shape)*noise_sd
    ii, jj = np.where(rng.random((participants, stimuli)) < density)
    trials = pd.DataFrame({"trial_uid": [f"{i}:{j}" for i, j in zip(ii, jj)],
        "subject_uid": ii.astype(str), "stimulus_uid": jj.astype(str),
        "dataset_id": "synthetic", "context": "simulation",
        "target_valence": y[ii, jj, 0], "target_arousal": y[ii, jj, 1]})
    if scenario == "labels_only":
        x = np.zeros((len(ii), 2))
    elif scenario == "trial_signal":
        x = z[ii, jj]
    elif scenario == "participant_constant":
        x = a[ii]
    elif scenario == "independent_noise":
        x = rng.normal(size=(len(ii), 2))
    else:
        raise ValueError("Unknown scenario")
    features = pd.DataFrame(x, columns=["feature_0000", "feature_0001"])
    features.insert(0, "trial_uid", trials.trial_uid)
    return trials, features


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return float("nan"), float("nan")
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return float(center-radius), float(center+radius)


def summarize_oracle(records):
    keys = ["scenario", "density", "noise_sd", "participant_dose", "stimulus_dose", "contrast"]
    rows = []
    for identity, group in records.groupby(keys, sort=True):
        row = dict(zip(keys, identity))
        valid = np.isfinite(group[["ccc_ci_low", "ccc_ci_high"]]).all(axis=1)
        usable = group.loc[valid]
        # Exact algebraic cancellation can leave roundoff-scale CCC differences.
        covered = ((usable.ccc_ci_low <= usable.true_delta+1e-12)
                   & (usable.true_delta <= usable.ccc_ci_high+1e-12))
        detected = usable.ccc_ci_low > 1e-12
        lo, hi = wilson_interval(int(covered.sum()), len(usable))
        dl, dh = wilson_interval(int(detected.sum()), len(usable))
        row.update(replicates=len(group), valid_replicates=len(usable),
            true_delta=float(group.true_delta.iloc[0]),
            mean_estimate=float(usable.ccc_delta.mean()),
            coverage=float(covered.mean()), coverage_mc_low=lo, coverage_mc_high=hi,
            positive_detection_rate=float(detected.mean()), detection_mc_low=dl, detection_mc_high=dh,
            positive_rate_interpretation="false_positive" if group.true_delta.iloc[0] <= 1e-12 else "power",
            mean_interval_width=float((usable.ccc_ci_high-usable.ccc_ci_low).mean()),
            mean_test_trials=float(group.n_test_trials.mean()))
        rows.append(row)
    return pd.DataFrame(rows)
