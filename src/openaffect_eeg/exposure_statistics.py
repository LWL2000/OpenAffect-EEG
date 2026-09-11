"""Paired crossed-identity uncertainty for resource-matched EEG added value.

Participants and stimuli receive shared bootstrap multiplicities across fixed
splits, cells and models. This keeps overlapping test observations correlated.
Fits are held fixed; intervals are not full training-population uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTORS = ("prior", "prior_personalized", "eeg_population", "eeg_personalized",
              "combined_population", "combined_personalized")
CONTRASTS = {
    "eeg_added_population": ("combined_population", "prior"),
    "eeg_added_matched_calibration": ("combined_personalized", "prior_personalized"),
    "prior_calibration_gain": ("prior_personalized", "prior"),
    "combined_calibration_gain": ("combined_personalized", "combined_population"),
    "total_minus_uncalibrated_prior": ("combined_personalized", "prior"),
}


def weighted_scores(truth, estimates, weights):
    """B x N weights, N x Q x D estimates -> B x Q macro CCC and MAE."""
    y = np.asarray(truth, dtype=float)
    x = np.asarray(estimates, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 3 or y.ndim != 2 or y.shape != (x.shape[0], x.shape[2]) or w.shape[1] != len(y):
        raise ValueError("Incompatible weighted prediction dimensions")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Non-finite truth or predictions")
    total = w.sum(1, keepdims=True)
    norm = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    mx = np.einsum("bn,nqt->bqt", norm, x, optimize=True)
    my = (norm @ y)[:, None, :]
    ex2 = np.einsum("bn,nqt->bqt", norm, x*x, optimize=True)
    ey2 = (norm @ (y*y))[:, None, :]
    exy = np.einsum("bn,nqt->bqt", norm, x*y[:, None, :], optimize=True)
    denominator = np.maximum(ex2-mx*mx, 0) + np.maximum(ey2-my*my, 0) + (mx-my)**2
    ccc = np.divide(2*(exy-mx*my), denominator, out=np.full_like(mx, np.nan),
                    where=denominator > 1e-15)
    count = np.isfinite(ccc).sum(-1)
    macro = np.divide(np.nansum(ccc, axis=-1), count,
                      out=np.full(count.shape, np.nan), where=count > 0)
    mae = np.einsum("bn,nqt->bqt", norm, np.abs(x-y[:, None, :]), optimize=True).mean(-1)
    macro[total[:, 0] == 0] = np.nan
    mae[total[:, 0] == 0] = np.nan
    return macro, mae


def _interval(draws, *, point=None, method="percentile", cluster_df=None):
    values = np.asarray(draws, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    lo, hi = np.quantile(finite, [0.025, 0.975])
    if method == "basic":
        if point is None:
            raise ValueError("Basic interval requires its paired point estimate")
        lo, hi = 2*point-hi, 2*point-lo
    elif method in {"normal_t", "block_t"}:
        if point is None or cluster_df is None or cluster_df < 1 or len(finite) < 2:
            return np.nan, np.nan, len(finite)
        from scipy.stats import t
        radius = t.ppf(.975, cluster_df)*finite.std(ddof=1)
        lo, hi = point-radius, point+radius
    elif method != "percentile":
        raise ValueError("Unknown interval method")
    return float(lo), float(hi), len(finite)


def analyze_predictions(
    predictions,
    *,
    iterations=2000,
    seed=20260902,
    paired_models=(),
    interval_method="percentile",
    target_names=None,
):
    """Analyze a complete model x p x s grid; fail on mismatched test support."""
    if iterations < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    if interval_method not in {"percentile", "basic", "normal_t", "block_t"}:
        raise ValueError("Unknown interval method")
    table = predictions.copy()
    subject = "subject_uid" if "subject_uid" in table else "subject_id"
    stimulus = "stimulus_uid" if "stimulus_uid" in table else "condition_id"
    keys = ["seed", "trial_uid"]
    models = sorted(table["representation"].unique())
    p_values = sorted(table["participant_dose"].unique())
    s_values = sorted(table["stimulus_dose"].unique())
    splits = sorted(table["seed"].unique())
    grid = [(p, s) for p in p_values for s in s_values]
    if target_names is None:
        target_names = tuple(
            name
            for name in ("valence", "arousal")
            if f"target_{name}" in table
            and all(f"{predictor}_{name}" in table for predictor in PREDICTORS)
        )
    else:
        target_names = tuple(str(name) for name in target_names)
    if not target_names or len(target_names) != len(set(target_names)):
        raise ValueError("Could not identify unique prediction targets")
    target_columns = [f"target_{name}" for name in target_names]
    metadata = keys + [subject, stimulus, *target_columns]
    reference = None
    arrays = []
    for model in models:
        for p, s in grid:
            cell = table.loc[table["representation"].eq(model)
                             & table["participant_dose"].eq(p)
                             & table["stimulus_dose"].eq(s)].sort_values(keys)
            if cell.duplicated(keys).any() or cell.empty:
                raise ValueError("Missing or duplicate grid cell")
            if reference is None:
                reference = cell[metadata].reset_index(drop=True)
            elif not reference.equals(cell[metadata].reset_index(drop=True)):
                raise ValueError("Models/cells do not share identical truth and test support")
            for pred in PREDICTORS:
                arrays.append(
                    cell[[f"{pred}_{name}" for name in target_names]].to_numpy(float)
                )
    estimate = np.stack(arrays, axis=1)
    truth = reference[target_columns].to_numpy(float)
    subjects, si = np.unique(reference[subject].astype(str), return_inverse=True)
    stimuli, ti = np.unique(reference[stimulus].astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    # A draw assigns one multiplicity per identity, shared even across split seeds.
    sw = rng.multinomial(len(subjects), np.full(len(subjects), 1/len(subjects)), size=iterations)
    tw = rng.multinomial(len(stimuli), np.full(len(stimuli), 1/len(stimuli)), size=iterations)
    weights = np.concatenate([np.ones((1, len(reference))), sw[:, si]*tw[:, ti]], axis=0)
    split_ccc, split_mae = [], []
    supports = []
    for split in splits:
        mask = reference["seed"].eq(split).to_numpy()
        ccc, mae = weighted_scores(truth[mask], estimate[mask], weights[:, mask])
        split_ccc.append(ccc)
        split_mae.append(mae)
        supports.append({"seed": int(split), "n_trials": int(mask.sum()),
            "n_participants": int(reference.loc[mask, subject].nunique()),
            "n_stimuli_or_conditions": int(reference.loc[mask, stimulus].nunique())})
    # Do not silently drop a seed in a degenerate bootstrap draw.
    ccc = np.mean(split_ccc, axis=0)
    mae = np.mean(split_mae, axis=0)
    shape = (iterations+1, len(models), len(grid), len(PREDICTORS))
    ccc, mae = ccc.reshape(shape), mae.reshape(shape)
    index = {name: i for i, name in enumerate(PREDICTORS)}
    cluster_df = min(min(s["n_participants"], s["n_stimuli_or_conditions"]) for s in supports)-1
    block_df = len(splits)-1
    def interval(draws, point):
        interval_df = block_df if interval_method == "block_t" else cluster_df
        return _interval(draws, point=point, method=interval_method, cluster_df=interval_df)
    cells, contrast_rows, summary = [], [], []
    for mi, model in enumerate(models):
        for gi, (p, s) in enumerate(grid):
            for pi, predictor in enumerate(PREDICTORS):
                lo, hi, valid = interval(ccc[1:, mi, gi, pi], ccc[0, mi, gi, pi])
                cells.append(dict(representation=model, participant_dose=int(p), stimulus_dose=int(s),
                    predictor=predictor, macro_ccc=float(ccc[0, mi, gi, pi]),
                    ccc_ci_low=lo, ccc_ci_high=hi, valid_bootstrap=valid,
                    macro_mae=float(mae[0, mi, gi, pi])))
        for name, (right, left) in CONTRASTS.items():
            delta = ccc[:, mi, :, index[right]] - ccc[:, mi, :, index[left]]
            d_mae = mae[:, mi, :, index[right]] - mae[:, mi, :, index[left]]
            finite_rows = np.isfinite(delta[1:]).all(1)
            errors = np.max(np.abs(delta[1:][finite_rows] - delta[0]), axis=1)
            radius = float(np.quantile(errors, .95)) if len(errors) else np.nan
            for gi, (p, s) in enumerate(grid):
                lo, hi, valid = interval(delta[1:, gi], delta[0, gi])
                ml, mh, _ = interval(d_mae[1:, gi], d_mae[0, gi])
                contrast_rows.append(dict(representation=model, contrast=name,
                    participant_dose=int(p), stimulus_dose=int(s), ccc_delta=float(delta[0, gi]),
                    ccc_ci_low=lo, ccc_ci_high=hi, valid_bootstrap=valid,
                    ccc_simultaneous_low=float(delta[0, gi]-radius),
                    ccc_simultaneous_high=float(delta[0, gi]+radius),
                    mae_delta=float(d_mae[0, gi]), mae_ci_low=ml, mae_ci_high=mh))
    def add_summary(model, name, values):
        lo, hi, valid = interval(values[1:], values[0])
        summary.append(dict(representation=model, contrast=name, estimate=float(values[0]),
                            ci_low=lo, ci_high=hi, valid_bootstrap=valid))
    def surface_auc(values):
        surface = values.reshape(iterations+1, len(p_values), len(s_values))
        if len(p_values) < 2 or len(s_values) < 2:
            return surface.mean((1, 2))
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        inner = integrate(surface, np.asarray(s_values, float), axis=2)
        return integrate(inner, np.asarray(p_values, float), axis=1) / (
            (p_values[-1]-p_values[0])*(s_values[-1]-s_values[0]))
    for mi, model in enumerate(models):
        for predictor in ("prior_personalized", "combined_personalized"):
            v = ccc[:, mi, :, index[predictor]]
            add_summary(model, f"{predictor}_surface_auc", surface_auc(v))
            surface = v.reshape(iterations+1, len(p_values), len(s_values))
            add_summary(model, f"{predictor}_participant_change", surface[:, -1, 0]-surface[:, 0, 0])
            add_summary(model, f"{predictor}_stimulus_change", surface[:, 0, -1]-surface[:, 0, 0])
            add_summary(model, f"{predictor}_joint_change", surface[:, -1, -1]-surface[:, 0, 0])
            interaction = surface[:, -1, -1]-surface[:, -1, 0]-surface[:, 0, -1]+surface[:, 0, 0]
            add_summary(model, f"{predictor}_endpoint_interaction", interaction)
        added = ccc[:, mi, :, index["combined_personalized"]]-ccc[:, mi, :, index["prior_personalized"]]
        add_summary(model, "matched_eeg_added_surface_auc", surface_auc(added))
        added_mae = mae[:, mi, :, index["combined_personalized"]]-mae[:, mi, :, index["prior_personalized"]]
        add_summary(model, "matched_eeg_added_uniform_mean_ccc", added.mean(1))
        add_summary(model, "matched_eeg_added_uniform_mean_mae", added_mae.mean(1))
    for name, right, left in paired_models:
        if right not in models or left not in models:
            raise ValueError(f"Missing paired model for {name}")
        a, b = models.index(right), models.index(left)
        for metric, values in (("ccc", ccc), ("mae", mae)):
            raw = values[:, a, :, index["combined_personalized"]] - values[:, b, :, index["combined_personalized"]]
            added = raw - (values[:, a, :, index["prior_personalized"]] - values[:, b, :, index["prior_personalized"]])
            add_summary(name, f"combined_uniform_mean_{metric}", raw.mean(1))
            add_summary(name, f"matched_added_uniform_mean_{metric}", added.mean(1))
    for name in models:
        if not name.endswith("_pretrained"):
            continue
        random = name.removesuffix("_pretrained") + "_random"
        if random not in models:
            continue
        a, b = models.index(name), models.index(random)
        diff = ccc[:, a, :, index["combined_personalized"]] - ccc[:, b, :, index["combined_personalized"]]
        add_summary(name, "pretrained_minus_random_surface_auc", surface_auc(diff))
        add_summary(name, "pretrained_minus_random_zero", diff[:, 0])
        add_summary(name, "pretrained_minus_random_maximum", diff[:, -1])
        reversal = diff[:, 0]*diff[:, -1] < 0
        finite = np.isfinite(diff[:, 0]) & np.isfinite(diff[:, -1])
        summary.append(dict(representation=name, contrast="pairwise_order_reversal_bootstrap_frequency",
            estimate=float(reversal[0]), ci_low=np.nan, ci_high=np.nan,
            valid_bootstrap=int(finite[1:].sum()),
            bootstrap_frequency=float(reversal[1:][finite[1:]].mean())))
    report = dict(iterations=iterations, bootstrap_seed=seed, fixed_seeds=[int(x) for x in splits],
        target_names=list(target_names),
        pointwise_interval_method=interval_method, conservative_cluster_df=int(cluster_df),
        identity_block_df=int(block_df),
        interval_boundary="The block-t interval uses crossed-bootstrap standard errors and a five-block t critical value; topology-matched coverage is reported for the uniform-grid primary estimand; cellwise ranges remain descriptive; max-deviation simultaneous bands unchanged",
        supports=supports, resampling="shared participant x stimulus multiplicities across fixed splits",
        degenerate_stimulus_factor=any(x["n_stimuli_or_conditions"] < 2 for x in supports),
        simultaneous_family="dose cells within each model and named contrast, not all analyses",
        paired_model_contrasts=[list(item) for item in paired_models],
        boundary="Conditional on fitted predictions; no training refits; no independent-seed replication claim")
    return pd.DataFrame(cells), pd.DataFrame(contrast_rows), pd.DataFrame(summary), report
