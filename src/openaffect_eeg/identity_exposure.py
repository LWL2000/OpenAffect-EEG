"""Identity-exposure response surfaces for trial-level EEG evaluation.

The module treats labelled participant calibration and repeated-stimulus
observations as distinct deployment resources.  It never interprets either
axis as latent emotion disentanglement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from openaffect_eeg.baselines import regression_metrics

TARGET_COLUMNS = ("target_valence", "target_arousal")
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


class ExposureError(ValueError):
    """Raised when an exposure cell cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class ExposureCell:
    """One labelled-participant by repeated-stimulus exposure cell."""

    participant_dose: int
    stimulus_dose: int

    @property
    def name(self) -> str:
        return f"participant-{self.participant_dose:02d}_stimulus-{self.stimulus_dose:02d}"


def _rank(seed: int, *values: object) -> str:
    joined = ":".join(str(value) for value in (seed, *values))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _require_columns(table: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ExposureError(f"Missing columns: {missing}")


def _match_donors(
    base_train: pd.DataFrame,
    additions: pd.DataFrame,
    *,
    seed: int,
    context_column: str,
    target_columns: tuple[str, ...] = TARGET_COLUMNS,
) -> list[str]:
    """Select deterministic metadata/target-matched rows to keep train size fixed."""

    if additions.empty:
        return []
    if len(additions) >= len(base_train):
        raise ExposureError("Stimulus exposure would exhaust the base training set")
    available = base_train.copy()
    selected: list[str] = []
    ordered = additions.assign(
        _rank=[_rank(seed, "addition", value) for value in additions["trial_uid"]]
    ).sort_values("_rank")
    for added in ordered.itertuples(index=False):
        added_series = pd.Series(added._asdict())
        candidates = available.copy()
        context = added_series.get(context_column)
        if context_column in candidates and pd.notna(context):
            same_context = candidates[context_column].eq(context)
            if same_context.any():
                candidates = candidates.loc[same_context]
        if candidates.empty:
            raise ExposureError("No donor row is available for train-size matching")
        target = added_series[list(target_columns)].to_numpy(dtype=float)
        candidate_targets = candidates[list(target_columns)].to_numpy(dtype=float)
        distances = np.linalg.norm(candidate_targets - target[None, :], axis=1)
        nearest = candidates.iloc[np.flatnonzero(distances == distances.min())]
        donor_uid = min(
            (
                _rank(seed, "donor", added_series["trial_uid"], trial_uid),
                str(trial_uid),
            )
            for trial_uid in nearest["trial_uid"]
        )[1]
        selected.append(donor_uid)
        available = available.loc[~available["trial_uid"].eq(donor_uid)]
    return selected


def _nested_identity_sample(
    candidates: pd.DataFrame,
    identities: list[str],
    *,
    identity_column: str,
    dose: int,
    seed: int,
    axis: str,
) -> pd.DataFrame:
    if dose == 0:
        return candidates.iloc[0:0].copy()
    selected = []
    for identity in identities:
        rows = candidates.loc[candidates[identity_column].astype(str).eq(identity)].copy()
        if len(rows) < dose:
            raise ExposureError(
                f"{axis} identity {identity!r} has {len(rows)} rows, below dose {dose}"
            )
        rows["_rank"] = [
            _rank(seed, axis, identity, value) for value in rows["trial_uid"]
        ]
        selected.append(rows.sort_values("_rank").head(dose).drop(columns="_rank"))
    return pd.concat(selected, ignore_index=True) if selected else candidates.iloc[0:0]


def compile_exposure_cell(
    trials: pd.DataFrame,
    strict_assignment: pd.DataFrame,
    *,
    participant_dose: int,
    stimulus_dose: int,
    maximum_participant_dose: int,
    maximum_stimulus_dose: int,
    seed: int,
    subject_column: str = "subject_uid",
    stimulus_column: str = "stimulus_uid",
    context_column: str = "context",
    stimulus_label: str = "stimulus",
    deployment_context: dict[str, object] | None = None,
    target_columns: tuple[str, ...] = TARGET_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compile one fixed-support exposure cell from a strict joint assignment."""

    for value, label in (
        (participant_dose, "participant dose"),
        (stimulus_dose, "stimulus dose"),
        (maximum_participant_dose, "maximum participant dose"),
        (maximum_stimulus_dose, "maximum stimulus dose"),
    ):
        if value < 0:
            raise ExposureError(f"{label} must be non-negative")
    if participant_dose > maximum_participant_dose:
        raise ExposureError("Participant dose exceeds the frozen maximum")
    if stimulus_dose > maximum_stimulus_dose:
        raise ExposureError("Stimulus dose exceeds the frozen maximum")
    targets = tuple(str(column) for column in target_columns)
    if not targets or len(targets) != len(set(targets)):
        raise ExposureError("Target columns must be unique and non-empty")
    _require_columns(
        trials,
        ["trial_uid", subject_column, stimulus_column, *targets],
    )
    _require_columns(strict_assignment, ["trial_uid", "split"])
    if trials["trial_uid"].duplicated().any() or strict_assignment["trial_uid"].duplicated().any():
        raise ExposureError("Trial identifiers must be unique")
    for column in ("trial_uid", subject_column, stimulus_column):
        if trials[column].isna().any() or trials[column].astype(str).str.strip().eq("").any():
            raise ExposureError(f"Missing identity in {column}")
    if not np.isfinite(trials[list(targets)].to_numpy(float)).all():
        raise ExposureError("Non-finite targets")
    if not set(strict_assignment["split"]).issubset({"train", "validation", "test", "excluded"}):
        raise ExposureError("Unknown strict split label")

    merged = trials.merge(
        strict_assignment[["trial_uid", "split"]],
        on="trial_uid",
        how="inner",
        validate="1:1",
    )
    strict_test = merged.loc[merged["split"].eq("test")].copy()
    if strict_test.empty:
        raise ExposureError("The strict assignment has no eligible test trials")
    held_subjects = sorted(strict_test[subject_column].astype(str).unique())
    held_stimuli = sorted(strict_test[stimulus_column].astype(str).unique())
    for column in (subject_column, stimulus_column):
        groups = [set(merged.loc[merged["split"].eq(part), column].astype(str))
                  for part in ("train", "validation", "test")]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ExposureError(f"Strict assignment does not isolate {column}")

    subject_candidates = merged.loc[
        merged[subject_column].astype(str).isin(held_subjects)
        & ~merged[stimulus_column].astype(str).isin(held_stimuli)
        & merged["split"].eq("excluded")
    ].copy()
    stimulus_candidates = merged.loc[
        ~merged[subject_column].astype(str).isin(held_subjects)
        & merged[stimulus_column].astype(str).isin(held_stimuli)
        & merged["split"].eq("excluded")
    ].copy()

    participant_support = subject_candidates.groupby(subject_column).size()
    stimulus_support = stimulus_candidates.groupby(stimulus_column).size()
    eligible_subjects = sorted(
        str(value)
        for value in participant_support.loc[
            participant_support.ge(maximum_participant_dose)
        ].index
    )
    eligible_stimuli = sorted(
        str(value)
        for value in stimulus_support.loc[
            stimulus_support.ge(maximum_stimulus_dose)
        ].index
    )
    test = strict_test.loc[
        strict_test[subject_column].astype(str).isin(eligible_subjects)
        & strict_test[stimulus_column].astype(str).isin(eligible_stimuli)
    ].copy()
    if len(test) < 4:
        raise ExposureError("Fewer than four fixed-support test trials remain")
    eligible_subjects = sorted(test[subject_column].astype(str).unique())
    eligible_stimuli = sorted(test[stimulus_column].astype(str).unique())

    calibration = _nested_identity_sample(
        subject_candidates,
        eligible_subjects,
        identity_column=subject_column,
        dose=participant_dose,
        seed=seed,
        axis="participant",
    )
    stimulus_exposure = _nested_identity_sample(
        stimulus_candidates,
        eligible_stimuli,
        identity_column=stimulus_column,
        dose=stimulus_dose,
        seed=seed,
        axis="stimulus",
    )
    base_train = merged.loc[merged["split"].eq("train")].copy()
    donor_uids = _match_donors(
        base_train,
        stimulus_exposure,
        seed=seed + stimulus_dose * 1009,
        context_column=context_column,
        target_columns=targets,
    )
    retained_train = base_train.loc[~base_train["trial_uid"].isin(donor_uids)]
    train_uids = set(retained_train["trial_uid"]) | set(stimulus_exposure["trial_uid"])
    validation_uids = set(merged.loc[merged["split"].eq("validation"), "trial_uid"])
    test_uids = set(test["trial_uid"])
    calibration_uids = set(calibration["trial_uid"])
    if train_uids & test_uids or validation_uids & test_uids or calibration_uids & test_uids:
        raise ExposureError("A test trial leaked into train, validation, or calibration")

    assignment = pd.DataFrame({"trial_uid": trials["trial_uid"].astype(str)})
    assignment["split"] = "excluded"
    assignment["role"] = "unsupported_or_unused"
    roles = (
        (train_uids, "train", "population_train"),
        (validation_uids, "validation", "strict_validation"),
        (calibration_uids, "calibration", "labelled_participant_calibration"),
        (test_uids, "test", "fixed_joint_test"),
    )
    for uids, split, role in roles:
        selected = assignment["trial_uid"].isin(uids)
        assignment.loc[selected, ["split", "role"]] = [split, role]
    donor_mask = assignment["trial_uid"].isin(donor_uids)
    assignment.loc[donor_mask, "role"] = "train_size_matched_donor"
    exposure_mask = assignment["trial_uid"].isin(stimulus_exposure["trial_uid"])
    assignment.loc[exposure_mask, "role"] = "repeated_stimulus_exposure"
    assignment["protocol"] = "identity_exposure_response"
    assignment["seed"] = int(seed)
    assignment["participant_dose"] = int(participant_dose)
    assignment["stimulus_dose"] = int(stimulus_dose)
    assignment = assignment.sort_values("trial_uid").reset_index(drop=True)

    assigned = trials.merge(assignment[["trial_uid", "split"]], on="trial_uid", validate="1:1")
    development = assigned.loc[assigned["split"].isin(["train", "validation", "calibration"])]
    test_rows = assigned.loc[assigned["split"].eq("test")]
    isolation_evidence = {}
    context = deployment_context or {}
    for axis, default in (("session", "session_uid"), ("dataset", "dataset_id"), ("device", "device_id")):
        column = str(context.get(f"{axis}_column", default))
        valid = column in assigned and not assigned[column].isna().any()
        valid = valid and not assigned[column].astype(str).str.strip().eq("").any()
        overlap = sorted(set(development[column].astype(str)) & set(test_rows[column].astype(str))) if valid else None
        isolation_evidence[axis] = dict(column=column, requested=bool(context.get(f"{axis}_holdout")),
                                       verified=bool(valid and not overlap), overlap=overlap)

    test_support_hash = hashlib.sha256(
        "\n".join(sorted(test_uids)).encode("utf-8")
    ).hexdigest()
    audit = {
        "cell": ExposureCell(participant_dose, stimulus_dose).name,
        "seed": int(seed),
        "participant_dose": int(participant_dose),
        "stimulus_dose": int(stimulus_dose),
        "maximum_participant_dose": int(maximum_participant_dose),
        "maximum_stimulus_dose": int(maximum_stimulus_dose),
        "fixed_test_support_sha256": test_support_hash,
        "counts": {
            split: int(assignment["split"].eq(split).sum())
            for split in ("train", "validation", "calibration", "test", "excluded")
        },
        "test_subject_count": len(eligible_subjects),
        "test_stimulus_count": len(eligible_stimuli),
        "calibration_rows_per_test_subject": {
            identity: int(
                calibration[subject_column].astype(str).eq(identity).sum()
            )
            for identity in eligible_subjects
        },
        "exposure_rows_per_test_stimulus": {
            identity: int(
                stimulus_exposure[stimulus_column].astype(str).eq(identity).sum()
            )
            for identity in eligible_stimuli
        },
        "train_size_preserved": len(train_uids) == len(base_train),
        "donor_count": len(donor_uids),
        "test_trial_leakage": False,
        "additional_identity_isolation": isolation_evidence,
        "claim_card": build_claim_card(
            participant_dose,
            stimulus_dose,
            stimulus_label=stimulus_label,
            deployment_context=deployment_context,
            verified_isolation={axis: item["verified"] for axis, item in isolation_evidence.items()},
        ),
    }
    if not audit["train_size_preserved"]:
        raise ExposureError("Population training size changed across exposure cells")
    return assignment, audit


def build_claim_card(
    participant_dose: int,
    stimulus_dose: int,
    *,
    stimulus_label: str = "stimulus",
    deployment_context: dict[str, object] | None = None,
    verified_isolation: dict[str, bool] | None = None,
) -> dict[str, object]:
    """Map an exposure cell to deterministic, evidence-bounded wording."""

    participant = (
        "zero labelled target-participant calibration"
        if participant_dose == 0
        else f"a known participant with {participant_dose} labelled calibration trials"
    )
    stimulus = (
        f"a {stimulus_label} absent from population training"
        if stimulus_dose == 0
        else (
            f"a {stimulus_label} represented by {stimulus_dose} "
            "other-participant ratings"
        )
    )
    context = {
        "target_unlabelled_data": "not used",
        "calibration_unit": "labelled complete trials",
        "calibration_session_relation": "temporal ordering not verified",
        "stimulus_metadata": "source-released normative prior retained when available",
        "session_holdout": False,
        "dataset_holdout": False,
        "device_holdout": False,
    }
    declared = deployment_context or {}
    for axis in ("session", "dataset", "device"):
        column_key = f"{axis}_column"
        if column_key in declared:
            context[column_key] = declared[column_key]
    requested = {axis: bool(declared.get(f"{axis}_holdout")) for axis in ("session", "dataset", "device")}
    for axis in requested:
        context[f"{axis}_holdout"] = bool((verified_isolation or {}).get(axis, False))
    blocked = [
        "latent or true emotion disentanglement",
        "causal removal of participant or stimulus identity",
        "fully subject-independent prediction when participant calibration is non-zero",
        f"unseen-{stimulus_label} prediction when {stimulus_label} exposure is non-zero",
    ]
    if not bool(context.get("session_holdout")):
        blocked.append("session-independent generalization")
    if not bool(context.get("dataset_holdout")):
        blocked.append("cross-dataset generalization")
    if not bool(context.get("device_holdout")):
        blocked.append("cross-device generalization")
    return {
        "supported_deployment": f"Prediction for {participant} and {stimulus}.",
        "participant_identity_id_required": participant_dose > 0,
        "participant_labels_required": participant_dose,
        f"same_{stimulus_label}_training_labels_required": stimulus_dose,
        "deployment_context": context,
        "requested_deployment_context": dict(declared),
        "requested_additional_holdouts": requested,
        "verification_boundary": "Observed table IDs and evaluator operations only; upstream feature learning, participant identity aliases, and raw acquisition provenance are not certified.",
        "blocked_claims": blocked,
    }


def _feature_columns(table: pd.DataFrame) -> list[str]:
    columns = [column for column in table if column.startswith("feature_")]
    if not columns:
        raise ExposureError("No feature_* columns were supplied")
    return columns


def _standardize(
    train_x: np.ndarray, requested_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return (train_x - mean) / scale, (requested_x - mean) / scale, mean, scale


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    requested_x: np.ndarray,
    *,
    alpha: float,
    backend: str,
) -> np.ndarray:
    """Fit an intercept Ridge and predict with NumPy or CUDA torch."""

    x = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    requested = np.asarray(requested_x, dtype=float)
    xz, rz, _, _ = _standardize(x, requested)
    target_mean = y.mean(axis=0)
    centered = y - target_mean
    if backend == "numpy":
        gram = xz.T @ xz + float(alpha) * np.eye(xz.shape[1])
        coef = np.linalg.solve(gram, xz.T @ centered)
        return target_mean + rz @ coef
    if backend != "torch":
        raise ExposureError(f"Unknown Ridge backend: {backend}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - server dependency
        raise ExposureError("Torch backend requested but torch is unavailable") from exc
    if not torch.cuda.is_available():  # pragma: no cover - server dependency
        raise ExposureError("Torch backend requested but CUDA is unavailable")
    device = torch.device("cuda")
    tx = torch.as_tensor(xz, dtype=torch.float64, device=device)
    ty = torch.as_tensor(centered, dtype=torch.float64, device=device)
    tr = torch.as_tensor(rz, dtype=torch.float64, device=device)
    eye = torch.eye(tx.shape[1], dtype=torch.float64, device=device)
    coef = torch.linalg.solve(tx.T @ tx + float(alpha) * eye, tx.T @ ty)
    prediction = torch.as_tensor(target_mean, dtype=torch.float64, device=device) + tr @ coef
    return prediction.detach().cpu().numpy()


def select_ridge_alpha(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    backend: str = "numpy",
    validation_prior: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for alpha in alphas:
        prediction = ridge_predict(
            train_x, train_y, validation_x, alpha=alpha, backend=backend
        )
        if validation_prior is not None:
            prediction = prediction + validation_prior
        scores[f"{alpha:g}"] = float(np.mean(np.abs(validation_y - prediction)))
    selected = min(scores, key=lambda value: (scores[value], float(value)))
    return float(selected), scores


def _fallback_prior(
    table: pd.DataFrame,
    global_mean: np.ndarray,
    *,
    target_names: tuple[str, ...] = ("valence", "arousal"),
) -> np.ndarray:
    fallback = np.repeat(np.asarray(global_mean, dtype=float)[None, :], len(table), axis=0)
    columns = tuple(f"stimulus_prior_{name}" for name in target_names)
    if fallback.shape[1] != len(columns):
        raise ExposureError("Prior dimensions do not match target names")
    if set(columns).issubset(table.columns):
        published = table[list(columns)].to_numpy(dtype=float)
        valid = np.isfinite(published).all(axis=1)
        fallback[valid] = published[valid]
    return fallback


def leave_one_out_prior(
    targets: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    identities = np.asarray(groups).astype(str)
    result = np.asarray(fallback, dtype=float).copy()
    for identity in np.unique(identities):
        selected = identities == identity
        count = int(selected.sum())
        if count > 1:
            group_sum = values[selected].sum(axis=0)
            result[selected] = (group_sum - values[selected]) / (count - 1)
    return result


def predict_prior(
    train_targets: np.ndarray,
    train_groups: np.ndarray,
    requested_groups: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train_targets, dtype=float)
    identities = np.asarray(train_groups).astype(str)
    requested = np.asarray(requested_groups).astype(str)
    result = np.asarray(fallback, dtype=float).copy()
    means = {
        identity: values[identities == identity].mean(axis=0)
        for identity in np.unique(identities)
    }
    seen = np.asarray([identity in means for identity in requested])
    for index, identity in enumerate(requested):
        if identity in means:
            result[index] = means[identity]
    return result, seen


def _participant_offsets(
    calibration: pd.DataFrame,
    calibration_prediction: np.ndarray,
    test: pd.DataFrame,
    *,
    subject_column: str,
    target_columns: tuple[str, ...] = TARGET_COLUMNS,
) -> np.ndarray:
    offsets = np.zeros((len(test), len(target_columns)), dtype=float)
    if calibration.empty:
        return offsets
    residual = calibration[list(target_columns)].to_numpy() - calibration_prediction
    by_subject = {
        str(identity): residual[calibration[subject_column].astype(str).eq(str(identity))].mean(axis=0)
        for identity in calibration[subject_column].astype(str).unique()
    }
    for index, identity in enumerate(test[subject_column].astype(str)):
        offsets[index] = by_subject.get(identity, np.zeros(len(target_columns)))
    return offsets


def _nearest_centroid_probe(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    *,
    identity_column: str,
    feature_columns: list[str],
    seed: int,
) -> dict[str, object]:
    identities = sorted(set(reference[identity_column].astype(str)) & set(test[identity_column].astype(str)))
    if len(identities) < 2:
        return {"status": "not_applicable", "reason": "fewer than two supported identities"}
    reference = reference.loc[reference[identity_column].astype(str).isin(identities)]
    test = test.loc[test[identity_column].astype(str).isin(identities)]
    x_ref = reference[feature_columns].to_numpy(dtype=float)
    x_test = test[feature_columns].to_numpy(dtype=float)
    x_ref, x_test, _, _ = _standardize(x_ref, x_test)
    labels = reference[identity_column].astype(str).to_numpy()
    test_labels = test[identity_column].astype(str).to_numpy()

    def score(training_labels: np.ndarray) -> float:
        centroids = np.stack([x_ref[training_labels == identity].mean(axis=0) for identity in identities])
        distance = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
        prediction = np.asarray(identities)[np.argmin(distance, axis=1)]
        return float(balanced_accuracy_score(test_labels, prediction))

    rng = np.random.default_rng(seed)
    return {
        "status": "ok",
        "identity_count": len(identities),
        "reference_count": len(reference),
        "test_count": len(test),
        "chance_balanced_accuracy": 1.0 / len(identities),
        "observed_balanced_accuracy": score(labels),
        "permuted_balanced_accuracy": score(rng.permutation(labels)),
    }


def evaluate_exposure_cell(
    trials: pd.DataFrame,
    assignment: pd.DataFrame,
    features: pd.DataFrame,
    *,
    backend: str = "numpy",
    subject_column: str = "subject_uid",
    stimulus_column: str = "stimulus_uid",
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    target_columns: tuple[str, ...] = TARGET_COLUMNS,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate prior, population EEG, and labelled-personalization estimands."""

    targets = tuple(str(column) for column in target_columns)
    if not targets or len(targets) != len(set(targets)):
        raise ExposureError("Target columns must be unique and non-empty")
    target_names = tuple(column.removeprefix("target_") for column in targets)
    _require_columns(trials, ["trial_uid", subject_column, stimulus_column, *targets])
    _require_columns(assignment, ["trial_uid", "split", "seed", "participant_dose", "stimulus_dose"])
    _require_columns(features, ["trial_uid"])
    feature_columns = _feature_columns(features)
    experiment = (
        trials.merge(features, on="trial_uid", validate="1:1")
        .merge(assignment, on="trial_uid", validate="1:1")
    )
    parts = {
        split: experiment.loc[experiment["split"].eq(split)].copy()
        for split in ("train", "validation", "calibration", "test")
    }
    if min(len(parts["train"]), len(parts["validation"]), len(parts["test"])) < 2:
        raise ExposureError("Train, validation, and test each require at least two rows")
    train, validation, calibration, test = (
        parts["train"], parts["validation"], parts["calibration"], parts["test"]
    )
    train_x = train[feature_columns].to_numpy(dtype=float)
    validation_x = validation[feature_columns].to_numpy(dtype=float)
    train_y = train[list(targets)].to_numpy(dtype=float)
    validation_y = validation[list(targets)].to_numpy(dtype=float)
    direct_alpha, direct_scores = select_ridge_alpha(
        train_x,
        train_y,
        validation_x,
        validation_y,
        alphas=alphas,
        backend=backend,
    )

    train_global = train_y.mean(axis=0)
    train_fallback = _fallback_prior(train, train_global, target_names=target_names)
    train_prior = leave_one_out_prior(
        train_y, train[stimulus_column].to_numpy(), train_fallback
    )
    validation_fallback = _fallback_prior(
        validation, train_global, target_names=target_names
    )
    validation_prior, _ = predict_prior(
        train_y,
        train[stimulus_column].to_numpy(),
        validation[stimulus_column].to_numpy(),
        validation_fallback,
    )
    residual_alpha, residual_scores = select_ridge_alpha(
        train_x,
        train_y - train_prior,
        validation_x,
        validation_y,
        alphas=alphas,
        backend=backend,
        validation_prior=validation_prior,
    )

    fit = pd.concat([train, validation], ignore_index=True)
    fit_x = fit[feature_columns].to_numpy(dtype=float)
    fit_y = fit[list(targets)].to_numpy(dtype=float)
    fit_global = fit_y.mean(axis=0)
    fit_prior = leave_one_out_prior(
        fit_y,
        fit[stimulus_column].to_numpy(),
        _fallback_prior(fit, fit_global, target_names=target_names),
    )
    requested = pd.concat([test, calibration], ignore_index=True)
    requested_x = requested[feature_columns].to_numpy(dtype=float)
    direct_requested = ridge_predict(
        fit_x, fit_y, requested_x, alpha=direct_alpha, backend=backend
    )
    requested_prior, stimulus_seen = predict_prior(
        fit_y,
        fit[stimulus_column].to_numpy(),
        requested[stimulus_column].to_numpy(),
        _fallback_prior(requested, fit_global, target_names=target_names),
    )
    residual_requested = ridge_predict(
        fit_x,
        fit_y - fit_prior,
        requested_x,
        alpha=residual_alpha,
        backend=backend,
    )
    combined_requested = requested_prior + residual_requested
    test_count = len(test)
    direct_test = direct_requested[:test_count]
    prior_test = requested_prior[:test_count]
    combined_test = combined_requested[:test_count]
    direct_calibration = direct_requested[test_count:]
    combined_calibration = combined_requested[test_count:]
    prior_offset = _participant_offsets(
        calibration, requested_prior[test_count:], test, subject_column=subject_column,
        target_columns=targets,
    )
    direct_offset = _participant_offsets(
        calibration, direct_calibration, test, subject_column=subject_column,
        target_columns=targets,
    )
    combined_offset = _participant_offsets(
        calibration, combined_calibration, test, subject_column=subject_column,
        target_columns=targets,
    )
    predictions = {
        "prior": prior_test,
        "prior_personalized": prior_test + prior_offset,
        "eeg_population": direct_test,
        "eeg_personalized": direct_test + direct_offset,
        "combined_population": combined_test,
        "combined_personalized": combined_test + combined_offset,
    }
    truth = test[list(targets)].to_numpy(dtype=float)
    metrics = {
        name: regression_metrics(truth, value, target_names=target_names)
        for name, value in predictions.items()
    }
    seed = int(assignment["seed"].iloc[0])
    probes = {
        "participant": _nearest_centroid_probe(
            calibration,
            test,
            identity_column=subject_column,
            feature_columns=feature_columns,
            seed=seed + 17,
        ),
        "stimulus": _nearest_centroid_probe(
            train.loc[train["role"].eq("repeated_stimulus_exposure")],
            test,
            identity_column=stimulus_column,
            feature_columns=feature_columns,
            seed=seed + 31,
        ),
    }
    result = {
        "seed": seed,
        "participant_dose": int(assignment["participant_dose"].iloc[0]),
        "stimulus_dose": int(assignment["stimulus_dose"].iloc[0]),
        "backend": backend,
        "target_names": list(target_names),
        "counts": {name: len(value) for name, value in parts.items()},
        "selected_alpha": {"direct": direct_alpha, "residual": residual_alpha},
        "validation_mae": {"direct": direct_scores, "residual": residual_scores},
        "metrics": metrics,
        "eeg_added_value_ccc": (
            metrics["combined_population"]["macro"]["ccc"]
            - metrics["prior"]["macro"]["ccc"]
        ),
        "eeg_added_value_personalized_ccc": (
            metrics["combined_personalized"]["macro"]["ccc"]
            - metrics["prior_personalized"]["macro"]["ccc"]
        ),
        "prior_personalization_gain_ccc": (
            metrics["prior_personalized"]["macro"]["ccc"]
            - metrics["prior"]["macro"]["ccc"]
        ),
        "prior_contract": (
            "Training/validation stimulus means; source-released normative ratings "
            "when available for unseen stimuli, otherwise fit-global mean. "
            "No semantic-text model is fitted by this response-surface evaluator."
        ),
        "personalization_gain_ccc": (
            metrics["combined_personalized"]["macro"]["ccc"]
            - metrics["combined_population"]["macro"]["ccc"]
        ),
        "test_stimulus_seen_fraction": float(stimulus_seen[:test_count].mean()),
        "identity_probes": probes,
        "interpretation_boundary": (
            "Scores quantify prediction of observed self-report under a declared "
            "labelled-calibration and repeated-stimulus regime. They do not identify "
            "latent emotion or causal identity dependence."
        ),
    }
    prediction_table = test[
        ["trial_uid", subject_column, stimulus_column, *targets]
    ].reset_index(drop=True).copy()
    prediction_table.insert(0, "seed", seed)
    prediction_table.insert(1, "participant_dose", result["participant_dose"])
    prediction_table.insert(2, "stimulus_dose", result["stimulus_dose"])
    for name, values in predictions.items():
        for index, target_name in enumerate(target_names):
            prediction_table[f"{name}_{target_name}"] = values[:, index]
    return result, prediction_table
