"""Outcome-blind AMIGOS structural QC, ingestion, and crossed assignments."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
from scipy.signal import resample_poly, welch


CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")
BANDS = ((2.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(root: Path, subject: int) -> Path:
    return root / f"Data_Preprocessed_P{subject:02d}.mat"


def _signal_cells(path: Path) -> np.ndarray:
    value = loadmat(
        path,
        variable_names=["joined_data"],
        verify_compressed_data_integrity=True,
    ).get("joined_data")
    if value is None:
        raise ValueError("joined_data is missing")
    return np.asarray(value, dtype=object).reshape(-1)


def _selected_signal(cell: object) -> np.ndarray:
    signal = np.asarray(cell, dtype=float)
    if signal.ndim != 2 or signal.shape[1] < len(CHANNELS):
        raise ValueError("expected time-by-channel signal with at least 14 channels")
    post = signal[640:, : len(CHANNELS)]
    if len(post) < 3840:
        raise ValueError("fewer than 3840 post-baseline samples")
    start = (len(post) - 3840) // 2
    selected = post[start : start + 3840].T
    if not np.isfinite(selected).all():
        raise ValueError("selected EEG contains non-finite values")
    return selected


def structural_qc(
    root: Path,
    *,
    subject_ids: Iterable[int] = range(1, 41),
    short_trials: int = 16,
) -> dict[str, object]:
    """Inspect structure and EEG scale without loading self-assessment values."""
    records: list[dict[str, object]] = []
    scale_values: list[float] = []
    complete: list[int] = []
    for subject in subject_ids:
        path = _file(root, int(subject))
        record: dict[str, object] = {
            "subject_id": int(subject), "file": path.name, "status": "invalid", "errors": []
        }
        if not path.is_file():
            record["errors"] = ["missing_file"]
            records.append(record)
            continue
        record["sha256"] = sha256_file(path)
        variables = {name: {"shape": list(shape), "class": kind} for name, shape, kind in whosmat(path)}
        record["variables"] = variables
        missing = {"joined_data", "labels_selfassessment"} - set(variables)
        if missing:
            record["errors"] = ["missing_variables:" + ",".join(sorted(missing))]
            records.append(record)
            continue
        trial_records = []
        try:
            cells = _signal_cells(path)
            if len(cells) < short_trials:
                raise ValueError("fewer than 16 signal cells")
            for trial in range(short_trials):
                try:
                    selected = _selected_signal(cells[trial])
                    median_sd = float(np.median(np.std(selected, axis=1, ddof=1)))
                    if not np.isfinite(median_sd) or median_sd <= 0:
                        raise ValueError("non-positive signal scale")
                    scale_values.append(median_sd)
                    trial_records.append({"trial": trial + 1, "status": "valid", "median_channel_sd": median_sd})
                except ValueError as error:
                    trial_records.append({"trial": trial + 1, "status": "invalid", "error": str(error)})
        except (OSError, ValueError) as error:
            record["errors"] = [str(error)]
            records.append(record)
            continue
        record["trials"] = trial_records
        if all(item["status"] == "valid" for item in trial_records):
            record["status"] = "signal_complete"
            complete.append(int(subject))
        else:
            record["errors"] = ["one_or_more_invalid_short_trials"]
        records.append(record)
    if not scale_values:
        raise ValueError("No structurally valid EEG trials")
    median_scale = float(np.median(scale_values))
    if 0.1 <= median_scale <= 1000.0:
        source_units, factor = "microvolts", 1.0
    elif 1e-7 <= median_scale <= 1e-3:
        source_units, factor = "volts", 1_000_000.0
    else:
        source_units, factor = "unresolved", None
    return {
        "schema_version": "1.0",
        "status": "pass" if factor is not None else "stop_for_unit_documentation",
        "outcome_values_loaded": False,
        "required_variables": ["joined_data", "labels_selfassessment"],
        "source_sampling_hz": 128,
        "channels": list(CHANNELS),
        "complete_signal_subjects": complete,
        "complete_signal_subject_count": len(complete),
        "median_selected_window_channel_sd": median_scale,
        "source_units": source_units,
        "microvolt_scale_factor": factor,
        "records": records,
    }


def _bandpower(signal: np.ndarray) -> np.ndarray:
    frequencies, spectrum = welch(signal, fs=100.0, nperseg=200, axis=1)
    features = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        if not mask.any():
            raise ValueError("empty prespecified frequency band")
        features.append(np.log10(spectrum[:, mask].mean(axis=1) + 1e-12))
    return np.stack(features, axis=1)


def ingest_amigos(
    root: Path,
    structural_report: dict[str, object],
    *,
    minimum_participants: int = 25,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if structural_report.get("status") != "pass" or structural_report.get("outcome_values_loaded") is not False:
        raise ValueError("A passing outcome-blind structural report is required")
    scale_factor = structural_report.get("microvolt_scale_factor")
    if scale_factor not in (1.0, 1_000_000.0):
        raise ValueError("Unresolved EEG unit scale")
    hash_by_subject = {
        int(item["subject_id"]): item.get("sha256") for item in structural_report["records"]
    }
    candidate_subjects = [int(value) for value in structural_report["complete_signal_subjects"]]
    per_subject: dict[int, list[tuple]] = {}
    exclusions: list[dict[str, object]] = []
    for subject in candidate_subjects:
        path = _file(root, subject)
        if sha256_file(path) != hash_by_subject.get(subject):
            raise ValueError(f"Source hash changed after structural lock: {path.name}")
        data = loadmat(
            path,
            variable_names=["joined_data", "labels_selfassessment"],
            verify_compressed_data_integrity=True,
        )
        signals = np.asarray(data["joined_data"], dtype=object).reshape(-1)
        labels = np.asarray(data["labels_selfassessment"], dtype=object).reshape(-1)
        trials = []
        for trial in range(16):
            try:
                rating = np.asarray(labels[trial], dtype=float).reshape(-1)
                if len(rating) < 2 or not np.isfinite(rating[:2]).all() or not ((rating[:2] >= 1) & (rating[:2] <= 9)).all():
                    raise ValueError("invalid arousal/valence rating")
                raw = _selected_signal(signals[trial]) * float(scale_factor)
                tensor = resample_poly(raw, 25, 32, axis=1).astype(np.float32)
                if tensor.shape != (14, 3000) or not np.isfinite(tensor).all():
                    raise ValueError("unexpected resampled tensor")
                features = _bandpower(tensor).astype(np.float32)
                arousal, valence = (float(rating[0] - 5.0) / 4.0, float(rating[1] - 5.0) / 4.0)
                trials.append((trial + 1, valence, arousal, tensor, features))
            except (IndexError, TypeError, ValueError) as error:
                exclusions.append({"subject_id": subject, "trial": trial + 1, "reason": str(error)})
        if len(trials) == 16:
            per_subject[subject] = trials
        else:
            exclusions.append({"subject_id": subject, "trial": None, "reason": "participant incomplete across 16 short trials"})
    if len(per_subject) < minimum_participants:
        raise ValueError(
            f"Only {len(per_subject)} complete participants; confirmation requires {minimum_participants}"
        )
    rows, tensors, features, uids = [], [], [], []
    for subject in sorted(per_subject):
        for trial, valence, arousal, tensor, feature in per_subject[subject]:
            uid = f"amigos:P{subject:02d}:V{trial:02d}"
            uids.append(uid)
            tensors.append(tensor)
            features.append(feature)
            rows.append({
                "dataset_id": "amigos_confirmation_v14",
                "trial_uid": uid,
                "subject_id": f"P{subject:02d}",
                "subject_uid": f"amigos:P{subject:02d}",
                "stimulus_uid": f"amigos:V{trial:02d}",
                "target_valence": valence,
                "target_arousal": arousal,
                "target_available": True,
                "feature_alignment_status": "verified",
            })
    report = {
        "status": "complete",
        "participants": len(per_subject),
        "stimuli": 16,
        "trials": len(rows),
        "excluded": exclusions,
        "source_units": structural_report["source_units"],
        "microvolt_scale_factor": scale_factor,
        "target_transform": "(rating - 5) / 4",
        "tensor_shape": [len(rows), 14, 3000],
        "bandpower_shape": [len(rows), 14, 5],
    }
    return (
        pd.DataFrame(rows),
        np.stack(tensors),
        np.asarray(uids),
        np.stack(features),
        report,
    )


def crossed_outer_assignments(
    trials: pd.DataFrame,
    *,
    rotation_seed: int,
    participant_folds: int = 5,
    stimulus_folds: int = 4,
) -> list[tuple[dict[str, int], pd.DataFrame]]:
    if trials.trial_uid.duplicated().any():
        raise ValueError("Duplicate trial UID")

    def blocks(column: str, count: int) -> dict[str, int]:
        values = trials[column].astype(str).unique()
        ordered = sorted(
            values,
            key=lambda value: hashlib.sha256(
                f"{rotation_seed}:{column}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        return {value: index % count for index, value in enumerate(ordered)}

    participant_map = blocks("subject_uid", participant_folds)
    stimulus_map = blocks("stimulus_uid", stimulus_folds)
    p = trials.subject_uid.astype(str).map(participant_map).to_numpy()
    s = trials.stimulus_uid.astype(str).map(stimulus_map).to_numpy()
    result = []
    observed_test: set[str] = set()
    for participant_block in range(participant_folds):
        for stimulus_block in range(stimulus_folds):
            validation_p = (participant_block + 1) % participant_folds
            validation_s = (stimulus_block + 1) % stimulus_folds
            roles = np.full(len(trials), "excluded", dtype=object)
            roles[
                (p != participant_block) & (p != validation_p)
                & (s != stimulus_block) & (s != validation_s)
            ] = "train"
            roles[(p == validation_p) & (s == validation_s)] = "validation"
            roles[(p == participant_block) & (s == stimulus_block)] = "test"
            assignment = pd.DataFrame({"trial_uid": trials.trial_uid, "split": roles})
            test_ids = set(assignment.loc[assignment.split.eq("test"), "trial_uid"])
            if observed_test & test_ids:
                raise ValueError("Test trials overlap within a rotation")
            observed_test |= test_ids
            result.append(({
                "rotation_seed": rotation_seed,
                "participant_block": participant_block,
                "stimulus_block": stimulus_block,
            }, assignment))
    if observed_test != set(trials.trial_uid):
        raise ValueError("Cross-products do not cover every trial exactly once")
    return result

