"""Trial reconstruction for the ds006866 emotion-regulation dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_CONDITIONS = {
    18: ("nonsocial", "negative", "reappraise"),
    19: ("nonsocial", "negative", "watch"),
    20: ("nonsocial", "neutral", "watch"),
    21: ("social", "negative", "reappraise"),
    22: ("social", "negative", "watch"),
    23: ("social", "neutral", "watch"),
}
_CUE_FOR_STIMULUS = {stimulus: stimulus - 6 for stimulus in _CONDITIONS}
_CUE_CODES = set(_CUE_FOR_STIMULUS.values())
_RATING_CODES = set(range(1, 10))


class EmotionRegulationError(ValueError):
    pass


def _rating_after(values: np.ndarray, marker: int, stop_marker: int | None) -> float:
    positions = np.flatnonzero(values == marker)
    if not len(positions):
        return np.nan
    start = int(positions[0]) + 1
    stop = len(values)
    if stop_marker is not None:
        stops = np.flatnonzero(values[start:] == stop_marker)
        if len(stops):
            stop = start + int(stops[0])
    ratings = [int(value) for value in values[start:stop] if value in _RATING_CODES]
    return float(ratings[0]) if ratings else np.nan


def parse_event_trials(events: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """Reconstruct trials from cue, stimulus, and rating trigger sequences."""
    required = {"onset", "sample", "value"}
    if not required.issubset(events.columns):
        missing = sorted(required - set(events.columns))
        raise EmotionRegulationError(f"Event table is missing columns: {missing}")
    frame = events.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="raise").astype(int)
    values = frame["value"].to_numpy()
    stimulus_positions = np.flatnonzero(np.isin(values, list(_CONDITIONS)))
    rows: list[dict[str, object]] = []

    previous_stimulus = -1
    for trial_index, stimulus_position in enumerate(stimulus_positions, start=1):
        stimulus_position = int(stimulus_position)
        before = values[previous_stimulus + 1 : stimulus_position]
        cue_offsets = np.flatnonzero(np.isin(before, list(_CUE_CODES)))
        if not len(cue_offsets):
            raise EmotionRegulationError(
                f"No cue before {subject_id} trial {trial_index}"
            )
        cue_position = previous_stimulus + 1 + int(cue_offsets[-1])
        cue_code = int(values[cue_position])
        source_stimulus_code = int(values[stimulus_position])
        stimulus_code = cue_code + 6
        condition_alignment_status = (
            "verified"
            if source_stimulus_code == stimulus_code
            else "corrected_from_cue"
        )

        next_stimulus = (
            int(stimulus_positions[trial_index])
            if trial_index < len(stimulus_positions)
            else len(values)
        )
        after = values[stimulus_position + 1 : next_stimulus]
        next_cues = np.flatnonzero(np.isin(after, list(_CUE_CODES)))
        if len(next_cues):
            after = after[: int(next_cues[0])]
        arousal = _rating_after(after, 49, 48)
        valence = _rating_after(after, 48, None)
        missing = []
        if np.isnan(arousal):
            missing.append("arousal")
        if np.isnan(valence):
            missing.append("valence")
        status = "complete" if not missing else f"missing_{'_and_'.join(missing)}"
        content, nominal_valence, regulation = _CONDITIONS[stimulus_code]
        source = frame.iloc[stimulus_position]
        rows.append(
            {
                "trial_uid": f"ds006866:{subject_id}:trial-{trial_index:03d}",
                "dataset_id": "ds006866",
                "subject_id": subject_id,
                "session_id": "ses-01",
                "trial_index": trial_index,
                "cue_code": cue_code,
                "source_stimulus_code": source_stimulus_code,
                "stimulus_code": stimulus_code,
                "condition_alignment_status": condition_alignment_status,
                "condition_id": f"{content}_{nominal_valence}_{regulation}",
                "content": content,
                "nominal_valence": nominal_valence,
                "regulation": regulation,
                "eeg_onset_seconds": float(source["onset"]),
                "eeg_onset_s": float(source["onset"]),
                "eeg_sample": float(source["sample"]),
                "eeg_duration_s": 5.0,
                "arousal": arousal,
                "valence": valence,
                "arousal_available": bool(np.isfinite(arousal)),
                "valence_available": bool(np.isfinite(valence)),
                "rating_alignment_status": status,
                "extra_trigger_2048_count": int(np.count_nonzero(after == 2048)),
                "stimulus_id": "",
                "stimulus_identity_status": "not_distributed",
            }
        )
        previous_stimulus = stimulus_position
    return pd.DataFrame(rows)


def build_emotion_regulation_trial_table(dataset_root: Path) -> pd.DataFrame:
    dataset_root = dataset_root.resolve()
    participants = pd.read_csv(dataset_root / "participants.tsv", sep="\t")
    expected_subjects = set(participants["participant_id"].astype(str))
    event_paths = sorted(dataset_root.glob("sub-*/eeg/*_events.tsv"))
    observed_subjects = {path.parts[-3] for path in event_paths}
    if observed_subjects != expected_subjects:
        missing = sorted(expected_subjects - observed_subjects)
        unexpected = sorted(observed_subjects - expected_subjects)
        raise EmotionRegulationError(
            f"Participant/event mismatch; missing={missing}, unexpected={unexpected}"
        )

    tables = []
    for event_path in event_paths:
        subject_id = event_path.parts[-3]
        table = parse_event_trials(pd.read_csv(event_path, sep="\t"), subject_id)
        eeg_name = event_path.name.replace("_events.tsv", "_eeg.set")
        eeg_path = event_path.parent / eeg_name
        table["event_path"] = str(event_path.relative_to(dataset_root))
        table["eeg_path"] = str(eeg_path.relative_to(dataset_root))
        table["eeg_file_available"] = eeg_path.is_file()
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    if result["trial_uid"].duplicated().any():
        raise EmotionRegulationError("Trial identifiers are not unique")
    return result.sort_values(["subject_id", "trial_index"]).reset_index(drop=True)


def emotion_regulation_audit(table: pd.DataFrame) -> dict[str, object]:
    subject_trials = table.groupby("subject_id").size()
    condition_counts = table["condition_id"].value_counts().sort_index()
    noise = table.groupby("subject_id")["extra_trigger_2048_count"].sum()
    return {
        "dataset_id": "ds006866",
        "trial_count": len(table),
        "subject_count": int(table["subject_id"].nunique()),
        "condition_counts": {key: int(value) for key, value in condition_counts.items()},
        "subjects_with_nonstandard_trial_count": {
            key: int(value) for key, value in subject_trials[subject_trials.ne(240)].items()
        },
        "complete_rating_trial_count": int(
            table["rating_alignment_status"].eq("complete").sum()
        ),
        "missing_arousal_trial_count": int(table["arousal"].isna().sum()),
        "missing_valence_trial_count": int(table["valence"].isna().sum()),
        "corrected_condition_trial_count": int(
            table["condition_alignment_status"].eq("corrected_from_cue").sum()
        ),
        "subjects_with_trigger_2048": {
            key: int(value) for key, value in noise[noise.gt(0)].items()
        },
        "raw_eeg_available_subject_count": int(
            table.loc[table["eeg_file_available"], "subject_id"].nunique()
        ),
        "stimulus_identity_available": False,
        "stimulus_identity_limitation": (
            "The BIDS snapshot exposes condition triggers but no per-trial image IDs."
        ),
    }
