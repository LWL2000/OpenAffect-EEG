"""Trial-level parsing for the OpenNeuro DENS dataset (ds003751)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATASET_ID = "ds003751"
STIMULUS_DURATION_S = 60.0
_TRIAL_SUFFIX = re.compile(r"_\d+$")
_BEHAVIOR_RENAME = {
    "stimuliName": "behavior_stimulus_name",
    "valence": "self_valence",
    "arousal": "self_arousal",
    "dominance": "self_dominance",
    "liking": "self_liking",
    "familiarity": "self_familiarity",
    "relevance": "self_relevance",
    "emotionCateg": "self_emotion_category",
    "Quadrant": "self_quadrant",
    "FivePointScale": "self_emotion_intensity",
    "MouseClick": "self_emotion_mouse_click",
}


class DensEventError(ValueError):
    pass


def _normalize_event_label(value: object) -> str:
    return _TRIAL_SUFFIX.sub("", str(value).strip())


def _normalize_behavior_name(value: object) -> str:
    return Path(str(value).strip()).stem


def parse_dens_events(
    events: pd.DataFrame,
    *,
    sampling_frequency: float,
) -> pd.DataFrame:
    required = {"onset", "trial_type", "label"}
    if not required.issubset(events.columns):
        missing = ", ".join(sorted(required - set(events.columns)))
        raise DensEventError(f"Missing required event columns: {missing}")
    if sampling_frequency <= 0:
        raise DensEventError("Sampling frequency must be positive")

    stimulation = events.loc[events["trial_type"].eq("stm")].copy()
    stimulation["onset"] = pd.to_numeric(stimulation["onset"], errors="raise")
    stimulation = stimulation.sort_values("onset", kind="mergesort").reset_index(
        drop=True
    )
    if stimulation.empty:
        raise DensEventError("No stm events found")

    return pd.DataFrame(
        {
            "trial_index": range(1, len(stimulation) + 1),
            "event_label": stimulation["label"].astype(str),
            "stimulus_id": stimulation["label"].map(_normalize_event_label),
            "onset_samples": stimulation["onset"].astype(float),
            "eeg_onset_s": stimulation["onset"].astype(float)
            / float(sampling_frequency),
            "eeg_duration_s": STIMULUS_DURATION_S,
        }
    )


def attach_dens_behavior(
    trials: pd.DataFrame,
    behavior: pd.DataFrame | None,
) -> pd.DataFrame:
    output_columns = list(_BEHAVIOR_RENAME.values())
    if behavior is None or behavior.empty:
        merged = trials.copy()
        for column in output_columns:
            merged[column] = pd.NA
        merged["label_available"] = False
        return merged

    required = {
        "stimuliName",
        "valence",
        "arousal",
        "dominance",
        "liking",
        "familiarity",
        "relevance",
    }
    if not required.issubset(behavior.columns):
        missing = ", ".join(sorted(required - set(behavior.columns)))
        raise DensEventError(f"Missing required behavior columns: {missing}")

    selected = behavior.copy()
    for source in _BEHAVIOR_RENAME:
        if source not in selected.columns:
            selected[source] = pd.NA
    selected["stimulus_id"] = selected["stimuliName"].map(
        _normalize_behavior_name
    )
    if selected["stimulus_id"].duplicated().any():
        duplicates = sorted(
            selected.loc[selected["stimulus_id"].duplicated(False), "stimulus_id"]
            .astype(str)
            .unique()
        )
        raise DensEventError(f"Duplicate behavior stimuli: {duplicates}")

    selected = selected[["stimulus_id", *_BEHAVIOR_RENAME]].rename(
        columns=_BEHAVIOR_RENAME
    )
    merged = trials.merge(
        selected,
        on="stimulus_id",
        how="left",
        validate="m:1",
        indicator=True,
    )
    merged["label_available"] = merged.pop("_merge").eq("both")
    return merged


def _deduplicate_participants(participants: pd.DataFrame) -> pd.DataFrame:
    if "participant_id" not in participants.columns:
        raise DensEventError("participants.tsv has no participant_id column")
    duplicates = participants.loc[
        participants["participant_id"].duplicated(False)
    ]
    for subject_id, group in duplicates.groupby("participant_id", sort=False):
        if any(group[column].nunique(dropna=False) > 1 for column in group.columns):
            raise DensEventError(f"Conflicting participant rows for {subject_id}")
    return participants.drop_duplicates("participant_id").rename(
        columns={"participant_id": "subject_id"}
    )


def _one_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise DensEventError(f"Expected one {description}; found {len(paths)}")
    return paths[0]


def build_dens_trial_table(dataset_root: Path) -> pd.DataFrame:
    dataset_root = dataset_root.resolve()
    participants = _deduplicate_participants(
        pd.read_csv(dataset_root / "participants.tsv", sep="\t")
    )
    rows: list[pd.DataFrame] = []

    for event_path in sorted(dataset_root.glob("sub-*/eeg/*_events.tsv")):
        subject_id = event_path.parents[1].name
        eeg_dir = event_path.parent
        sidecar_path = _one_path(
            list(eeg_dir.glob("*_eeg.json")), f"EEG sidecar for {subject_id}"
        )
        eeg_path = _one_path(
            list(eeg_dir.glob("*_eeg.set")), f"EEGLAB set file for {subject_id}"
        )
        fdt_path = _one_path(
            list(eeg_dir.glob("*_eeg.fdt")), f"EEGLAB fdt file for {subject_id}"
        )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sampling_frequency = float(sidecar["SamplingFrequency"])
        trials = parse_dens_events(
            pd.read_csv(event_path, sep="\t"),
            sampling_frequency=sampling_frequency,
        )

        behavior_paths = sorted((dataset_root / subject_id / "beh").glob("*_beh.tsv"))
        if len(behavior_paths) > 1:
            raise DensEventError(
                f"Expected at most one behavior file for {subject_id}; "
                f"found {len(behavior_paths)}"
            )
        behavior = (
            pd.read_csv(behavior_paths[0], sep="\t") if behavior_paths else None
        )
        trials = attach_dens_behavior(trials, behavior)
        trials.insert(
            0,
            "trial_uid",
            [
                f"{DATASET_ID}:{subject_id}:trial-{index:02d}"
                for index in trials["trial_index"]
            ],
        )
        trials.insert(1, "dataset_id", DATASET_ID)
        trials.insert(2, "subject_id", subject_id)
        trials["context"] = "naturalistic_video"
        trials["stimulus_modality"] = "audiovisual"
        trials["stimulus_uid"] = "dens-" + trials["stimulus_id"].astype(str)
        trials["stimulus_mapping"] = "event_label_without_trial_suffix"
        trials["sampling_frequency_hz"] = sampling_frequency
        trials["eeg_path"] = eeg_path.relative_to(dataset_root).as_posix()
        trials["fdt_path"] = fdt_path.relative_to(dataset_root).as_posix()
        trials["events_path"] = event_path.relative_to(dataset_root).as_posix()
        trials["behavior_path"] = (
            behavior_paths[0].relative_to(dataset_root).as_posix()
            if behavior_paths
            else pd.NA
        )
        rows.append(trials)

    if not rows:
        raise DensEventError(f"No DENS event files found under {dataset_root}")
    table = pd.concat(rows, ignore_index=True)
    table = table.merge(participants, on="subject_id", how="left", validate="m:1")
    return table.sort_values(["subject_id", "trial_index"]).reset_index(drop=True)


def dens_trial_audit(table: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset_id": DATASET_ID,
        "trial_count": len(table),
        "subject_count": int(table["subject_id"].nunique()),
        "unique_stimulus_count": int(table["stimulus_id"].nunique()),
        "labeled_trial_count": int(table["label_available"].sum()),
        "unlabeled_trial_count": int((~table["label_available"]).sum()),
        "subjects_with_behavior": int(
            table.loc[table["label_available"], "subject_id"].nunique()
        ),
        "event_time_unit": "samples",
        "stimulus_duration_s": STIMULUS_DURATION_S,
        "metadata_notes": [
            "BIDS onset and duration fields contain sample indices, not seconds.",
            "Behavior rows are joined by normalized stimulus name, not row order.",
            "Missing behavior labels are retained and explicitly flagged.",
        ],
    }
