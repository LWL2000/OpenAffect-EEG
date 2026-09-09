"""Trial-level parsing for the OpenNeuro MusicEEG dataset (ds002721)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MUSIC_START = 788
FIXATION_START = 786
MUSIC_DURATION_S = 12.0
STIMULUS_CODE_MIN = 301
STIMULUS_CODE_MAX = 660
QUESTION_NAMES = {
    800: "pleasant",
    801: "energetic",
    802: "tense",
    803: "angry",
    804: "afraid",
    805: "happy",
    806: "sad",
    807: "tender",
}
NORMATIVE_COLUMNS = {
    "pleasant": "valence",
    "energetic": "energy",
    "tense": "tension",
    "angry": "anger",
    "afraid": "fear",
    "happy": "happy",
    "sad": "sad",
    "tender": "tender",
}
_EVENT_FILE = re.compile(
    r"(?P<subject>sub-\d+)_task-(?P<run>run\d+)_events\.tsv$"
)


class MusicEventError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedMusicTrial:
    trial_index: int
    fixation_onset_s: float
    onset_s: float
    stimulus_code: int
    stimulus_id: int
    ratings: dict[str, int]


def _parse_ratings(events: pd.DataFrame, trial_index: int) -> dict[str, int]:
    active_question: int | None = None
    ratings: dict[int, int] = {}

    for onset, group in events.groupby("onset", sort=True):
        codes = [int(value) for value in group["trial_type"]]
        responses = [code for code in codes if 833 <= code <= 841]
        questions = sorted({code for code in codes if code in QUESTION_NAMES})

        if responses:
            if len(responses) != 1:
                raise MusicEventError(
                    f"Trial {trial_index} has multiple responses at {onset}: {responses}"
                )
            if active_question is None:
                raise MusicEventError(
                    f"Trial {trial_index} has a response without an active question"
                )
            if active_question in ratings:
                raise MusicEventError(
                    f"Trial {trial_index} repeats question {active_question}"
                )
            ratings[active_question] = responses[0] - 832
            active_question = None

        unseen = [question for question in questions if question not in ratings]
        if len(unseen) > 1:
            raise MusicEventError(
                f"Trial {trial_index} activates multiple questions at {onset}: {unseen}"
            )
        if unseen:
            question = unseen[0]
            if active_question is not None and active_question != question:
                raise MusicEventError(
                    f"Trial {trial_index} replaces unanswered question "
                    f"{active_question} with {question}"
                )
            active_question = question

    if len(ratings) != len(QUESTION_NAMES):
        raise MusicEventError(
            f"Trial {trial_index} has {len(ratings)} ratings; expected 8 ratings"
        )
    return {QUESTION_NAMES[question]: ratings[question] for question in QUESTION_NAMES}


def parse_music_events(events: pd.DataFrame) -> list[ParsedMusicTrial]:
    required = {"onset", "trial_type"}
    if not required.issubset(events.columns):
        missing = ", ".join(sorted(required - set(events.columns)))
        raise MusicEventError(f"Missing required columns: {missing}")

    ordered = events.sort_values("onset", kind="mergesort").copy()
    ordered["onset"] = pd.to_numeric(ordered["onset"], errors="raise")
    ordered["trial_type"] = pd.to_numeric(
        ordered["trial_type"], errors="raise"
    ).astype(int)
    music_onsets = ordered.loc[
        ordered["trial_type"].eq(MUSIC_START), "onset"
    ].tolist()
    fixation_onsets = ordered.loc[
        ordered["trial_type"].eq(FIXATION_START), "onset"
    ].tolist()

    trials: list[ParsedMusicTrial] = []
    for index, onset in enumerate(music_onsets, start=1):
        prior_fixations = [value for value in fixation_onsets if value <= onset]
        if not prior_fixations:
            raise MusicEventError(f"Trial {index} has no preceding fixation")
        fixation_onset = max(prior_fixations)
        next_fixation = min(
            (value for value in fixation_onsets if value > onset),
            default=float("inf"),
        )
        stimulus_window_start = (
            music_onsets[index - 2] + MUSIC_DURATION_S if index > 1 else 0.0
        )
        stimulus_codes = ordered.loc[
            ordered["onset"].between(stimulus_window_start, onset, inclusive="both")
            & ordered["trial_type"].between(
                STIMULUS_CODE_MIN, STIMULUS_CODE_MAX, inclusive="both"
            ),
            "trial_type",
        ].tolist()
        if len(stimulus_codes) != 1:
            raise MusicEventError(
                f"Trial {index} has {len(stimulus_codes)} stimulus codes; expected 1"
            )
        stimulus_code = int(stimulus_codes[0])
        questionnaire = ordered.loc[
            ordered["onset"].gt(onset) & ordered["onset"].lt(next_fixation)
        ]
        ratings = _parse_ratings(questionnaire, index)
        trials.append(
            ParsedMusicTrial(
                trial_index=index,
                fixation_onset_s=float(fixation_onset),
                onset_s=float(onset),
                stimulus_code=stimulus_code,
                stimulus_id=stimulus_code - 300,
                ratings=ratings,
            )
        )
    return trials


def build_music_trial_table(
    dataset_root: Path,
    normative_ratings_path: Path,
) -> pd.DataFrame:
    dataset_root = dataset_root.resolve()
    participant_path = dataset_root / "participants.tsv"
    participants = pd.read_csv(participant_path, sep="\t").rename(
        columns={"participant_id": "subject_id"}
    )
    rows: list[dict[str, object]] = []

    for event_path in sorted(dataset_root.glob("sub-*/eeg/*_events.tsv")):
        match = _EVENT_FILE.match(event_path.name)
        if match is None:
            raise MusicEventError(f"Unexpected event filename: {event_path.name}")
        events = pd.read_csv(event_path, sep="\t")
        parsed = parse_music_events(events)
        if not parsed:
            continue
        subject_id = match.group("subject")
        run_id = match.group("run")
        run_index = int(run_id.removeprefix("run"))
        eeg_path = event_path.with_name(event_path.name.replace("_events.tsv", "_eeg.edf"))
        if not eeg_path.exists():
            raise FileNotFoundError(eeg_path)

        for trial in parsed:
            row: dict[str, object] = {
                "trial_uid": (
                    f"ds002721:{subject_id}:{run_id}:trial-{trial.trial_index:02d}"
                ),
                "dataset_id": "ds002721",
                "subject_id": subject_id,
                "run_id": run_id,
                "run_index": run_index,
                "trial_index": trial.trial_index,
                "context": "music_listening",
                "stimulus_modality": "audio",
                "stimulus_uid": f"eerola-set1-{trial.stimulus_id:03d}",
                "stimulus_id": trial.stimulus_id,
                "stimulus_code": trial.stimulus_code,
                "stimulus_mapping": "inferred_set1_code_minus_300",
                "normative_source": "osf:p6vkg:mean_ratings_set1",
                "fixation_onset_s": trial.fixation_onset_s,
                "eeg_onset_s": trial.onset_s,
                "eeg_duration_s": MUSIC_DURATION_S,
                "eeg_path": eeg_path.relative_to(dataset_root).as_posix(),
                "events_path": event_path.relative_to(dataset_root).as_posix(),
            }
            row.update(
                {f"self_{name}": value for name, value in trial.ratings.items()}
            )
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        raise MusicEventError(f"No music trials found under {dataset_root}")

    normative = pd.read_csv(normative_ratings_path).rename(
        columns={"number": "stimulus_id", "TARGET": "normative_target"}
    )
    normative["stimulus_id"] = pd.to_numeric(
        normative["stimulus_id"], errors="raise"
    ).astype(int)
    normative = normative.rename(
        columns={
            source: f"normative_{target}"
            for target, source in NORMATIVE_COLUMNS.items()
        }
    )
    keep = [
        "stimulus_id",
        "normative_target",
        *(f"normative_{name}" for name in NORMATIVE_COLUMNS),
    ]
    table = table.merge(normative[keep], on="stimulus_id", how="left", validate="m:1")
    table = table.merge(participants, on="subject_id", how="left", validate="m:1")
    if table["normative_target"].isna().any():
        missing = sorted(table.loc[table["normative_target"].isna(), "stimulus_id"].unique())
        raise MusicEventError(f"Missing normative ratings for stimuli: {missing}")

    for name in NORMATIVE_COLUMNS:
        table[f"residual_{name}"] = (
            table[f"self_{name}"] - table[f"normative_{name}"]
        )
    return table.sort_values(
        ["subject_id", "run_index", "trial_index"]
    ).reset_index(drop=True)


def music_trial_audit(table: pd.DataFrame) -> dict[str, object]:
    duplicate_pairs = int(
        table.duplicated(["subject_id", "stimulus_id"], keep=False).sum()
    )
    return {
        "dataset_id": "ds002721",
        "trial_count": len(table),
        "subject_count": int(table["subject_id"].nunique()),
        "task_run_count": int(table[["subject_id", "run_id"]].drop_duplicates().shape[0]),
        "unique_stimulus_count": int(table["stimulus_id"].nunique()),
        "stimulus_id_min": int(table["stimulus_id"].min()),
        "stimulus_id_max": int(table["stimulus_id"].max()),
        "subject_stimulus_duplicate_rows": duplicate_pairs,
        "self_rating_count": int(
            table[[f"self_{name}" for name in QUESTION_NAMES.values()]].count().sum()
        ),
        "self_rating_min": int(
            table[[f"self_{name}" for name in QUESTION_NAMES.values()]].min().min()
        ),
        "self_rating_max": int(
            table[[f"self_{name}" for name in QUESTION_NAMES.values()]].max().max()
        ),
        "metadata_note": (
            "Observed stimulus codes span the Set 1 convention 301-660; "
            "OpenNeuro's events JSON incorrectly documents 301-360."
        ),
    }
