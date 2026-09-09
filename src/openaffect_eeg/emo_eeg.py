"""Trial alignment and differential-entropy features for EmoEEG-MC (ds005540)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

CANONICAL_TRIAL_ORDER = (
    "sad4",
    "sad5",
    "sad8",
    "dis4",
    "dis5",
    "dis8",
    "fear4",
    "fear5",
    "fear8",
    "neu4",
    "neu5",
    "neu8",
    "joy4",
    "joy5",
    "joy8",
    "ten4",
    "ten5",
    "ten8",
    "ins4",
    "ins5",
    "ins8",
)
MISSING_TRIAL_NUMBERS = {
    "sub-55": (19, 20, 21, 40, 41, 42),
    "sub-56": (20, 21),
    "sub-57": (6, 8, 13, 21, 23, 24, 36, 37, 38, 42),
    "sub-58": (9, 20, 21),
    "sub-59": (2, 4, 6, 12, 19, 21, 29, 37, 39, 42),
    "sub-60": (*range(8, 22), *range(31, 43)),
}
SCORE_NAMES = {
    "score_1": "joy",
    "score_2": "inspiration",
    "score_3": "tenderness",
    "score_4": "sadness",
    "score_5": "fear",
    "score_6": "disgust",
    "score_7": "arousal",
    "score_8": "valence",
    "score_9": "familiarity",
    "score_10": "liking",
}
NOMINAL_CATEGORIES = {
    "sad": "sadness",
    "dis": "disgust",
    "fear": "fear",
    "neu": "neutral",
    "joy": "joy",
    "ten": "tenderness",
    "ins": "inspiration",
}
_MATERIAL_CATEGORY_OFFSETS = {
    "joy": 0,
    "ins": 3,
    "ten": 6,
    "sad": 9,
    "fear": 12,
    "dis": 15,
    "neu": 18,
}
_EXEMPLAR_OFFSETS = {4: 1, 5: 2, 8: 3}

_NUMBERED_SEQUENCE = re.compile(r"^(?P<number>\d+)\.\s+`(?P<value>\[.*\])`\s*$")
_SUBJECT_HEADING = re.compile(r"sub-?(?P<number>\d+)", re.IGNORECASE)
_LIST_LITERAL = re.compile(r"`(?P<value>\[.*\])`")


class EmoEEGError(ValueError):
    pass


def stimulus_material_number(token: str) -> int:
    """Map a canonical token such as sad8 to the 1-based stimulus workbook row."""

    match = re.fullmatch(r"(sad|dis|fear|neu|joy|ten|ins)([458])", token)
    if match is None:
        raise EmoEEGError(f"Invalid stimulus token: {token}")
    category, exemplar = match.groups()
    return _MATERIAL_CATEGORY_OFFSETS[category] + _EXEMPLAR_OFFSETS[int(exemplar)]


def pool_video_embedding(path: Path) -> np.ndarray:
    """Pool published per-second visual and audio embeddings into one vector."""

    payload = loadmat(path)
    features = []
    width: int | None = None
    for name in ("visual_feature", "audio_feature"):
        if name not in payload:
            raise EmoEEGError(f"Video embedding has no {name}: {path}")
        values = np.asarray(payload[name], dtype=np.float64)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise EmoEEGError(f"Invalid {name} array {values.shape}: {path}")
        if width is None:
            width = values.shape[1]
        elif values.shape[1] != width:
            raise EmoEEGError(f"Audio/visual embedding widths differ: {path}")
        features.append(values.mean(axis=0))
    return np.concatenate(features)


def _parse_sequence(raw: str) -> list[str]:
    try:
        values = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise EmoEEGError(f"Invalid trial sequence: {raw}") from error
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise EmoEEGError("Trial sequence must be a list of strings")
    unknown = sorted(set(values) - set(CANONICAL_TRIAL_ORDER))
    if unknown:
        raise EmoEEGError(f"Unknown trial tokens: {unknown}")
    if len(values) != len(set(values)):
        raise EmoEEGError(f"Trial sequence contains duplicate tokens: {values}")
    return values


def parse_trial_sequences(readme_text: str) -> dict[str, dict[str, list[str]]]:
    """Extract the per-subject acquisition sequences documented in the README."""

    sequences: dict[str, dict[str, list[str]]] = {}
    active_subject: str | None = None
    active_context: str | None = None

    for raw_line in readme_text.splitlines():
        line = raw_line.strip()
        numbered = _NUMBERED_SEQUENCE.match(line)
        if numbered is not None:
            subject = f"sub-{int(numbered.group('number')):02d}"
            sequence = _parse_sequence(numbered.group("value"))
            sequences[subject] = {"ima": sequence, "vid": list(sequence)}
            active_subject = None
            active_context = None
            continue

        subject_match = _SUBJECT_HEADING.search(line) if line.startswith("-") else None
        if subject_match is not None:
            active_subject = f"sub-{int(subject_match.group('number')):02d}"
            sequences.setdefault(active_subject, {})
            lowered = line.lower()
            if "imagery and video" in lowered:
                active_context = "both"
            elif "imagery" in lowered:
                active_context = "ima"
            elif "video" in lowered:
                active_context = "vid"
            else:
                active_context = None
            continue

        lowered = line.lower()
        if "imagery sequence" in lowered:
            active_context = "ima"
            continue
        if "video sequence" in lowered:
            active_context = "vid"
            continue

        literal = _LIST_LITERAL.search(line)
        if literal is None or active_subject is None or active_context is None:
            continue
        sequence = _parse_sequence(literal.group("value"))
        if active_context == "both":
            sequences[active_subject]["ima"] = sequence
            sequences[active_subject]["vid"] = list(sequence)
        else:
            sequences[active_subject][active_context] = sequence

    incomplete = {
        subject: sorted({"ima", "vid"} - set(contexts))
        for subject, contexts in sequences.items()
        if set(contexts) != {"ima", "vid"}
    }
    if incomplete:
        raise EmoEEGError(f"Incomplete trial sequences: {incomplete}")
    if not sequences:
        raise EmoEEGError("No per-subject trial sequences found")
    return sequences


def _parse_stimulus_code(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if match is None:
        raise EmoEEGError(f"Invalid global stimulus code: {value}")
    code = int(match.group())
    if not 1 <= code <= 42:
        raise EmoEEGError(f"Global stimulus code is outside 1-42: {value}")
    return code


def _token_from_material_number(material_number: int) -> str:
    matches = [
        token
        for token in CANONICAL_TRIAL_ORDER
        if stimulus_material_number(token) == material_number
    ]
    if len(matches) != 1:
        raise EmoEEGError(f"No unique token for material {material_number}")
    return matches[0]


def _stimulus_code(context_code: str, token: str) -> int:
    material_number = stimulus_material_number(token)
    return material_number if context_code == "ima" else material_number + 21


def align_behavior_to_de(
    behavior: pd.DataFrame,
    sequences: dict[str, list[str]],
) -> pd.DataFrame:
    """Map global stimulus-coded ratings onto the canonical DE trial order."""

    required = {"trial_number", "video_name", *SCORE_NAMES}
    if not required.issubset(behavior.columns):
        missing = sorted(required - set(behavior.columns))
        raise EmoEEGError(f"Behavior table is missing required columns: {missing}")
    if set(sequences) != {"ima", "vid"}:
        raise EmoEEGError("Sequences must contain exactly ima and vid contexts")

    expected_codes = {
        _stimulus_code(context, token)
        for context in ("ima", "vid")
        for token in sequences[context]
    }
    frame = behavior.copy()
    try:
        frame["stimulus_code"] = frame["video_name"].map(_parse_stimulus_code)
        codes_are_usable = (
            len(frame) == len(expected_codes)
            and not frame["stimulus_code"].duplicated().any()
            and set(frame["stimulus_code"]) == expected_codes
        )
    except EmoEEGError:
        codes_are_usable = False
    rows: list[dict[str, object]] = []

    if codes_are_usable:
        for source in frame.to_dict("records"):
            code = int(source["stimulus_code"])
            context = "ima" if code <= 21 else "vid"
            material_number = code if code <= 21 else code - 21
            token = _token_from_material_number(material_number)
            if token not in sequences[context]:
                raise EmoEEGError(
                    f"Behavior code {code} maps to unavailable {context}:{token}"
                )
            row = dict(source)
            row.update(
                {
                    "context_code": context,
                    "stimulus_token": token,
                    "experimental_trial_index": sequences[context].index(token) + 1,
                    "behavior_trial_number": int(source["trial_number"]),
                    "canonical_order": CANONICAL_TRIAL_ORDER.index(token),
                    "label_alignment_status": "verified_global_stimulus_code",
                    "label_available": True,
                }
            )
            rows.append(row)
    else:
        for context in ("ima", "vid"):
            for token in sequences[context]:
                rows.append(
                    {
                        "trial_number": np.nan,
                        "video_name": "",
                        "stimulus_code": _stimulus_code(context, token),
                        **{column: np.nan for column in SCORE_NAMES},
                        "context_code": context,
                        "stimulus_token": token,
                        "experimental_trial_index": sequences[context].index(token) + 1,
                        "behavior_trial_number": np.nan,
                        "canonical_order": CANONICAL_TRIAL_ORDER.index(token),
                        "label_alignment_status": "source_stimulus_code_unusable",
                        "label_available": False,
                    }
                )

    aligned = pd.DataFrame(rows).sort_values(
        ["context_code", "canonical_order"], kind="mergesort"
    )
    aligned["de_trial_index"] = aligned.groupby("context_code").cumcount() + 1
    return aligned.reset_index(drop=True)


def summarize_de_trials(
    de_path: Path,
    trial_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate 1-second DE values into channel-level and global trial features."""

    values, _ = _load_de_values(de_path)
    expected_seconds = trial_count * 30
    if values.shape != (64, expected_seconds, 5):
        raise EmoEEGError(
            f"DE shape {values.shape} is inconsistent; expected 60-second axis "
            f"length {expected_seconds} for {trial_count} trials"
        )
    if not np.isfinite(values).all():
        raise EmoEEGError(f"DE file contains non-finite values: {de_path}")

    trials = values.reshape(64, trial_count, 30, 5).transpose(1, 0, 2, 3)
    channel_mean = trials.mean(axis=2)
    global_mean = trials.mean(axis=(1, 2))
    global_std = trials.std(axis=(1, 2))
    return channel_mean, np.concatenate([global_mean, global_std], axis=1)


def _load_de_values(de_path: Path) -> tuple[np.ndarray, str]:
    loaded = np.load(de_path, allow_pickle=True)
    if loaded.shape == () and loaded.dtype == object:
        payload = loaded.item()
        if not isinstance(payload, dict) or "de" not in payload:
            raise EmoEEGError(f"DE file has no 'de' array: {de_path}")
        values = np.asarray(payload["de"], dtype=np.float64)
        storage_format = "object_dictionary"
    elif loaded.ndim == 4 and loaded.shape[0] == 1:
        values = np.asarray(loaded[0], dtype=np.float64)
        storage_format = "numeric_leading_batch"
    elif loaded.ndim == 3:
        values = np.asarray(loaded, dtype=np.float64)
        storage_format = "numeric_array"
    else:
        raise EmoEEGError(f"Unsupported DE storage shape {loaded.shape}: {de_path}")
    if values.ndim != 3 or values.shape[0] != 64 or values.shape[2] != 5:
        raise EmoEEGError(f"Unsupported DE feature geometry {values.shape}: {de_path}")
    return values, storage_format


def inspect_de_file(de_path: Path, expected_trial_count: int) -> dict[str, object]:
    """Report whether a DE file can be aligned without repairing source data."""

    values, storage_format = _load_de_values(de_path)
    complete_trials, trailing_seconds = divmod(values.shape[1], 30)
    if trailing_seconds:
        nonfinite_trial_count = 0
    else:
        trials = values.reshape(64, complete_trials, 30, 5).transpose(1, 0, 2, 3)
        nonfinite_trial_count = int(
            (~np.isfinite(trials).all(axis=(1, 2, 3))).sum()
        )
    valid = (
        trailing_seconds == 0
        and complete_trials == expected_trial_count
        and nonfinite_trial_count == 0
    )
    return {
        "storage_format": storage_format,
        "array_shape": list(values.shape),
        "expected_trial_count": expected_trial_count,
        "observed_trial_count": complete_trials,
        "trailing_second_count": trailing_seconds,
        "nonfinite_trial_count": nonfinite_trial_count,
        "alignment_status": "verified" if valid else "source_inconsistent_excluded",
    }


def _find_de_path(dataset_root: Path, subject_id: str, context_code: str) -> Path:
    filename = f"{subject_id}_ses-{context_code}_task-emotion_de.npy"
    candidates = (
        dataset_root / "derivatives" / subject_id / f"ses-{context_code}" / "eeg" / filename,
        dataset_root / "derivatives" / subject_id / f"ses-{context_code}" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


def _stimulus_metadata(dataset_root: Path) -> pd.DataFrame:
    imagery_path = dataset_root / "stimuli" / "ses-ima" / "imagery_guidence.xlsx"
    video_path = dataset_root / "stimuli" / "ses-vid" / "video_description.xlsx"
    imagery = pd.read_excel(imagery_path, sheet_name="imagery")
    video = pd.read_excel(video_path, sheet_name="video")
    if len(imagery) != 21 or len(video) != 21:
        raise EmoEEGError("Expected exactly 21 imagery and 21 video stimulus rows")

    records: list[dict[str, object]] = []
    for context_code, frame, metadata_path in (
        ("ima", imagery, imagery_path),
        ("vid", video, video_path),
    ):
        indexed = frame.set_index("number", verify_integrity=True)
        for token in CANONICAL_TRIAL_ORDER:
            material_number = stimulus_material_number(token)
            source = indexed.loc[material_number]
            nominal_category = NOMINAL_CATEGORIES[
                token.removesuffix(token[-1])
            ]
            observed_category = str(source["emotion category"]).strip().lower()
            if observed_category != nominal_category:
                raise EmoEEGError(
                    f"Stimulus {context_code}:{token} maps to category "
                    f"{observed_category}, expected {nominal_category}"
                )
            if context_code == "ima":
                text_en = source["the guidance (English version)"]
                text_zh = source["the guidance (Chinese version)"]
                description = text_en
                resource = "guided_imagery_text"
                source_reference = ""
                duration_s = source["duration/seconds"]
                embedding_path = ""
            else:
                text_en = ""
                text_zh = ""
                description = source["description"]
                resource = source["resource"]
                source_reference = source["source"]
                duration_s = source["duration (seconds)"]
                embedding_path = (
                    dataset_root
                    / "stimuli"
                    / "ses-vid"
                    / f"video_{material_number:02d}_embedding.mat"
                ).relative_to(dataset_root).as_posix()
            records.append(
                {
                    "context_code": context_code,
                    "stimulus_token": token,
                    "stimulus_material_number": material_number,
                    "nominal_category": nominal_category,
                    "stimulus_description": str(description).strip(),
                    "stimulus_text_en": str(text_en).strip(),
                    "stimulus_text_zh": str(text_zh).strip(),
                    "stimulus_resource": str(resource).strip(),
                    "stimulus_source_reference": str(source_reference).strip(),
                    "stimulus_duration_s": float(duration_s),
                    "stimulus_metadata_path": metadata_path.relative_to(
                        dataset_root
                    ).as_posix(),
                    "stimulus_embedding_path": embedding_path,
                }
            )
    return pd.DataFrame(records)


def build_emo_trial_table(dataset_root: Path) -> pd.DataFrame:
    """Build an auditable trial manifest without hiding source inconsistencies."""

    dataset_root = dataset_root.resolve()
    sequences = parse_trial_sequences((dataset_root / "README.md").read_text())
    stimuli = _stimulus_metadata(dataset_root)
    participants = pd.read_csv(dataset_root / "participants.tsv", sep="\t").rename(
        columns={"participant_id": "subject_id"}
    )
    rows: list[dict[str, object]] = []

    for subject_id in sorted(sequences):
        behavior_path = (
            dataset_root / subject_id / "beh" / f"{subject_id}_task-emotion_beh.tsv"
        )
        behavior = pd.read_csv(behavior_path, sep="\t")
        aligned = align_behavior_to_de(
            behavior,
            sequences[subject_id],
        )
        context_audits: dict[str, tuple[Path, dict[str, object]]] = {}
        for context_code in ("ima", "vid"):
            de_path = _find_de_path(dataset_root, subject_id, context_code)
            audit = inspect_de_file(
                de_path, expected_trial_count=len(sequences[subject_id][context_code])
            )
            context_audits[context_code] = (de_path, audit)

        for source in aligned.to_dict("records"):
            context_code = str(source["context_code"])
            token = str(source["stimulus_token"])
            de_path, de_audit = context_audits[context_code]
            row = dict(source)
            row.update(
                {
                    "trial_uid": f"ds005540:{subject_id}:ses-{context_code}:{token}",
                    "dataset_id": "ds005540",
                    "subject_id": subject_id,
                    "session_id": f"ses-{context_code}",
                    "context": (
                        "guided_imagery" if context_code == "ima" else "video_viewing"
                    ),
                    "stimulus_modality": (
                        "text_guided_imagery"
                        if context_code == "ima"
                        else "audiovisual_video"
                    ),
                    "stimulus_uid": f"ds005540:{context_code}:{token}",
                    "stimulus_intensity_code": int(token[-1]),
                    "behavior_path": behavior_path.relative_to(dataset_root).as_posix(),
                    "de_path": de_path.relative_to(dataset_root).as_posix(),
                    "eeg_path": (
                        Path(subject_id) / "eeg" / f"{subject_id}_task-emotion_eeg.edf"
                    ).as_posix(),
                    "de_storage_format": de_audit["storage_format"],
                    "de_expected_trial_count": de_audit["expected_trial_count"],
                    "de_observed_trial_count": de_audit["observed_trial_count"],
                    "de_nonfinite_trial_count": de_audit["nonfinite_trial_count"],
                    "feature_alignment_status": de_audit["alignment_status"],
                    "source_video_name": str(source.get("video_name", "")),
                }
            )
            for score_column, score_name in SCORE_NAMES.items():
                row[f"self_{score_name}"] = float(source[score_column])
            rows.append(row)

    table = pd.DataFrame(rows).merge(
        stimuli,
        on=["context_code", "stimulus_token"],
        how="left",
        validate="m:1",
    )
    table = table.merge(participants, on="subject_id", how="left", validate="m:1")
    if table["stimulus_description"].isna().any():
        raise EmoEEGError("Some trials have no stimulus semantics")
    if table["trial_uid"].duplicated().any():
        raise EmoEEGError("Trial identifiers are not unique")
    return table.sort_values(
        ["subject_id", "context_code", "de_trial_index"]
    ).reset_index(drop=True)


def emo_trial_audit(table: pd.DataFrame) -> dict[str, object]:
    session_columns = [
        "subject_id",
        "context_code",
        "de_path",
        "de_storage_format",
        "de_expected_trial_count",
        "de_observed_trial_count",
        "de_nonfinite_trial_count",
        "feature_alignment_status",
    ]
    sessions = table[session_columns].drop_duplicates()
    excluded = sessions.loc[
        sessions["feature_alignment_status"].ne("verified")
    ].to_dict("records")
    constant_video_name_subjects = int(
        table.groupby("subject_id")["source_video_name"].nunique().eq(1).sum()
    )
    supervised = table["label_available"] & table["feature_alignment_status"].eq(
        "verified"
    )
    return {
        "dataset_id": "ds005540",
        "trial_count": len(table),
        "subject_count": int(table["subject_id"].nunique()),
        "session_count": len(sessions),
        "context_counts": table["context"].value_counts().sort_index().to_dict(),
        "unique_stimulus_count": int(table["stimulus_uid"].nunique()),
        "feature_verified_trial_count": int(
            table["feature_alignment_status"].eq("verified").sum()
        ),
        "feature_excluded_trial_count": int(
            table["feature_alignment_status"].ne("verified").sum()
        ),
        "feature_verified_subject_count": int(
            table.loc[
                table["feature_alignment_status"].eq("verified"), "subject_id"
            ].nunique()
        ),
        "fully_verified_subject_count": int(
            sessions.groupby("subject_id")["feature_alignment_status"]
            .apply(lambda values: values.eq("verified").all())
            .sum()
        ),
        "verified_label_trial_count": int(table["label_available"].sum()),
        "verified_label_subject_count": int(
            table.loc[table["label_available"], "subject_id"].nunique()
        ),
        "supervised_trial_count": int(supervised.sum()),
        "supervised_subject_count": int(table.loc[supervised, "subject_id"].nunique()),
        "label_alignment_status_counts": table["label_alignment_status"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "de_storage_format_session_counts": sessions["de_storage_format"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "source_inconsistent_sessions": excluded,
        "constant_video_name_subject_count": constant_video_name_subjects,
        "self_rating_min": float(
            table[[f"self_{name}" for name in SCORE_NAMES.values()]].min().min()
        ),
        "self_rating_max": float(
            table[[f"self_{name}" for name in SCORE_NAMES.values()]].max().max()
        ),
        "alignment_policy": (
            "Behavior ratings are joined by the global stimulus code stored in "
            "video_name: 1-21 identify guided-imagery materials and 22-42 identify "
            "the corresponding video materials. This direction is confirmed by "
            "all 42 sub-54 raw EDF vid/ima triggers. Subjects whose exported code "
            "is constant or incomplete retain EEG rows but no supervised labels. "
            "Source-inconsistent DE sessions remain in the manifest and are "
            "excluded from feature modeling."
        ),
    }


def collect_emo_de_features(
    table: pd.DataFrame,
    dataset_root: Path,
) -> dict[str, np.ndarray]:
    """Collect verified DE features in manifest order."""

    verified = table.loc[
        table["feature_alignment_status"].eq("verified")
    ].copy()
    channel_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    trial_uids: list[str] = []

    for de_path, group in verified.groupby("de_path", sort=True):
        ordered = group.sort_values("de_trial_index")
        expected_indexes = np.arange(1, len(ordered) + 1)
        if not np.array_equal(ordered["de_trial_index"].to_numpy(), expected_indexes):
            raise EmoEEGError(f"Non-contiguous DE indexes for {de_path}")
        channel_features, global_features = summarize_de_trials(
            dataset_root / de_path, len(ordered)
        )
        channel_rows.extend(channel_features)
        global_rows.extend(global_features)
        trial_uids.extend(ordered["trial_uid"].astype(str))

    return {
        "trial_uids": np.asarray(trial_uids),
        "channel_features": np.asarray(channel_rows, dtype=np.float64),
        "global_features": np.asarray(global_rows, dtype=np.float64),
    }
