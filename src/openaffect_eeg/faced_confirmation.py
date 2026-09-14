"""Outcome-blind access and structural checks for the pinned FACED release."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://data.nemar.org/nm000112/v1.1.3"
SUBJECTS = tuple(f"sub-{index:03d}" for index in range(123))
LEGACY_CHANNELS = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
    "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
    "O1", "O2",
)
MODERN_CHANNELS_WITH_EOG = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
    "HEOR", "HEOL",
)
CHANNELS = MODERN_CHANNELS_WITH_EOG[:30]
REQUIRED_EVENT_COLUMNS = {
    "onset", "duration", "video_index", "Valence", "Arousal"
}
VIDEO_INDICES = {str(index) for index in range(1, 29)}


def requests_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        backoff_factor=0.5,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_manifest(
    session: requests.Session | None = None,
) -> tuple[list[dict[str, object]], str]:
    client = session or requests_session()
    response = client.get(f"{BASE_URL}/manifest.json", timeout=60)
    response.raise_for_status()
    return response.json(), hashlib.sha256(response.content).hexdigest()


def _ascii_number(block: bytes, start: int, stop: int, kind: type) -> int | float:
    text = block[start:stop].decode("ascii").strip()
    if not text:
        raise ValueError(f"empty numeric BDF header field at {start}:{stop}")
    return kind(text)


def parse_bdf_header(block: bytes, *, file_size: int) -> dict[str, object]:
    """Parse enough BDF header data to detect truncation without EEG samples."""
    if len(block) < 256:
        raise ValueError("BDF fixed header is incomplete")
    header_bytes = int(_ascii_number(block, 184, 192, int))
    records = int(_ascii_number(block, 236, 244, int))
    record_seconds = float(_ascii_number(block, 244, 252, float))
    signal_count = int(_ascii_number(block, 252, 256, int))
    if records < 0:
        raise ValueError("BDF record count is unknown")
    if signal_count <= 0 or header_bytes != 256 + 256 * signal_count:
        raise ValueError("inconsistent BDF header length")
    if len(block) < header_bytes:
        raise ValueError("BDF signal header is incomplete")

    offset = 256
    labels = [
        block[offset + 16 * index : offset + 16 * (index + 1)]
        .decode("ascii")
        .strip()
        for index in range(signal_count)
    ]
    offset += signal_count * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples_per_record = [
        int(
            block[offset + 8 * index : offset + 8 * (index + 1)]
            .decode("ascii")
            .strip()
        )
        for index in range(signal_count)
    ]
    expected_size = header_bytes + records * sum(samples_per_record) * 3
    if expected_size != file_size:
        raise ValueError(
            f"BDF size mismatch: header expects {expected_size}, manifest has {file_size}"
        )
    sampling_rates = [value / record_seconds for value in samples_per_record]
    return {
        "header_bytes": header_bytes,
        "records": records,
        "record_seconds": record_seconds,
        "recording_seconds": records * record_seconds,
        "signal_count": signal_count,
        "labels": labels,
        "samples_per_record": samples_per_record,
        "sampling_rates": sampling_rates,
        "expected_file_size": expected_size,
    }


def canonical_eeg_indices(labels: Iterable[str]) -> tuple[list[int], str]:
    """Return a fixed 30-scalp-channel view for either documented FACED cohort."""
    source = tuple(labels)
    if source == LEGACY_CHANNELS:
        legacy_to_modern = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
        renamed = [legacy_to_modern.get(label, label) for label in source]
        indices = [renamed.index(label) for label in CHANNELS]
        return indices, "legacy_10_20_with_a1_a2"
    if source == MODERN_CHANNELS_WITH_EOG:
        return list(range(30)), "modern_10_20_with_horizontal_eog"
    raise ValueError("unrecognised FACED cohort channel schema")


def _range_header(
    session: requests.Session, url: str, *, minimum_bytes: int = 16384
) -> bytes:
    response = session.get(
        url,
        headers={"Range": f"bytes=0-{minimum_bytes - 1}"},
        timeout=60,
    )
    response.raise_for_status()
    block = response.content
    if len(block) < 256:
        raise ValueError("remote range did not contain a BDF header")
    header_bytes = int(_ascii_number(block, 184, 192, int))
    if len(block) >= header_bytes:
        return block
    response = session.get(
        url,
        headers={"Range": f"bytes=0-{header_bytes - 1}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.content


def _nonmissing(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {"", "n/a", "na"}


def _event_structure(text: str, *, recording_seconds: float) -> dict[str, object]:
    rows = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff")), delimiter="\t"))
    columns = set(rows[0]) if rows else set()
    missing_columns = sorted(REQUIRED_EVENT_COLUMNS - columns)
    if missing_columns:
        raise ValueError("missing event columns: " + ",".join(missing_columns))
    video_rows = [row for row in rows if _nonmissing(row.get("video_index"))]
    counts = Counter(row["video_index"].strip() for row in video_rows)
    if set(counts) != VIDEO_INDICES or any(count != 1 for count in counts.values()):
        raise ValueError("video_index 1--28 must each occur exactly once")
    missing_rating_rows = sum(
        not (_nonmissing(row.get("Valence")) and _nonmissing(row.get("Arousal")))
        for row in video_rows
    )
    invalid_windows = 0
    for row in video_rows:
        try:
            onset = float(row["onset"])
            duration = float(row["duration"])
        except (TypeError, ValueError):
            invalid_windows += 1
            continue
        if duration < 30.0 or onset < 0.0 or onset + duration > recording_seconds + 1e-6:
            invalid_windows += 1
    return {
        "row_count": len(rows),
        "video_row_count": len(video_rows),
        "video_index_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "missing_valence_or_arousal_rows": missing_rating_rows,
        "invalid_30_second_windows": invalid_windows,
        "outcome_values_retained": False,
    }


def remote_structural_preflight(
    *, subjects: Iterable[str] = SUBJECTS, session: requests.Session | None = None
) -> dict[str, object]:
    """Check pinned remote headers/events without reading any EEG sample or rating."""
    client = session or requests_session()
    manifest, manifest_sha256 = fetch_manifest(client)
    by_path = {str(record["path"]): record for record in manifest}
    records: list[dict[str, object]] = []
    complete: list[str] = []
    for subject in subjects:
        prefix = f"{subject}/eeg/{subject}_task-watchingVideoClips"
        bdf_path = f"{prefix}_eeg.bdf"
        events_path = f"{prefix}_events.tsv"
        record: dict[str, object] = {"subject": subject, "status": "invalid"}
        try:
            bdf_entry = by_path[bdf_path]
            events_entry = by_path[events_path]
            block = _range_header(client, str(bdf_entry["bytes_url"]))
            header = parse_bdf_header(block, file_size=int(bdf_entry["size"]))
            selected_indices, source_schema = canonical_eeg_indices(header["labels"])
            rates = {round(float(value), 8) for value in header["sampling_rates"]}
            if rates not in ({250.0}, {1000.0}):
                raise ValueError(f"unexpected or mixed sampling rates: {sorted(rates)}")
            response = client.get(str(events_entry["bytes_url"]), timeout=60)
            response.raise_for_status()
            events = _event_structure(
                response.text,
                recording_seconds=float(header["recording_seconds"]),
            )
            if events["missing_valence_or_arousal_rows"]:
                raise ValueError("one or more trials has a missing target field")
            if events["invalid_30_second_windows"]:
                raise ValueError("one or more trials cannot supply the frozen 30-second window")
            record.update(
                {
                    "status": "structurally_complete",
                    "sampling_hz": next(iter(rates)),
                    "source_channel_schema": source_schema,
                    "selected_channel_indices": selected_indices,
                    "selected_channels": list(CHANNELS),
                    "recording_seconds": header["recording_seconds"],
                    "bdf_size": bdf_entry["size"],
                    "bdf_manifest_checksum": bdf_entry["checksum"],
                    "events_manifest_checksum": events_entry["checksum"],
                    "events": events,
                }
            )
            complete.append(subject)
        except (KeyError, requests.RequestException, TypeError, ValueError) as error:
            record["error"] = str(error)
        records.append(record)
    return {
        "schema_version": "1.0",
        "stage": "remote_structural_preflight",
        "dataset": "nm000112",
        "version": "v1.1.3",
        "base_url": BASE_URL,
        "manifest_sha256": manifest_sha256,
        "outcome_values_loaded": False,
        "eeg_samples_loaded": False,
        "complete_subjects": complete,
        "complete_subject_count": len(complete),
        "minimum_required_subjects": 25,
        "status": "pass" if len(complete) >= 25 else "fail_insufficient_support",
        "records": records,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
