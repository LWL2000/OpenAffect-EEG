#!/usr/bin/env python3
"""Download selected files from pinned FACED v1.1.3 with resume and checksums."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from openaffect_eeg.faced_confirmation import (
    BASE_URL,
    SUBJECTS,
    fetch_manifest,
    git_blob_sha1,
    requests_session,
    sha256_file,
)

ROOT_METADATA = {"dataset_description.json", "participants.tsv", "README.md"}


def _wanted(
    path: str, *, subjects: set[str], include_events: bool, include_bdf: bool
) -> bool:
    if path in ROOT_METADATA:
        return True
    subject = path.split("/", 1)[0]
    if subject not in subjects:
        return False
    if path.endswith(("_channels.tsv", "_eeg.json", "_events.json")):
        return True
    if include_events and path.endswith("_events.tsv"):
        return True
    return include_bdf and path.endswith("_eeg.bdf")


def _verify(path: Path, entry: dict[str, object]) -> bool:
    if not path.is_file() or path.stat().st_size != int(entry["size"]):
        return False
    algorithm = str(entry["checksum_algorithm"]).lower()
    expected = str(entry["checksum"]).lower()
    if algorithm == "sha256":
        return sha256_file(path).lower() == expected
    if algorithm == "git":
        return git_blob_sha1(path).lower() == expected
    raise ValueError(f"Unsupported checksum algorithm: {algorithm}")


def _download(
    client: requests.Session, entry: dict[str, object], destination: Path
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verify(destination, entry):
        return "verified_existing"
    expected_size = int(entry["size"])
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.get(
        str(entry["bytes_url"]), headers=headers, stream=True, timeout=120
    ) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            partial.unlink(missing_ok=True)
            return _download(client, entry, destination)
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                if block:
                    handle.write(block)
    if partial.stat().st_size != expected_size:
        raise ValueError(
            f"Incomplete download for {entry['path']}: "
            f"{partial.stat().st_size} of {expected_size} bytes"
        )
    partial.replace(destination)
    if not _verify(destination, entry):
        raise ValueError(f"Checksum failed for {entry['path']}")
    return "downloaded_verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--subjects",
        default=",".join(SUBJECTS),
        help="Comma-separated BIDS subject IDs",
    )
    parser.add_argument(
        "--include-events",
        action="store_true",
        help="Include trial event tables containing the sealed ratings",
    )
    parser.add_argument(
        "--include-bdf", action="store_true", help="Include the large EEG BDF files"
    )
    args = parser.parse_args()
    subjects = {value.strip() for value in args.subjects.split(",") if value.strip()}
    unknown = subjects - set(SUBJECTS)
    if unknown:
        raise ValueError(f"Unknown FACED subject IDs: {sorted(unknown)}")
    client = requests_session()
    manifest, manifest_sha256 = fetch_manifest(client)
    selected = [
        entry
        for entry in manifest
        if _wanted(
            str(entry["path"]),
            subjects=subjects,
            include_events=args.include_events,
            include_bdf=args.include_bdf,
        )
    ]
    records = []
    for entry in sorted(selected, key=lambda item: str(item["path"])):
        destination = args.output_root / str(entry["path"])
        state = _download(client, entry, destination)
        records.append(
            {
                "path": entry["path"],
                "size": entry["size"],
                "checksum_algorithm": entry["checksum_algorithm"],
                "checksum": entry["checksum"],
                "state": state,
            }
        )
        print(json.dumps(records[-1], sort_keys=True))
    receipt = {
        "dataset": "nm000112",
        "version": "v1.1.3",
        "base_url": BASE_URL,
        "manifest_sha256": manifest_sha256,
        "subjects": sorted(subjects),
        "include_events": args.include_events,
        "include_bdf": args.include_bdf,
        "files": records,
    }
    receipt_path = args.output_root / "faced_v1_1_3_download_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
