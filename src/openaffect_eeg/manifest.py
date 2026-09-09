"""Version-pinned OpenNeuro dataset manifest utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DATASET_ID = re.compile(r"^ds\d{6}$")
SNAPSHOT = re.compile(r"^\d+\.\d+\.\d+$")
ALLOWED_TIERS = {"core", "external"}


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    name: str
    snapshot: str
    size_bytes: int
    tier: str
    role: str
    subjects: int

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024**3)


def load_manifest(path: str | Path) -> list[DatasetRecord]:
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema")

    records: list[DatasetRecord] = []
    seen: set[str] = set()
    for item in payload.get("datasets", []):
        dataset_id = str(item["id"])
        snapshot = str(item["snapshot"])
        tier = str(item["tier"])
        if not DATASET_ID.fullmatch(dataset_id):
            raise ValueError(f"Invalid OpenNeuro dataset ID: {dataset_id}")
        if dataset_id in seen:
            raise ValueError(f"Duplicate OpenNeuro dataset ID: {dataset_id}")
        if not SNAPSHOT.fullmatch(snapshot):
            raise ValueError(f"Invalid snapshot for {dataset_id}: {snapshot}")
        if tier not in ALLOWED_TIERS:
            raise ValueError(f"Invalid tier for {dataset_id}: {tier}")
        if int(item["size_bytes"]) <= 0 or int(item["subjects"]) <= 0:
            raise ValueError(f"Non-positive size or subject count for {dataset_id}")

        seen.add(dataset_id)
        records.append(
            DatasetRecord(
                dataset_id=dataset_id,
                name=str(item["name"]),
                snapshot=snapshot,
                size_bytes=int(item["size_bytes"]),
                tier=tier,
                role=str(item["role"]),
                subjects=int(item["subjects"]),
            )
        )

    if not records:
        raise ValueError("Dataset manifest is empty")
    return records


def total_size_gib(records: list[DatasetRecord]) -> float:
    return sum(record.size_bytes for record in records) / (1024**3)
