"""Version-pinned OpenNeuro GraphQL manifests and direct downloads."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

GRAPHQL_ENDPOINT = "https://openneuro.org/crn/graphql"
_DATASET_ID = re.compile(r"^ds\d{6}$")


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    file_id: str
    path: str
    size_bytes: int
    annexed: bool
    urls: tuple[str, ...]


def prefix_tree_files(
    root_directory: str, payload: Sequence[dict[str, Any]]
) -> list[SnapshotFile]:
    files: list[SnapshotFile] = []
    for item in payload:
        if item.get("directory"):
            continue
        relative = PurePosixPath(str(item["filename"]))
        path = str(PurePosixPath(root_directory) / relative)
        files.append(
            SnapshotFile(
                file_id=str(item["id"]),
                path=path,
                size_bytes=int(item.get("size") or 0),
                annexed=bool(item.get("annexed")),
                urls=tuple(item.get("urls") or ()),
            )
        )
    return files


def safe_target(root: Path, snapshot_path: str) -> Path:
    relative = PurePosixPath(snapshot_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe snapshot path: {snapshot_path}")
    root = root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Unsafe snapshot path: {snapshot_path}")
    return target


def select_files(
    files: Sequence[SnapshotFile],
    *,
    include_patterns: Sequence[str] = (),
    annexed: bool | None = None,
) -> list[SnapshotFile]:
    return [
        item
        for item in files
        if (annexed is None or item.annexed is annexed)
        and (
            not include_patterns
            or any(fnmatchcase(item.path, pattern) for pattern in include_patterns)
        )
    ]


def trusted_download_url(dataset_id: str, urls: Sequence[str]) -> str:
    required_prefix = f"/openneuro.org/{dataset_id}/"
    for url in urls:
        parsed = urlparse(url)
        if (
            parsed.scheme == "https"
            and parsed.hostname == "s3.amazonaws.com"
            and parsed.path.startswith(required_prefix)
            and parse_qs(parsed.query).get("versionId")
        ):
            return url
    raise RuntimeError("No versioned OpenNeuro S3 URL found")


def _graphql(query: str, variables: dict[str, str]) -> dict[str, Any]:
    response = requests.post(
        GRAPHQL_ENDPOINT,
        json={"query": query, "variables": variables},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"OpenNeuro GraphQL error: {payload['errors']}")
    return payload["data"]


def fetch_snapshot_files(
    dataset_id: str,
    snapshot: str,
    *,
    directory_prefixes: Sequence[str] = ("sub-",),
) -> list[SnapshotFile]:
    if not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError(f"Invalid OpenNeuro dataset ID: {dataset_id}")
    fields = "id filename size directory annexed urls"
    root_query = (
        "query($id:ID!,$tag:String!){"
        f"snapshot(datasetId:$id,tag:$tag){{files{{{fields}}}}}"
        "}"
    )
    root = _graphql(root_query, {"id": dataset_id, "tag": snapshot})["snapshot"]
    root_items = root["files"]
    files = prefix_tree_files("", root_items)
    directories = [
        item
        for item in root_items
        if item.get("directory")
        and any(
            str(item["filename"]).startswith(prefix)
            for prefix in directory_prefixes
        )
    ]
    if not directories:
        return files

    aliases = {
        f"tree_{index}": str(item["filename"])
        for index, item in enumerate(directories)
    }
    tree_fields = " ".join(
        f'{alias}:files(tree:"{item["id"]}",recursive:true){{{fields}}}'
        for alias, item in zip(aliases, directories, strict=True)
    )
    tree_query = (
        "query($id:ID!,$tag:String!){"
        f"snapshot(datasetId:$id,tag:$tag){{{tree_fields}}}"
        "}"
    )
    trees = _graphql(tree_query, {"id": dataset_id, "tag": snapshot})[
        "snapshot"
    ]
    for alias, directory in aliases.items():
        files.extend(prefix_tree_files(directory, trees[alias]))
    return sorted(files, key=lambda item: item.path)


def write_manifest(
    path: Path,
    *,
    dataset_id: str,
    snapshot: str,
    files: Sequence[SnapshotFile],
) -> None:
    paths = [item.path for item in files]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate snapshot paths in OpenNeuro manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "snapshot": snapshot,
        "file_count": len(files),
        "size_bytes": sum(item.size_bytes for item in files),
        "files": [asdict(item) for item in files],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> tuple[str, str, list[SnapshotFile]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = [
        SnapshotFile(
            file_id=item["file_id"],
            path=item["path"],
            size_bytes=int(item["size_bytes"]),
            annexed=bool(item["annexed"]),
            urls=tuple(item["urls"]),
        )
        for item in payload["files"]
    ]
    return payload["dataset_id"], payload["snapshot"], files


def _run(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def download_file(dataset_id: str, root: Path, item: SnapshotFile) -> Path:
    target = safe_target(root, item.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != item.size_bytes:
            raise RuntimeError(f"Existing file has the wrong size: {target}")
        return target

    partial = target.with_name(f"{target.name}.part")
    if partial.exists() and partial.stat().st_size > item.size_bytes:
        raise RuntimeError(f"Partial file is oversized: {partial}")
    _run(
        [
            "curl",
            "-4",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "8",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--speed-time",
            "120",
            "--speed-limit",
            "1024",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            trusted_download_url(dataset_id, item.urls),
        ]
    )
    actual_size = partial.stat().st_size
    if actual_size != item.size_bytes:
        raise RuntimeError(
            f"Size mismatch for {item.path}: expected {item.size_bytes}, "
            f"got {actual_size}"
        )
    partial.replace(target)
    return target


def download_files(
    dataset_id: str,
    root: Path,
    files: Sequence[SnapshotFile],
    *,
    jobs: int = 4,
) -> list[Path]:
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    completed: list[Path] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(download_file, dataset_id, root, item): item
            for item in files
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                completed.append(future.result())
                print(f"verified {item.path}", flush=True)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                failures.append(f"{item.path}: {error}")
    if failures:
        raise RuntimeError("Snapshot downloads failed:\n" + "\n".join(failures))
    return sorted(completed)
