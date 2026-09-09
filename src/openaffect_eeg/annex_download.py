"""Fast, resumable downloads for version-pinned OpenNeuro annex objects."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

_KEY_SIZE = re.compile(r"^[^-]+-s(?P<size>\d+)--")
_KEY_SHA256 = re.compile(r"^SHA256E?-s\d+--(?P<digest>[0-9a-f]{64})(?:\.|$)")
_REINJECT_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class AnnexObject:
    file: str
    key: str
    size_bytes: int


def _capture(command: Sequence[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def expected_size(key: str) -> int:
    match = _KEY_SIZE.match(key)
    if match is None:
        raise ValueError(f"Annex key does not encode its size: {key}")
    return int(match.group("size"))


def expected_sha256(key: str) -> str:
    match = _KEY_SHA256.match(key)
    if match is None:
        raise ValueError(f"Annex key does not encode a SHA-256 digest: {key}")
    return match.group("digest")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_objects(dataset: Path) -> list[AnnexObject]:
    output = _capture(
        [
            "git",
            "-C",
            str(dataset),
            "annex",
            "find",
            "--not",
            "--in",
            "here",
            "--format=${key}\t${file}\\n",
        ]
    )
    objects: list[AnnexObject] = []
    seen: set[str] = set()
    for line in output.splitlines():
        if not line:
            continue
        key, file = line.split("\t", maxsplit=1)
        if key in seen:
            continue
        seen.add(key)
        objects.append(AnnexObject(file, key, expected_size(key)))
    return objects


def filter_objects(
    objects: Sequence[AnnexObject],
    *,
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
) -> list[AnnexObject]:
    return [
        obj
        for obj in objects
        if (
            not include_patterns
            or any(fnmatchcase(obj.file, pattern) for pattern in include_patterns)
        )
        and not any(fnmatchcase(obj.file, pattern) for pattern in exclude_patterns)
    ]


def public_s3_url(dataset: Path, file: str) -> str:
    payload = json.loads(
        _capture(
            [
                "git",
                "-C",
                str(dataset),
                "annex",
                "whereis",
                "--json",
                "--",
                file,
            ]
        )
    )
    for location in payload.get("whereis", []):
        for url in location.get("urls", []):
            parsed = urlparse(url)
            if (
                parsed.scheme == "https"
                and parsed.hostname == "s3.amazonaws.com"
                and parsed.path.startswith("/openneuro.org/")
            ):
                return url
    raise RuntimeError(f"No public OpenNeuro S3 URL found for {file}")


def _object_present(dataset: Path, obj: AnnexObject) -> bool:
    output = _capture(
        [
            "git",
            "-C",
            str(dataset),
            "annex",
            "find",
            "--in",
            "here",
            f"--include={obj.file}",
            "--format=${key}",
        ]
    )
    return output == obj.key


def download_object(dataset: Path, temp_dir: Path, obj: AnnexObject) -> AnnexObject:
    if _object_present(dataset, obj):
        return obj

    url = public_s3_url(dataset, obj.file)
    temp_dir.mkdir(parents=True, exist_ok=True)
    partial = temp_dir / obj.key
    if partial.exists() and partial.stat().st_size > obj.size_bytes:
        raise RuntimeError(f"Oversized partial download: {partial}")

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
            url,
        ]
    )
    actual_size = partial.stat().st_size
    if actual_size != obj.size_bytes:
        raise RuntimeError(
            f"Size mismatch for {obj.file}: expected {obj.size_bytes}, got {actual_size}"
        )

    with _REINJECT_LOCK:
        if not _object_present(dataset, obj):
            _run(
                [
                    "git",
                    "-C",
                    str(dataset),
                    "annex",
                    "reinject",
                    str(partial),
                    obj.file,
                ]
            )
        if not _object_present(dataset, obj):
            raise RuntimeError(f"Reinjection did not register content for {obj.file}")
    return obj


def reinject_local_object(
    dataset: Path,
    source: Path,
    obj: AnnexObject,
) -> AnnexObject:
    """Verify a local transfer against its annex key and reinject it."""
    if not source.is_file():
        raise RuntimeError(f"Transferred annex object is unavailable: {source}")
    actual_size = source.stat().st_size
    if actual_size != obj.size_bytes:
        raise RuntimeError(
            f"Size mismatch for {obj.file}: expected {obj.size_bytes}, got {actual_size}"
        )
    actual_sha256 = file_sha256(source)
    required_sha256 = expected_sha256(obj.key)
    if actual_sha256 != required_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {obj.file}: expected {required_sha256}, "
            f"got {actual_sha256}"
        )

    with _REINJECT_LOCK:
        if not _object_present(dataset, obj):
            _run(
                [
                    "git",
                    "-C",
                    str(dataset),
                    "annex",
                    "reinject",
                    str(source),
                    obj.file,
                ]
            )
        if not _object_present(dataset, obj):
            raise RuntimeError(f"Reinjection did not register content for {obj.file}")
    return obj


def download_missing(
    dataset: Path,
    temp_dir: Path,
    *,
    jobs: int = 4,
    limit: int | None = None,
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> list[AnnexObject]:
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    objects = filter_objects(
        missing_objects(dataset),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    if limit is not None:
        objects = objects[:limit]
    if not objects:
        return []

    completed: list[AnnexObject] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(download_object, dataset, temp_dir, obj): obj
            for obj in objects
        }
        for future in as_completed(futures):
            obj = futures[future]
            try:
                completed.append(future.result())
                message = (
                    f"[{len(completed) + len(failures)}/{len(objects)}] "
                    f"verified {obj.file} ({obj.size_bytes / 1024**2:.1f} MiB)"
                )
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                failures.append(f"{obj.file}: {error}")
                message = (
                    f"[{len(completed) + len(failures)}/{len(objects)}] "
                    f"failed {obj.file}: {error}"
                )
            if progress is not None:
                progress(message)

    if failures:
        details = "\n".join(failures)
        raise RuntimeError(f"{len(failures)} annex downloads failed:\n{details}")
    return completed
