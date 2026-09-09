"""Deterministic, allowlisted, and anonymity-scanned public release archives."""

from __future__ import annotations

import fnmatch
import gzip
import io
import json
import re
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from openaffect_eeg.artifacts import sha256_file


class ReleaseError(ValueError):
    """Raised when a candidate public release violates its policy."""


@dataclass(frozen=True)
class ReleasePolicy:
    version: str
    archive_root: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    forbidden_suffixes: tuple[str, ...]
    require_git_tracked: bool = True
    required_files: tuple[str, ...] = ()


def load_release_policy(path: Path) -> ReleasePolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ReleaseError("Release policy must contain a mapping")
    version = str(raw.get("version", "")).strip()
    archive_root = str(raw.get("archive_root", "")).strip().strip("/")
    include = raw.get("include")
    exclude = raw.get("exclude", [])
    forbidden_suffixes = raw.get("forbidden_suffixes")
    required_files = raw.get("required_files", [])
    if not version or not archive_root:
        raise ReleaseError("Release version and archive_root are required")
    if not isinstance(include, list) or not include:
        raise ReleaseError("Release include must be a non-empty list")
    if not isinstance(exclude, list):
        raise ReleaseError("Release exclude must be a list")
    if not isinstance(forbidden_suffixes, list) or not forbidden_suffixes:
        raise ReleaseError("Release forbidden_suffixes must be a non-empty list")
    if not isinstance(required_files, list):
        raise ReleaseError("Release required_files must be a list")
    return ReleasePolicy(
        version=version,
        archive_root=archive_root,
        include=tuple(str(item) for item in include),
        exclude=tuple(str(item) for item in exclude),
        forbidden_suffixes=tuple(str(item) for item in forbidden_suffixes),
        require_git_tracked=bool(raw.get("require_git_tracked", True)),
        required_files=tuple(str(item) for item in required_files),
    )


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:])
    )


def _tracked_files(source: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=source,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ReleaseError("Release policy requires a Git working tree")
    return [Path(item.decode("utf-8")) for item in completed.stdout.split(b"\0") if item]


def _candidate_files(source: Path, policy: ReleasePolicy) -> list[Path]:
    candidates = (
        _tracked_files(source)
        if policy.require_git_tracked
        else [path.relative_to(source) for path in source.rglob("*") if path.is_file()]
    )
    selected: list[Path] = []
    for relative in candidates:
        name = relative.as_posix()
        if not any(_matches(name, pattern) for pattern in policy.include):
            continue
        if any(_matches(name, pattern) for pattern in policy.exclude):
            continue
        selected.append(relative)
    if not selected:
        raise ReleaseError("Release allowlist selected no files")
    return sorted(set(selected), key=lambda item: item.as_posix())


def _sensitive_patterns(extra_forbidden: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    expressions = [
        "/" + "home" + r"/[^/\s]+",
        "/" + "opt" + r"/(?:[^/\s]+/)*[^/\s]+",
        r"(?i)[A-Z]:\\Users\\[^\\\s]+",
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        *(re.escape(value) for value in extra_forbidden if value),
    ]
    return tuple(re.compile(expression) for expression in expressions)


def _scan_file(path: Path, relative: Path, patterns: tuple[re.Pattern, ...]) -> None:
    payload = path.read_bytes()
    if b"\0" in payload[:8192]:
        return
    text = payload.decode("utf-8", errors="replace")
    for pattern in patterns:
        if pattern.search(text):
            raise ReleaseError(f"Found sensitive text in {relative.as_posix()}")


def _validate_files(
    source: Path,
    files: list[Path],
    policy: ReleasePolicy,
    extra_forbidden: tuple[str, ...],
) -> None:
    root = source.resolve()
    suffixes = {suffix.lower() for suffix in policy.forbidden_suffixes}
    patterns = _sensitive_patterns(extra_forbidden)
    for relative in files:
        path = source / relative
        if path.is_symlink():
            raise ReleaseError(f"Release may not contain symlinks: {relative.as_posix()}")
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ReleaseError(
                f"Release member escapes source root: {relative.as_posix()}"
            ) from error
        if path.suffix.lower() in suffixes:
            raise ReleaseError(
                f"Release member has forbidden suffix: {relative.as_posix()}"
            )
        _scan_file(path, relative, patterns)


def _validate_required_files(files: list[Path], policy: ReleasePolicy) -> None:
    selected = {path.as_posix() for path in files}
    missing = sorted(set(policy.required_files).difference(selected))
    if missing:
        raise ReleaseError(
            "Missing required release member(s): " + ", ".join(missing)
        )


def _file_manifest(source: Path, files: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in files:
        path = source / relative
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "mode": "755" if path.stat().st_mode & 0o111 else "644",
            }
        )
    return records


def _tar_info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_public_release(
    source: Path,
    policy: ReleasePolicy,
    output: Path,
    *,
    extra_forbidden: tuple[str, ...] = (),
) -> dict[str, object]:
    source = source.resolve()
    files = _candidate_files(source, policy)
    _validate_required_files(files, policy)
    _validate_files(source, files, policy, extra_forbidden)
    file_records = _file_manifest(source, files)
    internal_manifest = {
        "version": policy.version,
        "archive_root": policy.archive_root,
        "file_count": len(file_records),
        "files": file_records,
    }
    manifest_bytes = (
        json.dumps(internal_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_handle, \
        gzip.GzipFile(
            fileobj=raw_handle, mode="wb", filename="", mtime=0
        ) as gzip_handle, \
        tarfile.open(
            fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT
        ) as tar:
        for record in file_records:
            relative = Path(str(record["path"]))
            payload = (source / relative).read_bytes()
            name = f"{policy.archive_root}/{relative.as_posix()}"
            tar.addfile(
                _tar_info(name, len(payload), int(str(record["mode"]), 8)),
                io.BytesIO(payload),
            )
        manifest_name = f"{policy.archive_root}/PUBLIC_RELEASE_MANIFEST.json"
        tar.addfile(
            _tar_info(manifest_name, len(manifest_bytes), 0o644),
            io.BytesIO(manifest_bytes),
        )

    result = {
        **internal_manifest,
        "archive_sha256": sha256_file(output),
        "archive_size_bytes": output.stat().st_size,
    }
    sidecar = output.with_name(output.name + ".manifest.json")
    sidecar.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
