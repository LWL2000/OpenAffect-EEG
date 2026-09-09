"""Create a Git-indexed anonymous snapshot without mutating a source checkout."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from openaffect_eeg.release import (
    ReleaseError,
    ReleasePolicy,
    _candidate_files,
    _validate_files,
    _validate_required_files,
    load_release_policy,
)


class ReleaseSnapshotError(ValueError):
    """Raised when a release snapshot cannot be safely materialized."""


_OFFICIAL_STYLE = Path("paper/neurips2026/neurips_2026.sty")
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def _run_git(arguments: list[str], destination: Path) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=destination,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReleaseSnapshotError(f"Git snapshot setup failed: {detail}")


def _snapshot_members(source: Path, policy: ReleasePolicy) -> list[Path]:
    selection_policy = replace(policy, require_git_tracked=False)
    try:
        members = _candidate_files(source, selection_policy)
        _validate_required_files(members, policy)
    except ReleaseError as error:
        raise ReleaseSnapshotError(str(error)) from error
    return members


def _copy_snapshot_member(source: Path, destination: Path, relative: Path) -> None:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative != _OFFICIAL_STYLE:
        shutil.copy2(source / relative, target)
        return

    lines = (source / relative).read_text(encoding="utf-8").splitlines(keepends=True)
    sanitized = [
        _EMAIL.sub("[template-maintainer-email-redacted]", line)
        if line.lstrip().startswith("%")
        else line
        for line in lines
    ]
    target.write_text("".join(sanitized), encoding="utf-8")


def freeze_anonymous_release(
    source: Path,
    policy_path: Path,
    destination: Path,
) -> Path:
    """Copy selected current files to a clean Git-indexed release snapshot.

    The caller can then run the ordinary strict release builder against the
    snapshot. This captures selected uncommitted manuscript assets without
    staging, committing, or otherwise changing the development checkout.
    """

    source = source.resolve()
    destination = destination.resolve()
    policy_path = policy_path.resolve()
    if not source.is_dir():
        raise ReleaseSnapshotError(f"Source directory does not exist: {source}")
    if not policy_path.is_file():
        raise ReleaseSnapshotError(f"Release policy does not exist: {policy_path}")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise ReleaseSnapshotError("Snapshot destination must be outside the source tree")
    if destination.exists() and any(destination.iterdir()):
        raise ReleaseSnapshotError("Snapshot destination must be absent or empty, not non-empty")

    policy = load_release_policy(policy_path)
    members = _snapshot_members(source, policy)
    destination.mkdir(parents=True, exist_ok=True)
    for relative in members:
        _copy_snapshot_member(source, destination, relative)

    try:
        _validate_files(destination, members, policy, ())
    except ReleaseError as error:
        raise ReleaseSnapshotError(str(error)) from error

    _run_git(["init", "--quiet"], destination)
    _run_git(["add", "--all"], destination)
    return destination
