"""Safe, version-pinned OpenNeuro dataset preparation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from openaffect_eeg.manifest import DatasetRecord

RESERVE_GIB = 100.0


def select_records(
    records: list[DatasetRecord], dataset_ids: list[str] | None, tier: str
) -> list[DatasetRecord]:
    if dataset_ids:
        index = {record.dataset_id: record for record in records}
        unknown = sorted(set(dataset_ids) - index.keys())
        if unknown:
            raise ValueError(f"Datasets are not in the manifest: {', '.join(unknown)}")
        return [index[dataset_id] for dataset_id in dataset_ids]
    return [record for record in records if record.tier == tier]


def dataset_target(data_root: Path, dataset_id: str) -> Path:
    parent = (data_root / "raw" / "openneuro").resolve()
    target = (parent / dataset_id).resolve()
    if target.parent != parent:
        raise ValueError(f"Unsafe dataset target: {target}")
    return target


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("GIT_AUTHOR_NAME", "OpenAffect-EEG")
    environment.setdefault("GIT_AUTHOR_EMAIL", "openaffect-eeg@localhost")
    environment.setdefault("GIT_COMMITTER_NAME", "OpenAffect-EEG")
    environment.setdefault("GIT_COMMITTER_EMAIL", "openaffect-eeg@localhost")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def clone_snapshot(record: DatasetRecord, target: Path) -> None:
    """Clone one pinned snapshot and bootstrap its public annex metadata."""
    target.parent.mkdir(parents=True, exist_ok=True)
    repository = f"https://github.com/OpenNeuroDatasets/{record.dataset_id}.git"
    run(
        [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            record.snapshot,
            "--filter=blob:none",
            repository,
            str(target),
        ]
    )
    run(
        [
            "git",
            "-C",
            str(target),
            "-c",
            "http.version=HTTP/1.1",
            "fetch",
            "--depth=1",
            "origin",
            "refs/heads/git-annex:refs/remotes/origin/git-annex",
        ]
    )
    run(
        [
            "git",
            "-C",
            str(target),
            "branch",
            "-f",
            "git-annex",
            "origin/git-annex",
        ]
    )
    run(["git", "-C", str(target), "annex", "init"])


def prepare_metadata(record: DatasetRecord, target: Path) -> None:
    if not target.exists():
        clone_snapshot(record, target)
    if not (target / ".git").exists():
        raise RuntimeError(f"Existing target is not a Git dataset: {target}")
    configure_repository_identity(target)
    run(["git", "-C", str(target), "checkout", "--detach", record.snapshot])


def configure_repository_identity(target: Path) -> None:
    identity = {
        "user.name": "OpenAffect-EEG",
        "user.email": "openaffect-eeg@localhost",
    }
    for key, value in identity.items():
        current = subprocess.run(
            ["git", "-C", str(target), "config", "--local", "--get", key],
            check=False,
            capture_output=True,
            text=True,
        )
        if current.returncode != 0:
            run(["git", "-C", str(target), "config", "--local", key, value])


def get_annexed_data(record: DatasetRecord, target: Path, data_root: Path) -> None:
    free_gib = shutil.disk_usage(data_root).free / (1024**3)
    required_gib = record.size_gib + RESERVE_GIB
    if free_gib < required_gib:
        raise RuntimeError(
            f"Refusing {record.dataset_id}: {free_gib:.1f} GiB free, "
            f"but {required_gib:.1f} GiB is required including reserve"
        )
    run(["datalad", "-C", str(target), "get", "."])
