from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from openaffect_eeg.release import build_public_release, load_release_policy
from openaffect_eeg.release_snapshot import (
    ReleaseSnapshotError,
    freeze_anonymous_release,
)


def _write_policy(path: Path) -> Path:
    policy_path = path / "release.yaml"
    policy_path.write_text(
        json.dumps(
            {
                "version": "test",
                "archive_root": "OpenAffect-EEG-anonymous",
                "include": ["README.md", "paper/**"],
                "exclude": [],
                "forbidden_suffixes": [".set", ".npy"],
                "required_files": ["README.md", "paper/main.tex"],
                "require_git_tracked": True,
            }
        ),
        encoding="utf-8",
    )
    return policy_path


def _write_source(path: Path) -> Path:
    source = path / "source"
    (source / "paper").mkdir(parents=True)
    (source / "README.md").write_text("Anonymous benchmark.\n", encoding="utf-8")
    (source / "paper" / "main.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
    (source / "private-notes.txt").write_text("do not release\n", encoding="utf-8")
    return source


def test_freeze_anonymous_release_stages_only_allowlisted_members(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    policy_path = _write_policy(tmp_path)
    snapshot = freeze_anonymous_release(source, policy_path, tmp_path / "snapshot")

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=snapshot,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert tracked == ["README.md", "paper/main.tex"]
    assert not (snapshot / "private-notes.txt").exists()

    policy = load_release_policy(policy_path)
    manifest = build_public_release(snapshot, policy, tmp_path / "release.tar.gz")
    assert manifest["file_count"] == 2


def test_freeze_anonymous_release_refuses_nonempty_destination(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    policy_path = _write_policy(tmp_path)
    destination = tmp_path / "snapshot"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(ReleaseSnapshotError, match="non-empty"):
        freeze_anonymous_release(source, policy_path, destination)


def test_freeze_anonymous_release_redacts_official_style_comment_email(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    policy_path = _write_policy(tmp_path)
    style = source / "paper" / "neurips2026" / "neurips_2026.sty"
    style.parent.mkdir()
    maintainer_email = "maintainer" + "@example.org"
    style.write_text(
        "% Template maintainer (" + maintainer_email + ")\n"
        "\\ProvidesPackage{neurips_2026}\n",
        encoding="utf-8",
    )

    snapshot = freeze_anonymous_release(source, policy_path, tmp_path / "snapshot")

    copied_style = (snapshot / "paper" / "neurips2026" / "neurips_2026.sty").read_text(
        encoding="utf-8"
    )
    assert maintainer_email not in copied_style
    assert "\\ProvidesPackage{neurips_2026}" in copied_style
    manifest = build_public_release(
        snapshot, load_release_policy(policy_path), tmp_path / "release.tar.gz"
    )
    assert manifest["file_count"] == 3


def test_freeze_anonymous_release_excludes_generated_egg_info(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    package_info = source / "src" / "openaffect_eeg.egg-info" / "PKG-INFO"
    package_info.parent.mkdir(parents=True)
    package_info.write_text(
        "Installed from /" + "home" + "/researcher/source\n", encoding="utf-8"
    )
    cache = source / "src" / "__pycache__" / "audit.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_text("Cached /" + "home" + "/researcher/source\n", encoding="utf-8")
    policy_path = tmp_path / "release.yaml"
    policy_path.write_text(
        json.dumps(
            {
                "version": "test",
                "archive_root": "OpenAffect-EEG-anonymous",
                "include": ["README.md", "src/**"],
                "exclude": ["src/**.egg-info/**", "**/__pycache__/**"],
                "forbidden_suffixes": [".set", ".npy"],
                "required_files": ["README.md"],
                "require_git_tracked": True,
            }
        ),
        encoding="utf-8",
    )

    snapshot = freeze_anonymous_release(source, policy_path, tmp_path / "snapshot")

    assert not (snapshot / "src" / "openaffect_eeg.egg-info").exists()
    assert not (snapshot / "src" / "__pycache__").exists()
    manifest = build_public_release(
        snapshot, load_release_policy(policy_path), tmp_path / "release.tar.gz"
    )
    assert manifest["file_count"] == 1
