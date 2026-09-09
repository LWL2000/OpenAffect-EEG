from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.release import (
    ReleaseError,
    ReleasePolicy,
    build_public_release,
    load_release_policy,
)


def policy() -> ReleasePolicy:
    return ReleasePolicy(
        version="test",
        archive_root="OpenAffect-EEG-anonymous",
        include=("**/*",),
        exclude=(),
        forbidden_suffixes=(".set", ".pth", ".npz"),
        require_git_tracked=False,
    )


def test_project_public_release_policy_loads() -> None:
    project_root = Path(__file__).parents[1]

    loaded = load_release_policy(project_root / "configs" / "public_release.yaml")

    assert loaded.require_git_tracked is True
    assert "**/__pycache__/**" in loaded.exclude
    assert "paper/neurips2026/figures/qa/**" in loaded.exclude
    assert "paper/final_figures/qa/**" in loaded.exclude


def test_release_rejects_sensitive_text(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text(
        "Private data live at /" + "home" + "/researcher/project.\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseError, match="sensitive text"):
        build_public_release(source, policy(), tmp_path / "release.tar.gz")


def test_release_allows_anonymity_detector_source_without_a_real_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "detector.py").write_text(
        'WINDOWS_USER = r"[A-Z]:\\\\" + "Users" + r"\\\\[^\\\\s]+"\n',
        encoding="utf-8",
    )

    output = tmp_path / "release.tar.gz"
    build_public_release(source, policy(), output)

    assert output.is_file()


def test_release_rejects_raw_or_checkpoint_members(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Public benchmark.\n", encoding="utf-8")
    (source / "participant.set").write_bytes(b"raw EEG")

    with pytest.raises(ReleaseError, match="forbidden suffix"):
        build_public_release(source, policy(), tmp_path / "release.tar.gz")


def test_release_rejects_missing_required_members(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Public benchmark.\n", encoding="utf-8")
    required_policy = ReleasePolicy(
        version="test",
        archive_root="OpenAffect-EEG-anonymous",
        include=("README.md", "paper/main.tex"),
        exclude=(),
        forbidden_suffixes=(".set",),
        required_files=("README.md", "paper/main.tex"),
        require_git_tracked=False,
    )

    with pytest.raises(ReleaseError, match="required release member"):
        build_public_release(source, required_policy, tmp_path / "release.tar.gz")


def test_release_archive_is_deterministic_and_manifested(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "README.md").write_text("Public benchmark.\n", encoding="utf-8")
    script = source / "src" / "main.py"
    script.write_text("print('ready')\n", encoding="utf-8")
    script.chmod(0o755)

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_manifest = build_public_release(source, policy(), first)
    second_manifest = build_public_release(source, policy(), second)

    assert sha256_file(first) == sha256_file(second)
    assert first_manifest["archive_sha256"] == second_manifest["archive_sha256"]
    assert first_manifest["file_count"] == 2
    assert first_manifest["files"] == second_manifest["files"]


def test_release_cli_builds_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Public benchmark.\n", encoding="utf-8")
    config = tmp_path / "release.yaml"
    config.write_text(
        json.dumps(
            {
                "version": "test",
                "archive_root": "OpenAffect-EEG-anonymous",
                "include": ["README.md"],
                "exclude": [],
                "forbidden_suffixes": [".set", ".pth"],
                "require_git_tracked": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "public.tar.gz"
    script = Path(__file__).parents[1] / "scripts" / "build_public_release.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(source), str(config), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    manifest = json.loads(
        output.with_name(output.name + ".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["file_count"] == 1


def test_release_cli_reports_required_member_failure_without_traceback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Public benchmark.\n", encoding="utf-8")
    config = tmp_path / "release.yaml"
    config.write_text(
        json.dumps(
            {
                "version": "test",
                "archive_root": "OpenAffect-EEG-anonymous",
                "include": ["README.md", "paper/main.tex"],
                "exclude": [],
                "forbidden_suffixes": [".set", ".pth"],
                "required_files": ["README.md", "paper/main.tex"],
                "require_git_tracked": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "public.tar.gz"
    script = Path(__file__).parents[1] / "scripts" / "build_public_release.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(source), str(config), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Missing required release member" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()
