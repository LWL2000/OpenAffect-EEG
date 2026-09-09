import json
from pathlib import Path

import pytest

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.reviewer_acceptance import (
    ReviewerAcceptanceError,
    verify_release_manifest,
)


def write_manifest(root: Path, payload: bytes = b"ready\n") -> Path:
    source = root / "README.md"
    source.write_bytes(payload)
    manifest = {
        "version": "test",
        "files": [
            {
                "path": "README.md",
                "sha256": sha256_file(source),
                "size_bytes": len(payload),
                "mode": "644",
            }
        ],
    }
    path = root / "PUBLIC_RELEASE_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_release_manifest_verifies_all_recorded_files(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    result = verify_release_manifest(tmp_path)

    assert result["version"] == "test"
    assert result["file_count"] == 1
    assert result["manifest_sha256"] == sha256_file(manifest)


def test_release_manifest_rejects_modified_member(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ReviewerAcceptanceError, match="manifest mismatch"):
        verify_release_manifest(tmp_path)
