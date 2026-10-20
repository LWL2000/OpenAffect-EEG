"""Evidence integrity tests: no training, bootstrap or network is invoked."""
import json
import shutil
from pathlib import Path

import pytest

from openaffect_eeg.v14_evidence import build_assets, digest, load_archive, verify_assets

ROOT = Path(__file__).resolve().parents[1]


def test_rebuild_v14_from_public_aggregates_without_private_data(tmp_path):
    result = verify_assets(ROOT, tmp_path / "rebuilt")
    assert result["analysis_count"] == 10
    assert result["source_file_count"] == 20
    assert result["ridge_executed_cells"] == 2500
    assert result["experimental_computation_run"] is False


def copy_archive(tmp_path):
    target = tmp_path / "paper/generated/evidence_v14"
    target.parent.mkdir(parents=True)
    shutil.copytree(ROOT / "paper/generated/evidence_v14/archive", target / "archive")
    shutil.copyfile(ROOT / "paper/generated/evidence_v14/archive_manifest.json", target / "archive_manifest.json")
    return target


def test_modified_archived_cell_is_rejected(tmp_path):
    target = copy_archive(tmp_path)
    path = target / "archive/faced_labram_observed/cells.csv"
    path.write_bytes(path.read_bytes().replace(b"0.017026981219064756", b"0.117026981219064756"))
    with pytest.raises(ValueError, match="Archive hash mismatch"):
        load_archive(tmp_path)


def test_wrong_equivalence_claim_is_rejected_even_with_updated_digest(tmp_path):
    target = copy_archive(tmp_path)
    path = target / "archive/faced_eegnet_observed/summary.json"
    summary = json.loads(path.read_text())
    summary["primary"]["equivalence_primary"]["established"] = True
    path.write_text(json.dumps(summary))
    manifest_path = target / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for analysis in manifest["analyses"]:
        for record in analysis["files"]:
            if record["path"] == path.relative_to(tmp_path).as_posix():
                record["sha256"] = digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Equivalence decision mismatch"):
        load_archive(tmp_path)
