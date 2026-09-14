from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openaffect_eeg.confirmatory_execution import split_records


def test_split_manifest_is_loaded_in_explicit_order(tmp_path: Path) -> None:
    pd.DataFrame([
        {"split_index": 1, "assignment_seed": 12, "fold_rotation_seed": 8, "folder": "assignments/b"},
        {"split_index": 0, "assignment_seed": 11, "fold_rotation_seed": 7, "folder": "assignments/a"},
    ]).to_csv(tmp_path / "split_manifest.csv", index=False)
    records = split_records({}, tmp_path)
    assert [record["split_index"] for record in records] == [0, 1]
    assert records[0]["folder_path"] == tmp_path / "assignments/a"
    assert records[0]["fold_rotation_seed"] == 7


def test_split_manifest_rejects_duplicate_seed(tmp_path: Path) -> None:
    pd.DataFrame([
        {"split_index": 0, "assignment_seed": 11, "folder": "a"},
        {"split_index": 1, "assignment_seed": 11, "folder": "b"},
    ]).to_csv(tmp_path / "split_manifest.csv", index=False)
    with pytest.raises(ValueError, match="unique"):
        split_records({}, tmp_path)
