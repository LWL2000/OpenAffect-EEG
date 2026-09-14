"""Shared execution contracts for the locked v14 confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_records(config: dict, base: Path) -> list[dict]:
    """Load explicit crossed splits, with a legacy v9 fallback."""
    manifest_path = base / "split_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        required = {"split_index", "assignment_seed", "folder"}
        if required - set(manifest):
            raise ValueError("Malformed split manifest")
        if manifest.split_index.duplicated().any() or manifest.assignment_seed.duplicated().any():
            raise ValueError("Split indices and assignment seeds must be unique")
        records = []
        for _, row in manifest.sort_values("split_index").iterrows():
            record = row.to_dict()
            record.update(
                split_index=int(row.split_index),
                assignment_seed=int(row.assignment_seed),
                folder_path=base / str(row.folder),
            )
            if "fold_rotation_seed" in record:
                record["fold_rotation_seed"] = int(record["fold_rotation_seed"])
            records.append(record)
        return records
    return [
        {
            "split_index": fold,
            "assignment_seed": int(config["design_seed"]) + fold,
            "folder_path": base / "assignments" / f"seed-{int(config['design_seed']) + fold}",
        }
        for fold in range(int(config["fold_count"]))
    ]


def feature_table(path: Path, key: str) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        uid_key = "trial_uid" if "trial_uid" in data else "trial_uids"
        uids = data[uid_key].astype(str)
        values = np.asarray(data[key], dtype=float).reshape(len(uids), -1)
    if len(uids) != len(set(uids)) or not np.isfinite(values).all():
        raise ValueError("Malformed feature archive")
    result = pd.DataFrame(
        values, columns=[f"feature_{index:04d}" for index in range(values.shape[1])]
    )
    result.insert(0, "trial_uid", uids)
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
