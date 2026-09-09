"""Dataset-scope accounting derived from frozen trial tables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from openaffect_eeg.artifacts import sha256_file


class BenchmarkScopeError(ValueError):
    """Raised when a scope specification cannot be resolved."""


def _labeled_mask(table: pd.DataFrame, specification: Mapping[str, object]) -> pd.Series:
    availability_column = specification.get("label_available_column")
    if availability_column:
        column = str(availability_column)
        if column not in table:
            raise BenchmarkScopeError(f"Missing label availability column: {column}")
        values = table[column]
        if values.dtype == bool:
            return values.fillna(False)
        return values.astype(str).str.lower().isin({"true", "1", "yes"})

    raw_columns = specification.get("label_columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise BenchmarkScopeError(
            "Each dataset requires label_available_column or label_columns"
        )
    columns = [str(column) for column in raw_columns]
    missing = [column for column in columns if column not in table]
    if missing:
        raise BenchmarkScopeError(f"Missing label columns: {missing}")
    return table[columns].notna().all(axis=1)


def build_scope_payload(
    config: Mapping[str, object], data_root: Path
) -> dict[str, object]:
    """Compute deterministic benchmark counts and source hashes."""
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise BenchmarkScopeError("Scope configuration requires datasets")

    rows: list[dict[str, object]] = []
    for dataset_id, raw_specification in sorted(datasets.items()):
        if not isinstance(raw_specification, Mapping):
            raise BenchmarkScopeError(f"Invalid dataset specification: {dataset_id}")
        path = data_root / str(raw_specification["path"])
        table = pd.read_csv(path, sep="\t")
        subject_column = str(raw_specification.get("subject_column", "subject_id"))
        if subject_column not in table:
            raise BenchmarkScopeError(f"Missing subject column: {subject_column}")
        stimulus_column = raw_specification.get("stimulus_column")
        if stimulus_column and str(stimulus_column) not in table:
            raise BenchmarkScopeError(
                f"Missing stimulus column: {stimulus_column}"
            )
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "name": str(raw_specification.get("name", dataset_id)),
                "tier": str(raw_specification.get("tier", "core")),
                "participant_count": int(table[subject_column].nunique()),
                "trial_count": len(table),
                "stimulus_count": (
                    int(table[str(stimulus_column)].nunique())
                    if stimulus_column
                    else None
                ),
                "labeled_trial_count": int(
                    _labeled_mask(table, raw_specification).sum()
                ),
                "stimulus_identity_status": str(
                    raw_specification.get("stimulus_identity_status", "available")
                ),
                "source_path": str(raw_specification["path"]),
                "source_sha256": sha256_file(path),
            }
        )

    def summarize(selected: list[dict[str, object]]) -> dict[str, object]:
        stimulus_counts = [row["stimulus_count"] for row in selected]
        return {
            "dataset_count": len(selected),
            "participant_count": sum(int(row["participant_count"]) for row in selected),
            "trial_count": sum(int(row["trial_count"]) for row in selected),
            "stimulus_count": (
                sum(int(value) for value in stimulus_counts)
                if all(value is not None for value in stimulus_counts)
                else None
            ),
            "labeled_trial_count": sum(
                int(row["labeled_trial_count"]) for row in selected
            ),
        }

    core = [row for row in rows if row["tier"] == "core"]
    return {
        "version": str(config.get("version", "")),
        "datasets": rows,
        "groups": {"all": summarize(rows), "core": summarize(core)},
    }
