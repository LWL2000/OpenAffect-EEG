"""Auditable frequency-band and scalp-region feature views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml


class AblationError(ValueError):
    pass


def ensure_audit_entity_uids(table: pd.DataFrame) -> pd.DataFrame:
    """Add stable entity IDs, marking unavailable stimulus identity explicitly."""
    required = {"dataset_id", "subject_id"}
    missing = required - set(table.columns)
    if missing:
        raise AblationError(f"Trial table is missing identity columns: {sorted(missing)}")
    result = table.copy()
    dataset = result["dataset_id"].astype(str)
    subject_default = dataset + ":" + result["subject_id"].astype(str)
    if "subject_uid" not in result:
        result["subject_uid"] = subject_default
    else:
        absent = result["subject_uid"].isna() | result["subject_uid"].astype(str).eq("")
        result.loc[absent, "subject_uid"] = subject_default.loc[absent]

    stimulus_default = dataset + ":unavailable"
    if "stimulus_id" in result:
        stimulus_available = (
            result["stimulus_id"].notna()
            & result["stimulus_id"].astype(str).ne("")
        )
        stimulus_default.loc[stimulus_available] = (
            dataset.loc[stimulus_available]
            + ":"
            + result.loc[stimulus_available, "stimulus_id"].astype(str)
        )
    if "stimulus_uid" not in result:
        result["stimulus_uid"] = stimulus_default
    else:
        absent = result["stimulus_uid"].isna() | result["stimulus_uid"].astype(str).eq("")
        result.loc[absent, "stimulus_uid"] = stimulus_default.loc[absent]
    return result


@dataclass(frozen=True)
class AblationView:
    axis: Literal["full", "band", "region"]
    name: str
    mode: Literal["all", "only", "leave_out"]

    def __post_init__(self) -> None:
        if self.axis == "full":
            if self.name != "full" or self.mode != "all":
                raise AblationError("Full view must use name=full and mode=all")
        elif self.axis in {"band", "region"}:
            if not self.name or self.mode not in {"only", "leave_out"}:
                raise AblationError(
                    "Band and region views require a name and only/leave_out mode"
                )
        else:
            raise AblationError(f"Unknown ablation axis: {self.axis}")

    @property
    def identifier(self) -> str:
        if self.axis == "full":
            return "full"
        return f"{self.axis}_{self.name}_{self.mode}"


def _unique_names(values: Sequence[str], label: str) -> list[str]:
    names = [str(value) for value in values]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise AblationError(f"{label} must be unique non-empty names")
    return names


def _validate_regions(
    channel_names: Sequence[str],
    regions: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    channels = _unique_names(channel_names, "Channel names")
    if not isinstance(regions, Mapping) or not regions:
        raise AblationError("Region mapping must be a non-empty mapping")
    normalized: dict[str, list[str]] = {}
    assigned: list[str] = []
    for region, members in regions.items():
        if not isinstance(region, str) or not region:
            raise AblationError("Region names must be non-empty strings")
        normalized[region] = _unique_names(members, f"Region {region} channels")
        assigned.extend(normalized[region])
    if len(assigned) != len(set(assigned)) or set(assigned) != set(channels):
        missing = sorted(set(channels) - set(assigned))
        extra = sorted(set(assigned) - set(channels))
        raise AblationError(
            "Region mapping must partition the archive channels exactly; "
            f"missing={missing}, extra={extra}"
        )
    return normalized


def feature_view_indices(
    channel_names: Sequence[str],
    band_names: Sequence[str],
    view: AblationView,
    regions: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Return flattened channel-major feature indices for one ablation view."""
    channels = _unique_names(channel_names, "Channel names")
    bands = _unique_names(band_names, "Band names")
    selected = np.ones((len(channels), len(bands)), dtype=bool)

    if view.axis == "band":
        if view.name not in bands:
            raise AblationError(f"Unknown band view: {view.name}")
        target = np.asarray([band == view.name for band in bands])
        selected &= target[None, :] if view.mode == "only" else ~target[None, :]
    elif view.axis == "region":
        mapping = _validate_regions(channels, regions)
        if view.name not in mapping:
            raise AblationError(f"Unknown region view: {view.name}")
        target = np.asarray([channel in mapping[view.name] for channel in channels])
        selected &= target[:, None] if view.mode == "only" else ~target[:, None]

    indices = np.flatnonzero(selected.reshape(-1))
    if not len(indices):
        raise AblationError(f"Ablation view {view.identifier} selects no features")
    return indices


def build_ablation_views(
    band_names: Sequence[str],
    region_names: Sequence[str],
) -> list[AblationView]:
    bands = _unique_names(band_names, "Band names")
    regions = _unique_names(region_names, "Region names")
    views = [AblationView(axis="full", name="full", mode="all")]
    for band in bands:
        views.extend(
            [
                AblationView(axis="band", name=band, mode="only"),
                AblationView(axis="band", name=band, mode="leave_out"),
            ]
        )
    for region in regions:
        views.extend(
            [
                AblationView(axis="region", name=region, mode="only"),
                AblationView(axis="region", name=region, mode="leave_out"),
            ]
        )
    return views


def load_region_mapping(
    path: Path,
    dataset_id: str,
    archive_channel_names: Sequence[str],
) -> dict[str, list[str]]:
    """Load a region partition only when its channel order matches the archive."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AblationError(f"Cannot read region mapping: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise AblationError("Region mapping schema_version must be 1")
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping) or dataset_id not in datasets:
        raise AblationError(f"No region mapping for dataset {dataset_id}")
    dataset = datasets[dataset_id]
    if not isinstance(dataset, Mapping):
        raise AblationError(f"Region mapping for {dataset_id} must be a mapping")
    configured_order = _unique_names(
        dataset.get("channel_order", []), f"{dataset_id} channel order"
    )
    archive_order = [str(name) for name in archive_channel_names]
    if configured_order != archive_order:
        raise AblationError(
            f"{dataset_id} channel order does not match the feature archive"
        )
    regions = dataset.get("regions")
    if not isinstance(regions, Mapping):
        raise AblationError(f"No region mapping for dataset {dataset_id}")
    return _validate_regions(configured_order, regions)
