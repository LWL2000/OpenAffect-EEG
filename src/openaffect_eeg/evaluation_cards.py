"""Validation and tabulation for the public benchmark evaluation contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

VALID_PROTOCOL_STATUSES = {
    "supported",
    "not_applicable",
    "blocked_by_source_metadata",
}
REQUIRED_SEEDS = [11, 23, 47, 71, 101]


class BenchmarkContractError(ValueError):
    pass


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkContractError(f"{field} must be a non-empty string")
    return value.strip()


def validate_benchmark_contract(contract: Mapping[str, Any]) -> None:
    """Reject incomplete or ambiguous protocol applicability metadata."""
    if not isinstance(contract, Mapping):
        raise BenchmarkContractError("Benchmark contract must be a mapping")
    if contract.get("schema_version") != 1:
        raise BenchmarkContractError("schema_version must be 1")
    _require_nonempty_string(contract.get("benchmark_id"), "benchmark_id")

    seeds = contract.get("seeds")
    if seeds != REQUIRED_SEEDS:
        raise BenchmarkContractError(
            f"seeds must be the fixed benchmark seeds {REQUIRED_SEEDS}"
        )

    protocols = contract.get("protocols")
    if (
        not isinstance(protocols, list)
        or not protocols
        or any(not isinstance(protocol, str) or not protocol for protocol in protocols)
        or len(protocols) != len(set(protocols))
    ):
        raise BenchmarkContractError("protocols must be unique non-empty strings")

    datasets = contract.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise BenchmarkContractError("datasets must be a non-empty mapping")
    required_protocols = set(protocols)
    for dataset_id, dataset in datasets.items():
        _require_nonempty_string(dataset_id, "dataset_id")
        if not isinstance(dataset, Mapping):
            raise BenchmarkContractError(f"Dataset {dataset_id} must be a mapping")
        _require_nonempty_string(dataset.get("name"), f"{dataset_id}.name")
        cards = dataset.get("protocols")
        if not isinstance(cards, Mapping):
            raise BenchmarkContractError(
                f"{dataset_id}.protocols must be a mapping"
            )
        observed_protocols = set(cards)
        if observed_protocols != required_protocols:
            missing = sorted(required_protocols - observed_protocols)
            extra = sorted(observed_protocols - required_protocols)
            raise BenchmarkContractError(
                f"{dataset_id}.protocols mismatch; missing={missing}, extra={extra}"
            )
        for protocol, card in cards.items():
            if not isinstance(card, Mapping):
                raise BenchmarkContractError(
                    f"{dataset_id}.{protocol} card must be a mapping"
                )
            status = card.get("status")
            if status not in VALID_PROTOCOL_STATUSES:
                raise BenchmarkContractError(
                    f"{dataset_id}.{protocol} status must be one of "
                    f"{sorted(VALID_PROTOCOL_STATUSES)}"
                )
            _require_nonempty_string(
                card.get("reason"), f"{dataset_id}.{protocol}.reason"
            )


def load_benchmark_contract(path: Path) -> dict[str, Any]:
    """Load and validate a YAML benchmark contract."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BenchmarkContractError(f"Cannot read benchmark contract: {path}") from error
    validate_benchmark_contract(payload)
    return dict(payload)


def build_protocol_rows(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    """Flatten protocol cards into deterministic tabular rows."""
    validate_benchmark_contract(contract)
    rows = []
    for dataset_id, dataset in sorted(contract["datasets"].items()):
        for protocol, card in sorted(dataset["protocols"].items()):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset["name"],
                    "protocol": protocol,
                    "status": card["status"],
                    "reason": card["reason"],
                }
            )
    return rows
