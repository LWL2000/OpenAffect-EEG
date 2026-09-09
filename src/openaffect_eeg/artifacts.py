"""Validated result registries and deterministic paper artifact compilation."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


class ArtifactError(ValueError):
    """Raised when a registered result cannot support reproducible claims."""


@dataclass(frozen=True)
class ResultAsset:
    name: str
    path: Path
    kind: Literal["summary", "predictions"]
    expected_seeds: tuple[int, ...]
    sha256: str
    key_columns: tuple[str, ...] = ("protocol", "seed", "trial_uid")


@dataclass(frozen=True)
class ResultRegistry:
    version: str
    assets: tuple[ResultAsset, ...]
    allowed_protocols: frozenset[str]
    source_path: Path
    source_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_finite(node: object, location: str = "result") -> None:
    if isinstance(node, float) and not math.isfinite(node):
        raise ArtifactError(f"Found non-finite value at {location}")
    if isinstance(node, Mapping):
        for key, value in node.items():
            _validate_finite(value, f"{location}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _validate_finite(value, f"{location}[{index}]")


def _protocols(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, Mapping):
        protocol = node.get("protocol")
        if protocol is not None:
            found.add(str(protocol))
        for value in node.values():
            found.update(_protocols(value))
    elif isinstance(node, list):
        for value in node:
            found.update(_protocols(value))
    return found


def _seed_groups(
    node: object,
    inherited_protocol: str | None = None,
) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = defaultdict(set)

    def visit(value: object, protocol: str | None) -> None:
        if isinstance(value, Mapping):
            current_protocol = str(value.get("protocol", protocol or "<global>"))
            if "seed" in value:
                try:
                    groups[current_protocol].add(int(value["seed"]))
                except (TypeError, ValueError) as error:
                    raise ArtifactError(
                        f"Invalid seed value for protocol {current_protocol}"
                    ) from error
            for child in value.values():
                visit(child, current_protocol)
        elif isinstance(value, list):
            for child in value:
                visit(child, protocol)

    visit(node, inherited_protocol)
    return groups


def validate_seed_coverage(
    result: Mapping[str, object],
    expected_seeds: Sequence[int],
) -> None:
    expected = {int(seed) for seed in expected_seeds}
    if not expected:
        return
    groups = _seed_groups(result)
    if not groups:
        missing = ", ".join(str(seed) for seed in sorted(expected))
        raise ArtifactError(f"Result is missing seeds: {missing}")
    for protocol, observed in sorted(groups.items()):
        missing = expected - observed
        extra = observed - expected
        if missing:
            values = ", ".join(str(seed) for seed in sorted(missing))
            raise ArtifactError(f"Protocol {protocol} is missing seeds: {values}")
        if extra:
            values = ", ".join(str(seed) for seed in sorted(extra))
            raise ArtifactError(f"Protocol {protocol} has unexpected seeds: {values}")


def _validate_protocols(
    protocols: set[str],
    allowed_protocols: frozenset[str],
) -> None:
    unknown = protocols - allowed_protocols
    if unknown:
        raise ArtifactError(f"Unknown protocol(s): {sorted(unknown)}")


def _load_summary(asset: ResultAsset, allowed_protocols: frozenset[str]) -> dict:
    try:
        result = json.loads(asset.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError(f"Invalid result JSON: {asset.path}") from error
    if not isinstance(result, dict):
        raise ArtifactError(f"Summary asset must contain a JSON object: {asset.path}")
    _validate_finite(result)
    _validate_protocols(_protocols(result), allowed_protocols)
    validate_seed_coverage(result, asset.expected_seeds)
    return result


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _validate_predictions(
    asset: ResultAsset,
    allowed_protocols: frozenset[str],
) -> dict[str, object]:
    required = {"protocol", "seed", "trial_uid", *asset.key_columns}
    keys: set[tuple[str, ...]] = set()
    seeds: dict[str, set[int]] = defaultdict(set)
    protocol_counts: dict[str, int] = defaultdict(int)
    row_count = 0
    with _open_text(asset.path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ArtifactError(f"Prediction archive has no header: {asset.path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ArtifactError(
                f"Prediction archive is missing columns {sorted(missing)}: {asset.path}"
            )
        numeric_columns = [
            column
            for column in reader.fieldnames
            if "valence" in column or "arousal" in column
        ]
        for row in reader:
            protocol = str(row["protocol"])
            try:
                seed = int(row["seed"])
            except (TypeError, ValueError) as error:
                raise ArtifactError(f"Invalid prediction seed in {asset.path}") from error
            key = tuple(str(row[column]) for column in asset.key_columns)
            if key in keys:
                raise ArtifactError(f"Found duplicate prediction key {key} in {asset.path}")
            keys.add(key)
            seeds[protocol].add(seed)
            protocol_counts[protocol] += 1
            row_count += 1
            for column in numeric_columns:
                raw_value = row[column]
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise ArtifactError(
                        f"Invalid numeric prediction at {key} column {column}"
                    ) from error
                if not math.isfinite(value):
                    raise ArtifactError(
                        f"Found non-finite prediction at {key} column {column}"
                    )
    _validate_protocols(set(seeds), allowed_protocols)
    expected = set(asset.expected_seeds)
    for protocol, observed in sorted(seeds.items()):
        missing = expected - observed
        extra = observed - expected
        if missing:
            raise ArtifactError(
                f"Protocol {protocol} is missing seeds: {sorted(missing)}"
            )
        if extra:
            raise ArtifactError(
                f"Protocol {protocol} has unexpected seeds: {sorted(extra)}"
            )
    return {
        "row_count": row_count,
        "protocol_counts": dict(sorted(protocol_counts.items())),
        "columns": reader.fieldnames,
    }


def load_result_registry(path: Path) -> ResultRegistry:
    source_path = path.resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ArtifactError(f"Invalid registry YAML: {source_path}") from error
    if not isinstance(raw, dict):
        raise ArtifactError("Result registry must contain a mapping")
    version = str(raw.get("version", "")).strip()
    if not version:
        raise ArtifactError("Result registry version is required")
    raw_root = str(raw.get("root", "."))
    expanded_root = os.path.expandvars(raw_root)
    if "$" in expanded_root:
        raise ArtifactError(f"Unresolved environment variable in registry root: {raw_root}")
    root = Path(expanded_root)
    if not root.is_absolute():
        root = source_path.parent / root
    allowed = raw.get("allowed_protocols")
    if not isinstance(allowed, list) or not allowed:
        raise ArtifactError("Registry allowed_protocols must be a non-empty list")
    allowed_protocols = frozenset(str(protocol) for protocol in allowed)
    raw_assets = raw.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ArtifactError("Registry assets must be a non-empty list")

    assets: list[ResultAsset] = []
    names: set[str] = set()
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, dict):
            raise ArtifactError("Each registry asset must be a mapping")
        name = str(raw_asset.get("name", "")).strip()
        if not name:
            raise ArtifactError("Asset name is required")
        if name in names:
            raise ArtifactError(f"Duplicate asset name: {name}")
        names.add(name)
        kind = str(raw_asset.get("kind", ""))
        if kind not in {"summary", "predictions"}:
            raise ArtifactError(f"Unknown asset kind for {name}: {kind}")
        asset_path = Path(str(raw_asset.get("path", "")))
        if not asset_path.is_absolute():
            asset_path = root / asset_path
        asset_path = asset_path.resolve()
        if not asset_path.is_file():
            raise ArtifactError(f"Registered asset is missing: {asset_path}")
        expected_hash = str(raw_asset.get("sha256", "")).lower()
        if len(expected_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_hash
        ):
            raise ArtifactError(f"Invalid SHA-256 for asset {name}")
        observed_hash = sha256_file(asset_path)
        if observed_hash != expected_hash:
            raise ArtifactError(
                f"SHA-256 mismatch for {name}: expected {expected_hash}, "
                f"observed {observed_hash}"
            )
        raw_seeds = raw_asset.get("expected_seeds", [])
        if not isinstance(raw_seeds, list):
            raise ArtifactError(f"expected_seeds must be a list for {name}")
        raw_key_columns = raw_asset.get(
            "key_columns", ["protocol", "seed", "trial_uid"]
        )
        if not isinstance(raw_key_columns, list) or not raw_key_columns:
            raise ArtifactError(f"key_columns must be a non-empty list for {name}")
        key_columns = tuple(str(column) for column in raw_key_columns)
        required_key_columns = {"protocol", "seed", "trial_uid"}
        if kind == "predictions" and not required_key_columns.issubset(key_columns):
            raise ArtifactError(
                f"Prediction key for {name} must include "
                "protocol, seed, and trial_uid"
            )
        if len(set(key_columns)) != len(key_columns):
            raise ArtifactError(f"Prediction key columns must be unique for {name}")
        asset = ResultAsset(
            name=name,
            path=asset_path,
            kind=kind,  # type: ignore[arg-type]
            expected_seeds=tuple(int(seed) for seed in raw_seeds),
            sha256=expected_hash,
            key_columns=key_columns,
        )
        if kind == "summary":
            _load_summary(asset, allowed_protocols)
        else:
            _validate_predictions(asset, allowed_protocols)
        assets.append(asset)
    return ResultRegistry(
        version=version,
        assets=tuple(assets),
        allowed_protocols=allowed_protocols,
        source_path=source_path,
        source_sha256=sha256_file(source_path),
    )


def _numeric_rows(asset_name: str, node: object, path: str = "$") -> list[dict]:
    rows: list[dict] = []
    if isinstance(node, bool):
        return rows
    if isinstance(node, (int, float)):
        rows.append({"asset": asset_name, "json_path": path, "value": node})
    elif isinstance(node, Mapping):
        for key, value in sorted(node.items(), key=lambda item: str(item[0])):
            rows.extend(_numeric_rows(asset_name, value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            rows.extend(_numeric_rows(asset_name, value, f"{path}[{index}]"))
    return rows


def _benchmark_rows(asset_name: str, result: Mapping[str, object]) -> list[dict]:
    rows: list[dict] = []
    entries = result.get("results", [])
    if not isinstance(entries, list):
        return rows
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("models"), Mapping):
            continue
        protocol = str(entry.get("protocol", ""))
        seed = entry.get("seed", "")
        for model, contexts in entry["models"].items():
            if not isinstance(contexts, Mapping):
                continue
            for context, targets in contexts.items():
                if not isinstance(targets, Mapping):
                    continue
                for target, metrics in targets.items():
                    if not isinstance(metrics, Mapping):
                        continue
                    for metric, value in metrics.items():
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            continue
                        rows.append(
                            {
                                "asset": asset_name,
                                "protocol": protocol,
                                "seed": seed,
                                "model": model,
                                "context": context,
                                "target": target,
                                "metric": metric,
                                "value": value,
                            }
                        )
    return rows


def _paired_rows(asset_name: str, result: Mapping[str, object]) -> list[dict]:
    rows: list[dict] = []
    entries = result.get("results", [])
    if not isinstance(entries, list):
        return rows
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("comparisons"), Mapping
        ):
            continue
        for comparison, statistics in entry["comparisons"].items():
            if not isinstance(statistics, Mapping):
                continue
            for statistic, value in statistics.items():
                row = {
                    "asset": asset_name,
                    "protocol": str(entry.get("protocol", "")),
                    "comparison": comparison,
                    "statistic": statistic,
                    "value": "",
                    "ci_low": "",
                    "ci_high": "",
                }
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    row["value"] = value
                elif (
                    isinstance(value, list)
                    and len(value) == 2
                    and all(isinstance(item, (int, float)) for item in value)
                ):
                    row["ci_low"], row["ci_high"] = value
                else:
                    continue
                rows.append(row)
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: tuple(str(row.get(field, "")) for field in fieldnames),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def _repository_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def compile_artifacts(registry: ResultRegistry, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    numeric_rows: list[dict] = []
    benchmark_rows: list[dict] = []
    paired_rows: list[dict] = []
    prediction_rows: list[dict] = []
    inputs: dict[str, dict[str, object]] = {}

    for asset in registry.assets:
        observed_hash = sha256_file(asset.path)
        if observed_hash != asset.sha256:
            raise ArtifactError(f"SHA-256 mismatch after registry load: {asset.name}")
        inputs[asset.name] = {
            "name": asset.name,
            "kind": asset.kind,
            "sha256": observed_hash,
            "expected_seeds": list(asset.expected_seeds),
            "key_columns": list(asset.key_columns),
        }
        if asset.kind == "summary":
            result = _load_summary(asset, registry.allowed_protocols)
            numeric_rows.extend(_numeric_rows(asset.name, result))
            benchmark_rows.extend(_benchmark_rows(asset.name, result))
            paired_rows.extend(_paired_rows(asset.name, result))
        else:
            details = _validate_predictions(asset, registry.allowed_protocols)
            prediction_rows.append(
                {
                    "asset": asset.name,
                    "sha256": observed_hash,
                    "row_count": details["row_count"],
                    "protocol_counts_json": json.dumps(
                        details["protocol_counts"], sort_keys=True, separators=(",", ":")
                    ),
                    "columns_json": json.dumps(
                        details["columns"], separators=(",", ":")
                    ),
                    "key_columns_json": json.dumps(
                        asset.key_columns, separators=(",", ":")
                    ),
                }
            )

    tables = {
        "summary_numeric_values.csv": (
            ["asset", "json_path", "value"],
            numeric_rows,
        ),
        "benchmark_metrics.csv": (
            [
                "asset",
                "protocol",
                "seed",
                "model",
                "context",
                "target",
                "metric",
                "value",
            ],
            benchmark_rows,
        ),
        "paired_comparisons.csv": (
            [
                "asset",
                "protocol",
                "comparison",
                "statistic",
                "value",
                "ci_low",
                "ci_high",
            ],
            paired_rows,
        ),
        "prediction_index.csv": (
            [
                "asset",
                "sha256",
                "row_count",
                "protocol_counts_json",
                "columns_json",
                "key_columns_json",
            ],
            prediction_rows,
        ),
    }
    outputs: dict[str, dict[str, object]] = {}
    for filename, (fieldnames, rows) in tables.items():
        destination = output / filename
        _write_csv(destination, fieldnames, rows)
        outputs[filename] = {
            "sha256": sha256_file(destination),
            "row_count": len(rows),
        }

    manifest: dict[str, object] = {
        "registry_version": registry.version,
        "registry_sha256": registry.source_sha256,
        "repository_commit": _repository_commit(),
        "inputs": dict(sorted(inputs.items())),
        "outputs": dict(sorted(outputs.items())),
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
