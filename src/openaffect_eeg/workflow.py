"""Typed public workflows for metadata, synthetic, and full benchmark modes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from openaffect_eeg.artifacts import (
    compile_artifacts,
    load_result_registry,
    sha256_file,
)
from openaffect_eeg.croissant_validation import validate_croissant_metadata
from openaffect_eeg.evaluation_cards import (
    build_protocol_rows,
    load_benchmark_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WorkflowError(ValueError):
    """Raised when a requested workflow cannot be constructed safely."""


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    command: tuple[str, ...]


def _metadata_steps(output_root: Path) -> list[WorkflowStep]:
    module = (sys.executable, "-m", "openaffect_eeg.workflow")
    contract = PROJECT_ROOT / "configs" / "benchmark.yaml"
    croissant = PROJECT_ROOT / "croissant.json"
    return [
        WorkflowStep(
            "validate_contract",
            (
                *module,
                "validate-contract",
                str(contract),
                str(output_root / "contract_validation.json"),
            ),
        ),
        WorkflowStep(
            "validate_croissant",
            (
                *module,
                "validate-croissant",
                str(croissant),
                str(output_root / "croissant_validation.json"),
            ),
        ),
        WorkflowStep(
            "build_evaluation_cards",
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_evaluation_cards.py"),
                str(contract),
                str(output_root / "evaluation_cards"),
            ),
        ),
    ]


def build_workflow(
    mode: Literal["metadata", "smoke", "full"],
    data_root: Path,
    output_root: Path,
) -> list[WorkflowStep]:
    if mode not in {"metadata", "smoke", "full"}:
        raise WorkflowError(f"Unknown workflow mode: {mode}")
    output_root = output_root.resolve()
    steps = _metadata_steps(output_root)
    if mode == "metadata":
        return steps
    if mode == "smoke":
        tests = [
            PROJECT_ROOT / "tests" / filename
            for filename in [
                "test_splits.py",
                "test_baselines.py",
                "test_probes.py",
                "test_artifacts.py",
            ]
        ]
        steps.extend(
            [
                WorkflowStep(
                    "test_synthetic_modules",
                    (sys.executable, "-m", "pytest", *(str(path) for path in tests)),
                ),
                WorkflowStep(
                    "compile_synthetic_registry",
                    (
                        sys.executable,
                        "-m",
                        "openaffect_eeg.workflow",
                        "synthetic-registry",
                        str(output_root / "synthetic"),
                    ),
                ),
            ]
        )
        return steps

    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise WorkflowError(f"Full mode requires an existing data root: {data_root}")
    registry = PROJECT_ROOT / "configs" / "paper_artifacts.yaml"
    steps.append(
        WorkflowStep(
            "compile_registered_results",
            (
                sys.executable,
                "-m",
                "openaffect_eeg.workflow",
                "compile-registry",
                str(registry),
                str(data_root),
                str(output_root / "paper"),
            ),
        )
    )
    return steps


def execute_steps(steps: Sequence[WorkflowStep]) -> None:
    for step in steps:
        print(f"[{step.name}] {' '.join(step.command)}", flush=True)
        subprocess.run(step.command, check=True)


def validate_contract(contract_path: Path, output: Path) -> dict[str, object]:
    contract = load_benchmark_contract(contract_path)
    rows = build_protocol_rows(contract)
    report = {
        "benchmark_id": contract["benchmark_id"],
        "contract_sha256": sha256_file(contract_path),
        "dataset_count": len(contract["datasets"]),
        "evaluation_cell_count": len(rows),
        "status": "valid",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def validate_croissant(croissant_path: Path, output: Path) -> dict[str, object]:
    try:
        metadata = json.loads(croissant_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkflowError(f"Invalid Croissant JSON: {croissant_path}") from error
    try:
        profile = validate_croissant_metadata(metadata)
    except ValueError as error:
        raise WorkflowError(str(error)) from error
    report = {
        "croissant_sha256": sha256_file(croissant_path),
        **profile,
        "raw_recordings_distributed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def compile_synthetic_registry(output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = output_root / "synthetic_result.json"
    summary.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "protocol": "subject_holdout",
                        "seed": 11,
                        "models": {
                            "synthetic_mean": {
                                "all": {
                                    "macro": {
                                        "ccc": 0.0,
                                        "mae": 0.25,
                                        "rmse": 0.3,
                                    }
                                }
                            }
                        },
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registry_path = output_root / "synthetic_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "version": "synthetic_v1",
                "root": str(output_root),
                "allowed_protocols": ["subject_holdout"],
                "assets": [
                    {
                        "name": "synthetic_summary",
                        "path": summary.name,
                        "kind": "summary",
                        "expected_seeds": [11],
                        "sha256": sha256_file(summary),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    registry = load_result_registry(registry_path)
    return compile_artifacts(registry, output_root / "paper")


def compile_registered_results(
    registry_path: Path,
    data_root: Path,
    output: Path,
) -> dict[str, object]:
    previous = os.environ.get("OPENAFFECT_DATA_ROOT")
    os.environ["OPENAFFECT_DATA_ROOT"] = str(data_root.resolve())
    try:
        registry = load_result_registry(registry_path)
    finally:
        if previous is None:
            os.environ.pop("OPENAFFECT_DATA_ROOT", None)
        else:
            os.environ["OPENAFFECT_DATA_ROOT"] = previous
    return compile_artifacts(registry, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    contract = subparsers.add_parser("validate-contract")
    contract.add_argument("contract", type=Path)
    contract.add_argument("output", type=Path)
    croissant = subparsers.add_parser("validate-croissant")
    croissant.add_argument("croissant", type=Path)
    croissant.add_argument("output", type=Path)
    synthetic = subparsers.add_parser("synthetic-registry")
    synthetic.add_argument("output", type=Path)
    registry = subparsers.add_parser("compile-registry")
    registry.add_argument("registry", type=Path)
    registry.add_argument("data_root", type=Path)
    registry.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "validate-contract":
        validate_contract(args.contract, args.output)
    elif args.action == "validate-croissant":
        validate_croissant(args.croissant, args.output)
    elif args.action == "synthetic-registry":
        compile_synthetic_registry(args.output)
    elif args.action == "compile-registry":
        compile_registered_results(args.registry, args.data_root, args.output)
    else:
        raise WorkflowError(f"Unknown workflow action: {args.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
