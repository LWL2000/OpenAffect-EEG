"""Validated, resumable recomputation plans for benchmark releases."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
VALID_ENVIRONMENTS = {"statistics", "foundation"}


class RecomputeError(ValueError):
    """Raised when a recomputation plan is invalid or cannot be materialized."""


@dataclass(frozen=True)
class RecomputeStepTemplate:
    name: str
    stage: str
    environment: str
    command: tuple[str, ...]
    after: tuple[str, ...]
    requires: tuple[str, ...]
    produces: tuple[str, ...]


@dataclass(frozen=True)
class RecomputePlan:
    version: str
    stages: tuple[str, ...]
    steps: tuple[RecomputeStepTemplate, ...]


@dataclass(frozen=True)
class RecomputeStep:
    name: str
    stage: str
    environment: str
    command: tuple[str, ...]
    after: tuple[str, ...]
    requires: tuple[Path, ...]
    produces: tuple[Path, ...]


def _strings(value: object, field: str, step_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RecomputeError(f"Step {step_name!r} field {field!r} must be a string list")
    return tuple(value)


def _assert_acyclic(steps: Sequence[RecomputeStepTemplate]) -> None:
    dependencies = {step.name: set(step.after) for step in steps}
    remaining = set(dependencies)
    while remaining:
        ready = {name for name in remaining if not (dependencies[name] & remaining)}
        if not ready:
            raise RecomputeError(
                f"Recompute dependency cycle involves: {sorted(remaining)}"
            )
        remaining -= ready


def load_recompute_plan(path: Path) -> RecomputePlan:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecomputeError("Recompute plan must be a mapping")
    version = payload.get("version")
    raw_stages = payload.get("stages")
    raw_steps = payload.get("steps")
    if not isinstance(version, str) or not version:
        raise RecomputeError("Recompute plan requires a version")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise RecomputeError("Recompute plan requires an ordered stage list")
    if not all(isinstance(stage, str) and stage for stage in raw_stages):
        raise RecomputeError("Recompute stages must be non-empty strings")
    if len(raw_stages) != len(set(raw_stages)):
        raise RecomputeError("Recompute stage names must be unique")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RecomputeError("Recompute plan requires at least one step")

    stages = tuple(raw_stages)
    steps: list[RecomputeStepTemplate] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise RecomputeError("Each recompute step must be a mapping")
        name = raw_step.get("name")
        stage = raw_step.get("stage")
        environment = raw_step.get("environment")
        if not isinstance(name, str) or not name:
            raise RecomputeError("Each recompute step requires a name")
        if stage not in stages:
            raise RecomputeError(f"Step {name!r} has unknown stage: {stage!r}")
        if environment not in VALID_ENVIRONMENTS:
            raise RecomputeError(
                f"Step {name!r} has unknown environment: {environment!r}"
            )
        steps.append(
            RecomputeStepTemplate(
                name=name,
                stage=str(stage),
                environment=str(environment),
                command=_strings(raw_step.get("command"), "command", name),
                after=_strings(raw_step.get("after", []), "after", name),
                requires=_strings(raw_step.get("requires", []), "requires", name),
                produces=_strings(raw_step.get("produces", []), "produces", name),
            )
        )

    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise RecomputeError("Recompute step names must be unique")
    known = set(names)
    for item in steps:
        missing = set(item.after) - known
        if missing:
            raise RecomputeError(
                f"Step {item.name!r} has unknown dependencies: {sorted(missing)}"
            )
        if item.name in item.after:
            raise RecomputeError(f"Step {item.name!r} depends on itself")
    _assert_acyclic(steps)
    return RecomputePlan(version=version, stages=stages, steps=tuple(steps))


def _resolve(value: str, variables: Mapping[str, str]) -> str:
    missing = sorted(set(VARIABLE_PATTERN.findall(value)) - set(variables))
    if missing:
        raise RecomputeError(f"Missing recompute variables: {missing}")
    return VARIABLE_PATTERN.sub(lambda match: str(variables[match.group(1)]), value)


def materialize_steps(
    plan: RecomputePlan,
    variables: Mapping[str, str],
    selected_stages: Sequence[str] | None = None,
) -> tuple[RecomputeStep, ...]:
    selected = set(selected_stages or plan.stages)
    unknown = selected - set(plan.stages)
    if unknown:
        raise RecomputeError(f"Unknown selected stages: {sorted(unknown)}")
    materialized: list[RecomputeStep] = []
    for template in plan.steps:
        if template.stage not in selected:
            continue
        python_key = f"{template.environment}_python"
        if python_key not in variables:
            raise RecomputeError(f"Missing recompute variable: {python_key}")
        command = (
            str(variables[python_key]),
            *(_resolve(item, variables) for item in template.command),
        )
        materialized.append(
            RecomputeStep(
                name=template.name,
                stage=template.stage,
                environment=template.environment,
                command=command,
                after=template.after,
                requires=tuple(Path(_resolve(item, variables)) for item in template.requires),
                produces=tuple(Path(_resolve(item, variables)) for item in template.produces),
            )
        )
    return tuple(materialized)


def recompute_readiness(steps: Sequence[RecomputeStep]) -> dict[str, object]:
    missing = sorted(
        {
            str(path)
            for item in steps
            for path in item.requires
            if not path.exists()
        }
    )
    return {
        "ready": not missing,
        "step_count": len(steps),
        "missing_requirements": missing,
        "declared_output_count": len(
            {str(path) for item in steps for path in item.produces}
        ),
    }


def execute_recompute_steps(
    steps: Sequence[RecomputeStep],
    *,
    resume: bool = False,
) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for item in steps:
        missing = [str(path) for path in item.requires if not path.exists()]
        if missing:
            raise RecomputeError(f"Step {item.name!r} is missing inputs: {missing}")
        if resume and item.produces and all(path.exists() for path in item.produces):
            report.append({"name": item.name, "status": "skipped_existing"})
            continue
        for output in item.produces:
            output.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        subprocess.run(item.command, check=True)
        absent = [str(path) for path in item.produces if not path.exists()]
        if absent:
            raise RecomputeError(f"Step {item.name!r} did not produce outputs: {absent}")
        report.append(
            {
                "name": item.name,
                "status": "completed",
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        )
    return report
