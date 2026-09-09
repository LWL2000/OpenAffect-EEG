"""Supported-runtime checks for reproducible OpenAffect-EEG workflows."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path


class EnvironmentSpecificationError(ValueError):
    """Raised when the project cannot state a supported runtime range."""


def _version_tuple(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in text.split(".")]
    if not 1 <= len(parts) <= 3:
        raise EnvironmentSpecificationError(f"Unsupported Python version: {text}")
    padded = parts + [0] * (3 - len(parts))
    return padded[0], padded[1], padded[2]


def python_is_supported(
    version_info: tuple[int, int, int],
    requires_python: str,
) -> bool:
    """Evaluate the simple version range used in the project metadata."""
    version = tuple(version_info[:3])
    comparisons = [part.strip() for part in requires_python.split(",") if part.strip()]
    if not comparisons:
        raise EnvironmentSpecificationError("requires-python must not be empty")
    for comparison in comparisons:
        match = re.fullmatch(r"(>=|<=|==|>|<)\s*(\d+(?:\.\d+){0,2})", comparison)
        if match is None:
            raise EnvironmentSpecificationError(
                f"Unsupported requires-python comparison: {comparison!r}"
            )
        operator, expected_text = match.groups()
        expected = _version_tuple(expected_text)
        valid = {
            ">=": version >= expected,
            "<=": version <= expected,
            "==": version == expected,
            ">": version > expected,
            "<": version < expected,
        }[operator]
        if not valid:
            return False
    return True


def project_requires_python(project_root: str | Path) -> str:
    """Read the supported Python range from pyproject.toml."""
    source = Path(project_root) / "pyproject.toml"
    try:
        metadata = tomllib.loads(source.read_text(encoding="utf-8"))
        value = metadata["project"]["requires-python"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as error:
        raise EnvironmentSpecificationError(
            f"Cannot read project requires-python from {source}"
        ) from error
    if not isinstance(value, str):
        raise EnvironmentSpecificationError("project.requires-python must be text")
    return value


def build_preflight_report(
    project_root: str | Path,
    data_root: str | Path,
    *,
    require_data_link: bool = False,
    version_info: tuple[int, int, int] | None = None,
    import_names: Iterable[str] = ("numpy", "pandas", "sklearn", "mne", "yaml"),
) -> dict[str, object]:
    """Produce a serializable, non-mutating supported-runtime report."""
    project = Path(project_root).resolve()
    data = Path(data_root).resolve()
    requires_python = project_requires_python(project)
    runtime_version = version_info or tuple(sys.version_info[:3])
    dependencies = {name: importlib.util.find_spec(name) is not None for name in import_names}
    checks: dict[str, bool] = {
        "project_exists": project.is_dir(),
        "python_supported": python_is_supported(runtime_version, requires_python),
        "data_exists": data.is_dir(),
        "data_writable": data.is_dir() and os.access(data, os.W_OK),
        "required_imports": all(dependencies.values()),
    }
    if require_data_link:
        checks["data_link"] = (project / "data").is_symlink()
    return {
        "project_root": str(project),
        "data_root": str(data),
        "python_version": ".".join(str(part) for part in runtime_version),
        "requires_python": requires_python,
        "dependency_imports": dependencies,
        "checks": checks,
        "status": "valid" if all(checks.values()) else "failed",
    }
