"""Selection-gated synchronization of approved paper figure artifacts."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from openaffect_eeg.artifacts import sha256_file


class FigureFinalizationError(ValueError):
    """Raised when a final figure selection is malformed or cannot be traced."""


_DESTINATION_INDEX = {"F1": 1, "F2": 2, "F3": 3, "F4": 4, "F5": 5}
_CANDIDATE_FIELDS = ("candidates", "b_refinements", "v2_candidates")


def parse_selection(values: Sequence[str]) -> dict[str, str]:
    """Parse unique ``F2=B2`` selections from CLI-friendly strings."""
    selections: dict[str, str] = {}
    for raw in values:
        if raw.count("=") != 1:
            raise FigureFinalizationError(
                f"Selection must use FIGURE=CANDIDATE syntax: {raw!r}"
            )
        raw_figure, raw_candidate = raw.split("=", 1)
        figure = raw_figure.strip().upper()
        candidate = raw_candidate.strip().upper()
        if not figure or not candidate:
            raise FigureFinalizationError(
                f"Selection must use FIGURE=CANDIDATE syntax: {raw!r}"
            )
        if figure in selections:
            raise FigureFinalizationError(f"Figure {figure} was selected twice")
        selections[figure] = candidate
    if not selections:
        raise FigureFinalizationError("At least one figure selection is required")
    return selections


def _load_plan(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FigureFinalizationError(f"Invalid figure plan: {path}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("figures"), list):
        raise FigureFinalizationError("Figure plan must contain a figures list")
    return raw


def _figure_records(plan: Mapping[str, object]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    figures = plan.get("figures")
    if not isinstance(figures, list):
        raise FigureFinalizationError("Figure plan must contain a figures list")
    for raw in figures:
        if not isinstance(raw, dict):
            raise FigureFinalizationError("Each figure plan record must be a mapping")
        figure = str(raw.get("id", "")).upper()
        if figure not in _DESTINATION_INDEX:
            raise FigureFinalizationError(f"Unknown figure plan ID: {figure!r}")
        if figure in records:
            raise FigureFinalizationError(f"Duplicate figure plan ID: {figure}")
        records[figure] = raw
    return records


def _candidate_path(record: Mapping[str, object], candidate: str) -> str:
    candidate_paths: dict[str, str] = {}
    for field in _CANDIDATE_FIELDS:
        raw = record.get(field, {})
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise FigureFinalizationError(f"Figure candidate field {field} must be a mapping")
        for raw_key, raw_path in raw.items():
            key = str(raw_key).upper()
            path = str(raw_path)
            if key in candidate_paths and candidate_paths[key] != path:
                raise FigureFinalizationError(f"Ambiguous candidate {key}")
            candidate_paths[key] = path
    if candidate not in candidate_paths:
        raise FigureFinalizationError(
            f"Candidate {candidate} is not registered for {record.get('id', '<unknown>')}"
        )
    return candidate_paths[candidate]


def _inside_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise FigureFinalizationError(
            f"Candidate artifact escapes project root: {relative}"
        ) from error
    return candidate


def _copy_pair(source_pdf: Path, destination_prefix: Path) -> tuple[Path, Path]:
    source_png = source_pdf.with_suffix(".png")
    if not source_pdf.is_file():
        raise FigureFinalizationError(f"Candidate PDF is missing: {source_pdf}")
    if not source_png.is_file():
        raise FigureFinalizationError(f"Candidate PNG is missing PNG: {source_png}")
    destination_prefix.parent.mkdir(parents=True, exist_ok=True)
    destination_pdf = destination_prefix.with_suffix(".pdf")
    destination_png = destination_prefix.with_suffix(".png")
    shutil.copyfile(source_pdf, destination_pdf)
    shutil.copyfile(source_png, destination_png)
    return destination_pdf, destination_png


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def finalize_figure_selection(
    project_root: str | Path,
    selections: Mapping[str, str],
) -> dict[str, object]:
    """Copy approved candidates to the LaTeX and release locations.

    The manifest intentionally contains only paths and content hashes so it is
    stable across repeated executions with the same selected candidate artifacts.
    """
    root = Path(project_root).resolve()
    plan_path = root / "paper" / "figure_plan_zh.json"
    plan = _load_plan(plan_path)
    records = _figure_records(plan)
    normalized = {str(figure).upper(): str(candidate).upper() for figure, candidate in selections.items()}
    unknown = sorted(set(normalized).difference(records))
    if unknown:
        raise FigureFinalizationError("Unknown figure selection(s): " + ", ".join(unknown))

    manifest_selections: dict[str, dict[str, str]] = {}
    pending_updates: list[tuple[dict[str, object], str]] = []
    for figure in sorted(normalized, key=lambda item: _DESTINATION_INDEX[item]):
        record = records[figure]
        candidate = normalized[figure]
        relative_source = _candidate_path(record, candidate)
        source_pdf = _inside_root(root, relative_source)
        if source_pdf.suffix.lower() != ".pdf":
            raise FigureFinalizationError(f"Candidate must be a PDF: {relative_source}")
        source_png = source_pdf.with_suffix(".png")
        if not source_pdf.is_file():
            raise FigureFinalizationError(f"Candidate PDF is missing: {source_pdf}")
        if not source_png.is_file():
            raise FigureFinalizationError(f"Candidate PNG is missing PNG: {source_png}")

        index = _DESTINATION_INDEX[figure]
        build_pdf, build_png = _copy_pair(
            source_pdf, root / "paper" / "neurips2026" / "figures" / f"figure{index}"
        )
        release_pdf, release_png = _copy_pair(
            source_pdf, root / "paper" / "final_figures" / f"figure{index}"
        )
        manifest_selections[figure] = {
            "candidate": candidate,
            "source_pdf": _relative(root, source_pdf),
            "source_png": _relative(root, source_png),
            "source_pdf_sha256": sha256_file(source_pdf),
            "source_png_sha256": sha256_file(source_png),
            "build_pdf": _relative(root, build_pdf),
            "build_png": _relative(root, build_png),
            "release_pdf": _relative(root, release_pdf),
            "release_png": _relative(root, release_png),
        }
        pending_updates.append((record, candidate))

    for record, candidate in pending_updates:
        previous_candidate = record.get("selected_candidate")
        record["selected_candidate"] = candidate
        if not (
            previous_candidate == candidate
            and record.get("candidate_status") == "finalized_page_audited"
        ):
            record["candidate_status"] = "selected_pending_page_audit"
    approval_policy = plan.get("approval_policy")
    if isinstance(approval_policy, dict):
        approval_policy["current_final_selection_count"] = sum(
            1 for record in records.values() if record.get("selected_candidate")
        )
    _write_json(plan_path, plan)
    manifest: dict[str, object] = {
        "figure_plan": _relative(root, plan_path),
        "figure_plan_sha256": sha256_file(plan_path),
        "selections": manifest_selections,
    }
    _write_json(root / "paper" / "generated" / "final_figure_selection.json", manifest)
    return manifest
