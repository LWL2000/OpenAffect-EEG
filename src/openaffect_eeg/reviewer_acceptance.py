"""Independent-machine acceptance checks for the anonymous review artifact."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.audit_cli import main as audit_main
from openaffect_eeg.submission_closure import build_submission_closure


class ReviewerAcceptanceError(RuntimeError):
    """Raised when a review artifact fails an acceptance check."""


def verify_release_manifest(project_root: Path) -> dict[str, object]:
    """Verify every file recorded by the archive's internal manifest."""

    manifest_path = project_root / "PUBLIC_RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        raise ReviewerAcceptanceError("PUBLIC_RELEASE_MANIFEST.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ReviewerAcceptanceError("Release manifest has no file records")

    failures: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            failures.append("malformed manifest record")
            continue
        relative = Path(str(record.get("path", "")))
        path = project_root / relative
        if not path.is_file():
            failures.append(f"missing:{relative.as_posix()}")
            continue
        observed = sha256_file(path)
        expected = str(record.get("sha256", ""))
        if observed != expected:
            failures.append(f"hash:{relative.as_posix()}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            failures.append(f"size:{relative.as_posix()}")
    if failures:
        preview = ", ".join(failures[:10])
        raise ReviewerAcceptanceError(f"Release manifest mismatch: {preview}")
    return {
        "version": manifest.get("version"),
        "file_count": len(records),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _record_check(checks: list[dict[str, object]], name: str, function) -> object:
    try:
        detail = function()
    except Exception as error:  # The report must survive a failed acceptance run.
        checks.append(
            {
                "name": name,
                "status": "fail",
                "error_type": type(error).__name__,
                "detail": str(error),
            }
        )
        return None
    checks.append({"name": name, "status": "pass", "detail": detail})
    return detail


def run_reviewer_acceptance(project_root: Path, output: Path) -> dict[str, object]:
    """Run artifact-integrity, install-state, toy, and evidence checks."""

    project_root = project_root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ReviewerAcceptanceError("Output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    def python_check() -> dict[str, object]:
        supported = (3, 11) <= sys.version_info[:2] < (3, 13)
        if not supported:
            raise ReviewerAcceptanceError("Python 3.11 or 3.12 is required")
        return {"version": platform.python_version(), "implementation": platform.python_implementation()}

    _record_check(checks, "supported_python", python_check)
    release = _record_check(
        checks,
        "release_manifest",
        lambda: verify_release_manifest(project_root),
    )

    def pip_check() -> str:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ReviewerAcceptanceError(completed.stdout.strip() or completed.stderr.strip())
        return completed.stdout.strip()

    _record_check(checks, "dependency_consistency", pip_check)

    def toy_check() -> dict[str, object]:
        toy_output = output / "toy-audit"
        status = audit_main(["toy", "--output", str(toy_output)])
        if status != 0:
            raise ReviewerAcceptanceError(f"Toy audit returned {status}")
        manifest_path = toy_output / "reproduction_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cards = json.loads((toy_output / "claim_cards.json").read_text(encoding="utf-8"))
        return {
            "cell_count": len(cards),
            "fixed_test_support_sha256": manifest["fixed_test_support_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
        }

    _record_check(checks, "synthetic_toy_audit", toy_check)

    def closure_check() -> dict[str, object]:
        rebuilt = output / "rebuilt-closure"
        rebuilt_manifest = build_submission_closure(project_root, rebuilt)
        frozen_path = (
            project_root
            / "paper"
            / "generated"
            / "submission_closure_v11"
            / "manifest.json"
        )
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if rebuilt_manifest["outputs"] != frozen["outputs"]:
            raise ReviewerAcceptanceError("Rebuilt closure hashes differ from the frozen record")
        return {"output_sha256": rebuilt_manifest["outputs"]}

    _record_check(checks, "source_linked_evidence_rebuild", closure_check)

    status = "pass" if checks and all(item["status"] == "pass" for item in checks) else "fail"
    report = {
        "schema_version": "1.0",
        "status": status,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "machine_architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "release": release,
        "checks": checks,
        "interpretation": (
            "A passing report establishes integrity and executable behavior on this machine. "
            "It is independent-user evidence only when the operator did not participate in development."
        ),
    }
    report_path = output / "independent_acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
