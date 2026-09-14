"""Validate a returned independent-user artifact acceptance packet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_acceptance_packet(
    form_path: Path,
    report_path: Path,
    transcript_path: Path,
) -> dict[str, object]:
    for path in (form_path, report_path, transcript_path):
        if not path.is_file():
            raise ValueError(f"Missing packet file: {path}")
    form = json.loads(form_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    transcript = transcript_path.read_text(encoding="utf-8", errors="replace")

    required_text = (
        "anonymous_tester_id", "test_date_utc", "repository_commit",
        "release_version", "release_manifest_sha256",
    )
    errors = [f"empty:{field}" for field in required_text if not str(form.get(field, "")).strip()]
    for field in ("participated_in_project_development", "received_guidance_beyond_quickstart"):
        if not isinstance(form.get(field), bool):
            errors.append(f"not_boolean:{field}")
    minutes = form.get("minutes_to_first_result")
    if not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or minutes <= 0:
        errors.append("invalid:minutes_to_first_result")
    final = form.get("final_attempt") or {}
    if final.get("verifier_exit_code") != 0 or final.get("status") != "pass":
        errors.append("final_attempt_not_pass")
    if report.get("status") != "pass":
        errors.append("report_not_pass")
    release = report.get("release") or {}
    if str(release.get("version", "")) != str(form.get("release_version", "")):
        errors.append("release_version_mismatch")
    if str(release.get("manifest_sha256", "")) != str(form.get("release_manifest_sha256", "")):
        errors.append("release_manifest_mismatch")
    for marker in ("START_UTC=", "END_UTC=", "VERIFY_EXIT_CODE=0"):
        if marker not in transcript:
            errors.append(f"transcript_missing:{marker}")
    commit = str(form.get("repository_commit", ""))
    if commit and commit not in transcript:
        errors.append("repository_commit_absent_from_transcript")

    participated = form.get("participated_in_project_development")
    guided = form.get("received_guidance_beyond_quickstart")
    if participated is False and guided is False:
        acceptance_type = "independent_user"
    elif guided is True:
        acceptance_type = "author_assisted"
    else:
        acceptance_type = "independent_machine"
    packet_status = "valid" if not errors else "invalid"
    return {
        "schema_version": "1.0",
        "packet_status": packet_status,
        "acceptance_type": acceptance_type,
        "independent_user_claim_allowed": bool(
            packet_status == "valid" and acceptance_type == "independent_user"
        ),
        "errors": errors,
        "attachments_sha256": {
            "form": _sha256(form_path),
            "report": _sha256(report_path),
            "transcript": _sha256(transcript_path),
        },
        "boundary": (
            "This validator checks internal consistency and attachment hashes; "
            "it cannot authenticate the tester's declarations."
        ),
    }

