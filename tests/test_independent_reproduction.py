from __future__ import annotations

import json
from pathlib import Path

from openaffect_eeg.independent_reproduction import validate_acceptance_packet


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_valid_independent_packet(tmp_path: Path) -> None:
    form = write_json(tmp_path / "form.json", {
        "anonymous_tester_id": "T01", "test_date_utc": "2026-09-15T00:00:00Z",
        "participated_in_project_development": False,
        "received_guidance_beyond_quickstart": False,
        "repository_commit": "a" * 40, "release_version": "V14",
        "release_manifest_sha256": "b" * 64, "minutes_to_first_result": 12,
        "final_attempt": {"verifier_exit_code": 0, "status": "pass"},
    })
    report = write_json(tmp_path / "report.json", {
        "status": "pass", "release": {"version": "V14", "manifest_sha256": "b" * 64}
    })
    transcript = tmp_path / "terminal.txt"
    transcript.write_text(
        f"START_UTC=x\n{'a' * 40}\nVERIFY_EXIT_CODE=0\nEND_UTC=y\n", encoding="utf-8"
    )
    result = validate_acceptance_packet(form, report, transcript)
    assert result["packet_status"] == "valid"
    assert result["independent_user_claim_allowed"] is True


def test_guided_packet_is_not_independent_user(tmp_path: Path) -> None:
    form = write_json(tmp_path / "form.json", {
        "anonymous_tester_id": "T02", "test_date_utc": "2026-09-15T00:00:00Z",
        "participated_in_project_development": False,
        "received_guidance_beyond_quickstart": True,
        "repository_commit": "c" * 40, "release_version": "V14",
        "release_manifest_sha256": "d" * 64, "minutes_to_first_result": 4,
        "final_attempt": {"verifier_exit_code": 0, "status": "pass"},
    })
    report = write_json(tmp_path / "report.json", {
        "status": "pass", "release": {"version": "V14", "manifest_sha256": "d" * 64}
    })
    transcript = tmp_path / "terminal.txt"
    transcript.write_text(
        f"START_UTC=x\n{'c' * 40}\nVERIFY_EXIT_CODE=0\nEND_UTC=y\n", encoding="utf-8"
    )
    result = validate_acceptance_packet(form, report, transcript)
    assert result["packet_status"] == "valid"
    assert result["acceptance_type"] == "author_assisted"
    assert result["independent_user_claim_allowed"] is False

