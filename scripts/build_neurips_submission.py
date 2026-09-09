#!/usr/bin/env python3
"""Compile and audit the anonymous NeurIPS E&D submission."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from openaffect_eeg.submission import build_submission
from openaffect_eeg.evidence_argument import build_argument
from openaffect_eeg.submission_closure import build_submission_closure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--require-final-figures", action="store_true")
    parser.add_argument("--forbid", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    build_argument(project_root / "paper/generated/final_closure_v9",
                   project_root / "paper/generated/evidence_argument_v10")
    build_submission_closure(
        project_root,
        project_root / "paper/generated/submission_closure_v11",
    )
    claim_build = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "append_identity_exposure_claims.py")],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if claim_build.returncode != 0:
        raise RuntimeError(
            "Identity-exposure claim build failed:\n"
            + claim_build.stdout
            + claim_build.stderr
        )
    report = build_submission(
        project_root,
        require_final_figures=args.require_final_figures,
        forbidden_patterns=tuple(args.forbid),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
