#!/usr/bin/env python3
"""Build the anonymous English submission and its paired Chinese reading PDF."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from openaffect_eeg.chinese_reading import build_chinese_reading_pdf
from openaffect_eeg.submission import build_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--require-final-figures", action="store_true")
    parser.add_argument("--forbid", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
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
    chinese_pdf = build_chinese_reading_pdf(project_root)
    report["chinese_reading_pdf"] = str(chinese_pdf)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
