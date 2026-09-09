#!/usr/bin/env python3
"""Run the anonymous artifact's independent-machine acceptance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.reviewer_acceptance import (
    ReviewerAcceptanceError,
    run_reviewer_acceptance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_reviewer_acceptance(args.project_root, args.output)
    except ReviewerAcceptanceError as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
