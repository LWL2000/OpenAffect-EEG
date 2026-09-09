#!/usr/bin/env python3
"""Build the final source-linked method-validation and claim-repair summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.submission_closure import build_submission_closure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/generated/submission_closure_v11"),
    )
    args = parser.parse_args()
    print(json.dumps(build_submission_closure(args.project_root, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

