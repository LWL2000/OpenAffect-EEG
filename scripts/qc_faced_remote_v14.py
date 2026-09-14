#!/usr/bin/env python3
"""Run outcome-blind remote FACED BDF/event structural preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.faced_confirmation import remote_structural_preflight, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--subjects",
        help="Comma-separated BIDS subject IDs; default is all sub-000 through sub-122",
    )
    args = parser.parse_args()
    subjects = None
    if args.subjects:
        subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    report = (
        remote_structural_preflight(subjects=subjects)
        if subjects is not None
        else remote_structural_preflight()
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "stage",
                    "outcome_values_loaded",
                    "eeg_samples_loaded",
                    "complete_subject_count",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
