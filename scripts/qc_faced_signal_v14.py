#!/usr/bin/env python3
"""Run the frozen local FACED signal-window QC without reading ratings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.faced_confirmation import local_signal_qc, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("remote_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--subjects", help="Optional comma-separated BIDS subject IDs")
    parser.add_argument("--minimum-subjects", type=int, default=25)
    args = parser.parse_args()
    remote = json.loads(args.remote_report.read_text(encoding="utf-8"))
    subjects = None
    if args.subjects:
        subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    report = local_signal_qc(
        args.dataset_root,
        remote,
        subjects=subjects,
        minimum_subjects=args.minimum_subjects,
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "outcome_values_loaded",
                    "eeg_samples_loaded",
                    "complete_subject_count",
                    "trial_count",
                    "median_trial_channel_sd_microvolts",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
