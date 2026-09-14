#!/usr/bin/env python3
"""Run label-blind structural and signal-scale QC on authorised AMIGOS files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.amigos_confirmation import structural_qc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_preprocessed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = structural_qc(args.data_preprocessed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "outcome_values_loaded", "complete_signal_subject_count",
        "source_units", "microvolt_scale_factor",
    )}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

