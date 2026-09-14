#!/usr/bin/env python3
"""Validate and hash an independent artifact acceptance packet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.independent_reproduction import validate_acceptance_packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("form", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = validate_acceptance_packet(args.form, args.report, args.transcript)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "packet_status", "acceptance_type", "independent_user_claim_allowed"
    )}, sort_keys=True))
    return 0 if result["packet_status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())

