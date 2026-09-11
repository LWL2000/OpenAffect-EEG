"""Validate OpenAffect-EEG Croissant metadata without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.croissant_validation import validate_croissant_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("croissant", type=Path)
    parser.add_argument("--require-publishable-url", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.croissant.read_text(encoding="utf-8"))
    report = validate_croissant_metadata(
        metadata,
        require_publishable_url=args.require_publishable_url,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
