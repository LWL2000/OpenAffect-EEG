#!/usr/bin/env python3
"""Materialize an allowlisted Git-indexed snapshot for anonymous packaging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openaffect_eeg.release_snapshot import (
    ReleaseSnapshotError,
    freeze_anonymous_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        snapshot = freeze_anonymous_release(args.source, args.policy, args.destination)
    except ReleaseSnapshotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Created anonymous Git-indexed snapshot: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
