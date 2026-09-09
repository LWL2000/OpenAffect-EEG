#!/usr/bin/env python3
"""Build a deterministic, anonymity-scanned public benchmark archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openaffect_eeg.release import (
    ReleaseError,
    build_public_release,
    load_release_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--forbid",
        action="append",
        default=[],
        help="Additional literal string that must not occur in public files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        policy = load_release_policy(args.policy)
        manifest = build_public_release(
            args.source,
            policy,
            args.output,
            extra_forbidden=tuple(args.forbid),
        )
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"Built {manifest['file_count']} public files: "
        f"{manifest['archive_sha256']}"
    )
    print(f"Archive: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
