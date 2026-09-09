#!/usr/bin/env python3
"""Compile the local Chinese reading PDF paired with the English submission."""

from __future__ import annotations

import argparse

from openaffect_eeg.chinese_reading import build_chinese_reading_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(build_chinese_reading_pdf(args.project_root, output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
