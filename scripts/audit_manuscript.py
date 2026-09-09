#!/usr/bin/env python3
"""Audit manuscript citations, frozen claim references, and anonymity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.manuscript_audit import audit_manuscript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("bibliography", type=Path)
    parser.add_argument("claim_ledger", type=Path)
    parser.add_argument("citation_audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--forbid", action="append", default=[])
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_manuscript(
        args.manuscript,
        args.bibliography,
        args.claim_ledger,
        args.citation_audit,
        forbidden_patterns=tuple(args.forbid),
        source_root=args.source_root,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
