#!/usr/bin/env python3
"""Rebuild manuscript tables from frozen v14 aggregates, without experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from openaffect_eeg.v14_evidence import archive_results, build_assets, verify_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--import-existing-results", action="store_true", help="Copy already computed local aggregate files into the public archive; never fit a model or resample.")
    parser.add_argument("--verify", action="store_true", help="Rebuild in --output and compare to frozen public outputs.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.import_existing_results:
        archive_results(root)
    if args.verify:
        if args.output is None:
            parser.error("--verify requires a separate --output directory")
        result = verify_assets(root, args.output.resolve())
    else:
        result = build_assets(root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
