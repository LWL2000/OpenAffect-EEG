#!/usr/bin/env python3
"""Unseal and ingest pinned FACED after the outcome-blind signal QC is locked."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openaffect_eeg.faced_confirmation import ingest_faced_to_disk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("signal_report", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = ingest_faced_to_disk(
        args.dataset_root,
        args.signal_report,
        args.output_root,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "participants", "stimuli", "trials")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
