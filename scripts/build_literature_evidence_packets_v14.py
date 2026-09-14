#!/usr/bin/env python3
"""Create non-coding search packets for the two independent human coders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


PATTERNS = {
    "dataset_and_labels": re.compile(
        r"\b(DEAP|DREAMER|AMIGOS|MAHNOB(?:-HCI)?|self[- ]assessment|valence|arousal|rating|label)\b",
        re.IGNORECASE,
    ),
    "partition_and_support": re.compile(
        r"\b(subject[- ]dependent|subject[- ]independent|cross[- ]subject|leave[- ]one|train(?:ing)?|validation|test(?:ing)?|split|fold|window|segment|trial)\b",
        re.IGNORECASE,
    ),
    "comparator_and_claim": re.compile(
        r"\b(baseline|ablation|comparison|compare|without EEG|non[- ]EEG|multimodal|peripheral|audio|video|face|fusion|increment|improv)\w*\b",
        re.IGNORECASE,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excerpts(text: str, pattern: re.Pattern[str], limit: int = 24) -> list[dict[str, object]]:
    pages = text.split("\f")
    found: list[dict[str, object]] = []
    seen: set[str] = set()
    for page_number, page in enumerate(pages, start=1):
        lines = [" ".join(line.split()) for line in page.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if not pattern.search(line):
                continue
            context = " ".join(lines[max(0, index - 1) : min(len(lines), index + 2)])
            key = context.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append({"page": page_number, "excerpt": context})
            if len(found) >= limit:
                return found
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_set", type=Path)
    parser.add_argument("text_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sample = pd.read_csv(args.paper_set, keep_default_na=False)
    main = sample.loc[sample.sample_role.eq("main")]
    args.output.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for row in main.itertuples(index=False):
        text_path = args.text_root / f"{row.paper_id}.txt"
        if not text_path.exists():
            index_rows.append({"paper_id": row.paper_id, "status": "full_text_missing"})
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        packet = {
            "paper_id": row.paper_id,
            "title": row.title,
            "doi": row.doi,
            "document_version": "locally_acquired_full_text",
            "source_text_sha256": sha256_file(text_path),
            "purpose": (
                "Search aid only. Each human coder must inspect the cited page and "
                "make every judgment independently under the locked coding manual."
            ),
            "sections": {
                name: excerpts(text, pattern) for name, pattern in PATTERNS.items()
            },
        }
        destination = args.output / f"{row.paper_id}.json"
        destination.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index_rows.append({
            "paper_id": row.paper_id,
            "status": "packet_created",
            "packet_sha256": sha256_file(destination),
            "source_text_sha256": packet["source_text_sha256"],
        })
    index = pd.DataFrame(index_rows)
    index.to_csv(args.output / "index.csv", index=False)
    print(index.status.value_counts().to_json())


if __name__ == "__main__":
    main()
