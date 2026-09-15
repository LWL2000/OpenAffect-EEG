#!/usr/bin/env python3
"""Retrieve a declared set of literature PDFs and record every access attempt."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url_map", type=Path, help="CSV with paper_id and url columns")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("attempt_log", type=Path)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with args.url_map.open(newline="", encoding="utf-8-sig") as handle:
        sources = list(csv.DictReader(handle))

    for source in sources:
        paper_id = source["paper_id"].strip()
        url = source["url"].strip()
        row: dict[str, object] = {
            "paper_id": paper_id,
            "url": url,
            "status": "",
            "http_status": "",
            "content_type": "",
            "bytes": 0,
            "sha256": "",
            "error": "",
        }
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; OpenAffect literature audit/1.0)",
                    "Accept": "application/pdf,text/html;q=0.8,*/*;q=0.5",
                },
            )
            with urlopen(request, timeout=args.timeout) as response:
                payload = response.read()
                row["http_status"] = response.status
                row["content_type"] = response.headers.get("Content-Type", "")
            row["bytes"] = len(payload)
            if payload.lstrip().startswith(b"%PDF-"):
                destination = args.output_root / f"{paper_id}.pdf"
                destination.write_bytes(payload)
                row["status"] = "downloaded_pdf"
                row["sha256"] = sha256(payload)
            else:
                row["status"] = "non_pdf_response"
        except HTTPError as error:
            row["status"] = "http_error"
            row["http_status"] = error.code
            row["error"] = str(error)
        except (URLError, TimeoutError, OSError) as error:
            row["status"] = "network_error"
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        print(f"{paper_id}: {row['status']} ({row['bytes']} bytes)")

    args.attempt_log.parent.mkdir(parents=True, exist_ok=True)
    with args.attempt_log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
