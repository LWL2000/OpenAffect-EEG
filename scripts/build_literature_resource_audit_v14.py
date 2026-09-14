#!/usr/bin/env python3
"""Retrieve, screen, and deterministically sample the locked v14 literature frame."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

from openaffect_eeg.literature_audit import (
    dataset_mentions,
    reconstruct_abstract,
    select_stratified_sample,
    stable_order,
    strict_title_match,
)


SELECT = ",".join(
    [
        "id", "doi", "title", "publication_year", "publication_date", "type",
        "language", "primary_location", "best_oa_location", "open_access",
        "authorships", "abstract_inverted_index",
    ]
)


def fetch_query(query: str, start: str, end: str, raw_dir: Path) -> list[dict]:
    cursor = "*"
    page = 0
    records: list[dict] = []
    while cursor:
        parameters = {
            "filter": f"from_publication_date:{start},to_publication_date:{end},title.search:{query}",
            "per-page": "200",
            "cursor": cursor,
            "select": SELECT,
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(parameters)
        request = urllib.request.Request(url, headers={"User-Agent": "OpenAffect-literature-audit-v14"})
        with urllib.request.urlopen(request, timeout=60) as handle:
            payload = json.load(handle)
        page += 1
        query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        (raw_dir / f"query-{query_id}-page-{page:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.extend(payload.get("results", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not payload.get("results"):
            break
        time.sleep(0.1)
    return records


def retrieve(config_path: Path, output: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw_openalex"
    raw_dir.mkdir(exist_ok=True)
    all_records = []
    for query in config["retrieval"]["title_queries"]:
        all_records.extend(
            fetch_query(
                query,
                config["scope"]["publication_start"].isoformat(),
                config["scope"]["publication_end"].isoformat(),
                raw_dir,
            )
        )
    deduplicated: dict[str, dict] = {}
    for record in all_records:
        persistent = (record.get("doi") or record.get("id") or "").lower()
        if persistent:
            deduplicated.setdefault(persistent, record)
    rows = []
    seed = config["sampling"]["order_seed"]
    for persistent, record in deduplicated.items():
        title = record.get("title") or ""
        abstract = reconstruct_abstract(record.get("abstract_inverted_index"))
        mentions = dataset_mentions(title + " " + abstract)
        year = record.get("publication_year")
        location = record.get("best_oa_location") or record.get("primary_location") or {}
        authors = "; ".join(
            item.get("author", {}).get("display_name", "") for item in record.get("authorships", [])
        )
        rows.append(
            {
                "candidate_id": persistent,
                "openalex_id": record.get("id", ""),
                "doi": record.get("doi", ""),
                "title": title,
                "authors": authors,
                "publication_year": year,
                "publication_date": record.get("publication_date", ""),
                "type": record.get("type", ""),
                "language": record.get("language", ""),
                "landing_page_url": location.get("landing_page_url", "") if location else "",
                "pdf_url": location.get("pdf_url", "") if location else "",
                "open_access_status": (record.get("open_access") or {}).get("oa_status", ""),
                "strict_title_match": strict_title_match(title),
                "dataset_mentions": ";".join(mentions),
                "automatic_candidate": bool(strict_title_match(title) and mentions),
                "sample_order": stable_order(seed, int(year or 0), persistent, title),
                "screening_decision": "",
                "exclusion_reason": "",
                "screening_evidence": "",
                "abstract": abstract,
            }
        )
    table = pd.DataFrame(rows).sort_values(["publication_year", "sample_order", "candidate_id"])
    table.to_csv(output / "candidate_screening.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {
        "retrieved_before_deduplication": len(all_records),
        "deduplicated": len(table),
        "automatic_candidates": int(table.automatic_candidate.sum()),
        "raw_pages": len(list(raw_dir.glob("*.json"))),
    }
    (output / "retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


def sample(config_path: Path, screening_path: Path, output: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    screened = pd.read_csv(screening_path, keep_default_na=False)
    invalid = set(screened.screening_decision) - {
        "eligible", "excluded", "not_screened_after_quota"
    }
    if invalid:
        raise ValueError(f"Unresolved or invalid screening decisions: {sorted(invalid)}")
    selected = select_stratified_sample(
        screened,
        years=config["sampling"]["year_strata"],
        main_per_year=int(config["sampling"]["main_per_year"]),
        reserve_per_year=int(config["sampling"]["reserve_per_year"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "paper_set.csv", index=False)
    paper_columns = [
        "paper_id", "coder_id", "document_version", "datasets_evaluated",
        "prediction_target", "claim_type", "no_eeg_comparator", "overall_parity",
        "confidence", "evidence_locator", "notes",
    ]
    evaluation_columns = [
        "paper_id", "coder_id", "dataset", "protocol_id", "target_label_source",
        "prediction_unit", "participant_isolation", "stimulus_isolation",
        "window_trial_leakage", "eeg_label_resources", "baseline_label_resources",
        "participant_calibration", "stimulus_label_count", "population_label_count",
        "test_label_model_selection", "comparator_predictions", "metric_rows",
        "overall_parity", "evidence_locator", "notes",
    ]
    main_ids = selected.loc[selected.sample_role.eq("main"), "paper_id"].tolist()
    for coder in ("A", "B"):
        pd.DataFrame({"paper_id": main_ids}).reindex(columns=paper_columns).assign(coder_id=coder).to_csv(
            output / f"coder_{coder}_paper.csv", index=False
        )
        pd.DataFrame(columns=evaluation_columns).to_csv(output / f"coder_{coder}_evaluations.csv", index=False)
    print(f"Selected {len(main_ids)} main papers and {len(selected) - len(main_ids)} reserves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("retrieve", "sample"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--screening", type=Path)
    args = parser.parse_args()
    if args.mode == "retrieve":
        retrieve(args.config, args.output)
    else:
        if args.screening is None:
            parser.error("sample mode requires --screening")
        sample(args.config, args.screening, args.output)


if __name__ == "__main__":
    main()
