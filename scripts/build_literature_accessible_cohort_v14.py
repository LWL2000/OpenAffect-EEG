#!/usr/bin/env python3
"""Build and validate the access-complete 24-paper human-coding cohort."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def doi_key(value: object) -> str:
    key = str(value).strip().lower()
    prefix = "https://doi.org/"
    return key[len(prefix) :] if key.startswith(prefix) else key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_set", type=Path)
    parser.add_argument("candidate_screening", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("full_text_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    original = pd.read_csv(args.paper_set, keep_default_na=False)
    candidates = pd.read_csv(args.candidate_screening, keep_default_na=False)
    selection = pd.read_csv(args.selection, keep_default_na=False)
    if selection.paper_id.duplicated().any() or selection.doi.map(doi_key).duplicated().any():
        raise ValueError("Accessible cohort contains duplicate identifiers")

    metadata = pd.concat([original, candidates], ignore_index=True, sort=False)
    metadata["doi_key"] = metadata.doi.map(doi_key)
    by_doi = metadata.drop_duplicates("doi_key", keep="first").set_index("doi_key")
    rows: list[dict[str, object]] = []
    for chosen in selection.itertuples(index=False):
        key = doi_key(chosen.doi)
        if key not in by_doi.index:
            raise ValueError(f"No frozen-frame metadata for {chosen.paper_id}: {key}")
        source = by_doi.loc[key].to_dict()
        pdf = args.full_text_root / f"{chosen.paper_id}.pdf"
        text = args.full_text_root / f"{chosen.paper_id}.txt"
        if not pdf.exists() or not text.exists():
            raise FileNotFoundError(f"Missing validated full text for {chosen.paper_id}")
        if not pdf.read_bytes()[:5] == b"%PDF-":
            raise ValueError(f"Non-PDF payload for {chosen.paper_id}")
        rows.append(
            {
                "paper_id": chosen.paper_id,
                "doi": source["doi"],
                "title": source["title"],
                "authors": source["authors"],
                "publication_year": int(source["publication_year"]),
                "publication_date": source["publication_date"],
                "type": source["type"],
                "language": source["language"],
                "dataset_mentions": source["dataset_mentions"],
                "sample_role": "main",
                "selection_source": chosen.selection_source,
                "pdf_sha256": sha256_file(pdf),
                "text_sha256": sha256_file(text),
            }
        )
    cohort = pd.DataFrame(rows).sort_values(["publication_year", "paper_id"])
    counts = cohort.groupby("publication_year").size().to_dict()
    if counts != {year: 4 for year in range(2021, 2027)}:
        raise ValueError(f"Expected four accessible papers per year, got {counts}")
    if not set(cohort.type).issubset({"article", "conference-paper"}):
        raise ValueError("Accessible cohort contains an ineligible publication type")

    args.output.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output / "paper_set.csv", index=False)
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
    for coder in ("A", "B"):
        template = pd.DataFrame(
            {
                "paper_id": cohort.paper_id,
                "coder_id": coder,
                "document_version": "pdf_sha256:" + cohort.pdf_sha256,
            }
        ).reindex(columns=paper_columns)
        template.to_csv(args.output / f"coder_{coder}_paper.csv", index=False)
        pd.DataFrame(columns=evaluation_columns).to_csv(
            args.output / f"coder_{coder}_evaluations.csv", index=False
        )
    print(f"validated {len(cohort)} papers; strata={counts}")


if __name__ == "__main__":
    main()
