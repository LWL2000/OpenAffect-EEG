"""Deterministic sampling and agreement utilities for the v14 literature audit."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable

import pandas as pd


DATASET_PATTERNS = {
    "DEAP": re.compile(r"\bDEAP\b", re.IGNORECASE),
    "DREAMER": re.compile(r"\bDREAMER\b", re.IGNORECASE),
    "AMIGOS": re.compile(r"\bAMIGOS\b", re.IGNORECASE),
    "MAHNOB-HCI": re.compile(r"\bMAHNOB(?:[- ]HCI)?\b", re.IGNORECASE),
}


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = [(int(position), word) for word, values in index.items() for position in values]
    return " ".join(word for _, word in sorted(positions))


def strict_title_match(title: str) -> bool:
    value = " ".join(str(title).split()).lower()
    return bool(
        re.search(r"\b(eeg|electroencephal\w*)\b", value)
        and re.search(r"\b(emotion\w*|affect\w*)\b", value)
        and re.search(r"\b(recogni\w*|classif\w*|predict\w*|decod\w*)\b", value)
    )


def dataset_mentions(text: str) -> list[str]:
    return [name for name, pattern in DATASET_PATTERNS.items() if pattern.search(str(text))]


def stable_order(seed: str, year: int, persistent_id: str, title: str) -> str:
    material = f"{seed}|{int(year)}|{persistent_id.lower()}|{' '.join(title.lower().split())}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select_stratified_sample(
    screened: pd.DataFrame,
    *,
    years: Iterable[int],
    main_per_year: int,
    reserve_per_year: int,
) -> pd.DataFrame:
    required = {"candidate_id", "publication_year", "sample_order", "screening_decision"}
    missing = required - set(screened.columns)
    if missing:
        raise ValueError(f"Missing screening columns: {sorted(missing)}")
    eligible = screened.loc[screened.screening_decision.eq("eligible")].copy()
    chosen = []
    for year in years:
        pool = eligible.loc[eligible.publication_year.eq(int(year))].sort_values(
            ["sample_order", "candidate_id"]
        )
        needed = main_per_year + reserve_per_year
        if len(pool) < needed:
            raise ValueError(f"Year {year} has {len(pool)} eligible papers; {needed} required")
        block = pool.head(needed).copy()
        block["sample_role"] = ["main"] * main_per_year + ["reserve"] * reserve_per_year
        block["year_rank"] = range(1, needed + 1)
        chosen.append(block)
    result = pd.concat(chosen, ignore_index=True)
    result = result.sort_values(["sample_role", "sample_order", "candidate_id"]).reset_index(drop=True)
    result.insert(0, "paper_id", [f"P{index:03d}" for index in range(1, len(result) + 1)])
    return result


def nominal_agreement(left: Iterable[str], right: Iterable[str]) -> dict[str, float | int]:
    left_values = [str(value) for value in left]
    right_values = [str(value) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("Coding vectors have different lengths")
    pairs = list(zip(left_values, right_values))
    if not pairs:
        raise ValueError("No paired coding values")
    observed = sum(a == b for a, b in pairs) / len(pairs)
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    labels = set(ca) | set(cb)
    expected = sum((ca[label] / len(pairs)) * (cb[label] / len(pairs)) for label in labels)
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else float("nan")
    return {"n": len(pairs), "percent_agreement": observed * 100.0, "cohen_kappa": kappa}
