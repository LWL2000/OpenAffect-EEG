from __future__ import annotations

import pandas as pd

from openaffect_eeg.literature_audit import (
    dataset_mentions,
    nominal_agreement,
    reconstruct_abstract,
    select_stratified_sample,
    strict_title_match,
)


def test_metadata_helpers() -> None:
    assert reconstruct_abstract({"EEG": [0], "emotion": [1], "DEAP": [2]}) == "EEG emotion DEAP"
    assert strict_title_match("EEG-based emotion classification with a transformer")
    assert not strict_title_match("EEG seizure classification")
    assert dataset_mentions("DEAP and MAHNOB-HCI") == ["DEAP", "MAHNOB-HCI"]


def test_stratified_selection_uses_frozen_order() -> None:
    table = pd.DataFrame(
        [
            {"candidate_id": f"x{year}{rank}", "publication_year": year,
             "sample_order": f"{rank:02d}", "screening_decision": "eligible"}
            for year in (2024, 2025) for rank in range(4)
        ]
    )
    result = select_stratified_sample(table, years=(2024, 2025), main_per_year=2, reserve_per_year=1)
    assert (result.sample_role == "main").sum() == 4
    assert (result.sample_role == "reserve").sum() == 2
    assert set(result.year_rank) == {1, 2, 3}


def test_nominal_agreement() -> None:
    result = nominal_agreement(["matched", "unclear", "matched"], ["matched", "unclear", "unclear"])
    assert result["n"] == 3
    assert 0 < result["cohen_kappa"] < 1
    try:
        nominal_agreement(["matched"], ["matched", "unclear"])
    except ValueError as error:
        assert "different lengths" in str(error)
    else:
        raise AssertionError("unequal coder vectors should fail")
