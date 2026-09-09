"""Component-wise audit utilities for the structured semantic stimulus prior."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMNS = ("target_valence", "target_arousal")
COMPONENT_COLUMNS = {
    "category": "nominal_category",
    "context": "context_code",
    "intensity": "stimulus_intensity_code",
    "description": "stimulus_description",
}


def _semantic_model(alpha: float, components: tuple[str, ...]) -> object:
    transformers: list[tuple[str, object, object]] = []
    for component in components:
        column = COMPONENT_COLUMNS[component]
        if component == "description":
            transformer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=1024,
                sublinear_tf=True,
            )
            selector: object = column
        else:
            transformer = OneHotEncoder(handle_unknown="ignore")
            selector = [column]
        transformers.append((component, transformer, selector))
    return make_pipeline(
        ColumnTransformer(transformers),
        Ridge(alpha=alpha, solver="lsqr"),
    )


def _aggregate_stimuli(table: pd.DataFrame) -> pd.DataFrame:
    semantic_columns = list(COMPONENT_COLUMNS.values())
    required = {"stimulus_uid", *semantic_columns, *TARGET_COLUMNS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Semantic prior table is missing columns: {missing}")
    semantics = table.groupby("stimulus_uid", sort=True)[semantic_columns].first()
    targets = table.groupby("stimulus_uid", sort=True)[list(TARGET_COLUMNS)].mean()
    return semantics.join(targets).reset_index()


def fit_semantic_prior(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    components: Sequence[str],
    alphas: Sequence[float],
) -> tuple[np.ndarray, dict[str, object]]:
    """Select Ridge alpha on validation data and predict a held-out test set."""

    selected_components = tuple(dict.fromkeys(str(value) for value in components))
    if not selected_components:
        raise ValueError("At least one semantic component is required")
    unknown = sorted(set(selected_components) - set(COMPONENT_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown semantic components: {unknown}")
    alpha_grid = tuple(float(value) for value in alphas)
    if not alpha_grid or any(not np.isfinite(value) or value <= 0 for value in alpha_grid):
        raise ValueError("Semantic-prior alphas must be finite and positive")

    train_stimuli = _aggregate_stimuli(train)
    validation_y = validation[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    validation_scores: dict[float, float] = {}
    for alpha in alpha_grid:
        model = _semantic_model(alpha, selected_components)
        model.fit(
            train_stimuli,
            train_stimuli[list(TARGET_COLUMNS)].to_numpy(dtype=float),
        )
        prediction = model.predict(validation)
        validation_scores[alpha] = float(np.mean(np.abs(prediction - validation_y)))
    selected_alpha = min(validation_scores, key=validation_scores.get)

    fit_stimuli = _aggregate_stimuli(
        pd.concat([train, validation], ignore_index=True)
    )
    model = _semantic_model(selected_alpha, selected_components)
    model.fit(
        fit_stimuli,
        fit_stimuli[list(TARGET_COLUMNS)].to_numpy(dtype=float),
    )
    prediction = np.asarray(model.predict(test), dtype=float)
    return prediction, {
        "components": list(selected_components),
        "selected_alpha": selected_alpha,
        "validation_mae": {
            str(alpha): validation_scores[alpha] for alpha in alpha_grid
        },
    }
