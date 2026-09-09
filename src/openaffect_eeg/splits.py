"""Deterministic trial-level splits with explicit identity-isolation audits."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

SPLITS = ("train", "validation", "test")


class SplitError(ValueError):
    pass


def load_trial_manifests(paths: list[Path]) -> pd.DataFrame:
    required = {
        "trial_uid",
        "dataset_id",
        "subject_id",
        "stimulus_uid",
        "context",
    }
    tables: list[pd.DataFrame] = []
    for path in paths:
        table = pd.read_csv(path, sep="\t")
        missing = sorted(required - set(table.columns))
        if missing:
            raise SplitError(f"{path} is missing columns: {', '.join(missing)}")
        if "label_available" not in table.columns:
            table["label_available"] = True
        table["subject_uid"] = (
            table["dataset_id"].astype(str) + ":" + table["subject_id"].astype(str)
        )
        tables.append(table)
    if not tables:
        raise SplitError("At least one trial manifest is required")
    combined = pd.concat(tables, ignore_index=True, sort=False)
    if combined["trial_uid"].duplicated().any():
        raise SplitError("Duplicate trial_uid values across manifests")
    return combined


def _validate_trials(table: pd.DataFrame, columns: Iterable[str]) -> None:
    required = {"trial_uid", *columns}
    missing = sorted(required - set(table.columns))
    if missing:
        raise SplitError(f"Missing split columns: {', '.join(missing)}")
    if table["trial_uid"].duplicated().any():
        raise SplitError("Duplicate trial_uid values are not allowed")
    if table[list(columns)].isna().any().any():
        raise SplitError("Split identity columns cannot contain missing values")


def _partition_groups(
    values: pd.Series,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[object, str]:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise SplitError("Validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise SplitError("Validation and test fractions must sum to less than one")

    groups = sorted(values.unique(), key=lambda value: str(value))
    if len(groups) < 3:
        raise SplitError("A three-way split requires at least three identity groups")
    random.Random(seed).shuffle(groups)
    test_count = max(1, round(len(groups) * test_fraction))
    validation_count = max(1, round(len(groups) * validation_fraction))
    if test_count + validation_count >= len(groups):
        test_count = max(1, len(groups) - validation_count - 1)
    if test_count + validation_count >= len(groups):
        validation_count = 1
        test_count = len(groups) - 2

    test = set(groups[:test_count])
    validation = set(groups[test_count : test_count + validation_count])
    return {
        group: (
            "test" if group in test else "validation" if group in validation else "train"
        )
        for group in groups
    }


def _partition_groups_by_stratum(
    table: pd.DataFrame,
    group_column: str,
    *,
    stratify_column: str | None,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[object, str]:
    if stratify_column is None:
        return _partition_groups(
            table[group_column],
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    group_strata = table.groupby(group_column)[stratify_column].nunique()
    if group_strata.gt(1).any():
        invalid = sorted(group_strata.loc[group_strata.gt(1)].index, key=str)
        raise SplitError(f"Identity groups span multiple strata: {invalid}")

    partition: dict[object, str] = {}
    strata = sorted(table[stratify_column].unique(), key=str)
    for index, stratum in enumerate(strata):
        subset = table.loc[table[stratify_column].eq(stratum), group_column]
        partition.update(
            _partition_groups(
                subset,
                seed=seed + index * 1009,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
            )
        )
    return partition


def build_group_holdout_split(
    table: pd.DataFrame,
    group_column: str,
    *,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    stratify_column: str | None = None,
) -> pd.DataFrame:
    columns = [group_column]
    if stratify_column is not None:
        columns.append(stratify_column)
    _validate_trials(table, columns)
    partition = _partition_groups_by_stratum(
        table,
        group_column,
        stratify_column=stratify_column,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    assignments = pd.DataFrame(
        {
            "trial_uid": table["trial_uid"],
            "split": table[group_column].map(partition),
            "protocol": f"{group_column}_held_out",
            "seed": seed,
        }
    )
    return assignments.sort_values("trial_uid").reset_index(drop=True)


def build_double_holdout_split(
    table: pd.DataFrame,
    *,
    first_group: str,
    second_group: str,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    stratify_column: str | None = None,
) -> pd.DataFrame:
    if first_group == second_group:
        raise SplitError("Double holdout requires two different identity columns")
    columns = [first_group, second_group]
    if stratify_column is not None:
        columns.append(stratify_column)
    _validate_trials(table, columns)
    first_partition = _partition_groups_by_stratum(
        table,
        first_group,
        stratify_column=stratify_column,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    second_partition = _partition_groups_by_stratum(
        table,
        second_group,
        stratify_column=stratify_column,
        seed=seed + 100_003,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    first_assignment = table[first_group].map(first_partition)
    second_assignment = table[second_group].map(second_partition)
    split = first_assignment.where(first_assignment.eq(second_assignment), "excluded")
    assignments = pd.DataFrame(
        {
            "trial_uid": table["trial_uid"],
            "split": split,
            "protocol": f"{first_group}_and_{second_group}_held_out",
            "seed": seed,
        }
    )
    return assignments.sort_values("trial_uid").reset_index(drop=True)


def build_context_transfer_split(
    table: pd.DataFrame,
    *,
    subject_column: str,
    context_column: str,
    source_context: str,
    target_context: str,
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.2,
) -> pd.DataFrame:
    """Train in one context and test in another with disjoint subjects."""

    if source_context == target_context:
        raise SplitError("Context transfer requires two different contexts")
    _validate_trials(table, [subject_column, context_column])
    observed_contexts = set(table[context_column])
    missing_contexts = {source_context, target_context} - observed_contexts
    if missing_contexts:
        raise SplitError(f"Missing context values: {sorted(missing_contexts)}")
    subject_partition = _partition_groups(
        table[subject_column],
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    identity_split = table[subject_column].map(subject_partition)
    source = table[context_column].eq(source_context)
    target = table[context_column].eq(target_context)
    split = pd.Series("excluded", index=table.index, dtype=object)
    split.loc[source & identity_split.eq("train")] = "train"
    split.loc[source & identity_split.eq("validation")] = "validation"
    split.loc[target & identity_split.eq("test")] = "test"
    assignments = pd.DataFrame(
        {
            "trial_uid": table["trial_uid"],
            "split": split,
            "protocol": f"{source_context}_to_{target_context}_subject_held_out",
            "seed": seed,
        }
    )
    return assignments.sort_values("trial_uid").reset_index(drop=True)


def split_audit(
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    isolation_columns: list[str],
) -> dict[str, object]:
    _validate_trials(table, isolation_columns)
    required = {"trial_uid", "split"}
    if not required.issubset(assignments.columns):
        raise SplitError("Assignments must contain trial_uid and split")
    if assignments["trial_uid"].duplicated().any():
        raise SplitError("Assignments contain duplicate trial_uid values")
    if set(assignments["trial_uid"]) != set(table["trial_uid"]):
        raise SplitError("Assignments must cover every trial exactly once")

    audit_columns = list(dict.fromkeys(["trial_uid", *isolation_columns]))
    merged = table[audit_columns].merge(
        assignments[["trial_uid", "split"]],
        on="trial_uid",
        validate="1:1",
    )
    identity_overlap: dict[str, dict[str, int]] = {}
    for column in isolation_columns:
        identities = {
            split: set(merged.loc[merged["split"].eq(split), column])
            for split in SPLITS
        }
        identity_overlap[column] = {
            "train_validation": len(identities["train"] & identities["validation"]),
            "train_test": len(identities["train"] & identities["test"]),
            "validation_test": len(identities["validation"] & identities["test"]),
        }

    return {
        "trial_count": len(table),
        "split_counts": {
            str(split): int(count)
            for split, count in assignments["split"].value_counts().items()
        },
        "identity_overlap": identity_overlap,
    }
