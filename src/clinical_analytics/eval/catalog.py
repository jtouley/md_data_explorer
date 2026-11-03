"""Generated golden questions for the committed synthetic cohorts.

Ground truth is computed from the frame at load time and never stored, so the suite
needs no patient data and cannot drift from the file it checks. Datasets live in
tests/fixtures/synthetic (scripts/make_synthetic_data.py).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl

from clinical_analytics.core.column_parser import parse_column_name

# Dataset id -> CSV path relative to the repo root.
DATASET_FILES: dict[str, str] = {
    "cohort": "tests/fixtures/synthetic/cohort.csv",
}

_STAT_PHRASINGS = ("what is the {stat} {phrase}?", "what is the {stat} of {phrase}?", "{stat} {phrase}?")
_STATS = ("average", "mean", "median", "minimum", "maximum", "min", "max", "std")
_STAT_COLUMN = {
    "average": "mean",
    "mean": "mean",
    "median": "median",
    "minimum": "min",
    "min": "min",
    "maximum": "max",
    "max": "max",
    "std": "std",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dataset_path(dataset_id: str, workspace: Path | None = None) -> Path:
    """CSV path for a golden dataset id."""
    if dataset_id not in DATASET_FILES:
        known = ", ".join(sorted(DATASET_FILES))
        raise KeyError(f"Unknown dataset '{dataset_id}'. Known: {known}")
    path = (workspace or repo_root()) / DATASET_FILES[dataset_id]
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset file not found: {path}")
    return path


def load_dataset_frame(dataset_id: str, workspace: Path | None = None) -> pl.DataFrame:
    """Load one golden dataset."""
    return pl.read_csv(dataset_path(dataset_id, workspace))


def column_phrase(column: str) -> str:
    """Phrase that matches the semantic-layer alias for this column."""
    display = parse_column_name(column).display_name
    normalized = display.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _numeric_columns(frame: pl.DataFrame) -> list[str]:
    numeric: list[str] = []
    for name, dtype in frame.schema.items():
        if not dtype.is_numeric() or dtype == pl.Boolean:
            continue
        series = frame.get_column(name).drop_nulls()
        if series.len() == 0:
            continue
        if dtype.is_integer():
            distinct = series.n_unique()
            maximum = series.cast(pl.Int64).max()
            if distinct <= 20 and isinstance(maximum, int) and maximum <= 20:
                continue
        numeric.append(name)
    return numeric


def _grouping_columns(frame: pl.DataFrame) -> list[str]:
    groups: list[str] = []
    for name, dtype in frame.schema.items():
        if name in {"patient_id", "subject_id"} or name.endswith("_id"):
            continue
        series = frame.get_column(name).drop_nulls()
        distinct = series.n_unique()
        if not 2 <= distinct <= 12:
            continue
        if dtype == pl.Utf8 or dtype == pl.String or (dtype.is_integer() and series.max() is not None):
            groups.append(name)
    return groups


def _stat_value(frame: pl.DataFrame, column: str, stat: str) -> float | None:
    series = frame.get_column(column).drop_nulls()
    if series.len() == 0:
        return None
    reducer = _STAT_COLUMN[stat]
    value = getattr(series, reducer)()
    if value is None:
        return None
    return float(value)


def _mean_by(frame: pl.DataFrame, metric: str, group: str) -> dict[str, float | None]:
    aggregated = (
        frame.group_by(group, maintain_order=True).agg(pl.col(metric).mean().alias("mean")).sort(group, nulls_last=True)
    )
    expected: dict[str, float | None] = {}
    for row in aggregated.iter_rows(named=True):
        key = "null" if row[group] is None else str(row[group])
        mean = row["mean"]
        expected[key] = None if mean is None else float(mean)
    return expected


def _count_by(frame: pl.DataFrame, group: str) -> dict[str, int]:
    aggregated = frame.group_by(group, maintain_order=True).len().sort(group, nulls_last=True)
    expected: dict[str, int] = {}
    for row in aggregated.iter_rows(named=True):
        key = "null" if row[group] is None else str(row[group])
        expected[key] = int(row["len"])
    return expected


def build_questions(dataset_id: str, frame: pl.DataFrame) -> list[dict[str, Any]]:
    """One question per (statistic, column), mean-by and count-by per grouping, and totals.

    Phrasings rotate so each template is exercised without multiplying near-duplicates.
    """
    questions: list[dict[str, Any]] = []
    numerics = _numeric_columns(frame)
    groups = _grouping_columns(frame)

    def add(query: str, kind: str, **fields: Any) -> None:
        questions.append(
            {
                "id": f"{dataset_id}_{len(questions) + 1:03d}",
                "dataset_id": dataset_id,
                "query": query,
                "kind": kind,
                **fields,
            }
        )

    turn = 0
    for stat in _STATS:
        for column in numerics:
            template = _STAT_PHRASINGS[turn % len(_STAT_PHRASINGS)]
            turn += 1
            value = _stat_value(frame, column, stat)
            add(
                template.format(stat=stat, phrase=column_phrase(column)),
                "stat",
                metric=column,
                stat=stat,
                expected=value,
            )

    for column in numerics:
        phrase = column_phrase(column)
        for group in groups:
            group_phrase = column_phrase(group)
            expected = _mean_by(frame, column, group)
            if turn % 2 == 0:
                add(
                    f"what is the average {phrase} by {group_phrase}?",
                    "mean_by",
                    metric=column,
                    group_by=group,
                    expected=expected,
                )
            else:
                add(f"compare {phrase} by {group_phrase}", "compare", metric=column, group_by=group, expected=expected)
            turn += 1

    for group in groups:
        counts = _count_by(frame, group)
        for template in ("how many patients by {g}?", "number of patients by {g}?", "count patients by {g}"):
            add(template.format(g=column_phrase(group)), "count_by", group_by=group, expected=counts)

    for query in ("how many patients?", "number of patients?", "count patients"):
        add(query, "count", expected=frame.height)

    return questions


def questions_for_dataset(dataset_id: str, workspace: Path | None = None) -> list[dict[str, Any]]:
    return build_questions(dataset_id, load_dataset_frame(dataset_id, workspace))
