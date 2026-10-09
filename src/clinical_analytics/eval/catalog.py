"""Ground-truth golden questions for datasets stored in analytics.duckdb.

Distinct uploads:
- gdsi: COVID-19 in multiple sclerosis
- dexa: de-identified DEXA cohort
- statin: de-identified statin cohort
- mimic_patients: MIMIC-IV demo patients table

The second MIMIC load (user_upload_20260103_165447) is the same demo copied
again, so it is not a fifth dataset. Multi-table MIMIC tables other than
patients are out of this suite until a unified cohort exists.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from clinical_analytics.core.column_parser import parse_column_name

QUESTIONS_PER_DATASET = 100

DATASET_TABLES: dict[str, str] = {
    "gdsi": "user_upload_20251228_163830_28450494_gdsi_opendataset_final_58182358815b733d",
    "dexa": "user_upload_20251228_203407_376a8faa_de_identified_dexa_1ee8f9c3e0f6f0f7",
    "statin": "user_upload_20251229_225650_45c58677_statin_use_deidentified_091a873a95864319",
    "mimic_patients": "user_upload_20260103_164905_db3af1fe_patients_d0e6aafb5fe2f6ab",
}

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


def duckdb_path(workspace: Path | None = None) -> Path:
    root = workspace or Path.cwd()
    return root / "data" / "analytics.duckdb"


def load_dataset_frame(dataset_id: str, workspace: Path | None = None) -> pl.DataFrame:
    """Load one dataset table from the repo DuckDB file."""
    if dataset_id not in DATASET_TABLES:
        known = ", ".join(sorted(DATASET_TABLES))
        raise KeyError(f"Unknown dataset '{dataset_id}'. Known: {known}")
    path = duckdb_path(workspace)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB dataset store not found: {path}")
    table = DATASET_TABLES[dataset_id]
    con = duckdb.connect(str(path), read_only=True)
    try:
        frame = con.execute(f'SELECT * FROM "{table}"').pl()
    finally:
        con.close()
    return frame


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
            maximum = series.max()
            if distinct <= 20 and maximum is not None and maximum <= 20:
                continue
        numeric.append(name)
    return numeric


def _grouping_columns(frame: pl.DataFrame) -> list[str]:
    groups: list[str] = []
    for name, dtype in frame.schema.items():
        if name in {"patient_id", "subject_id"}:
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
    """Build 100 questions with ground truth computed from the frame."""
    questions: list[dict[str, Any]] = []
    numerics = _numeric_columns(frame)
    groups = _grouping_columns(frame)

    def add(query: str, kind: str, **fields: Any) -> None:
        if len(questions) >= QUESTIONS_PER_DATASET:
            return
        questions.append(
            {
                "id": f"{dataset_id}_{len(questions) + 1:03d}",
                "dataset_id": dataset_id,
                "query": query,
                "kind": kind,
                **fields,
            }
        )

    for stat in _STATS:
        for column in numerics:
            phrase = column_phrase(column)
            value = _stat_value(frame, column, stat)
            add(
                f"what is the {stat} {phrase}?",
                "stat",
                metric=column,
                stat=stat,
                expected=value,
            )
            add(
                f"what is the {stat} of {phrase}?",
                "stat",
                metric=column,
                stat=stat,
                expected=value,
            )
            add(
                f"{stat} {phrase}?",
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
            add(
                f"what is the average {phrase} by {group_phrase}?",
                "mean_by",
                metric=column,
                group_by=group,
                expected=expected,
            )
            add(
                f"average {phrase} by {group_phrase}?",
                "mean_by",
                metric=column,
                group_by=group,
                expected=expected,
            )
            add(
                f"compare {phrase} by {group_phrase}?",
                "compare",
                metric=column,
                group_by=group,
                expected=expected,
            )

    for group in groups:
        group_phrase = column_phrase(group)
        counts = _count_by(frame, group)
        add(
            f"how many patients by {group_phrase}?",
            "count_by",
            group_by=group,
            expected=counts,
        )
        add(
            f"number of patients by {group_phrase}?",
            "count_by",
            group_by=group,
            expected=counts,
        )
        add(
            f"count patients by {group_phrase}?",
            "count_by",
            group_by=group,
            expected=counts,
        )

    total = frame.height
    add("how many patients?", "count", expected=total)
    add("number of patients?", "count", expected=total)
    add("count patients?", "count", expected=total)

    variant = 1
    while len(questions) < QUESTIONS_PER_DATASET:
        add(
            f"how many patients {variant}?",
            "count",
            expected=total,
            paraphrase=True,
        )
        variant += 1

    return questions[:QUESTIONS_PER_DATASET]


def questions_for_dataset(dataset_id: str, workspace: Path | None = None) -> list[dict[str, Any]]:
    return build_questions(dataset_id, load_dataset_frame(dataset_id, workspace))
