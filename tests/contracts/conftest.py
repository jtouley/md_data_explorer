"""Fixtures for data-contract tests (``make_contract`` is shared: tests/conftest.py)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts.semantics import DatasetSemantics

DEXA = "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis"
STATIN = "Statin use 1: Yes 2: No"
FIXED_TS = datetime(2026, 11, 3, tzinfo=UTC)


@pytest.fixture
def cohort_with_defects(load_synthetic_dataset: Callable[[str], pl.DataFrame]) -> pl.DataFrame:
    """Synthetic cohort with known defects injected on top of its ~5.8% missing LDL.

    * 3 duplicated patients (rows 0-2 appended again)
    * DEXA code 4 (not in the header codebook) in 12 rows, and code 9 in 1 row
    * ages 150 (x2) and -5 (x1), outside the 0-122 plausibility bound
    """
    df = load_synthetic_dataset("cohort")
    df = pl.concat([df, df.head(3)])
    row = pl.int_range(pl.len())
    return df.with_columns(
        pl.when(row.is_between(10, 21)).then(4).when(row == 30).then(9).otherwise(pl.col(DEXA)).alias(DEXA),
        pl.when(row.is_in([40, 41])).then(150).when(row == 42).then(-5).otherwise(pl.col("age")).alias("age"),
    )


def applied_facts(semantics: DatasetSemantics, back: dict[str, str]) -> dict[str, Any]:
    """Every fact that changes a number, with columns named as in the original file."""
    agg = semantics.aggregate
    return {
        "columns": {
            back[name]: (
                sorted(str(m.value) for m in col.missing_codes),
                (col.date.format, col.date.has_time) if col.date else None,
                dict(col.labels),
                (col.stated_range.minimum, col.stated_range.maximum) if col.stated_range else None,
                sorted(str(c.value) for c in col.nonresponse_codes),
                sorted(str(c.value) for c in col.censored_codes),
                col.unit.value if col.unit is not None and col.unit.source == "dictionary" else None,
            )
            for name, col in semantics.columns.items()
        },
        "aggregate": (agg.kind, back.get(agg.count_column or "", None), agg.total_levels) if agg else None,
        "questions": sorted((q.kind, back.get(q.column or "", ""), str(q.value)) for q in semantics.open_questions),
    }
