"""Fixtures for data-contract tests (``make_contract`` is shared: tests/conftest.py)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import polars as pl
import pytest

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
