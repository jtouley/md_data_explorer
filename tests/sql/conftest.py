"""Fixtures for the governed SQL tests: a synthetic cohort as stored Parquet plus its contract.

The catalog comes from the shared ``make_sql_catalog`` factory in tests/conftest.py (default
frame: ``synthetic_cohort`` below, passed by ``ask_sql``).
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

from clinical_analytics.sql.answer import SqlAnswer, answer_query
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.sandbox import SandboxConnection
from clinical_analytics.sql.validate import validate_sql

N_PATIENTS = 300
ICU_PATIENTS = 3  # ward "ICU": a group of 1-10 patients
YOUNG_PATIENTS = 10  # age 18: removed by "age > 18", a small difference with the whole table
RARE_SCORE_PATIENTS = 4  # score 3: a value held by 1-10 patients


def synthetic_cohort(n: int = N_PATIENTS) -> pl.DataFrame:
    """One row per patient, with small groups planted (they must be shown exactly)."""
    ids = list(range(n))
    return pl.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in ids],
            "sex": ["F" if i % 2 == 0 else "M" for i in ids],
            "ward": ["ICU" if i < ICU_PATIENTS else ("A" if i % 3 else "B") for i in ids],
            "age": [18 if i < YOUNG_PATIENTS else 30 + (i * 7) % 50 for i in ids],
            # Deaths: every 5th patient, plus patient 1 (so the men's deaths include one extra).
            "died": [1 if i % 5 == 0 or i == 1 else 0 for i in ids],
            "score": [3 if n - RARE_SCORE_PATIENTS <= i else i % 3 for i in ids],
            "ldl": [None if i % 17 == 0 else 80.0 + (i * 13) % 97 / 1.3 for i in ids],
        }
    )


@pytest.fixture
def make_sql_catalog(make_sql_catalog: Callable[..., SqlCatalog]) -> Callable[..., SqlCatalog]:
    """The shared factory (tests/conftest.py) with this directory's default frame, ``synthetic_cohort()``."""
    shared = make_sql_catalog

    def _make(df: pl.DataFrame | None = None, levels: tuple = (), **contract_overrides: object) -> SqlCatalog:
        return shared(synthetic_cohort() if df is None else df, levels, **contract_overrides)

    return _make


@pytest.fixture
def ask_sql(make_sql_catalog: Callable[..., SqlCatalog]) -> Callable[..., SqlAnswer]:
    """Factory: the governed answer to ``sql`` on a catalog (default: the synthetic cohort).
    ``confirmed_patient_rows``: the row count the user confirmed as one row per patient."""
    default: list[SqlCatalog] = []

    def _ask(sql: str, catalog: SqlCatalog | None = None, confirmed_patient_rows: int | None = None) -> SqlAnswer:
        if catalog is None:
            if not default:
                default.append(make_sql_catalog())
            catalog = default[0]
        query = validate_sql(sql, catalog)
        with SandboxConnection(catalog) as sandbox:
            return answer_query(query, sandbox, catalog, confirmed_patient_rows=confirmed_patient_rows)

    return _ask
