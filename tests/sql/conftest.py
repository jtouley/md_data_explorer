"""Fixtures for the governed SQL tests: a synthetic cohort as stored Parquet plus its contract.

No fixture for this existed in tests/conftest.py; the contract comes from the shared
``make_contract`` factory.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from clinical_analytics.contracts.declarations import LevelMapping
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.privacy import SqlAnswer, answer_query
from clinical_analytics.sql.sandbox import SandboxConnection
from clinical_analytics.sql.validate import validate_sql

N_PATIENTS = 300
ICU_PATIENTS = 3  # ward "ICU": a group of 1-10 patients
YOUNG_PATIENTS = 10  # age 18: removed by "age > 18", a small difference with the whole table
RARE_SCORE_PATIENTS = 4  # score 3: a value held by 1-10 patients


def synthetic_cohort(n: int = N_PATIENTS) -> pl.DataFrame:
    """One row per patient. Small groups are planted where the attacks look for them."""
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
def make_sql_catalog(tmp_path: Path, make_contract: Callable[..., object]) -> Callable[..., SqlCatalog]:
    """Factory: the catalog of ``df`` (default: ``synthetic_cohort()``) stored as Parquet."""

    def _make(
        df: pl.DataFrame | None = None, levels: tuple[LevelMapping, ...] = (), **contract_overrides: object
    ) -> SqlCatalog:
        frame = synthetic_cohort() if df is None else df
        path = tmp_path / f"cohort_{len(list(tmp_path.glob('*.parquet')))}.parquet"
        frame.write_parquet(path)
        contract = make_contract(frame, **contract_overrides)
        return SqlCatalog.from_contract(contract, path, levels)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def ask_sql(make_sql_catalog: Callable[..., SqlCatalog]) -> Callable[..., SqlAnswer]:
    """Factory: the governed answer to ``sql`` on a catalog (default: the synthetic cohort)."""
    default: list[SqlCatalog] = []

    def _ask(sql: str, catalog: SqlCatalog | None = None) -> SqlAnswer:
        if catalog is None:
            if not default:
                default.append(make_sql_catalog())
            catalog = default[0]
        query = validate_sql(sql, catalog)
        with SandboxConnection(catalog) as sandbox:
            return answer_query(query, sandbox, catalog)

    return _ask
