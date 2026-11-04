"""Fixtures for data-contract tests (none existed in tests/conftest.py for contracts)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.contracts.builder import ContractInputs, build_contract

DEXA = "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis"
STATIN = "Statin use 1: Yes 2: No"
FIXED_TS = datetime(2026, 11, 3, tzinfo=UTC)


@pytest.fixture
def make_contract() -> Callable[..., OpenDataContractStandard]:
    """Factory: contract for ``df`` with test defaults (upload u1, version v1, id patient_id)."""

    def _make(df: pl.DataFrame, **overrides: Any) -> OpenDataContractStandard:
        fields: dict[str, Any] = {
            "upload_id": "u1",
            "dataset_version": "v1",
            "dataset_name": "test cohort",
            "id_column": "patient_id",
            "created_at": FIXED_TS,
        }
        fields.update(overrides)
        return build_contract(df, ContractInputs(**fields))

    return _make


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
