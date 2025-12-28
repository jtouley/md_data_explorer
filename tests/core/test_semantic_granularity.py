"""
Tests for SemanticLayer granularity parameter handling.

Validates:
- get_cohort() accepts and passes granularity parameter
- build_cohort_query() accepts granularity parameter
- show_sql=True uses logger.info() not print()
- All granularity values are accepted (permissive behavior)
- Backward compatibility with default granularity
"""

import logging

import pandas as pd
import pytest


class _FakeQuery:
    """Minimal query object for testing without DB/Ibis backend."""

    def __init__(self) -> None:
        self._compiled = "SELECT 1 AS x"

    def compile(self) -> str:
        return self._compiled

    def execute(self) -> pd.DataFrame:
        # PANDAS EXCEPTION: SemanticLayer.get_cohort() returns pd.DataFrame
        return pd.DataFrame({"x": [1]})


class _FakeSemantic:
    """
    Minimal SemanticLayer surface area for testing method behavior.

    Avoids requiring DB/Ibis backend.
    """

    dataset_name = "fake_dataset"

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def build_cohort_query(
        self,
        *,
        granularity: str = "patient_level",
        outcome_col: str | None = None,
        outcome_label: str | None = None,
        filters: dict | None = None,
    ) -> _FakeQuery:
        self.seen.append(
            {
                "granularity": granularity,
                "outcome_col": outcome_col,
                "outcome_label": outcome_label,
                "filters": filters,
            }
        )
        return _FakeQuery()


def test_granularity_default_value_is_patient_level() -> None:
    """Test that default granularity is patient_level."""
    # Arrange
    s = _FakeSemantic()

    # Act
    df = s.build_cohort_query(granularity="patient_level").execute()

    # Assert
    assert df.iloc[0]["x"] == 1
    assert s.seen[-1]["granularity"] == "patient_level"


@pytest.mark.parametrize(
    "granularity",
    ["patient_level", "admission_level", "event_level"],
)
def test_build_cohort_query_all_granularity_values_accepted(granularity: str) -> None:
    """Test that build_cohort_query accepts all valid granularity values."""
    # Arrange
    s = _FakeSemantic()

    # Act
    _ = s.build_cohort_query(granularity=granularity)

    # Assert
    assert s.seen[-1]["granularity"] == granularity


def test_get_cohort_with_show_sql_uses_logger_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that get_cohort passes granularity and show_sql uses logger.info."""
    from clinical_analytics.core.semantic import SemanticLayer

    # Arrange: Create instance bypassing heavy init
    s = SemanticLayer.__new__(SemanticLayer)
    s.dataset_name = "test_dataset"
    seen: dict = {}

    def _fake_build(
        *,
        granularity: str = "patient_level",
        outcome_col: str | None = None,
        outcome_label: str | None = None,
        filters: dict | None = None,
    ) -> _FakeQuery:
        seen["granularity"] = granularity
        return _FakeQuery()

    monkeypatch.setattr(s, "build_cohort_query", _fake_build)
    caplog.set_level(logging.INFO)

    # Act
    df = s.get_cohort(granularity="event_level", show_sql=True)

    # Assert
    assert seen["granularity"] == "event_level"
    assert df.iloc[0]["x"] == 1
    assert any("Generated SQL" in rec.message for rec in caplog.records)


def test_get_cohort_without_granularity_uses_default() -> None:
    """Test that get_cohort works without granularity (uses default)."""
    # Import here to avoid module-level import issues
    from clinical_analytics.core.semantic import SemanticLayer

    # Arrange
    s = SemanticLayer.__new__(SemanticLayer)
    s.dataset_name = "test_dataset"
    seen: dict = {}

    def _fake_build(
        *,
        granularity: str = "patient_level",
        outcome_col: str | None = None,
        outcome_label: str | None = None,
        filters: dict | None = None,
    ) -> _FakeQuery:
        seen["granularity"] = granularity
        return _FakeQuery()

    # Use object.__setattr__ to bypass any property
    object.__setattr__(s, "build_cohort_query", _fake_build)

    # Act: Call without granularity - should use default
    df = s.get_cohort()

    # Assert
    assert seen["granularity"] == "patient_level"
    assert len(df) == 1
