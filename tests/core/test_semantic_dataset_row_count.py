"""A filtered answer carries the dataset row count, so the presenter can suppress filtered-out rows."""

import polars as pl
import pytest

from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

# 12 synthetic rows, 8 of them aged 50 or over
SYNTHETIC_ROWS = {
    "patient_id": list(range(12)),
    "age": [40, 45, 50, 55, 60, 65] * 2,
    "ldl": [float(v) for v in range(12)],
    "died": [0, 1] * 6,
}
AGED_50_PLUS = FilterSpec(column="age", operator=">=", value=50)


def _execution_result() -> dict:
    return {"success": True, "result": pl.DataFrame({"mean": [1.0]}), "run_key": "k", "warnings": []}


@pytest.mark.parametrize(("filters", "expected"), [([AGED_50_PLUS], 12), ([], None)], ids=["filtered", "unfiltered"])
def test_format_execution_result_filters_addDatasetRowCount(make_semantic_layer, filters, expected):
    # Arrange
    layer = make_semantic_layer(data=SYNTHETIC_ROWS)
    plan = QueryPlan(intent="DESCRIBE", metric="ldl", filters=filters, confidence=0.9)
    context = AnalysisContext(
        primary_variable="ldl", inferred_intent=AnalysisIntent.DESCRIBE, query_plan=plan, query_text="describe ldl"
    )

    # Act
    formatted = layer.format_execution_result(_execution_result(), context)

    # Assert
    assert formatted.get("dataset_row_count") == expected
    assert formatted["row_count"] == (8 if filters else 12)


def test_format_execution_result_failedAnalysis_skipsDatasetRowCount(make_semantic_layer):
    # Arrange: 8 filtered rows are too few for a regression, so the analysis fails
    layer = make_semantic_layer(data=SYNTHETIC_ROWS)
    plan = QueryPlan(intent="FIND_PREDICTORS", metric="died", filters=[AGED_50_PLUS], confidence=0.9)
    context = AnalysisContext(
        primary_variable="died",
        predictor_variables=["ldl"],
        inferred_intent=AnalysisIntent.FIND_PREDICTORS,
        query_plan=plan,
        query_text="does ldl predict died?",
    )

    # Act
    formatted = layer.format_execution_result(_execution_result(), context)

    # Assert: a failed result is never presented, so no count query is run for it
    assert formatted["error"]
    assert "dataset_row_count" not in formatted
    assert layer._base_row_count is None


@pytest.mark.parametrize("calls", [1, 2, 5], ids=["once", "twice", "five_times"])
def test_get_base_row_count_repeatedCalls_countsBaseViewOnce(make_semantic_layer, monkeypatch, calls):
    # Arrange: record every use of the base view
    layer = make_semantic_layer(data=SYNTHETIC_ROWS)
    base_view = layer.get_base_view()
    uses: list[int] = []
    monkeypatch.setattr(layer, "get_base_view", lambda: uses.append(1) or base_view)

    # Act
    counts = [layer.get_base_row_count() for _ in range(calls)]

    # Assert
    assert counts == [12] * calls
    assert len(uses) == 1
