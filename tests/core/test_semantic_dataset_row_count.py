"""An answer carries the rows its filters removed, so the presenter can suppress that count."""

import polars as pl
import pytest

from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.result_presentation import present_analysis

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


@pytest.mark.parametrize(
    ("plan_filters", "legacy_filters", "expected"),
    [([AGED_50_PLUS], [], 4), ([], [AGED_50_PLUS], 4), ([AGED_50_PLUS], [AGED_50_PLUS], 4), ([], [], 0)],
    ids=["plan_filter", "legacy_context_filter", "both", "unfiltered"],
)
def test_format_execution_result_filters_addRowsFilteredOut(
    make_semantic_layer, plan_filters, legacy_filters, expected
):
    # Arrange: compute falls back to context.filters when the plan has none
    layer = make_semantic_layer(data=SYNTHETIC_ROWS)
    plan = QueryPlan(intent="DESCRIBE", metric="ldl", filters=plan_filters, confidence=0.9)
    context = AnalysisContext(
        primary_variable="ldl",
        inferred_intent=AnalysisIntent.DESCRIBE,
        query_plan=plan,
        query_text="describe ldl",
        filters=legacy_filters,
    )

    # Act
    formatted = layer.format_execution_result(_execution_result(), context)

    # Assert: the rows compute analysed and the rows the filters removed add up to the dataset
    assert formatted["n_filtered_out"] == expected
    assert formatted["row_count"] == 12 - expected


@pytest.mark.parametrize(
    ("max_age", "expected_total", "expected_filtered_out", "expected_claim"),
    [(200, 55, 0, "Total count: 55."), (81, 52, 3, "Total count hidden."), (60, 31, 24, "Total count: 31.")],
    ids=["filter_keeps_everyone", "three_patients_filtered_out", "many_filtered_out"],
)
def test_format_execution_result_distinctEntityCount_filteredOutCountsEntities(
    make_semantic_layer, max_age, expected_total, expected_filtered_out, expected_claim
):
    # Arrange: 60 visit rows for 55 patients (five patients have two visits), ages 30 to 84
    patients = [*range(55), *range(5)]
    layer = make_semantic_layer(data={"patient_id": patients, "age": [30 + p for p in patients]})
    plan = QueryPlan(
        intent="COUNT",
        entity_key="patient_id",
        filters=[FilterSpec(column="age", operator="<=", value=max_age)],
        confidence=0.9,
    )
    context = AnalysisContext(inferred_intent=AnalysisIntent.COUNT, query_plan=plan, query_text="how many patients?")
    execution_result = {"success": True, "result": layer._execute_plan(plan), "run_key": "k", "warnings": []}

    # Act
    formatted = layer.format_execution_result(execution_result, context)

    # Assert: the five repeat visits are not counted as filtered-out patients
    assert (formatted["total_count"], formatted["n_filtered_out"]) == (expected_total, expected_filtered_out)
    assert present_analysis(formatted)["claim"] == expected_claim


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
    assert layer._base_counts == {}


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
