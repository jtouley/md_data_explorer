"""A filtered answer carries the dataset row count, so the presenter can suppress filtered-out rows."""

import pandas as pd
import pytest

from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan


@pytest.mark.parametrize(
    ("filters", "expected"),
    [([FilterSpec(column="age", operator=">=", value=50)], 12), ([], None)],
    ids=["filtered", "unfiltered"],
)
def test_format_execution_result_filters_addDatasetRowCount(make_semantic_layer, filters, expected):
    # Arrange: 12 synthetic rows, 8 of them aged 50 or over
    layer = make_semantic_layer(
        data={"patient_id": list(range(12)), "age": [40, 45, 50, 55, 60, 65] * 2, "ldl": [float(v) for v in range(12)]}
    )
    plan = QueryPlan(intent="DESCRIBE", metric="ldl", filters=filters, confidence=0.9)
    context = AnalysisContext(
        primary_variable="ldl", inferred_intent=AnalysisIntent.DESCRIBE, query_plan=plan, query_text="describe ldl"
    )
    execution_result = {"success": True, "result": pd.DataFrame({"mean": [1.0]}), "run_key": "k", "warnings": []}

    # Act
    formatted = layer.format_execution_result(execution_result, context)

    # Assert
    assert formatted.get("dataset_row_count") == expected
    assert formatted["row_count"] == (8 if filters else 12)
