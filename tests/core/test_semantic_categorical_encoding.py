"""DESCRIBE picks a frequency table or summary statistics per column (_detect_categorical_encoding)."""

import pytest

from clinical_analytics.core.query_plan import QueryPlan

N = 40
STATS = ["mean", "median", "std", "min", "max"]


@pytest.mark.parametrize(
    "values",
    [
        [i % 11 for i in range(N)],  # 0-10 pain score
        [i % 2 for i in range(N)],  # 0/1 outcome: prevalence is the mean
        [1 + i % 14 for i in range(N)],  # length of stay 1-14 days
        [None if i % 5 == 0 else i % 7 for i in range(N)],  # comorbidity count with missing values
        list(range(N)),  # high cardinality
    ],
    ids=["score-0-10", "binary-outcome", "los-1-14", "count-with-nulls", "high-cardinality"],
)
def test_executePlan_describeWholeNumberColumn_returnsSummaryStatistics(make_semantic_layer, values) -> None:
    # Arrange
    layer = make_semantic_layer(data={"patient_id": list(range(N)), "measure": values})

    # Act
    out = layer._execute_plan(QueryPlan(intent="DESCRIBE", metric="measure"))

    # Assert
    assert list(out.columns) == STATS
    assert out.shape[0] == 1


def test_executePlan_describeLabelledCodes_returnsFrequencyTable(make_semantic_layer) -> None:
    # Arrange: "1: Yes" / "2: No" strings, as some exports encode categories.
    labels = ["1: Yes" if i % 3 else "2: No" for i in range(N)]
    layer = make_semantic_layer(data={"patient_id": list(range(N)), "measure": labels})

    # Act
    out = layer._execute_plan(QueryPlan(intent="DESCRIBE", metric="measure"))

    # Assert
    assert sorted(out.columns) == ["count", "measure"]
    assert dict(zip(out["measure"], out["count"], strict=True)) == {"1: Yes": 26, "2: No": 14}
