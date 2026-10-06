"""
Every compute_* analysis must honour the question's filters.

Regression: comparison / predictor / survival / relationship analyses ignored FilterSpecs, so
"compare outcome by treatment for patients over 60" returned statistics over the whole cohort.

Metamorphic check: analysing the cohort with a filter must equal analysing the pre-filtered cohort
with no filter. Holds regardless of each analysis's result shape.

Test name follows: test_unit_scenario_expectedBehavior
"""

import json

import polars as pl
import pytest

from clinical_analytics.analysis.compute import compute_analysis_by_type
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

N_PATIENTS = 120
AGE_FILTER = FilterSpec(column="age", operator=">", value=80)


@pytest.fixture
def filter_cohort(make_cohort_with_categorical) -> pl.DataFrame:
    base = make_cohort_with_categorical(
        n_patients=N_PATIENTS, treatment=["A" if i % 2 else "B" for i in range(N_PATIENTS)]
    )
    i = pl.int_range(pl.len())
    return base.with_columns(
        ((i // 3) % 2).alias("outcome"),
        (i * 7 % 23).cast(pl.Float64).alias("score"),
        (i % 11).cast(pl.Float64).alias("value"),
        ((i * 13 % 50) + 1).alias("time"),
        ((i // 2) % 2).alias("event"),
        (i % 2).cast(pl.Utf8).alias("category"),
    )


def _stable(result: dict) -> str:
    return json.dumps(result, sort_keys=True, default=str)


@pytest.mark.parametrize(
    "context_fixture",
    ["sample_context_compare", "sample_context_predictor", "sample_context_survival", "sample_context_relationship"],
)
@pytest.mark.parametrize("via", ["context_filters", "query_plan_filters"])
def test_compute_analysis_withFilter_equalsAnalysisOfPrefilteredCohort(request, filter_cohort, context_fixture, via):
    # Arrange
    context = request.getfixturevalue(context_fixture)
    prefiltered = filter_cohort.filter(pl.col("age") > AGE_FILTER.value)
    assert 0 < prefiltered.height < filter_cohort.height
    expected = _stable(compute_analysis_by_type(prefiltered, context))
    if via == "context_filters":
        context.filters = [AGE_FILTER]
    else:
        context.query_plan = QueryPlan(intent="DESCRIBE", filters=[AGE_FILTER], confidence=0.9)

    # Act
    actual = _stable(compute_analysis_by_type(filter_cohort, context))

    # Assert
    assert actual == expected


def test_compute_descriptive_withFilter_reportsExcludedRows(filter_cohort, sample_context_describe):
    """Describe keeps reporting original vs filtered counts (why filtering stays per-analysis)."""
    # Arrange
    sample_context_describe.filters = [AGE_FILTER]
    matching = filter_cohort.filter(pl.col("age") > AGE_FILTER.value).height

    # Act
    result = compute_analysis_by_type(filter_cohort, sample_context_describe)

    # Assert
    assert (result.get("original_count"), result.get("filtered_count")) == (N_PATIENTS, matching)


@pytest.mark.parametrize(
    ("context_fixture", "attr", "value"),
    [
        ("sample_context_compare", "grouping_variable", "sex"),
        ("sample_context_compare", "primary_variable", "sex"),
        ("sample_context_predictor", "predictor_variables", ["age", "sex"]),
        ("sample_context_survival", "event_variable", "sex"),
        ("sample_context_relationship", "predictor_variables", ["age", "sex"]),
    ],
)
def test_compute_analysis_unknownColumn_returnsErrorNotException(request, filter_cohort, context_fixture, attr, value):
    """Regression: a variable missing from the cohort raised ColumnNotFoundError and crashed the page."""
    # Arrange
    context = request.getfixturevalue(context_fixture)
    setattr(context, attr, value)

    # Act
    result = compute_analysis_by_type(filter_cohort, context)

    # Assert
    assert "sex" in result["error"]
    assert result["available_columns"] == filter_cohort.columns
