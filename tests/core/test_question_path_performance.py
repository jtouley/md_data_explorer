# SPDX-License-Identifier: MIT
"""Question path cost: values are indexed once per layer, identifiers are never matched as
words, and execution reads only the columns it needs. Counts calls, never wall-clock time."""

from __future__ import annotations

import cProfile
import pstats
from typing import Any

import polars as pl
import pytest

from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan
from clinical_analytics.core.semantic import _analysis_columns, _with_cohort_cte

_ROWS = 60


def _frame(rows: int) -> pl.DataFrame:
    """Deterministic synthetic cohort: string ids, a unique text column, codes and measures."""
    index = pl.col("index")
    return pl.DataFrame({"index": pl.int_range(rows, eager=True)}).select(
        pl.format("S{}", index.cast(pl.Utf8).str.zfill(6)).alias("patient_id"),
        pl.format("visit note {}", index).alias("note"),
        (18 + index * 7 % 72).alias("age"),
        pl.when(index % 2 == 0).then(pl.lit("F")).otherwise(pl.lit("M")).alias("sex"),
        pl.when(index % 3 == 0).then(pl.lit("placebo")).otherwise(pl.lit("drug")).alias("treatment"),
        pl.lit(pl.Series(["North", "South", "East", "West"])).gather(index % 4).alias("site"),
        (80.0 + (index * 13 % 1200) / 10).alias("ldl"),
        (index % 6).alias("visits"),
    )


def _calls(profile: cProfile.Profile, name: str) -> int:
    stats = pstats.Stats(profile)
    return sum(entry[1] for key, entry in stats.stats.items() if key[2] == name)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("patient_id", False),  # the patient id
        ("note", False),  # a different value in every row
        ("ldl", False),  # a measure with more values than a grouping
        ("age", False),
        ("sex", True),
        ("site", True),
        ("visits", True),  # six integer codes
    ],
)
def test_distinctValues_column_storedValuesMatchedOnlyWhenNameable(
    make_semantic_layer, nl_query_engine_with_cached_model, column: str, expected: bool
) -> None:
    # Arrange
    engine = nl_query_engine_with_cached_model(semantic_layer=make_semantic_layer(data=_frame(_ROWS)))

    # Act
    values = engine._distinct_values(column)

    # Assert
    assert values.match_stored is expected
    assert list(values) == sorted(values, key=lambda value: (value is None, value))


@pytest.mark.parametrize("rows", [3, 49])
def test_distinctValues_smallTable_uniqueColumnsStillMatched(
    make_semantic_layer, nl_query_engine_with_cached_model, rows: int
) -> None:
    # Arrange
    engine = nl_query_engine_with_cached_model(semantic_layer=make_semantic_layer(data=_frame(rows)))

    # Act
    matched = {column: engine._distinct_values(column).match_stored for column in ("patient_id", "note")}

    # Assert
    assert matched == {"patient_id": True, "note": True}


def test_refreshValueIndex_aliasAdded_dropsCachedValues(make_semantic_layer, nl_query_engine_with_cached_model) -> None:
    # Arrange
    layer = make_semantic_layer(data=_frame(_ROWS))
    engine = nl_query_engine_with_cached_model(semantic_layer=layer)
    columns = list(layer.get_base_view().columns)
    engine.refresh_value_index(columns)
    before = engine._distinct_values("site")
    engine.refresh_value_index(columns)
    kept = engine._distinct_values("site")

    # Act
    layer._user_aliases = {"region": "site"}
    layer._alias_index = None
    engine.refresh_value_index(columns)
    after = engine._distinct_values("site")

    # Assert
    assert kept is before
    assert after is not before
    assert list(after) == list(before)


def test_invalidateValueIndex_cachedValues_readAgain(make_semantic_layer, nl_query_engine_with_cached_model) -> None:
    # Arrange
    engine = nl_query_engine_with_cached_model(semantic_layer=make_semantic_layer(data=_frame(_ROWS)))
    before = engine._distinct_values("sex")

    # Act
    engine.invalidate_value_index()

    # Assert
    assert engine._distinct_values("sex") is not before


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT count(*) FROM _cohort_in", "WITH _cohort_in AS (SELECT 1) SELECT count(*) FROM _cohort_in"),
        (
            "WITH cut AS (SELECT 1 AS m) SELECT * FROM _cohort_in, cut",
            "WITH _cohort_in AS (SELECT 1), cut AS (SELECT 1 AS m) SELECT * FROM _cohort_in, cut",
        ),
        ("  with cut AS (SELECT 1) SELECT 2", "WITH _cohort_in AS (SELECT 1), cut AS (SELECT 1) SELECT 2"),
    ],
)
def test_withCohortCte_query_viewBoundAsFirstCte(sql: str, expected: str) -> None:
    # Arrange / Act
    combined = _with_cohort_cte("SELECT 1", sql)

    # Assert
    assert combined == expected


def test_withCohortCte_recursiveQuery_refused() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="recursive"):
        _with_cohort_cte("SELECT 1", "WITH RECURSIVE r AS (SELECT 1) SELECT * FROM r")


class _Context:
    def __init__(self, **values: Any) -> None:
        self.primary_variable = values.get("primary")
        self.grouping_variable = values.get("group")
        self.time_variable = values.get("time")
        self.event_variable = values.get("event")
        self.predictor_variables = values.get("predictors", [])
        self.filters = values.get("filters", [])


_VIEW = ["patient_id", "age", "sex", "ldl", "site", "time_zero"]


@pytest.mark.parametrize(
    ("context", "filters", "expected"),
    [
        (_Context(primary="ldl", group="sex"), [], ["sex", "ldl"]),
        (_Context(primary="ldl"), [FilterSpec(column="age", operator=">", value=65)], ["age", "ldl"]),
        (
            _Context(primary="ldl", filters=[FilterSpec(column="site", operator="==", value="North")]),
            [],
            ["ldl", "site"],
        ),
        (_Context(primary="ldl", predictors=["age", "sex"]), [], ["age", "sex", "ldl"]),
        (_Context(), [], None),  # nothing named: every column
        (_Context(primary="all"), [], None),  # describe everything
        (_Context(primary="LDL "), [], None),  # a fuzzy name resolved against every column
        (_Context(primary="ldl", group="missing"), [], None),  # the analysis reports the missing column
        (_Context(primary="ldl"), [FilterSpec(column="bmi", operator=">", value=30)], None),
    ],
)
def test_analysisColumns_plan_onlyExactlyNamedColumnsOrEveryColumn(
    context: _Context, filters: list[FilterSpec], expected: list[str] | None
) -> None:
    # Arrange
    plan = QueryPlan(intent="DESCRIBE", metric="ldl", filters=filters)

    # Act
    needed = _analysis_columns(plan, context, _VIEW)

    # Assert
    assert needed == expected


def test_cohortSql_medianCutoff_sameRowsAsPolars(make_semantic_layer) -> None:
    # Arrange
    frame = _frame(_ROWS)
    layer = make_semantic_layer(data=frame)
    steps = [
        CohortStep("eq", "sex", "F"),
        CohortStep("gt_median", "ldl"),
        CohortStep("ge_percent_rank", "age", quantile=0.5),
    ]
    plan = QueryPlan(intent="COUNT", confidence=0.9, cohort_steps=steps)

    # Act
    kept = layer._filtered_view(plan).select("patient_id").to_polars().sort("patient_id")

    # Assert
    women = frame.filter(pl.col("sex") == "F")
    above = women.filter(pl.col("ldl") > women.get_column("ldl").median())
    expected = above.filter(pl.col("age") >= above.get_column("age").quantile(0.5, interpolation="linear"))
    assert kept.get_column("patient_id").to_list() == expected.sort("patient_id").get_column("patient_id").to_list()


@pytest.mark.slow
@pytest.mark.integration
def test_askQuestion_100kRows_warmQuestionNeverScansStoredValues(make_semantic_layer) -> None:
    # Arrange
    from clinical_analytics.core.query_service import QueryService

    frame = _frame(100_000)
    service = QueryService(make_semantic_layer(data=frame))
    question = "compare average ldl by treatment for patients older than 65"
    cold = service.ask(question, dataset_id="perf", dataset_version="v1")
    profile = cProfile.Profile()

    # Act
    profile.enable()
    warm = service.ask(question, dataset_id="perf", dataset_version="v1")
    profile.disable()

    # Assert
    assert cold.requires_confirmation is False and warm.requires_confirmation is False
    assert warm.result is not None and warm.result.get("success") is True
    assert warm.plan is not None and warm.plan.group_by == "treatment" and warm.plan.metric == "ldl"
    # Before the value index: 163,211 label checks and 1.2M stem matches per question.
    print(
        {
            name: _calls(profile, name)
            for name in (
                "_label_at",
                "stem_match",
                "_distinct_values",
                "to_pyarrow",
                "register",
                "to_polars",
                "execute",
            )
        }
    )
    assert _calls(profile, "_label_at") <= 50
    assert _calls(profile, "stem_match") <= 5_000
    assert _calls(profile, "_distinct_values") <= 200
    # No view is exported to Arrow and registered back into DuckDB.
    assert _calls(profile, "register") == 0
