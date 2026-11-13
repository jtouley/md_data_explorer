"""PR #81 review findings 1-6: what the app shows for shares, rates and grouped statistics.

Each test runs a plan through the display path (execute_query_plan -> format_execution_result
-> present_analysis) on a small synthetic frame and reads the claim and table, not the raw
execution frame.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import polars.testing
import pytest

from clinical_analytics.analysis.numeric_text import WHITESPACE, missing_token_expr
from clinical_analytics.core.analysis_types import AnalysisContext
from clinical_analytics.core.query_plan import CohortStep, CohortStepError, QueryPlan
from clinical_analytics.core.query_service import NO_MATCH_REASON, all_null_statistics_reason
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED, present_analysis
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.core.word_accounting import unused_words

# 100 patients: statin taking 30, stopped 30, missing 40; site North 60 / South 40.
_SIZE = 100
_FRAME = pl.DataFrame(
    {
        "patient_id": [f"P{index:03d}" for index in range(_SIZE)],
        "sex": ["F" if index % 2 else "M" for index in range(_SIZE)],
        "site": ["North" if index < 60 else "South" for index in range(_SIZE)],
        "statin": ["taking" if index < 30 else ("stopped" if index < 60 else None) for index in range(_SIZE)],
        "diabetes": ["1: Yes" if index % 3 == 0 else "2: No" for index in range(_SIZE)],
        "age": [30 + index for index in range(_SIZE)],
        "flat": [5.0] * _SIZE,
    }
)


def _present(layer: Any, plan: QueryPlan) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a plan the way QueryService does and return (execution result, presentation)."""
    execution = layer.execute_query_plan(plan, confidence_threshold=0.0)
    context = AnalysisContext()
    context.query_plan = plan
    analysis = layer.format_execution_result(execution, context) if execution.get("success") else {}
    return execution, dict(present_analysis(analysis))


@pytest.fixture
def share_layer(make_semantic_layer):
    return make_semantic_layer(dataset_name="shares", data=_FRAME)


def test_proportion_plan_presentsTheShareNotACount(share_layer) -> None:
    # Arrange
    plan = QueryPlan(intent="COUNT", confidence=0.95, proportion_column="statin", proportion_value="taking")

    # Act
    _execution, shown = _present(share_layer, plan)

    # Assert
    assert shown["table"]["rows"] == [{"proportion": 0.5, "n": 60}]
    assert "Total count" not in shown["claim"]
    assert "50.0%" in shown["claim"]


def test_proportion_missingValues_leftOutOfTheDenominator(share_layer) -> None:
    # Arrange: 30 taking of 60 recorded; 40 patients have no statin value.
    plan = QueryPlan(intent="COUNT", confidence=0.95, proportion_column="statin", proportion_value="taking")

    # Act
    execution, shown = _present(share_layer, plan)

    # Assert
    row = execution["result"].row(0, named=True)
    assert (row["proportion"], row["n"], row["n_rows"]) == (0.5, 60, 100)
    assert "40 excluded for missing data" in shown["claim"]


@pytest.mark.parametrize(
    ("value", "hidden"),
    [
        pytest.param("taking", False, id="large_cells_shown"),
        pytest.param("never", False, id="zero_events_shown"),
    ],
)
def test_proportion_cellSizes_suppressedOnlyWhenSmall(share_layer, value: str, hidden: bool) -> None:
    # Arrange
    plan = QueryPlan(intent="COUNT", confidence=0.95, proportion_column="statin", proportion_value=value)

    # Act
    _execution, shown = _present(share_layer, plan)

    # Assert
    assert (shown["table"]["rows"][0]["proportion"] == SECONDARY_SUPPRESSED) is hidden


def test_proportion_smallNumerator_hidesTheShare(make_semantic_layer) -> None:
    # Arrange: 5 of 100 patients have the value, so share and n would reveal a cell of 5.
    frame = _FRAME.with_columns(pl.Series("rare", ["x" if index < 5 else "y" for index in range(_SIZE)]))
    layer = make_semantic_layer(dataset_name="rare", data=frame)
    plan = QueryPlan(intent="COUNT", confidence=0.95, proportion_column="rare", proportion_value="x")

    # Act
    _execution, shown = _present(layer, plan)

    # Assert
    assert shown["table"]["rows"][0]["proportion"] == SECONDARY_SUPPRESSED
    assert "0.05" not in shown["claim"] and "5.0%" not in shown["claim"]
    assert "not shown" in shown["claim"]


def test_rate_groupedPlan_presentsRatePerGroup(share_layer) -> None:
    # Arrange: statin taking is patients 0-29, all in North.
    plan = QueryPlan(intent="COUNT", confidence=0.95, group_by="site", rate_column="statin", rate_value="taking")

    # Act
    _execution, shown = _present(share_layer, plan)

    # Assert
    assert shown["table"]["columns"] == ["site", "rate", "n"]
    assert shown["table"]["rows"] == [{"site": "North", "rate": 0.5, "n": 60}, {"site": "South", "rate": None, "n": 0}]


def test_rate_smallCellInGroup_hidesThatRate(make_semantic_layer) -> None:
    # Arrange: South has 3 events of 40; North has 30 of 60.
    frame = _FRAME.with_columns(
        pl.Series("event", ["e" if index < 30 or 60 <= index < 63 else "n" for index in range(_SIZE)])
    )
    layer = make_semantic_layer(dataset_name="rates", data=frame)
    plan = QueryPlan(intent="COUNT", confidence=0.95, group_by="site", rate_column="event", rate_value="e")

    # Act
    _execution, shown = _present(layer, plan)

    # Assert
    rates = {row["site"]: row["rate"] for row in shown["table"]["rows"]}
    assert rates == {"North": 0.5, "South": SECONDARY_SUPPRESSED}
    assert "0.075" not in shown["claim"] and "7.5%" not in shown["claim"]


def test_rate_withoutGroup_isRefusedWithReason(share_layer) -> None:
    # Arrange
    plan = QueryPlan(intent="COUNT", confidence=0.95, rate_column="statin", rate_value="taking")

    # Act
    execution, shown = _present(share_layer, plan)

    # Assert
    assert execution["success"] is False
    assert "which groups to compare" in execution["failure_reason"]
    assert shown == {"claim": None, "table": None}


@pytest.mark.parametrize(
    ("statistic", "column"),
    [
        pytest.param("median", "median age", id="median"),
        pytest.param("min", "min age", id="min"),
        pytest.param("max", "max age", id="max"),
        pytest.param(None, "mean age", id="no_statistic_is_mean"),
    ],
)
def test_groupedDescribe_statistic_computesTheStatisticAsked(share_layer, statistic: str | None, column: str) -> None:
    # Arrange
    plan = QueryPlan(intent="DESCRIBE", metric="age", group_by="site", confidence=0.95, statistic=statistic)  # type: ignore[arg-type]
    expected = (
        _FRAME.group_by("site")
        .agg(getattr(pl.col("age"), statistic or "mean")().cast(pl.Float64).alias(column))
        .sort("site")
        .to_dicts()
    )

    # Act
    _execution, shown = _present(share_layer, plan)

    # Assert
    assert shown["table"]["columns"] == ["site", column, "n"]
    assert [{key: row[key] for key in ("site", column)} for row in shown["table"]["rows"]] == expected


@pytest.mark.parametrize(
    ("question", "statistic", "unused"),
    [
        pytest.param("what is the median age by sex?", "median", [], id="median_carried"),
        pytest.param("what is the median age by sex?", "mean", ["median"], id="median_on_mean_plan"),
        pytest.param("what is the median age by sex?", None, ["median"], id="median_no_statistic"),
        pytest.param("what is the maximum age by sex?", "max", [], id="maximum_carried"),
    ],
)
def test_wordAccounting_statisticWord_usedOnlyWhenPlanCarriesIt(
    question: str, statistic: str | None, unused: list[str]
) -> None:
    # Arrange
    plan = QueryPlan(intent="DESCRIBE", metric="age", group_by="sex", confidence=0.95, statistic=statistic)  # type: ignore[arg-type]
    values = {"age": [30, 40, 50], "sex": ["F", "M"], "patient_id": ["P1", "P2", "P3"]}

    # Act
    words = unused_words(question, plan, list(values), lambda column: values[column])

    # Assert
    assert words == unused


def test_wordAccounting_rateWithoutGroup_leavesRateWordUnused() -> None:
    # Arrange
    plan = QueryPlan(intent="COUNT", confidence=0.95, rate_column="statin", rate_value="Yes")
    values = {"statin": ["Yes", "No"], "smoker": ["Yes", "No"], "patient_id": ["P1", "P2"]}

    # Act
    words = unused_words("are smokers more likely to use statin?", plan, list(values), lambda column: values[column])

    # Assert
    assert "likely" in " ".join(words)


@pytest.mark.parametrize(
    ("value", "outcome"),
    [
        pytest.param(1, "refused", id="number_not_stored_as_text"),
        pytest.param(1.0, "refused", id="float_not_stored_as_text"),
        pytest.param("1: Yes", "answered", id="stored_text"),
    ],
)
def test_cohortStep_numberOnTextColumn_refusesPlainlyOrComparesAsText(share_layer, value: Any, outcome: str) -> None:
    # Arrange
    plan = QueryPlan(intent="COUNT", confidence=0.95, cohort_steps=[CohortStep("eq", "diabetes", value)])

    # Act
    execution = share_layer.execute_query_plan(plan, confidence_threshold=0.0)

    # Assert
    if outcome == "refused":
        assert execution["success"] is False
        assert "holds text" in execution["failure_reason"]
        assert "Execution error" not in execution["failure_reason"]
    else:
        assert execution["success"] is True


def test_cohortStep_numberStoredAsText_comparesAsText(make_semantic_layer) -> None:
    # Arrange: codes stored as text ("x" keeps the column text in the CSV).
    frame = pl.DataFrame({"patient_id": ["A", "B", "C", "D"], "code": ["1", "2", "1", "x"]})
    layer = make_semantic_layer(dataset_name="codes", data=frame)
    view = layer.get_base_view()

    # Act
    filtered = layer._apply_cohort_step(view, CohortStep("eq", "code", 1)).to_polars()

    # Assert
    assert filtered.height == 2


def test_cohortStep_orderComparisonOnText_raisesCohortStepError(share_layer) -> None:
    # Arrange
    view = share_layer.get_base_view()

    # Act / Assert
    with pytest.raises(CohortStepError, match="holds text"):
        share_layer._apply_cohort_step(view, CohortStep("gt", "site", 3.0))


@pytest.mark.parametrize(
    ("frame", "plan", "expected"),
    [
        pytest.param(
            pl.DataFrame({"corr": [None], "n_rows": [50], "n_pairs": [50]}),
            QueryPlan(intent="CORRELATIONS", metric="age", group_by="flat"),
            "same value",
            id="constant_column",
        ),
        pytest.param(
            pl.DataFrame({"corr": [None], "n_rows": [5], "n_pairs": [1]}),
            QueryPlan(intent="CORRELATIONS", metric="age", group_by="flat"),
            "fewer than 2",
            id="one_pair",
        ),
        pytest.param(
            pl.DataFrame({"corr": [None], "n_rows": [0], "n_pairs": [0]}),
            QueryPlan(intent="CORRELATIONS", metric="age", group_by="flat"),
            NO_MATCH_REASON,
            id="nobody_matched",
        ),
        pytest.param(
            pl.DataFrame({"site": ["North"], "median": [None], "n": [0], "n_rows": [12]}),
            QueryPlan(intent="DESCRIBE", metric="ldl", group_by="site", statistic="median"),
            "missing 'ldl'",
            id="metric_all_missing",
        ),
        pytest.param(
            pl.DataFrame({"proportion": [None], "events": [0], "n": [0], "n_rows": [30]}),
            QueryPlan(intent="COUNT", proportion_column="statin", proportion_value="Yes"),
            "'statin' recorded",
            id="share_column_all_missing",
        ),
    ],
)
def test_allNullStatisticsReason_patientsMatched_givesTheRealCause(
    frame: pl.DataFrame, plan: QueryPlan, expected: str
) -> None:
    # Arrange / Act
    reason = all_null_statistics_reason(frame, plan)

    # Assert
    assert reason is not None
    assert expected in reason
    if expected != NO_MATCH_REASON:
        assert "No patients matched" not in reason


# A numeric-text lab column: "2" 30 patients, "3" 30, and 40 missing written as tokens or null.
_TOKEN_FRAME = pl.DataFrame(
    {
        "patient_id": [f"P{index:03d}" for index in range(_SIZE)],
        "site": ["North" if index % 2 else "South" for index in range(_SIZE)],
        "grade": [
            ("2" if index < 30 else "3") if index < 60 else (" N/A", ".", "NA", None)[index % 4]
            for index in range(_SIZE)
        ],
        # A text category whose level "None" is a real answer, not a missing token.
        "diabetes": ["None" if index < 40 else "Type 2" for index in range(_SIZE)],
    }
)


@pytest.fixture
def token_layer(make_semantic_layer: Callable[..., SemanticLayer]) -> SemanticLayer:
    return make_semantic_layer(dataset_name="tokens", data=_TOKEN_FRAME)


@pytest.mark.parametrize("entity_key", [None, "patient_id"], ids=["rows", "patients"])
def test_proportion_numericTextMissingTokens_leftOutOfTheDenominator(
    token_layer: SemanticLayer, entity_key: str | None
) -> None:
    # Arrange: 30 of the 60 recorded grades are "2"; "N/A", "." and "NA" are missing, as null is.
    token_layer.count_entity_key = entity_key
    plan = QueryPlan(intent="COUNT", confidence=0.95, proportion_column="grade", proportion_value="2")

    # Act
    execution, shown = _present(token_layer, plan)

    # Assert
    row = execution["result"].row(0, named=True)
    assert (row["events"], row["n"], row["n_rows"]) == (30, 60, 100)
    assert row["proportion"] == pytest.approx(0.5)
    assert "40 excluded for missing data" in shown["claim"]


def test_rate_numericTextMissingTokens_leftOutOfEachGroupsDenominator(token_layer: SemanticLayer) -> None:
    # Arrange: each site has 30 recorded grades, 15 of them "2".
    plan = QueryPlan(intent="COUNT", confidence=0.95, rate_column="grade", rate_value="2", group_by="site")

    # Act
    execution, _shown = _present(token_layer, plan)

    # Assert
    rows = execution["result"].select("site", "events", "n").sort("site")
    expected = pl.DataFrame({"site": ["North", "South"], "events": [15, 15], "n": [30, 30]})
    pl.testing.assert_frame_equal(rows, expected, check_dtypes=False)


def test_proportion_textCategoryNamedNone_staysRecorded(token_layer: SemanticLayer) -> None:
    # Arrange: "None" is a level of a text category, not numeric text, so it is recorded.
    plan = QueryPlan(intent="COUNT", confidence=0.95, proportion_column="diabetes", proportion_value="Type 2")

    # Act
    execution, _shown = _present(token_layer, plan)

    # Assert
    row = execution["result"].row(0, named=True)
    assert (row["events"], row["n"]) == (60, 100)


@pytest.mark.parametrize("entity_key", [None, "patient_id"], ids=["rows", "patients"])
@pytest.mark.parametrize("value", ["N/A", "n/a ", ".", "NA"])
@pytest.mark.parametrize("share", ["proportion", "rate"])
def test_share_valueIsMissingToken_refused(
    token_layer: SemanticLayer, entity_key: str | None, value: str, share: str
) -> None:
    # Arrange: "N/A" is missing in a numeric-text column, so it is outside n; its share would
    # count events outside the denominator (40 of 60, or more than 1).
    token_layer.count_entity_key = entity_key
    asked = {"proportion_column": "grade", "proportion_value": value}
    if share == "rate":
        asked = {"rate_column": "grade", "rate_value": value, "group_by": "site"}
    plan = QueryPlan(intent="COUNT", confidence=0.95, **asked)

    # Act
    execution, _shown = _present(token_layer, plan)

    # Assert
    assert execution["success"] is False
    assert "missing" in execution["failure_reason"]


def test_proportion_eventsNeverOutsideN(token_layer: SemanticLayer) -> None:
    # Arrange: the share of a recorded value; its events are recorded rows only.
    plan = QueryPlan(intent="COUNT", confidence=0.95, proportion_column="grade", proportion_value="3")

    # Act
    execution, _shown = _present(token_layer, plan)

    # Assert
    row = execution["result"].row(0, named=True)
    assert row["events"] <= row["n"]
    assert 0 <= row["proportion"] <= 1


def test_whitespace_isExactlyWhatPolarsStrips() -> None:
    # Arrange: every code point outside the surrogates, wrapped on both sides.
    points = [chr(code) for code in range(0x30000) if not 0xD800 <= code < 0xE000]
    wrapped = pl.Series("text", [f"{point}x{point}" for point in points])

    # Act
    stripped = wrapped.str.strip_chars()

    # Assert
    stripped_points = {point for point, text in zip(points, stripped.to_list(), strict=True) if text == "x"}
    assert stripped_points == set(WHITESPACE)


@pytest.mark.parametrize("entity_key", [None, "patient_id"], ids=["rows", "patients"])
@pytest.mark.parametrize(
    "token",
    ["N/A ", "N/A\u00a0", "\u3000n/a", "\u2007.\u2007", "NA\u202f"],
    ids=["space", "nbsp", "ideographic", "figure_space", "narrow_nbsp"],
)
def test_proportion_missingTokenUnicodeWhitespace_sameNAsPolars(
    make_semantic_layer: Callable[..., SemanticLayer], entity_key: str | None, token: str
) -> None:
    # Arrange: 60 numbers and 40 tokens padded with Unicode whitespace (Excel exports use NBSP).
    frame = _TOKEN_FRAME.with_columns(
        pl.when(pl.col("grade").is_in(["2", "3"])).then(pl.col("grade")).otherwise(pl.lit(token)).alias("grade")
    )
    layer = make_semantic_layer(dataset_name="unicode_tokens", data=frame)
    layer.count_entity_key = entity_key
    plan = QueryPlan(intent="COUNT", confidence=0.95, proportion_column="grade", proportion_value="2")
    polars_n = frame.filter(pl.col("grade").is_not_null() & ~missing_token_expr("grade")).height

    # Act
    execution, _shown = _present(layer, plan)

    # Assert
    assert execution["result"].row(0, named=True)["n"] == polars_n == 60
