"""Review rounds of the over-holding fix: the gate may never let a wrong plan run.

Each test uses the reviewer's exact question. The gate does not infer meaning: a negation is
never read as "every other value" or as an inverted comparison, an "or" list is never bound,
a reversed range binds nothing. The plan holds: a narrower answer rate is acceptable, a wrong
answer is not.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from clinical_analytics.core import word_accounting
from clinical_analytics.core.cohort_constraints import peel_cohort
from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan
from clinical_analytics.core.query_service import QueryService
from clinical_analytics.core.question_vocabulary import negation_scope
from clinical_analytics.core.word_accounting import (
    coded_grouping_reason,
    measurement_grouping_reason,
    numeric_grouping_levels,
    tokenize,
    unjustified_elements,
    unused_words,
)

_COLUMNS = ["patient_id", "age", "sex", "treatment", "site", "ldl", "diabetes", "hypertension", "smoker"]
_VALUES: dict[str, list[Any]] = {
    "patient_id": ["a", "b", "c"],
    "age": [35, 50, 65, 84],
    "sex": ["F", "M"],
    "treatment": ["drug", "placebo"],
    "site": ["North", "South", "West"],
    "ldl": [80.5, 130.0, 175.7],
    "diabetes": [0, 1],
    "hypertension": [0, 1],
    "smoker": [0, 1],
}


def _values(column: str) -> list[Any]:
    return _VALUES[column]


def _steps(query: str) -> list[CohortStep]:
    return peel_cohort(query, _COLUMNS, _values).steps


def _count_plan(*steps: CohortStep) -> QueryPlan:
    return QueryPlan(intent="COUNT", cohort_steps=list(steps), confidence=0.9)


# ---------------------------------------------------------------------------
# Finding 1: a negation before two coordinated items
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("how many patients without diabetes and hypertension", [], id="without_x_and_y"),
        pytest.param("how many patients excluding women on placebo", [], id="excluding"),
        pytest.param("how many patients not on placebo with diabetes?", [], id="not_x_with_y"),
        pytest.param("how many patients not on placebo?", [], id="not_x"),
        # "placebo" comes before the negation and is bound as asked; "outcome" is not bound.
        pytest.param(
            "how many patients on placebo without outcome?",
            [CohortStep("eq", "treatment", "placebo")],
            id="negation_after_value",
        ),
    ],
)
def test_peelCohort_negatedQuestion_bindsNothingTheNegationMayReach(query: str, expected: list[CohortStep]) -> None:
    # Arrange: query above

    # Act
    steps = _steps(query)

    # Assert
    assert steps == expected


@pytest.mark.parametrize(
    ("query", "plan", "unused_word"),
    [
        pytest.param(
            "how many patients without diabetes and hypertension",
            _count_plan(CohortStep("eq", "diabetes", 0), CohortStep("eq", "hypertension", 1)),
            "hypertension",
            id="second_item_as_present",
        ),
        pytest.param(
            "how many patients without diabetes and hypertension",
            _count_plan(CohortStep("eq", "diabetes", 0), CohortStep("eq", "hypertension", 0)),
            "hypertension",
            id="second_item_as_absent",
        ),
        pytest.param(
            "how many patients excluding women on placebo",
            _count_plan(CohortStep("ne", "sex", "F"), CohortStep("eq", "treatment", "placebo")),
            "placebo",
            id="excluding_women_on_placebo",
        ),
        pytest.param(
            "how many patients excluding women on placebo",
            QueryPlan(
                intent="COUNT",
                filters=[FilterSpec("sex", "!=", "F"), FilterSpec("treatment", "==", "placebo")],
                confidence=0.9,
            ),
            "placebo",
            id="llm_filters",
        ),
    ],
)
def test_unusedWords_negationBeforeTwoItems_holdsEveryReading(query: str, plan: QueryPlan, unused_word: str) -> None:
    # Arrange: question and plan above

    # Act
    unused = unused_words(query, plan, _COLUMNS, _values)

    # Assert
    assert unused_word in unused


# ---------------------------------------------------------------------------
# Finding 2: a negation the comparison may be outside of
# ---------------------------------------------------------------------------


def test_peelCohort_negationBeforeComparisonPhrase_neverBindsTheAskedDirection() -> None:
    # Arrange
    query = "how many patients excluding those with ldl over 160"

    # Act
    steps = _steps(query)

    # Assert
    assert all(step.column != "ldl" for step in steps)


@pytest.mark.parametrize(
    "op",
    [pytest.param("gt", id="asked_direction"), pytest.param("le", id="inverted_direction")],
)
def test_unusedWords_negationBeforeComparisonPhrase_holdsAnyDirection(op: str) -> None:
    # Arrange
    query = "how many patients excluding those with ldl over 160"
    plan = _count_plan(CohortStep(op, "ldl", 160.0))  # type: ignore[arg-type]

    # Act
    unused = unused_words(query, plan, _COLUMNS, _values)

    # Assert
    assert {"over", "160"} <= set(unused)


def test_unjustifiedElements_negationBeforeComparisonPhrase_askedDirectionNotJustified() -> None:
    # Arrange
    query = "how many patients excluding those with ldl over 160"
    plan = _count_plan(CohortStep("gt", "ldl", 160.0))

    # Act
    unasked = unjustified_elements(query, plan, _COLUMNS, _values)

    # Assert
    assert unasked != []


# ---------------------------------------------------------------------------
# Finding 3: an "or" list after a share verb
# ---------------------------------------------------------------------------


def test_peelCohort_orListAfterShareVerb_neverBuildsAListProportion() -> None:
    # Arrange
    query = "what share of patients are in the north or west"

    # Act
    peeled = peel_cohort(query, _COLUMNS, _values)

    # Assert
    assert peeled.proportion_column is None
    assert not isinstance(peeled.proportion_value, list)
    assert peeled.steps == []


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(
            QueryPlan(intent="COUNT", proportion_column="site", proportion_value=["North", "West"]),  # type: ignore[arg-type]
            id="proportion",
        ),
        pytest.param(
            QueryPlan(intent="COUNT", group_by="sex", rate_column="site", rate_value=["North", "West"]),  # type: ignore[arg-type]
            id="rate",
        ),
    ],
)
def test_validateQueryPlan_listValuedShare_isInvalid(make_semantic_layer, plan: QueryPlan) -> None:
    # Arrange
    layer = make_semantic_layer(
        dataset_name="share_list",
        data={"patient_id": ["P1", "P2", "P3"], "sex": ["F", "M", "F"], "site": ["North", "West", "South"]},
    )

    # Act
    validation = layer._validate_query_plan(plan)

    # Assert
    assert validation["valid"] is False
    assert "one value at a time" in validation["error"]


# ---------------------------------------------------------------------------
# Finding 4: an invalid plan is refused, never offered for confirmation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confirmed", [pytest.param(False, id="first_ask"), pytest.param(True, id="confirmed")])
def test_ask_invalidPlanWithUnusedWord_refusedWithoutConfirmation(mock_semantic_layer, confirmed: bool) -> None:
    # Arrange
    layer = mock_semantic_layer(columns={"ldl": "ldl", "treatment": "treatment", "sex": "sex"})
    layer._generate_run_key.return_value = "run_key"
    layer._validate_query_plan.return_value = {"valid": False, "error": "Cannot use scope='all' with filters."}
    service = QueryService(layer)
    service.nl_engine.parse_query = MagicMock(
        return_value=MagicMock(
            intent_type="COUNT",
            confidence=0.95,
            filters=[],
            predictor_variables=[],
            time_variable=None,
            event_variable=None,
        )
    )
    plan = QueryPlan(intent="COUNT", filters=[FilterSpec("treatment", "==", "placebo")], confidence=0.95)
    service.nl_engine._intent_to_plan = MagicMock(return_value=plan)
    service.nl_engine._distinct_values = MagicMock(return_value=["drug", "placebo"])
    from clinical_analytics.core.execution_gate import plan_fingerprint

    plan_hash = plan_fingerprint(plan, service._intent_to_context(service.nl_engine.parse_query.return_value, plan))

    # Act
    result = service.ask(
        "how many patients on placebo in the hospital",
        dataset_id="ds",
        confirmed=confirmed,
        confirm_plan_hash=plan_hash if confirmed else None,
    )

    # Assert
    assert result.requires_confirmation is False
    assert result.plan_hash is None
    assert result.failure_reason
    layer.execute_query_plan.assert_not_called()


# ---------------------------------------------------------------------------
# Findings 5 and 6: hyphenated and reversed ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("how many patients aged 40-60", id="hyphen"),
        pytest.param("how many patients aged 40 - 60", id="spaced_hyphen"),
        pytest.param("how many patients aged 40–60", id="en_dash"),
        pytest.param("how many patients aged 40 to 60", id="to"),
    ],
)
def test_rangeSpellings_peelAndGate_agreeOnBothEnds(query: str) -> None:
    # Arrange
    expected = [CohortStep("ge", "age", 40.0), CohortStep("le", "age", 60.0)]

    # Act
    steps = _steps(query)
    unused = unused_words(query, _count_plan(*expected), _COLUMNS, _values)

    # Assert
    assert steps == expected
    assert unused == []


def test_peelCohort_reversedRange_bindsNothing() -> None:
    # Arrange
    query = "how many patients between 65 and 50"

    # Act
    steps = _steps(query)

    # Assert
    assert steps == []


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(_count_plan(CohortStep("ge", "age", 65.0), CohortStep("le", "age", 50.0)), id="as_written"),
        pytest.param(_count_plan(CohortStep("ge", "age", 50.0), CohortStep("le", "age", 65.0)), id="normalised"),
    ],
)
def test_unusedWords_reversedRange_holdsEveryPlan(plan: QueryPlan) -> None:
    # Arrange
    query = "how many patients between 65 and 50"

    # Act
    unused = unused_words(query, plan, _COLUMNS, _values)

    # Assert
    assert {"65", "50"} <= set(unused)


# ---------------------------------------------------------------------------
# Finding 7 (and 10): a negated comparison is never inverted, in the peel or the gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "inverted"),
    [
        pytest.param("how many patients exclude patients over 65", CohortStep("le", "age", 65.0), id="exclude"),
        pytest.param("how many patients not over 65", CohortStep("le", "age", 65.0), id="not_over"),
        pytest.param("how many patients not under 40", CohortStep("ge", "age", 40.0), id="not_under"),
        pytest.param("how many patients excluding those over 65", CohortStep("le", "age", 65.0), id="excluding"),
    ],
)
def test_negatedComparison_peelAndGate_neverInvertIt(query: str, inverted: CohortStep) -> None:
    # Arrange
    plan = _count_plan(inverted)

    # Act
    steps = _steps(query)
    unused = unused_words(query, plan, _COLUMNS, _values)
    unasked = unjustified_elements(query, plan, _COLUMNS, _values)

    # Assert
    assert steps == []
    assert str(int(inverted.value)) in unused  # type: ignore[arg-type]
    assert unasked != []


@pytest.mark.parametrize(
    ("question", "target", "expected"),
    [
        pytest.param("how many patients not on placebo", "placebo", "negated", id="direct"),
        pytest.param("how many patients exclude patients over 65", "over", "negated", id="subject_between"),
        pytest.param("how many patients without diabetes and hypertension", "hypertension", "uncertain", id="and"),
        pytest.param("how many patients excluding women on placebo", "placebo", "uncertain", id="phrase"),
        pytest.param("how many non-diabetic women", "women", "uncertain", id="non_prefix_may_reach_on"),
        pytest.param("how many non smokers and diabetics", "diabetics", "uncertain", id="non_then_and"),
        pytest.param("how many statin non-users on placebo", "placebo", "negated", id="postfix_non"),
        pytest.param("how many patients over 65", "over", "clear", id="no_negation"),
        pytest.param("", "", "clear", id="empty"),
    ],
)
def test_negationScope_words_readsWhatTheNegationReaches(question: str, target: str, expected: str) -> None:
    # Arrange
    tokens = tokenize(question)
    at = tokens.index(target) if target else 0

    # Act
    scope = negation_scope(tokens[:at])

    # Assert
    assert scope == expected


def test_unusedWords_noRateWord_buildsOneColumnNamer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    built: list[int] = []
    original = word_accounting._ColumnNamer

    class Counting(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(word_accounting, "_ColumnNamer", Counting)

    # Act
    unused_words("how many patients over 65", _count_plan(CohortStep("gt", "age", 65.0)), _COLUMNS, _values)

    # Assert
    assert len(built) == 1


# ---------------------------------------------------------------------------
# Finding 8: numeric-coded categories get a confirmation path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "row_count", "declared", "expected"),
    [
        pytest.param(list(range(1, 13)), 400, None, "coded", id="site_codes"),
        pytest.param([10001 + step for step in range(40)], 400, None, "coded", id="zip_like_codes"),
        pytest.param([80.5 + step for step in range(40)], 400, None, "measurement", id="non_integer"),
        pytest.param(list(range(300)), 400, None, "measurement", id="high_cardinality"),
        pytest.param(list(range(1, 13)), 400, "continuous", "measurement", id="declared_continuous"),
        pytest.param([80.5 + step for step in range(40)], 400, "categorical", "coded", id="declared_categorical"),
        pytest.param(list(range(1, 11)), 400, None, None, id="ten_levels_boundary"),
        pytest.param(list(range(1, 12)), 400, None, "coded", id="eleven_levels_boundary"),
        pytest.param([None, *range(1, 13)], None, None, "coded", id="nulls_no_row_count"),
        pytest.param([], 400, None, None, id="empty"),
        pytest.param(["a", "b"], 400, None, None, id="text"),
    ],
)
def test_numericGroupingLevels_values_separatesCodesFromMeasurements(
    values: list[Any], row_count: int | None, declared: str | None, expected: str | None
) -> None:
    # Arrange: values above

    # Act
    levels = numeric_grouping_levels(values, row_count, declared)

    # Assert
    assert levels == expected


def test_groupingReasons_siteIdCodes_heldForConfirmationNotRefused() -> None:
    # Arrange
    columns = ["patient_id", "site_id"]
    values = {"patient_id": ["a", "b"], "site_id": list(range(1, 13))}
    plan = QueryPlan(intent="COUNT", group_by="site_id")

    # Act
    refused = measurement_grouping_reason(plan, columns, values.__getitem__, row_count=400)
    held = coded_grouping_reason(plan, columns, values.__getitem__, row_count=400)

    # Assert
    assert refused is None
    assert held is not None and "Confirm the plan" in held


def test_ask_countBySiteIdCodes_holdsThenRunsOnConfirmation(make_semantic_layer) -> None:
    # Arrange
    rows = 240
    layer = make_semantic_layer(
        dataset_name="site_codes",
        data={
            "patient_id": [f"P{index:03d}" for index in range(rows)],
            "site_id": [1 + index % 12 for index in range(rows)],
        },
    )
    service = QueryService(layer)

    # Act
    held = service.ask("count patients by site_id", dataset_id="site_codes")
    released = service.ask(
        "count patients by site_id", dataset_id="site_codes", confirmed=True, confirm_plan_hash=held.plan_hash
    )

    # Assert
    assert held.plan.group_by == "site_id"
    assert held.requires_confirmation is True
    assert "measurement" not in (held.failure_reason or "")
    assert released.requires_confirmation is False
    assert isinstance(released.result, dict) and released.result.get("success") is True


# ---------------------------------------------------------------------------
# Finding 9: an "or" list after "more likely"
# ---------------------------------------------------------------------------


def test_peelCohort_orListAfterMoreLikely_neverLimitsTheCohort() -> None:
    # Arrange
    query = "are smokers more likely to be in the north or west"

    # Act
    peeled = peel_cohort(query, _COLUMNS, _values)

    # Assert
    assert all(step.column != "site" for step in peeled.steps)
    assert peeled.rate_column != "site"
