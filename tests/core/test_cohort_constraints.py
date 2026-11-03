"""Cohort binding must never widen or redirect a question.

Pure-function tests on peel_cohort / attach_cohort, plus the semantic layer's fail-closed
handling of cohort steps it cannot apply.
"""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core.cohort_constraints import PeelResult, attach_cohort, peel_cohort
from clinical_analytics.core.query_plan import CohortStep, CohortStepError, FilterSpec, QueryPlan

_COLUMNS = ["patient_id", "age", "gender", "treatment", "ldl", "Statin use 1: Yes 2: No"]
_VALUES: dict[str, list[Any]] = {
    "patient_id": ["a", "b"],
    "age": [40, 70],
    "gender": ["Female", "Male"],
    "treatment": ["drug", "placebo"],
    "ldl": [100.0, 150.0],
    "Statin use 1: Yes 2: No": [1, 2],
}


def _peel(query: str) -> PeelResult:
    return peel_cohort(query, _COLUMNS, lambda column: _VALUES[column])


@pytest.mark.parametrize(
    ("query", "expected_value"),
    [
        pytest.param("How many male patients?", "Male", id="male_is_not_female"),
        pytest.param("How many female patients?", "Female", id="female"),
        pytest.param("average ldl for male patients", "Male", id="male_in_stat_question"),
        pytest.param("how many women?", "Female", id="inflection_women"),
        pytest.param("how many men?", "Male", id="inflection_men"),
    ],
)
def test_peelCohort_sexWord_bindsTheNamedValue(query: str, expected_value: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert peeled.steps == [CohortStep("eq", "gender", expected_value)]


@pytest.mark.parametrize(
    ("query", "metric", "group"),
    [
        pytest.param("compare ldl by treatment", "ldl", "treatment", id="named_group_not_sex"),
        pytest.param("compare age by treatment", "age", "treatment", id="metric_before_by"),
        pytest.param("compare ldl by gender", "ldl", "gender", id="sex_group_when_named"),
        pytest.param("compare ldl between treatment", "ldl", "treatment", id="between"),
    ],
)
def test_peelCohort_compareQuestion_locksNamedMetricAndGroup(query: str, metric: str, group: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert (peeled.locked_metric, peeled.locked_group) == (metric, group)


def test_peelCohort_compareWithoutGroup_doesNotDefaultToSex() -> None:
    # Arrange / Act
    peeled = _peel("compare ldl")

    # Assert
    assert peeled.locked_group is None


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("how many patients by statin use 1 yes 2 no?", id="count_by_coded_yes_no"),
        pytest.param("what is the average ldl by statin use 1 yes 2 no?", id="mean_by_coded_yes_no"),
    ],
)
def test_peelCohort_groupColumn_isNotAlsoFiltered(query: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert [step for step in peeled.steps if step.column == "Statin use 1: Yes 2: No"] == []


def test_peelCohort_statPhraseBeforeBy_locksMetricNotGroup() -> None:
    # Arrange / Act
    peeled = _peel("what is the average ldl by treatment?")

    # Assert
    assert peeled.locked_metric == "ldl"


@pytest.mark.parametrize(
    "intent",
    [pytest.param(None, id="unparsed"), pytest.param("low", id="low_confidence")],
)
def test_attachCohort_nothingPeeled_doesNotFabricateConfidentCount(intent: str | None) -> None:
    # Arrange
    from clinical_analytics.core.nl_query_engine import QueryIntent

    parsed = None if intent is None else QueryIntent(intent_type="DESCRIBE", confidence=0.3)
    peeled = PeelResult(residual="tell me about the weather?")

    # Act
    attached = attach_cohort(parsed, peeled)

    # Assert
    assert attached is parsed
    assert attached is None or attached.confidence < 0.9


def test_attachCohort_peeledSteps_buildCountWithSteps() -> None:
    # Arrange
    peeled = PeelResult(residual="how many patients?", steps=[CohortStep("eq", "gender", "Male")])

    # Act
    attached = attach_cohort(None, peeled)

    # Assert
    assert attached.intent_type == "COUNT"
    assert attached.cohort_steps == [CohortStep("eq", "gender", "Male")]


def test_attachCohort_builtIntent_usesConfiguredPatternConfidence() -> None:
    # Arrange
    from clinical_analytics.core.nl_query_config import TIER_1_PATTERN_MATCH_THRESHOLD

    peeled = PeelResult(residual="how many patients?", steps=[CohortStep("eq", "gender", "Male")])

    # Act
    attached = attach_cohort(None, peeled)

    # Assert
    assert attached.confidence == TIER_1_PATTERN_MATCH_THRESHOLD


def test_attachCohort_correlation_setsPredictorVariables() -> None:
    # Arrange
    peeled = _peel("what is the correlation between age and ldl?")

    # Act
    attached = attach_cohort(None, peeled)

    # Assert
    assert attached.intent_type == "CORRELATIONS"
    assert sorted(attached.predictor_variables) == ["age", "ldl"]


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("remove male patients", id="remove"),
        pytest.param("exclude the placebo group", id="exclude"),
        pytest.param("how many patients without placebo?", id="without"),
        pytest.param("don't want female patients", id="dont_want"),
    ],
)
def test_peelCohort_negatedValue_isNeverBoundAsEquality(query: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert [step for step in peeled.steps if step.op == "eq"] == []


@pytest.mark.parametrize(
    ("query", "expected_code"),
    [
        pytest.param("how many patients on statin?", 1, id="affirmative_yes_code"),
        pytest.param("how many patients not on statin?", 2, id="not_no_code"),
        pytest.param("how many patients excluding statin users?", 2, id="excluding_no_code"),
    ],
)
def test_peelCohort_codedYesNoColumn_polarityFollowsNegation(query: str, expected_code: int) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert peeled.steps == [CohortStep("eq", "Statin use 1: Yes 2: No", expected_code)]


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("remove control group", id="refinement"),
        pytest.param("how many patients", id="plain_count"),
        pytest.param("tell me about the weather", id="unrelated"),
    ],
)
def test_peelCohort_nothingPeeled_residualIsTheQuestionAsAsked(query: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert peeled.residual == query


_LAYER_DATA = {"patient_id": [1, 2, 3, 4], "age": [40, 50, 60, 70], "ldl": [100.0, 120.0, 140.0, 160.0]}


@pytest.mark.parametrize(
    ("plan_kwargs", "reason"),
    [
        pytest.param(
            {"cohort_steps": [CohortStep("eq", "missing_col", 1)]}, "not in this dataset", id="missing_column"
        ),
        pytest.param(
            {"cohort_steps": [CohortStep("ge_quantile", "ldl", quantile=0.9)]}, "not supported", id="unsupported_op"
        ),
        pytest.param({"filters": [FilterSpec("missing_col", "==", 1)]}, "missing_col", id="missing_filter"),
    ],
)
def test_executeQueryPlan_unappliableRestriction_refusesWithReason(
    make_semantic_layer, plan_kwargs: dict[str, Any], reason: str
) -> None:
    # Arrange
    layer = make_semantic_layer(data=_LAYER_DATA)
    plan = QueryPlan(intent="COUNT", confidence=0.95, **plan_kwargs)

    # Act
    result = layer.execute_query_plan(plan)

    # Assert
    assert result["success"] is False
    assert reason in result["failure_reason"]


@pytest.mark.parametrize(
    "step",
    [
        pytest.param(CohortStep("eq", "missing_col", 1), id="missing_column"),
        pytest.param(CohortStep("ge_quantile", "ldl", quantile=0.9), id="unsupported_op"),
    ],
)
def test_applyCohortStep_unappliableStep_raisesInsteadOfSkipping(make_semantic_layer, step: CohortStep) -> None:
    # Arrange
    layer = make_semantic_layer(data=_LAYER_DATA)

    # Act / Assert
    with pytest.raises(CohortStepError):
        layer._apply_cohort_step(layer.get_base_view(), step)


def test_executeQueryPlan_supportedStep_stillAnswers(make_semantic_layer) -> None:
    # Arrange
    layer = make_semantic_layer(data=_LAYER_DATA)
    plan = QueryPlan(intent="COUNT", confidence=0.95, cohort_steps=[CohortStep("gt", "age", 55)])

    # Act
    result = layer.execute_query_plan(plan)

    # Assert
    assert result["success"] is True


def test_peelCohort_columnNameInsideStoredValue_isNotAValueFilter() -> None:
    # Arrange
    columns = ["patient_id", "Current Regimen"]
    values: dict[str, list[Any]] = {"patient_id": ["a"], "Current Regimen": ["regimen_1", "regimen_2"]}

    # Act
    peeled = peel_cohort("what was the most common Current Regimen", columns, lambda column: values[column])

    # Assert
    assert peeled.steps == []
    assert peeled.residual == "what was the most common Current Regimen"
