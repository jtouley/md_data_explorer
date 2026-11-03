"""Execution gate for low-confidence QueryPlans. Issue #52."""

from types import SimpleNamespace

import pytest

from clinical_analytics.core.execution_gate import (
    HOLD_FAILURE_REASON,
    apply_planned_query_result,
    classify_planned_query,
    evaluate_execution_gate,
    format_plan_line,
)
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan


@pytest.fixture
def compare_plan() -> QueryPlan:
    """Complete compare-groups plan. Confidence varies per test."""
    return QueryPlan(
        intent="COMPARE_GROUPS",
        metric="ldl",
        group_by="sex",
        filters=[FilterSpec(column="site", operator="IN", value=["A", "B"])],
        confidence=0.3,
    )


def test_execution_gate_confidence_below_threshold_requires_confirmation(compare_plan):
    """A complete valid plan below 0.75 holds for confirmation."""
    decision = evaluate_execution_gate(
        compare_plan,
        threshold=0.75,
        confirmed=False,
        is_complete=True,
        completeness_error="",
        validation_valid=True,
        validation_error="",
    )

    assert decision.allow is False
    assert decision.requires_confirmation is True
    assert decision.failure_reason == HOLD_FAILURE_REASON


def test_execution_gate_confidence_at_threshold_does_not_hold(compare_plan):
    """Confidence 0.75 auto-executes."""
    compare_plan.confidence = 0.75

    decision = evaluate_execution_gate(
        compare_plan,
        threshold=0.75,
        confirmed=False,
        is_complete=True,
        completeness_error="",
        validation_valid=True,
        validation_error="",
    )

    assert decision.allow is True
    assert decision.requires_confirmation is False


def test_execution_gate_confirmed_low_confidence_does_not_hold_when_complete(compare_plan):
    """Confirm releases a complete valid low-confidence plan."""
    decision = evaluate_execution_gate(
        compare_plan,
        threshold=0.75,
        confirmed=True,
        is_complete=True,
        completeness_error="",
        validation_valid=True,
        validation_error="",
    )

    assert decision.allow is True
    assert decision.requires_confirmation is False


def test_execution_gate_confirmed_incomplete_still_refuses(compare_plan):
    """Confirm cannot repair a missing field."""
    decision = evaluate_execution_gate(
        compare_plan,
        threshold=0.75,
        confirmed=True,
        is_complete=False,
        completeness_error="missing group_by",
        validation_valid=True,
        validation_error="",
    )

    assert decision.allow is False
    assert decision.requires_confirmation is False
    assert decision.failure_reason == "missing group_by"


def test_execution_gate_low_confidence_incomplete_does_not_confirm(compare_plan):
    """Incomplete wins over the confidence hold."""
    decision = evaluate_execution_gate(
        compare_plan,
        threshold=0.75,
        confirmed=False,
        is_complete=False,
        completeness_error="missing metric",
        validation_valid=True,
        validation_error="",
    )

    assert decision.allow is False
    assert decision.requires_confirmation is False
    assert decision.failure_reason == "missing metric"


def test_execution_gate_low_confidence_invalid_does_not_confirm(compare_plan):
    """Invalid wins over the confidence hold."""
    decision = evaluate_execution_gate(
        compare_plan,
        threshold=0.75,
        confirmed=False,
        is_complete=True,
        completeness_error="",
        validation_valid=False,
        validation_error="unknown column",
    )

    assert decision.allow is False
    assert decision.requires_confirmation is False
    assert decision.failure_reason == "unknown column"


def test_execution_gate_incomplete_compare_groups_refuses_without_confirmation(compare_plan):
    """High confidence does not create Confirm when the plan is incomplete."""
    compare_plan.confidence = 0.9

    decision = evaluate_execution_gate(
        compare_plan,
        threshold=0.75,
        confirmed=False,
        is_complete=False,
        completeness_error="missing group_by",
        validation_valid=True,
        validation_error="",
    )

    assert decision.allow is False
    assert decision.requires_confirmation is False
    assert decision.failure_reason == "missing group_by"


def test_execution_gate_plan_line_includes_intent_outcome_grouping_filters_and_test(compare_plan):
    """The hold line names intent, outcome, grouping, filters, and test."""
    line = format_plan_line(compare_plan)

    assert line == ("COMPARE_GROUPS · outcome ldl · grouping sex · filters site IN [A, B] · test group comparison")


@pytest.mark.parametrize(
    ("plan", "context_fields", "expected"),
    [
        (
            QueryPlan(intent="FIND_PREDICTORS", metric="outcome", confidence=0.5),
            {"predictor_variables": ["age", "diabetes"]},
            "FIND_PREDICTORS · outcome outcome · grouping none · predictors age, diabetes · filters none"
            " · test logistic regression",
        ),
        (
            QueryPlan(intent="SURVIVAL", group_by="arm", confidence=0.5),
            {"time_variable": "days", "event_variable": "death"},
            "SURVIVAL · outcome none · grouping arm · time days · event death · filters none · test Kaplan–Meier",
        ),
        (
            QueryPlan(
                intent="COUNT",
                filters=[FilterSpec(column="ldl", operator=">", value=130, exclude_nulls=False)],
                scope="filtered",
                confidence=0.5,
            ),
            {},
            "COUNT · outcome none · grouping none · filters ldl > 130 (missing kept) · scope filtered · test count",
        ),
    ],
    ids=["predictors", "survival_time_event", "nulls_kept_and_scope"],
)
def test_format_plan_line_contextInputs_namedSoConfirmationCoversThem(plan, context_fields, expected):
    """Every input that changes the analysis is in the line a confirmation is bound to."""
    # Arrange
    context = SimpleNamespace(
        predictor_variables=context_fields.get("predictor_variables", []),
        time_variable=context_fields.get("time_variable"),
        event_variable=context_fields.get("event_variable"),
    )

    # Act
    line = format_plan_line(plan, context)

    # Assert
    assert line == expected


def test_format_plan_line_predictorsChange_lineChanges():
    """A re-parse that adds covariates must not match the confirmed line."""
    # Arrange
    plan = QueryPlan(intent="FIND_PREDICTORS", metric="outcome", confidence=0.5)

    # Act
    shown = format_plan_line(plan, SimpleNamespace(predictor_variables=["age"]))
    reparsed = format_plan_line(plan, SimpleNamespace(predictor_variables=["age", "diabetes", "hypertension"]))

    # Assert
    assert shown != reparsed


def test_classify_planned_query_hold_refused_error_and_continue():
    """Hold is confirmation-only; incomplete/invalid without run_key is refused."""
    assert (
        classify_planned_query(
            {
                "success": False,
                "requires_confirmation": True,
                "failure_reason": HOLD_FAILURE_REASON,
            }
        )
        == "hold"
    )
    assert (
        classify_planned_query(
            {
                "success": False,
                "requires_confirmation": False,
                "failure_reason": "missing group_by",
            }
        )
        == "refused"
    )
    assert classify_planned_query({"success": False, "run_key": None}) == "refused"
    assert classify_planned_query({"success": False, "run_key": "rk"}) == "execution_error"
    assert classify_planned_query({"success": True}) == "continue"


def test_apply_planned_query_result_clears_force_rerun_on_hold_error_and_continue():
    """The helper returns the label and clears the rerun flag. It does not render."""
    for result, label in (
        (
            {
                "success": False,
                "requires_confirmation": True,
                "failure_reason": HOLD_FAILURE_REASON,
            },
            "hold",
        ),
        (
            {
                "success": False,
                "requires_confirmation": False,
                "failure_reason": "missing metric",
            },
            "refused",
        ),
        ({"success": False, "run_key": "rk"}, "execution_error"),
        ({"success": True, "run_key": "rk"}, "continue"),
    ):
        session = {"force_rerun:v1": True}
        assert apply_planned_query_result(result, session, "v1") == label
        assert session["force_rerun:v1"] is False
