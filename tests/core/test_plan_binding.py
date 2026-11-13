"""A confirmation releases exactly the plan the clinician reviewed, and nothing else.

The plan line must show every execution-relevant field, and the confirmation is bound to a
canonical hash of the full executable plan, not to the display line.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from clinical_analytics.core.execution_gate import (
    NON_EXECUTION_PLAN_FIELDS,
    PLAN_CHANGED_REASON,
    format_plan_line,
    plan_fingerprint,
)
from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan
from clinical_analytics.core.query_service import QueryService

_CONTEXT = SimpleNamespace(predictor_variables=["age"], time_variable="days", event_variable="event")


def _base_plan() -> QueryPlan:
    return QueryPlan(
        intent="COUNT",
        metric="ldl",
        group_by="site",
        filters=[FilterSpec("bmi", ">", 30)],
        cohort_steps=[CohortStep("eq", "sex", "F"), CohortStep("ge_percent_rank", "ldl", quantile=0.9)],
        proportion_column="statin",
        proportion_value=1,
        rate_column="event",
        rate_value=1,
        entity_key="patient_id",
        scope="filtered",
    )


def _changed(value: Any) -> Any:
    """A different value of the same kind, so the change is one an execution would see."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, float):
        return value + 0.05
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "_x"
    if value is None:
        return "set"
    if isinstance(value, list):
        return [*value, value[-1]] if value else ["extra"]
    raise TypeError(f"no change rule for {value!r}")


_SCOPE_CHANGE = {"scope": "all", "intent": "DESCRIBE"}


@pytest.mark.parametrize("name", [field.name for field in dataclasses.fields(QueryPlan)])
def test_planFingerprint_eachQueryPlanField_isHashedUnlessPresentationOnly(name: str) -> None:
    # Arrange
    plan = _base_plan()
    changed = dataclasses.replace(plan, **{name: _SCOPE_CHANGE.get(name, _changed(getattr(plan, name)))})

    # Act
    same = plan_fingerprint(plan, _CONTEXT) == plan_fingerprint(changed, _CONTEXT)

    # Assert: a new QueryPlan field is hashed unless it is explicitly listed as presentation-only
    assert same is (name in NON_EXECUTION_PLAN_FIELDS)


@pytest.mark.parametrize(
    ("owner", "name"),
    [("step", field.name) for field in dataclasses.fields(CohortStep)]
    + [("filter", field.name) for field in dataclasses.fields(FilterSpec)],
)
def test_planFingerprint_eachStepAndFilterField_isHashed(owner: str, name: str) -> None:
    # Arrange
    plan = _base_plan()
    if owner == "step":
        step = plan.cohort_steps[1]
        value = "gt" if name == "op" else _changed(getattr(step, name))
        changed = dataclasses.replace(
            plan, cohort_steps=[plan.cohort_steps[0], dataclasses.replace(step, **{name: value})]
        )
    else:
        spec = plan.filters[0]
        value = "<" if name == "operator" else _changed(getattr(spec, name))
        changed = dataclasses.replace(plan, filters=[dataclasses.replace(spec, **{name: value})])

    # Act / Assert
    assert plan_fingerprint(plan, _CONTEXT) != plan_fingerprint(changed, _CONTEXT)


def test_planFingerprint_stepOrder_isHashed() -> None:
    # Arrange: a percentile cutoff before or after an equality limit selects different patients
    plan = _base_plan()
    reordered = dataclasses.replace(plan, cohort_steps=list(reversed(plan.cohort_steps)))

    # Act / Assert
    assert plan_fingerprint(plan, _CONTEXT) != plan_fingerprint(reordered, _CONTEXT)


@pytest.mark.parametrize("name", ["predictor_variables", "time_variable", "event_variable"])
def test_planFingerprint_contextInputs_areHashed(name: str) -> None:
    # Arrange
    changed = SimpleNamespace(**{**vars(_CONTEXT), name: _changed(getattr(_CONTEXT, name))})

    # Act / Assert
    assert plan_fingerprint(_base_plan(), _CONTEXT) != plan_fingerprint(_base_plan(), changed)


@pytest.mark.parametrize(
    ("plan", "expected_parts"),
    [
        pytest.param(
            QueryPlan(
                intent="DESCRIBE",
                metric="age",
                cohort_steps=[CohortStep("eq", "Statin use 1: Yes 2: No", 1)],
            ),
            ["Statin use 1: Yes 2: No = 1 (Yes)"],
            id="coded_equality_step",
        ),
        pytest.param(
            QueryPlan(intent="COUNT", cohort_steps=[CohortStep("eq", "sex", "F"), CohortStep("gt", "age", 65.0)]),
            ["1) sex = F", "2) age > 65"],
            id="ordered_steps",
        ),
        pytest.param(
            QueryPlan(
                intent="DESCRIBE", metric="ldl", cohort_steps=[CohortStep("ge_percent_rank", "ldl", quantile=0.9)]
            ),
            ["ldl at or above its 90th percentile"],
            id="quantile_step",
        ),
        pytest.param(
            QueryPlan(
                intent="COUNT",
                cohort_steps=[CohortStep("eq", "sex", "F")],
                proportion_column="Statin use 1: Yes 2: No",
                proportion_value=1,
            ),
            ["proportion with Statin use 1: Yes 2: No = 1 (Yes)", "among patients matching the limits"],
            id="proportion_and_denominator",
        ),
        pytest.param(
            QueryPlan(intent="COUNT", group_by="sex", rate_column="event", rate_value=1),
            ["rate of event = 1 within each sex group"],
            id="rate",
        ),
    ],
)
def test_formatPlanLine_executionFields_shownInPlainLanguage(plan: QueryPlan, expected_parts: list[str]) -> None:
    # Arrange / Act
    line = format_plan_line(plan)

    # Assert
    for part in expected_parts:
        assert part in line


def test_formatPlanLine_differentLimits_produceDifferentLines() -> None:
    # Arrange: the reviewer's case, a north-or-west line must not stand for an age limit
    north = QueryPlan(intent="COUNT", cohort_steps=[CohortStep("eq", "site", "North")])
    over_65 = QueryPlan(intent="COUNT", cohort_steps=[CohortStep("gt", "age", 65.0)])

    # Act / Assert
    assert format_plan_line(north) != format_plan_line(over_65)


def _service(mock_semantic_layer, plan: QueryPlan) -> tuple[QueryService, Any]:
    layer = mock_semantic_layer()
    layer._generate_run_key.return_value = "run_key"
    layer._validate_query_plan.return_value = {"valid": True, "issues": []}
    layer.execute_query_plan.return_value = {"success": True, "run_key": "rk", "result": None}
    layer.format_execution_result.return_value = {"type": "count"}
    service = QueryService(layer)
    intent = MagicMock(
        intent_type=plan.intent,
        primary_variable=plan.metric,
        confidence=0.9,
        filters=[],
        predictor_variables=[],
        time_variable=None,
        event_variable=None,
    )
    service.nl_engine.parse_query = MagicMock(return_value=intent)
    service.nl_engine._intent_to_plan = MagicMock(return_value=plan)
    return service, layer


_HELD_PLAN = QueryPlan(intent="COUNT", cohort_steps=[CohortStep("eq", "site", "North")], confidence=0.9)
_OTHER_PLAN = QueryPlan(intent="COUNT", cohort_steps=[CohortStep("gt", "age", 65.0)], confidence=0.9)


def test_ask_confirmWithHashOfAnotherPlan_reHeldWithPlanChanged(mock_semantic_layer) -> None:
    # Arrange: a question whose words are all bound would normally run; a confirmation naming
    # another plan's hash must hold it instead of releasing or silently running it
    held_service, _ = _service(mock_semantic_layer, _HELD_PLAN)
    held = held_service.ask("how many patients in the north or west?", dataset_id="ds")
    service, layer = _service(mock_semantic_layer, _OTHER_PLAN)

    # Act
    result = service.ask(
        "how many patients over 65?",
        dataset_id="ds",
        confirmed=True,
        confirm_plan_line=held.plan_line,
        confirm_plan_hash=held.plan_hash,
    )

    # Assert
    assert result.requires_confirmation is True
    assert result.failure_reason is not None and result.failure_reason.startswith(PLAN_CHANGED_REASON)
    layer.execute_query_plan.assert_not_called()


def test_ask_confirmWithLineButNoHash_notReleased(mock_semantic_layer) -> None:
    # Arrange
    service, layer = _service(mock_semantic_layer, _HELD_PLAN)
    held = service.ask("how many patients in the north or west?", dataset_id="ds")

    # Act
    result = service.ask(
        "how many patients in the north or west?", dataset_id="ds", confirmed=True, confirm_plan_line=held.plan_line
    )

    # Assert
    assert result.requires_confirmation is True
    layer.execute_query_plan.assert_not_called()


def test_ask_confirmWithHashOfSamePlan_released(mock_semantic_layer) -> None:
    # Arrange
    service, layer = _service(mock_semantic_layer, _HELD_PLAN)
    held = service.ask("how many patients in the north or west?", dataset_id="ds")

    # Act
    result = service.ask(
        "how many patients in the north or west?",
        dataset_id="ds",
        confirmed=True,
        confirm_plan_line=held.plan_line,
        confirm_plan_hash=held.plan_hash,
    )

    # Assert
    assert held.requires_confirmation is True
    assert "'or'" in (held.failure_reason or "")
    assert result.requires_confirmation is False
    assert layer.execute_query_plan.call_args.kwargs["confirmed"] is True
