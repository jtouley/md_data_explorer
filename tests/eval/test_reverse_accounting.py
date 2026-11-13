"""Reverse accounting through the headless service: the plan never adds what the question did not ask.

"type 2 diabetes" once ran as diabetes = Type 2 AND Creatinine = 2.2: the "2" was taken by a
stored lab value. Each probe here must be answered with the Polars truth or held; a held
probe must say why in one sentence, and a success must never carry a limit the question did
not ask for.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pytest

from clinical_analytics.eval.headless import actual_value, ask, register_frame, service_for, values_match
from tests.eval.reverse_probes import reverse_frame, reverse_probes

_DATASET = "reverse_labs"
# Simple questions with one clear meaning: these must be answered, not held.
_MUST_ANSWER = {
    "how many patients with type 2 diabetes?",
    "average hba1c in type 2 diabetes",
    "how many patients with stage 3 kidney disease?",
    "how many patients on dialysis for over 2 years?",
    "how many patients with BMI 30 plus?",
}


@lru_cache(maxsize=1)
def _probes() -> list[tuple[str, Any]]:
    frame = reverse_frame()
    register_frame(_DATASET, frame)
    return reverse_probes(frame)


def _value(question: str, result: dict[str, Any]) -> Any:
    kind, stat = ("stat", "mean") if question.startswith("average") else ("count", None)
    return actual_value(kind, stat, None, result["payload"])


def _plan_limits(question: str) -> set[tuple[str, str, Any]]:
    plan = service_for(_DATASET).ask(question, dataset_id=_DATASET, dataset_version="golden").plan
    assert plan is not None
    limits = {(step.column, step.op, step.value) for step in plan.cohort_steps}
    return limits | {(spec.column, spec.operator, spec.value) for spec in plan.filters}


def _plan_columns(question: str) -> set[str]:
    return {column for column, _op, _value in _plan_limits(question)}


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("index", range(8), ids=lambda index: f"reverse{index:02d}")
def test_reverseProbe_question_answeredCorrectlyOrHeldWithReason(index: int) -> None:
    # Arrange
    question, truth = _probes()[index]

    # Act
    result = ask(_DATASET, question)

    # Assert
    if not result["success"]:
        assert question not in _MUST_ANSWER, f"over-held: {question!r}: {result['failure_reason']!r}"
        reason = result["failure_reason"] or ""
        assert result["held"] and result["plan_line"], "a hold shows the exact plan line"
        assert reason.endswith(".") and reason.count(". ") == 0, f"one plain sentence: {reason!r}"
        return
    assert truth is not None, f"must not run: {question!r}"
    assert values_match(truth, _value(question, result)), f"{question!r}: truth {truth!r}"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "question",
    [
        pytest.param("how many patients with type 2 diabetes?", id="type_2_diabetes"),
        pytest.param("average hba1c in type 2 diabetes", id="type_2_diabetes_mean"),
        pytest.param("how many patients with stage 3 kidney disease?", id="stage_3_kidney_disease"),
        pytest.param("how many patients on dialysis for over 2 years?", id="over_2_years"),
    ],
)
def test_reverseProbe_digitInLabel_neverBecomesCreatinineLimit(question: str) -> None:
    # Arrange
    _probes()

    # Act
    columns = _plan_columns(question)

    # Assert
    assert "Creatinine" not in columns


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("how many patients with type 2 diabetes?", {("diabetes", "eq", "Type 2")}, id="type_2_diabetes"),
        pytest.param(
            "how many patients with stage 3 kidney disease?",
            {("kidney disease", "eq", "Stage 3")},
            id="stage_3_kidney_disease",
        ),
        pytest.param(
            "how many patients on dialysis for over 2 years?", {("dialysis_years", "gt", 2.0)}, id="over_2_years"
        ),
        pytest.param("how many patients with BMI 30 plus?", {("BMI", "ge", 30.0)}, id="bmi_30_plus"),
    ],
)
def test_reverseProbe_namedCase_planHoldsExactlyTheAskedLimits(
    question: str, expected: set[tuple[str, str, Any]]
) -> None:
    # Arrange
    _probes()

    # Act
    limits = _plan_limits(question)

    # Assert
    assert limits == expected


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    ("question", "reason"),
    [
        pytest.param(
            "how many patients older than 200?",
            "200 is outside the plausible range for age (0 to 122 years).",
            id="age_above_range_held",
        ),
        pytest.param(
            "how many patients with BMI over 300?",
            "300 is outside the plausible range for BMI (10 to 90 kg/m2).",
            id="bmi_above_range_held",
        ),
        pytest.param("how many patients older than 80?", None, id="age_inside_range_answered"),
    ],
)
def test_reverseProbe_cutoffOutsidePlausibleRange_heldWithReason(question: str, reason: str | None) -> None:
    # Arrange
    _probes()

    # Act
    result = ask(_DATASET, question)

    # Assert
    assert result["failure_reason"] == reason
    assert result["held"] is (reason is not None)
    assert result["success"] is (reason is None)
