"""Reverse accounting: every plan element must be asked for by a word of the question.

The forward check (test_word_accounting) asks whether every word is used. This one asks the
opposite: whether every filter column, value, operator, grouping and measure on the plan is
justified by the question, an alias, or the column's own codebook. Pure-function tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan
from clinical_analytics.core.word_accounting import unjustified_elements, unjustified_reason

_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "treatment",
    "Region",
    "Creatinine",
    "diabetes",
    "kidney disease",
    "BMI",
    "dialysis_years",
    "Statin use 1: Yes 2: No",
]
_VALUES: dict[str, list[Any]] = {
    "patient_id": ["S1", "S2", "S3"],
    "age": [40, 55, 70],
    "sex": ["Female", "Male"],
    "treatment": ["drug", "placebo"],
    "Region": ["North", "North East", "South"],
    "Creatinine": [0.8, 2.2, 3.3],
    "diabetes": ["Type 1", "Type 2", "None"],
    "kidney disease": ["Stage 3", "Stage 4", "None"],
    "BMI": [22.0, 30.0, 35.0],
    "dialysis_years": [0, 2, 7],
    "Statin use 1: Yes 2: No": [1, 2],
}
_ALIASES = {"statin": "Statin use 1: Yes 2: No", "kidney": "kidney disease"}


def _unjustified(question: str, plan: QueryPlan) -> list[str]:
    return unjustified_elements(question, plan, _COLUMNS, lambda column: _VALUES[column], alias_index=_ALIASES)


def _count(*steps: CohortStep, filters: list[FilterSpec] | None = None, group_by: str | None = None) -> QueryPlan:
    return QueryPlan(intent="COUNT", confidence=0.9, cohort_steps=list(steps), filters=filters or [], group_by=group_by)


def _mean(metric: str, *steps: CohortStep, group_by: str | None = None) -> QueryPlan:
    return QueryPlan(
        intent="DESCRIBE", metric=metric, confidence=0.9, cohort_steps=list(steps), group_by=group_by, statistic="mean"
    )


TYPE_2 = CohortStep("eq", "diabetes", "Type 2")
CREAT_22 = CohortStep("eq", "Creatinine", 2.2)


@pytest.mark.parametrize(
    ("question", "plan", "expected"),
    [
        pytest.param(
            "how many patients with type 2 diabetes?", _count(TYPE_2, CREAT_22), ["Creatinine = 2.2"], id="type_2"
        ),
        pytest.param(
            "how many patients with stage 3 kidney disease?",
            _count(CohortStep("eq", "kidney disease", "Stage 3"), CohortStep("eq", "Creatinine", 3.3)),
            ["Creatinine = 3.3"],
            id="stage_3",
        ),
        pytest.param(
            "how many patients on dialysis for over 2 years?",
            _count(CohortStep("gt", "dialysis_years", 2.0), CREAT_22),
            ["Creatinine = 2.2"],
            id="over_2_years",
        ),
        pytest.param(
            "how many patients with BMI 30 plus?",
            _count(CohortStep("eq", "BMI", 30.0)),
            ["BMI = 30.0"],
            id="plus_is_not_equality",
        ),
        pytest.param(
            "how many patients with diabetes?",
            _count(filters=[FilterSpec("diabetes", "==", "Type 2"), FilterSpec("Creatinine", ">", 2.0)]),
            ["diabetes == Type 2", "Creatinine > 2.0"],
            id="llm_added_filters",
        ),
        pytest.param(
            "how many patients in the east?",
            _count(CohortStep("eq", "Region", "North East")),
            ["Region = North East"],
            id="partial_value_name",
        ),
        pytest.param(
            "how many women over 65?",
            _count(CohortStep("eq", "sex", "Female"), CohortStep("gt", "age", 56.0)),
            ["age > 56"],
            id="number_not_in_question",
        ),
        pytest.param(
            "how many patients with creatinine over 2?",
            _count(CohortStep("lt", "Creatinine", 2.0)),
            ["Creatinine < 2"],
            id="reversed_operator",
        ),
        pytest.param(
            "how many type 2 diabetes patients on dialysis?",
            _count(TYPE_2, CohortStep("eq", "dialysis_years", 2)),
            ["dialysis_years = 2"],
            id="label_digit_not_reused_as_value",
        ),
        pytest.param(
            "how many dialysis patients with stage 3 or more kidney disease?",
            _count(CohortStep("eq", "kidney disease", "Stage 3"), CohortStep("ge", "dialysis_years", 3.0)),
            ["dialysis_years ≥ 3"],
            id="label_digit_not_reused_as_cutoff",
        ),
        pytest.param(
            "how many patients?",
            _count(CohortStep("eq", "Statin use 1: Yes 2: No", 1)),
            ["Statin use 1: Yes 2: No = 1 (Yes)"],
            id="coded_value_shown_with_its_label",
        ),
        pytest.param("average age", _mean("Creatinine"), ["outcome Creatinine"], id="metric_not_asked"),
        pytest.param("how many patients?", _count(group_by="Region"), ["grouping Region"], id="group_not_asked"),
    ],
)
def test_unjustifiedElements_planAddsWhatQuestionDidNotAsk_namesEachElement(
    question: str, plan: QueryPlan, expected: list[str]
) -> None:
    # Arrange: question and plan above

    # Act
    elements = _unjustified(question, plan)

    # Assert
    assert elements == expected


@pytest.mark.parametrize(
    ("question", "plan"),
    [
        pytest.param("how many patients?", _count(), id="plain_count"),
        pytest.param("how many patients with type 2 diabetes?", _count(TYPE_2), id="type_2_only"),
        pytest.param("how many women?", _count(CohortStep("eq", "sex", "Female")), id="sex_inflection"),
        pytest.param("how many patients over 65?", _count(CohortStep("gt", "age", 65.0)), id="subject_age"),
        pytest.param("how many patients aged 65 or older?", _count(CohortStep("ge", "age", 65.0)), id="aged"),
        pytest.param("how many patients with BMI 30 plus?", _count(CohortStep("ge", "BMI", 30.0)), id="plus_ge"),
        pytest.param("how many placebo patients?", _count(CohortStep("eq", "treatment", "placebo")), id="value_names"),
        pytest.param("how many statin users?", _count(CohortStep("eq", "Statin use 1: Yes 2: No", 1)), id="polarity"),
        pytest.param(
            "how many patients not in the south?",
            _count(filters=[FilterSpec("Region", "!=", "South")]),
            id="negated_value",
        ),
        pytest.param(
            "how many patients in the north or south?",
            _count(filters=[FilterSpec("Region", "IN", ["North", "South"])]),
            id="or_values",
        ),
        pytest.param("average bmi by sex", _mean("BMI", group_by="sex"), id="grouped_mean"),
        pytest.param(
            "compare bmi between drug and placebo",
            QueryPlan(intent="COMPARE_GROUPS", metric="BMI", group_by="treatment", confidence=0.9),
            id="group_named_by_values",
        ),
        pytest.param(
            "how many patients with kidney disease stage 4?",
            _count(CohortStep("eq", "kidney disease", "Stage 4")),
            id="alias_and_value",
        ),
        pytest.param("how many patients with creatinine 2.2?", _count(CREAT_22), id="decimal_value"),
    ],
)
def test_unjustifiedElements_everyElementAsked_returnsNothing(question: str, plan: QueryPlan) -> None:
    # Arrange: question and plan above

    # Act
    elements = _unjustified(question, plan)

    # Assert
    assert elements == []


@pytest.mark.parametrize(
    ("elements", "expected"),
    [
        pytest.param(
            ["Creatinine = 2.2"], "The plan uses Creatinine = 2.2, which your question doesn't ask for.", id="one"
        ),
        pytest.param(
            ["Creatinine = 2.2", "age > 2"],
            "The plan uses Creatinine = 2.2 and age > 2, which your question doesn't ask for.",
            id="two",
        ),
        pytest.param(
            ["a = 1", "b = 2", "c = 3"],
            "The plan uses a = 1, b = 2 and c = 3, which your question doesn't ask for.",
            id="three",
        ),
    ],
)
def test_unjustifiedReason_elements_isOnePlainSentence(elements: list[str], expected: str) -> None:
    # Arrange: elements above

    # Act
    reason = unjustified_reason(elements)

    # Assert
    assert reason == expected


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(_count(TYPE_2, CREAT_22, CohortStep("ge", "BMI", 30.0)), id="cohort_steps"),
        pytest.param(
            _count(filters=[FilterSpec("Creatinine", ">", 2.0), FilterSpec("Region", "IN", ["North", "South"])]),
            id="filters",
        ),
        pytest.param(_count(filters=[FilterSpec("Region", "!=", "South", exclude_nulls=False)]), id="missing_kept"),
    ],
)
def test_unjustifiedElements_everyLimit_isQuotedExactlyFromThePlanLine(plan: QueryPlan) -> None:
    # Arrange
    from clinical_analytics.core.execution_gate import format_plan_line

    plan_line = format_plan_line(plan)

    # Act
    elements = _unjustified("how many patients?", plan)

    # Assert
    assert len(elements) == len(plan.filters) + len(plan.cohort_steps)
    assert all(element in plan_line for element in elements)
