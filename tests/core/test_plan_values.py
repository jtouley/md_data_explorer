"""Value presence: a plan limit on a value its column does not hold is named, not run."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core.plan_values import absent_value_reason
from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan

_STATIN = "Statin use 1: Yes 2: No"
_DEXA = "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis"
_VALUES: dict[str, list[Any]] = {
    "diabetes": ["Type 1", "Type 2", "None", None],
    "smoker": [0, 1],
    _STATIN: [1, 2],
    _DEXA: [1, 2],
    "age": [40, 55, 70, None],
    "BMI": [22.0, 30.0, 35.0],
    "Creatinine": [0.8, 2.2, 3.3],
    "notes": [None],
}
_COLUMNS = list(_VALUES)


def _reason(plan: QueryPlan) -> str | None:
    return absent_value_reason(plan, _COLUMNS, lambda column: _VALUES[column])


def _count(*steps: CohortStep, filters: list[FilterSpec] | None = None, **fields: Any) -> QueryPlan:
    return QueryPlan(intent="COUNT", confidence=0.9, cohort_steps=list(steps), filters=filters or [], **fields)


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        pytest.param(
            _count(CohortStep("eq", "diabetes", "Type 3")), "I couldn't find 'Type 3' in diabetes.", id="absent_label"
        ),
        pytest.param(
            _count(filters=[FilterSpec("diabetes", "==", "type 2")]),
            "I couldn't find 'type 2' in diabetes.",
            id="wrong_case_matches_no_row",
        ),
        pytest.param(
            _count(filters=[FilterSpec("diabetes", "IN", ["Type 1", "Gestational"])]),
            "I couldn't find 'Gestational' in diabetes.",
            id="one_of_in_list_absent",
        ),
        pytest.param(
            _count(filters=[FilterSpec("diabetes", "!=", "Type 9")]),
            "I couldn't find 'Type 9' in diabetes.",
            id="excluded_value_absent",
        ),
        pytest.param(
            _count(CohortStep("eq", _STATIN, 3)), "I couldn't find '3' in Statin use.", id="code_outside_codebook"
        ),
        pytest.param(_count(CohortStep("eq", "smoker", 2.0)), "I couldn't find '2' in smoker.", id="binary_flag"),
        pytest.param(
            _count(proportion_column="diabetes", proportion_value="Type 4"),
            "I couldn't find 'Type 4' in diabetes.",
            id="proportion_value_absent",
        ),
        pytest.param(
            _count(rate_column="smoker", rate_value=5, group_by="diabetes"),
            "I couldn't find '5' in smoker.",
            id="rate_value_absent",
        ),
        pytest.param(_count(CohortStep("eq", "notes", "x")), "I couldn't find 'x' in notes.", id="all_null_column"),
        pytest.param(
            _count(CohortStep("gt", "age", 200.0)),
            "200 is outside the plausible range for age (0 to 122 years).",
            id="age_cutoff_above_range",
        ),
        pytest.param(
            _count(filters=[FilterSpec("age", "<", -1)]),
            "-1 is outside the plausible range for age (0 to 122 years).",
            id="age_cutoff_below_range",
        ),
        pytest.param(
            _count(CohortStep("ge", "BMI", 300.0)),
            "300 is outside the plausible range for BMI (10 to 90 kg/m2).",
            id="bmi_cutoff_above_range",
        ),
        pytest.param(
            _count(CohortStep("eq", "BMI", 9.5)),
            "9.5 is outside the plausible range for BMI (10 to 90 kg/m2).",
            id="bmi_equality_below_range",
        ),
    ],
)
def test_absentValueReason_valueColumnCannotHold_namesValueAndColumn(plan: QueryPlan, expected: str) -> None:
    # Arrange: plan above

    # Act
    reason = _reason(plan)

    # Assert
    assert reason == expected


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(_count(), id="no_limits"),
        pytest.param(_count(CohortStep("eq", "diabetes", "Type 2")), id="stored_label"),
        pytest.param(_count(filters=[FilterSpec("diabetes", "IN", ["Type 1", "None"])]), id="stored_in_list"),
        pytest.param(_count(CohortStep("eq", _STATIN, 1.0)), id="code_as_float"),
        pytest.param(_count(CohortStep("eq", _STATIN, "2")), id="code_as_text"),
        pytest.param(_count(CohortStep("eq", _DEXA, 3)), id="codebook_code_with_no_rows"),
        pytest.param(_count(CohortStep("gt", "age", 122.0)), id="range_upper_boundary"),
        pytest.param(_count(CohortStep("ge", "age", 0)), id="range_lower_boundary"),
        pytest.param(_count(CohortStep("eq", "age", 41)), id="measure_value_not_stored_but_plausible"),
        pytest.param(_count(CohortStep("gt", "Creatinine", 500.0)), id="measure_without_known_range"),
        pytest.param(_count(CohortStep("not_null", "notes")), id="step_without_value"),
        pytest.param(_count(CohortStep("gt_median", "age")), id="median_cutoff"),
        pytest.param(_count(CohortStep("eq", "unknown_column", "x")), id="column_not_in_data"),
        pytest.param(_count(proportion_column="diabetes", proportion_value="Type 1"), id="stored_proportion_value"),
    ],
)
def test_absentValueReason_valueColumnCanHold_returnsNone(plan: QueryPlan) -> None:
    # Arrange: plan above

    # Act
    reason = _reason(plan)

    # Assert
    assert reason is None
