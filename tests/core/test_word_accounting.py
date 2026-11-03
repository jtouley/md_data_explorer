"""Word accounting gate: every content word must be bound by the plan, or the plan holds.

Pure-function tests: a plan, the dataset's columns and stored values, and the question.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan
from clinical_analytics.core.query_service import all_null_statistics_reason
from clinical_analytics.core.word_accounting import hold_reason, number_owner, tokenize, unused_words

_COLUMNS = ["patient_id", "age", "sex", "treatment", "site", "ldl", "bmi", "Statin use 1: Yes 2: No", "outcome"]
_VALUES: dict[str, list[Any]] = {
    "patient_id": ["S1", "S2", "S3"],
    "age": [40, 55, 70],
    "sex": ["F", "M"],
    "treatment": ["drug", "placebo"],
    "site": ["North", "South", "West"],
    "ldl": [90.5, 120.0, 160.2],
    "bmi": [20.1, 27.5, 33.0],
    "Statin use 1: Yes 2: No": [1, 2],
    "outcome": [0, 1],
}
_ALIASES = {"statin": "Statin use 1: Yes 2: No", "statin use 1 yes 2 no": "Statin use 1: Yes 2: No"}
WOMEN = CohortStep("eq", "sex", "F")


def _unused(question: str, plan: QueryPlan) -> list[str]:
    return unused_words(question, plan, _COLUMNS, lambda column: _VALUES[column], alias_index=_ALIASES)


def _count(*steps: CohortStep, filters: list[FilterSpec] | None = None) -> QueryPlan:
    return QueryPlan(intent="COUNT", confidence=0.9, cohort_steps=list(steps), filters=filters or [])


def _mean(metric: str, *steps: CohortStep) -> QueryPlan:
    return QueryPlan(intent="DESCRIBE", metric=metric, confidence=0.9, cohort_steps=list(steps))


@pytest.mark.parametrize(
    ("question", "plan"),
    [
        pytest.param("how many patients?", _count(), id="plain_count"),
        pytest.param("how many women?", _count(WOMEN), id="sex_inflection"),
        pytest.param("how many women over 65?", _count(WOMEN, CohortStep("gt", "age", 65.0)), id="subject_age"),
        pytest.param("how many patients older than 65?", _count(CohortStep("gt", "age", 65.0)), id="older_than"),
        pytest.param("how many patients with bmi over 30?", _count(CohortStep("gt", "bmi", 30.0)), id="named_measure"),
        pytest.param("how many patients 65+?", _count(filters=[FilterSpec("age", ">=", 65)]), id="plus_is_ge"),
        pytest.param(
            "how many patients not on statin?",
            _count(CohortStep("eq", "Statin use 1: Yes 2: No", 2)),
            id="not_bound_to_no_code",
        ),
        pytest.param(
            "how many patients outside the west?",
            _count(filters=[FilterSpec("site", "!=", "West")]),
            id="outside_with_not_equal",
        ),
        pytest.param(
            "how many patients in the north or south?",
            _count(filters=[FilterSpec("site", "IN", ["North", "South"])]),
            id="or_with_in",
        ),
        pytest.param("what is the average ldl in women?", _mean("ldl", WOMEN), id="stat_with_limit"),
        pytest.param(
            "how many patients by statin use 1 yes 2 no?",
            QueryPlan(intent="COUNT", group_by="Statin use 1: Yes 2: No", confidence=0.9),
            id="coded_column_name_phrase",
        ),
        pytest.param(
            "compare ldl between drug and placebo",
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="treatment", confidence=0.9),
            id="between_groups",
        ),
        pytest.param(
            "what proportion of women had the outcome?",
            QueryPlan(intent="COUNT", confidence=0.9, cohort_steps=[WOMEN], proportion_column="outcome"),
            id="proportion_output",
        ),
    ],
)
def test_unusedWords_planBindsEveryWord_returnsNothing(question: str, plan: QueryPlan) -> None:
    # Arrange / Act
    unused = _unused(question, plan)

    # Assert
    assert unused == []


@pytest.mark.parametrize(
    ("question", "plan", "expected"),
    [
        pytest.param("how many women are not on drug?", _count(WOMEN), ["not", "drug"], id="negation_dropped"),
        pytest.param(
            "how many patients outside the west?",
            _count(CohortStep("eq", "site", "West")),
            ["outside"],
            id="outside_as_eq",
        ),
        pytest.param("how many patients 65+?", _count(), ["65", "or more"], id="ge_dropped"),
        pytest.param("how many patients 65+?", _count(CohortStep("gt", "age", 65.0)), ["65", "or more"], id="ge_as_gt"),
        pytest.param("how many patients between 50 and 65?", _count(), ["between", "50", "65"], id="range_dropped"),
        pytest.param("how many women or men?", _count(WOMEN), ["or", "men"], id="or_as_one_value"),
        pytest.param(
            "how many patients in the north or south?",
            _count(CohortStep("eq", "site", "North")),
            ["or", "south"],
            id="or_values",
        ),
        pytest.param("what is the average weight in women?", _count(WOMEN), ["average", "weight"], id="missing_column"),
        pytest.param("how many women smoke?", _count(WOMEN), ["smoke"], id="unknown_verb"),
        pytest.param("show me women", _count(WOMEN), ["show"], id="listing_is_not_a_count"),
        pytest.param("what is the mortality in women?", _count(WOMEN), ["mortality"], id="unknown_measure"),
        pytest.param(
            "what is the average bmi in patients over 30?",
            _mean("bmi", CohortStep("gt", "age", 30.0)),
            ["over", "30"],
            id="number_owner_ambiguous",
        ),
        pytest.param(
            "what is the average ldl over 130?",
            _mean("ldl", CohortStep("gt", "age", 130.0)),
            ["over", "130"],
            id="cutoff_on_wrong_column",
        ),
        pytest.param(
            "median ldl over 70?",
            _mean("ldl", CohortStep("gt", "ldl", 70.0)),
            ["over", "70"],
            id="cutoff_on_summarised_measure_is_ambiguous",
        ),
        pytest.param(
            "what is the average ldl in non-statin users?",
            _mean("ldl", CohortStep("eq", "Statin use 1: Yes 2: No", 1)),
            ["non"],
            id="non_bound_to_yes",
        ),
        pytest.param(
            "what proportion of patients are on drug?",
            _count(CohortStep("eq", "treatment", "drug")),
            ["proportion"],
            id="proportion_without_output",
        ),
        pytest.param(
            "compare ldl by treatment in women",
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="treatment", confidence=0.9),
            ["women"],
            id="limit_dropped_from_compare",
        ),
        pytest.param("how many patients?", _mean("ldl"), ["how", "many"], id="count_words_need_count"),
    ],
)
def test_unusedWords_planMissesAWord_namesIt(question: str, plan: QueryPlan, expected: list[str]) -> None:
    # Arrange / Act
    unused = _unused(question, plan)

    # Assert
    assert unused == expected


def test_tokenize_logicPhrases_becomeSingleTokens() -> None:
    # Arrange / Act
    tokens = tokenize("patients 65+ or at least 50, older than 40, other than men, don't")

    # Assert
    assert tokens == [
        "patients",
        "65",
        "__ge__",
        "or",
        "__ge__",
        "50",
        "__gt_age__",
        "40",
        "__neg__",
        "men",
        "do",
        "not",
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("how many patients with bmi over 30", "bmi", id="named_before"),
        pytest.param("how many patients over 65", "age", id="subject_only"),
        pytest.param("average bmi in patients over 30", None, id="measure_named_earlier"),
        pytest.param("average bmi in patients older than 30", "age", id="older_than_says_age"),
        pytest.param("how many patients aged over 65", "age", id="aged"),
    ],
)
def test_numberOwner_question_bindsTheNamedColumnOrNone(question: str, expected: str | None) -> None:
    # Arrange
    tokens = tokenize(question)
    numeric = {"age", "ldl", "bmi"}

    # Act
    owner = number_owner(tokens, len(tokens) - 1, _COLUMNS, numeric)

    # Assert
    assert owner == expected


def test_holdReason_unusedWords_plainLanguageList() -> None:
    # Arrange / Act
    reason = hold_reason(["not", "__ge__", "smoke"])

    # Assert
    assert reason == "I didn't use: 'not', 'or more', 'smoke'. Check the plan before running it."


@pytest.mark.parametrize(
    ("frame", "plan", "refused"),
    [
        pytest.param(pl.DataFrame({"mean": [None], "median": [None]}), _mean("ldl"), True, id="all_null_stats"),
        pytest.param(pl.DataFrame({"mean": [], "median": []}), _mean("ldl"), True, id="empty_frame"),
        pytest.param(pl.DataFrame({"mean": [1.0], "median": [None]}), _mean("ldl"), False, id="one_stat_present"),
        pytest.param(pl.DataFrame({"count": [0]}), _count(), False, id="zero_count_is_an_answer"),
        pytest.param(
            pl.DataFrame({"sex": ["F", "M"], "mean": [None, None], "n": [0, 0]}),
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex", confidence=0.9),
            True,
            id="grouped_all_null",
        ),
    ],
)
def test_allNullStatisticsReason_result_refusesOnlyWhenEveryStatIsNull(
    frame: pl.DataFrame, plan: QueryPlan, refused: bool
) -> None:
    # Arrange / Act
    reason = all_null_statistics_reason(frame, plan)

    # Assert
    assert (reason is not None) is refused
