"""Word accounting gate: every content word must be bound by the plan, or the plan holds.

Pure-function tests: a plan, the dataset's columns and stored values, and the question.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan
from clinical_analytics.core.query_service import all_null_statistics_reason
from clinical_analytics.core.word_accounting import hold_reason, inflection_of, number_owner, tokenize, unused_words

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
    return QueryPlan(intent="DESCRIBE", metric=metric, confidence=0.9, cohort_steps=list(steps), statistic="mean")


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
            QueryPlan(
                intent="DESCRIBE",
                metric="ldl",
                cohort_steps=[CohortStep("gt", "ldl", 70.0)],
                confidence=0.9,
                statistic="median",
            ),
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
        "50",
        "__ge__",
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


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        pytest.param(
            ["recently"],
            "I couldn't use the word 'recently' in the plan; rephrase it with a column or value from your data.",
            id="one_word",
        ),
        pytest.param(
            ["not", "placebo"],
            "I couldn't use the words 'not' and 'placebo' in the plan; rephrase them with columns or values from "
            "your data.",
            id="two_words",
        ),
        pytest.param(
            ["not", "__ge__", "smoke"],
            "I couldn't use the words 'not', 'or more' and 'smoke' in the plan; rephrase them with columns or "
            "values from your data.",
            id="placeholder_shown_as_words",
        ),
    ],
)
def test_holdReason_unusedWords_isOnePlainSentence(words: list[str], expected: str) -> None:
    # Arrange: words above

    # Act
    reason = hold_reason(words)

    # Assert
    assert reason == expected
    assert reason.count(".") == 1


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


_SITE_IN = FilterSpec("site", "IN", ["North", "West"])


@pytest.mark.parametrize(
    ("question", "plan", "expected"),
    [
        pytest.param(
            "how many women not on drug?",
            _count(WOMEN, filters=[FilterSpec("site", "!=", "West")]),
            ["not", "drug"],
            id="negation_operator_on_other_column",
        ),
        pytest.param(
            "how many women in the north or west on drug?",
            _count(WOMEN, filters=[FilterSpec("treatment", "IN", ["drug", "placebo"])]),
            ["north", "or", "west"],
            id="or_operator_on_other_column",
        ),
        pytest.param(
            "compare ldl by sex for patients between 50 and 65",
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex", confidence=0.9),
            ["between", "50", "65"],
            id="between_needs_a_range_not_an_intent",
        ),
        pytest.param(
            "average ldl by site",
            QueryPlan(intent="DESCRIBE", metric="ldl", group_by="sex", confidence=0.9, statistic="mean"),
            ["by", "site"],
            id="by_names_another_grouping",
        ),
        pytest.param(
            "average ldl by site in the north",
            QueryPlan(intent="DESCRIBE", metric="ldl", group_by="site", confidence=0.9, statistic="mean"),
            ["north"],
            id="one_group_value_is_a_dropped_limit",
        ),
        pytest.param(
            "compare ldl between the north and the south",
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="site", confidence=0.9),
            ["north", "south"],
            id="two_of_three_group_values_is_a_dropped_limit",
        ),
        pytest.param(
            "average hba1c above 7",
            QueryPlan(
                intent="DESCRIBE",
                metric="hba1c",
                cohort_steps=[CohortStep("gt", "hba1c", 7.0)],
                confidence=0.9,
                statistic="mean",
            ),
            ["above", "7"],
            id="cutoff_on_only_mention_of_measure",
        ),
    ],
)
def test_unusedWords_logicWordOnTheWrongColumn_isUnused(question: str, plan: QueryPlan, expected: list[str]) -> None:
    # Arrange
    columns = [*_COLUMNS, "hba1c"]
    values = {**_VALUES, "hba1c": [5.1, 6.4, 8.2]}

    # Act
    unused = unused_words(question, plan, columns, lambda column: values[column], alias_index=_ALIASES)

    # Assert
    assert unused == expected


@pytest.mark.parametrize(
    ("question", "plan"),
    [
        pytest.param(
            "how many patients in the north or west?",
            _count(filters=[_SITE_IN]),
            id="or_with_in_on_named_column",
        ),
        pytest.param(
            "compare ldl between drug and placebo",
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="treatment", confidence=0.9),
            id="all_group_values_named",
        ),
        pytest.param(
            "what share of women are not on statin?",
            QueryPlan(
                intent="COUNT",
                cohort_steps=[WOMEN],
                proportion_column="Statin use 1: Yes 2: No",
                proportion_value=2,
                confidence=0.9,
            ),
            id="negation_as_no_code_proportion",
        ),
        pytest.param(
            "what is the average hdl in patients with hba1c above 7?",
            QueryPlan(
                intent="DESCRIBE",
                metric="hdl",
                cohort_steps=[CohortStep("gt", "hba1c", 7.0)],
                confidence=0.9,
                statistic="mean",
            ),
            id="alphanumeric_measure_cutoff",
        ),
    ],
)
def test_unusedWords_logicWordOnItsColumn_isUsed(question: str, plan: QueryPlan) -> None:
    # Arrange
    columns = [*_COLUMNS, "hdl", "hba1c"]
    values = {**_VALUES, "hdl": [40.0, 50.0, 60.0], "hba1c": [5.1, 6.4, 8.2]}

    # Act
    unused = unused_words(question, plan, columns, lambda column: values[column], alias_index=_ALIASES)

    # Assert
    assert unused == []


def test_tokenize_alphanumericMeasure_staysOneToken() -> None:
    # Arrange / Act / Assert
    assert tokenize("average hba1c above 7") == ["average", "hba1c", "above", "7"]


@pytest.mark.parametrize(
    ("word", "target", "expected"),
    [
        pytest.param("counted", "count", True, id="past_tense"),
        pytest.param("associated", "association", True, id="noun_to_participle"),
        pytest.param("predicting", "predict", True, id="gerund"),
        pytest.param("count", "count", True, id="same_word"),
        pytest.param("country", "count", False, id="longer_word_sharing_letters"),
        pytest.param("meaning", "mean", False, id="stem_below_minimum"),
        pytest.param("", "count", False, id="empty"),
    ],
)
def test_inflectionOf_wordAndIntentWord_matchesOnlyInflections(word: str, target: str, expected: bool) -> None:
    # Arrange / Act
    matched = inflection_of(word, target)

    # Assert
    assert matched is expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("how many patients were counted?", [], id="inflection_of_count_is_used"),
        pytest.param("how many patients per country?", ["country", "per"], id="country_is_not_count"),
    ],
)
def test_unusedWords_inflectedIntentWord_usedOnlyWhenItInflects(question: str, expected: list[str]) -> None:
    # Arrange
    plan = _count()

    # Act
    unused = _unused(question, plan)

    # Assert
    assert sorted(unused) == expected


@pytest.mark.parametrize(
    ("extra_columns", "expected"),
    [
        pytest.param({}, [], id="label_word_names_no_column"),
        pytest.param({"statistics": [1, 2]}, ["statistics"], id="label_word_names_a_column"),
    ],
)
def test_unusedWords_testLabelWord_usedOnlyWhenItNamesNoColumn(
    extra_columns: dict[str, list[Any]], expected: list[str]
) -> None:
    # Arrange: "descriptive statistics" is the plan line's label for a DESCRIBE plan. No statin
    # column: "statistics" shares a five-letter stem with "statin" and would name it.
    columns = ["age", "sex", "ldl", *extra_columns]
    values = {**_VALUES, **extra_columns}

    # Act
    unused = unused_words("ldl descriptive statistics", _mean("ldl"), columns, lambda column: values[column])

    # Assert
    assert unused == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("what proportion of patients had the outcome by sex?", ["by", "sex"], id="by"),
        pytest.param("what proportion of patients had the outcome in each sex?", ["each", "sex"], id="in_each"),
        pytest.param("what proportion of patients had the outcome per sex?", ["per", "sex"], id="per"),
    ],
)
def test_unusedWords_proportionBesideGrouping_groupingWordsUnused(question: str, expected: list[str]) -> None:
    # Arrange: execution gives one share of everyone, so the grouping is never applied
    plan = QueryPlan(intent="COUNT", group_by="sex", proportion_column="outcome", proportion_value=1, confidence=0.9)

    # Act
    unused = _unused(question, plan)

    # Assert
    assert unused == expected


def test_unusedWords_rateWithGrouping_groupingWordsUsed() -> None:
    # Arrange: a rate is computed within each group, so "by sex" is applied
    plan = QueryPlan(intent="COUNT", group_by="sex", rate_column="outcome", rate_value=1, confidence=0.9)

    # Act
    unused = _unused("what proportion of patients had the outcome by sex?", plan)

    # Assert
    assert unused == []


def test_unusedWords_countWithMetric_metricWordUnused() -> None:
    # Arrange: a COUNT never reads its metric, so "ldl" would be dropped from the answer
    plan = QueryPlan(intent="COUNT", metric="ldl", confidence=0.9)

    # Act
    unused = _unused("how many patients have an ldl?", plan)

    # Assert
    assert unused == ["ldl"]
