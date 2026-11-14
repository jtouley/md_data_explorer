"""How questions are read: hyphenated names, shared column prefixes, digit-suffixed names,
grouping phrases, "how many A and how many B", "A versus not A", shares within groups, and the
analysis columns the gap check counts. Pure-function tests on synthetic columns and values."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from clinical_analytics.core.cohort_constraints import peel_cohort, unapplied_constraints
from clinical_analytics.core.query_plan import CohortStep, QueryPlan
from clinical_analytics.core.question_vocabulary import complement_tail, join_hyphenated_names, plain_grouping
from clinical_analytics.core.word_accounting import (
    distribution_planned,
    share_planned,
    stem_match,
    tokenize,
    unused_words,
)

_COLUMNS = [
    "patient_id",
    "sex",
    "age_in_cat",
    "ms_type2",
    "year_onset",
    "covid19_confirmed_case",
    "covid19_admission_hospital",
    "covid19_icu_stay",
    "covid19_ventilation",
]
_VALUES: dict[str, list[Any]] = {
    "patient_id": [f"V{i:03d}" for i in range(60)],
    "sex": ["female", "male"],
    "age_in_cat": [1, 2],
    "ms_type2": ["relapsing_remitting", "progressive_MS", "other"],
    "year_onset": [1990, 2001, 2015],
    "covid19_confirmed_case": ["yes", "no"],
    "covid19_admission_hospital": ["yes", "no"],
    "covid19_icu_stay": ["yes", "no"],
    "covid19_ventilation": ["yes", "no"],
}
HOSPITAL = "covid19_admission_hospital"
CONFIRMED = "covid19_confirmed_case"


def _values(column: str) -> list[Any]:
    return _VALUES[column]


# Single-word aliases, as the semantic layer derives them from each column name's words.
_ALIASES = {
    "hospital": HOSPITAL,
    "admission": HOSPITAL,
    "icu": "covid19_icu_stay",
    "ventilation": "covid19_ventilation",
}


def _unused(question: str, plan: QueryPlan) -> list[str]:
    return unused_words(question, plan, _COLUMNS, _values, alias_index=_ALIASES)


def _count_by(group: str, *steps: CohortStep, **share: Any) -> QueryPlan:
    return QueryPlan(intent="COUNT", group_by=group, confidence=0.9, cohort_steps=list(steps), **share)


# ---------------------------------------------------------------------------
# Shared reading rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("had a confirmed covid-19 case", "had a confirmed covid19 case", id="column_word_joined"),
        pytest.param("il-6 over 7", "il-6 over 7", id="no_column_word_kept"),
        pytest.param("aged 18-65", "aged 18-65", id="range_untouched"),
        pytest.param("", "", id="empty"),
    ],
)
def test_joinHyphenatedNames_columnSpellsJoinedWord_joinsOnlyThatWord(question: str, expected: str) -> None:
    # Arrange: columns above

    # Act
    joined = join_hyphenated_names(question, _COLUMNS)

    # Assert
    assert joined == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("admitted to hospital versus not admitted", 3, id="repeat_after_not"),
        pytest.param("admitted to hospital vs not admitted to hospital", 3, id="vs_full_repeat"),
        pytest.param("admitted to hospital versus not admitted to the icu", None, id="new_item_after_not"),
        pytest.param("admitted to hospital versus admitted", None, id="no_negation"),
        pytest.param("versus not", None, id="nothing_repeated"),
    ],
)
def test_complementTail_versusNotRepeat_marksOnlyRepeatedTail(question: str, expected: int | None) -> None:
    # Arrange
    tokens = question.split()

    # Act
    tail = complement_tail(tokens)

    # Assert
    assert tail == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("count broken down by sex", "count by sex", id="broken_down"),
        pytest.param("rates split by site", "rates by site", id="split"),
        pytest.param("stratified by arm", "by arm", id="stratified"),
        pytest.param("broken bones by site", "broken bones by site", id="not_a_phrase"),
    ],
)
def test_plainGrouping_groupingPhrase_readsAsBy(text: str, expected: str) -> None:
    # Arrange: text above

    # Act
    plain = plain_grouping(text)

    # Assert
    assert plain == expected


@pytest.mark.parametrize(
    ("word", "target", "expected"),
    [
        pytest.param("type", "type2", True, id="digit_suffix"),
        pytest.param("covid", "covid19", True, id="two_digit_suffix"),
        pytest.param("ty", "ty2", False, id="too_short"),
        pytest.param("type", "type2a", False, id="not_only_digits"),
        pytest.param("type", "types", True, id="plural_unchanged"),
    ],
)
def test_stemMatch_nameWithAppendedNumber_namedByItsWord(word: str, target: str, expected: bool) -> None:
    # Arrange: words above

    # Act
    matched = stem_match(word, target)

    # Assert
    assert matched is expected


def test_tokenize_brokenDownBy_readsAsBy() -> None:
    # Arrange
    question = "how many patients, broken down by sex?"

    # Act
    tokens = tokenize(question)

    # Assert
    assert tokens == ["how", "many", "patients", "by", "sex"]


# ---------------------------------------------------------------------------
# The peel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("how many patients had covid?", id="prefix_word"),
        pytest.param("how many patients had covid19?", id="prefix_token"),
    ],
)
def test_peelCohort_wordSharedByManyColumns_bindsNone(question: str) -> None:
    # Arrange: four yes/no columns share "covid19"

    # Act
    peeled = peel_cohort(question, _COLUMNS, _values)

    # Assert
    assert peeled.steps == []


def test_peelCohort_sharedPrefixWithOwnWord_bindsThatColumnOnly() -> None:
    # Arrange
    question = "how many patients had a confirmed covid case?"

    # Act
    peeled = peel_cohort(question, _COLUMNS, _values)

    # Assert
    assert [step.column for step in peeled.steps] == [CONFIRMED]


def test_peelCohort_howManyEachValue_groupsByThatColumn() -> None:
    # Arrange
    question = "how many patients are male and how many are female?"

    # Act
    peeled = peel_cohort(question, _COLUMNS, _values)

    # Assert
    assert (peeled.locked_group, peeled.steps, peeled.force_intent) == ("sex", [], "COUNT")


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("how many patients are female and how many are not female?", id="negated_clause"),
        pytest.param("how many women and men?", id="one_clause"),
        pytest.param("how many are female and how many had a confirmed covid case?", id="two_columns"),
    ],
)
def test_peelCohort_howManyClausesNotOneColumnEach_noGroup(question: str) -> None:
    # Arrange: question above

    # Act
    peeled = peel_cohort(question, _COLUMNS, _values)

    # Assert
    assert peeled.locked_group != "sex"


def test_peelCohort_itemVersusNotItem_groupsByTheItem() -> None:
    # Arrange
    question = "how many patients were admitted to hospital for covid19 versus not admitted?"

    # Act
    peeled = peel_cohort(question, _COLUMNS, _values)

    # Assert
    assert (peeled.locked_group, peeled.steps) == (HOSPITAL, [])


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("how many women were admitted to hospital versus not admitted?", id="second_limit"),
        pytest.param("how many patients were not admitted to hospital versus admitted?", id="negated_first"),
    ],
)
def test_peelCohort_versusNotWithOtherLimitOrNegation_noGroup(question: str) -> None:
    # Arrange: question above

    # Act
    peeled = peel_cohort(question, _COLUMNS, _values)

    # Assert
    assert peeled.locked_group != HOSPITAL


def test_peelCohort_shareOfEachGroupNoLimit_countsEachGroup() -> None:
    # Arrange
    question = "what share of patients have each ms type?"

    # Act
    peeled = peel_cohort(question, _COLUMNS, _values)

    # Assert
    assert (peeled.locked_group, peeled.proportion_column, peeled.steps) == ("ms_type2", None, [])


@pytest.mark.parametrize(
    ("question", "group"),
    [
        pytest.param("what percentage of patients in each age group had a confirmed covid19 case?", "age_in_cat"),
        pytest.param("what share of patients were admitted to hospital, broken down by sex?", "sex"),
    ],
)
def test_peelCohort_shareWithinGroups_isRateByGroup(question: str, group: str) -> None:
    # Arrange: question above

    # Act
    peeled = peel_cohort(question, _COLUMNS, _values)

    # Assert
    assert peeled.proportion_column is None
    assert peeled.locked_group == group
    assert peeled.rate_column in {CONFIRMED, HOSPITAL}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_unusedWords_itemVersusNotItem_groupedCountUsesEveryWord() -> None:
    # Arrange
    question = "how many patients were admitted to hospital for covid19 versus not admitted?"

    # Act
    unused = _unused(question, _count_by(HOSPITAL))

    # Assert
    assert unused == []


def test_unusedWords_versusNotWithLimit_notUsed() -> None:
    # Arrange: the plan limits to women, so "not admitted" is no longer one of its groups only
    question = "how many women were admitted to hospital versus not admitted?"

    # Act
    unused = _unused(question, _count_by(HOSPITAL, CohortStep("eq", "sex", "female")))

    # Assert
    assert "not" in unused


@pytest.mark.parametrize(
    ("question", "plan", "expected"),
    [
        pytest.param("how many patients were admitted to hospital?", _count_by("sex"), ["admitted", "hospital"]),
        pytest.param(
            "how many patients were admitted to hospital?",
            QueryPlan(intent="COUNT", confidence=0.9, cohort_steps=[CohortStep("eq", HOSPITAL, "yes")]),
            [],
        ),
        pytest.param("how many patients were admitted?", _count_by("sex"), ["admitted"]),
    ],
)
def test_unusedWords_placeVerb_usedOnlyWithItsPlaceColumn(question: str, plan: QueryPlan, expected: list[str]) -> None:
    # Arrange: question and plan above

    # Act
    unused = _unused(question, plan)

    # Assert
    assert unused == expected


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        pytest.param(_count_by("ms_type2"), True, id="every_group"),
        pytest.param(_count_by("ms_type2", CohortStep("eq", "sex", "female")), False, id="limited"),
        pytest.param(_count_by("sex", proportion_column=HOSPITAL, proportion_value="yes"), False, id="proportion"),
        pytest.param(QueryPlan(intent="COUNT", confidence=0.9), False, id="no_group"),
    ],
)
def test_distributionPlanned_groupedCountWithoutLimits_only(plan: QueryPlan, expected: bool) -> None:
    # Arrange: plan above

    # Act
    planned = distribution_planned(plan)

    # Assert
    assert planned is expected


def test_sharePlanned_proportionBesideGrouping_notComputed() -> None:
    # Arrange: a proportion is one share of everyone; the grouping would be dropped
    plan = _count_by("sex", proportion_column=HOSPITAL, proportion_value="yes")

    # Act
    planned = share_planned(plan)

    # Assert
    assert planned is False


# ---------------------------------------------------------------------------
# The gap check
# ---------------------------------------------------------------------------


def test_unappliedConstraints_predictorsInContext_areApplied() -> None:
    # Arrange
    plan = QueryPlan(intent="FIND_PREDICTORS", metric="outcome", confidence=0.9)
    context = SimpleNamespace(predictor_variables=["age", "hypertension"], time_variable=None, event_variable=None)

    # Act
    gaps = unapplied_constraints(
        "do age and hypertension predict outcome", plan, ["age", "hypertension", "outcome"], context=context
    )

    # Assert
    assert gaps == []


def test_unappliedConstraints_predictorMissingFromContext_isGap() -> None:
    # Arrange
    plan = QueryPlan(intent="FIND_PREDICTORS", metric="outcome", confidence=0.9)
    context = SimpleNamespace(predictor_variables=["age"], time_variable=None, event_variable=None)

    # Act
    gaps = unapplied_constraints(
        "do age and hypertension predict outcome", plan, ["age", "hypertension", "outcome"], context=context
    )

    # Assert
    assert gaps == ["hypertension"]


@pytest.mark.parametrize(
    ("question", "plan"),
    [
        pytest.param("how many patients are male and how many are female?", _count_by("sex"), id="group_values"),
        pytest.param("how many patients by ms type?", _count_by("ms_type2"), id="digit_suffixed_name"),
        pytest.param(
            "how many patients were admitted to hospital versus not admitted?", _count_by(HOSPITAL), id="complement"
        ),
    ],
)
def test_unappliedConstraints_wordsOfTheGrouping_noGap(question: str, plan: QueryPlan) -> None:
    # Arrange: question and plan above

    # Act
    gaps = unapplied_constraints(question, plan, _COLUMNS, values_for=_values)

    # Assert
    assert gaps == []


# ---------------------------------------------------------------------------
# Variables named in a question (predictor questions)
# ---------------------------------------------------------------------------

_PREDICTOR_DATA = {
    "patient_id": [1, 2, 3, 4, 5, 6],
    "outcome": [0, 1, 0, 1, 0, 1],
    "age": [40, 50, 60, 70, 45, 65],
    "sex": ["F", "M", "F", "M", "F", "M"],
    "diabetes": [0, 1, 1, 0, 0, 1],
    "hypertension": [1, 0, 1, 0, 1, 0],
}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("do age and sex predict outcome", {"age", "sex", "outcome"}, id="and_joins_two"),
        pytest.param("do age and diabetes predict outcome", {"age", "diabetes", "outcome"}, id="and_coded"),
        pytest.param("do age and hypertension predict outcome", {"age", "hypertension", "outcome"}, id="three"),
    ],
)
def test_extractVariables_namesJoinedByAnd_eachNamedOnce(
    make_semantic_layer, nl_query_engine_with_cached_model, question: str, expected: set[str]
) -> None:
    # Arrange
    engine = nl_query_engine_with_cached_model(semantic_layer=make_semantic_layer(data=_PREDICTOR_DATA))

    # Act
    variables, _suggestions = engine._extract_variables_from_query(question)

    # Assert
    assert set(variables) == expected


def test_extractFilters_onInsideAWord_noFilter(make_semantic_layer, nl_query_engine_with_cached_model) -> None:
    # Arrange: "hypertension predict outcome" holds "on predict outcome" only inside a word
    engine = nl_query_engine_with_cached_model(semantic_layer=make_semantic_layer(data=_PREDICTOR_DATA))

    # Act
    filters = engine._extract_filters("do age and hypertension predict outcome")

    # Assert
    assert filters == []
