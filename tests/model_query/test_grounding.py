# SPDX-License-Identifier: MIT
"""The grounding check: every limit and breakdown of a compiled query is backed by the question."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.model_query.grounding import (
    find_doubts,
    unused_numbers,
    user_choices,
    without_columns,
)
from clinical_analytics.model_query.plan import ModelReply, compile_plan
from tests.model_query import tables


@pytest.fixture
def visits(make_query_table):
    return make_query_table(tables.visits(), doc_context=tables.VISITS_DICTIONARY)[1]


def _doubts(question: str, context: Any, answer: str = "count", **fields: str | None) -> list[str]:
    compiled = compile_plan(ModelReply(answer=answer, **fields), question, context)
    found = find_doubts(
        question,
        context,
        conditions=compiled.conditions,
        group_column=compiled.group_column,
        group_expression=compiled.group_expression,
    )
    return [doubt.kind for doubt in found]


@pytest.mark.parametrize(
    ("question", "fields"),
    [
        pytest.param("How many patients 65 and up tested positive?", {"where": "age_years >= 65"}, id="number_stated"),
        pytest.param("How many patients were screened?", {"where": "screened = 'yes'"}, id="yes_no_flag"),
        pytest.param("How many folks ended up with an antibiotic?", {"where": "abx = 'yes'"}, id="yes_no_paraphrased"),
        pytest.param("How many women tested positive?", {"where": "sex = 'female' AND result = 'positive'"}, id="sex"),
        pytest.param("How many patients came from the North?", {"where": "region = 'North'"}, id="value_stated"),
        pytest.param(
            "How many visits from 2022 through 2023?", {"where": "visit_year BETWEEN 2022 AND 2023"}, id="years"
        ),
        pytest.param("How many adult patients?", {"where": "age_years >= 18"}, id="adult_word"),
        pytest.param("How many patients had a result?", {"where": "result IS NOT NULL"}, id="presence"),
        pytest.param("How many patients were screened in each region?", {"group_by": "region"}, id="group_each"),
        pytest.param("How many men and women tested positive?", {"group_by": "sex"}, id="group_both_values"),
        pytest.param("How many patients tested positive versus negative?", {"group_by": "result"}, id="versus_grouped"),
    ],
)
def test_find_doubts_backedQuery_hasNone(visits, question, fields):
    # Act / Assert
    assert _doubts(question, visits, **fields) == []


@pytest.mark.parametrize(
    ("question", "fields", "kinds"),
    [
        pytest.param("How many patients were screened?", {"where": "age_years > 60"}, ["limit"], id="invented_cutoff"),
        pytest.param(
            "How many patients had a visit?", {"where": "result = 'positive'"}, ["limit"], id="invented_value"
        ),
        pytest.param("How many older patients tested positive?", {"where": "age_years >= 65"}, ["limit"], id="older"),
        pytest.param(
            "What share of patients tested positive?",
            {"answer": "share", "numerator": "result = 'positive'", "group_by": "region"},
            ["group"],
            id="group_not_named",
        ),
        pytest.param("How many patients were screened?", {"group_by": "screened"}, ["no_group"], id="one_number_asked"),
        pytest.param(
            "How many patients were screened versus not screened?",
            {"where": "screened = 'yes'"},
            ["missing_group"],
            id="versus_one_side",
        ),
    ],
)
def test_find_doubts_unbackedPart_isFound(visits, question, fields, kinds):
    # Act / Assert
    assert _doubts(question, visits, **fields) == kinds


def test_find_doubts_inventedLimit_offersWithAndWithoutAsChoices(visits):
    # Arrange
    question = "How many patients were screened?"
    compiled = compile_plan(ModelReply(answer="count", where="age_years > 60"), question, visits)

    # Act
    (doubt,) = find_doubts(question, visits, conditions=compiled.conditions, group_column=None)

    # Assert: both readings offered; each choice's words make the re-asked question say which
    assert doubt.choices == (
        ("Only where age_years > 60", "Only where age_years > 60."),
        ("Without that limit", "Do not limit it by age_years."),
    )
    assert "age_years > 60" in doubt.feedback


def test_find_doubts_choiceWordsReAsked_areBacked(visits):
    # Arrange: the question as the "Only where" choice re-asks it
    question = "How many patients were screened? Only where age_years > 60."

    # Act / Assert
    assert _doubts(question, visits, where="age_years > 60") == []


@pytest.mark.parametrize(
    ("question", "columns", "one_number"),
    [
        ("How many were screened?", frozenset(), False),
        ("How many were screened? Do not limit it by titer.", frozenset({"titer"}), False),
        ("How many were screened? do not limit it by titer, age_years.", frozenset({"titer", "age_years"}), False),
        ("How many were screened? Give one overall number.", frozenset(), True),
    ],
)
def test_user_choices_wordsAddedByAChoice_areRead(question, columns, one_number):
    # Act / Assert
    assert user_choices(question) == (columns, one_number)


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (None, None),
        ("titer > 120", None),
        ("titer > 120 AND screened = 'yes'", "screened = 'yes'"),
        ("screened = 'yes' AND (TITER > 1 OR titer IS NULL)", "screened = 'yes'"),
        ("screened = 'yes'", "screened = 'yes'"),
        ("not sql (((", "not sql ((("),
    ],
)
def test_without_columns_dropsOnlyTheTermsOnThoseColumns(condition, expected):
    # Act / Assert
    assert without_columns(condition, frozenset({"titer"})) == expected


def test_find_doubts_followUpDropsTheBreakdown_isFound(visits):
    # Arrange: "now only the women" after a question broken down by region
    question = "How many patients tested positive in each region? Now only the women."
    compiled = compile_plan(ModelReply(answer="count", where="sex = 'female'"), question, visits)

    # Act
    found = find_doubts(question, visits, conditions=compiled.conditions, group_column=None, kept_group="region")

    # Assert
    assert [doubt.kind for doubt in found] == ["missing_group"]
    assert "keep group_by region" in found[0].feedback


def test_find_doubts_listOfThreeValues_isABreakdown(visits):
    # Act / Assert: "North, South or East" asks for each of them
    assert (
        _doubts("What share of patients are from the North, the South, or the East?", visits, group_by="region") == []
    )


def test_find_doubts_unitConversionInsideTheLimit_onlyTheComparedValueMustBeStated(visits):
    # Arrange: the CASE's unit value and factor are the model's arithmetic, not a stated value
    where = "CASE WHEN titer_unit = 'IU/mL' THEN titer * 1000 ELSE titer END >= 120"

    # Act / Assert
    assert _doubts("How many patients had a titer of at least 120 mIU/mL?", visits, where=where) == []
    assert _doubts("How many patients had a titer drawn?", visits, where=where) == ["limit"]


@pytest.mark.parametrize(
    ("question", "sql", "unused"),
    [
        ("How many patients 70 and older?", "SELECT COUNT(*) FROM unified", {70.0}),
        ("How many patients 70 and older?", "SELECT COUNT(*) FROM unified WHERE age_years >= 70", set()),
        ("How many COVID-19 deaths, with a 95% CI?", "SELECT COUNT(*) FROM unified", set()),
        ("How many died within 30 days?", "SELECT COUNT(*) FROM unified WHERE \"died_30d\" = 'yes'", set()),
        (
            "Visits from 2022 through 2023?",
            "SELECT COUNT(*) FROM unified WHERE visit_year BETWEEN 2022 AND 2023",
            set(),
        ),
        ("How many had fewer than two doses?", "SELECT COUNT(*) FROM unified", set()),
    ],
)
def test_unused_numbers_numberStatedButNotUsed_isFound(question, sql, unused):
    # Act / Assert
    assert unused_numbers(question, sql) == unused
