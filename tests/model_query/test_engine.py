# SPDX-License-Identifier: MIT
"""The question loop with recorded model replies: answers, feedback, holds and ask-backs."""

from __future__ import annotations

import pytest
import structlog

from clinical_analytics.model_query.engine import LISTING_REASON, MODEL_UNAVAILABLE_REASON
from tests.model_query import tables

_reply = tables.reply


_cond = tables.cond


@pytest.fixture
def ask_cohort(make_query_table, make_model_ask):
    return make_model_ask(make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY))


@pytest.fixture
def ask_clinic(make_query_table, make_model_ask):
    return make_model_ask(make_query_table(tables.shots(), doc_context=tables.SHOTS_DICTIONARY))


def test_answer_question_invertedComparison_isAnsweredWithTheStatedOne(ask_cohort):
    # Arrange: "under 65" written as >= 65
    reply = _reply(filters=[_cond("age", ">=", 65)])

    # Act
    answer, prompts = ask_cohort("How many patients are under 65?", reply)

    # Assert: one call, and the count is of the patients under 65
    expected = tables.cohort().filter(tables.pl.col("age") < 65).height
    assert (answer.status, answer.analysis["total_count"], len(prompts)) == ("answered", expected, 1)
    assert "age in years at surgery under 65 years" in answer.reading


def test_answer_question_droppedNumber_fedBackOnceThenAnswered(ask_cohort):
    # Arrange
    first = _reply()
    second = _reply(filters=[_cond("age", ">", 70)])

    # Act
    answer, prompts = ask_cohort("How many patients are over 70?", first, second)

    # Assert
    assert answer.status == "answered" and len(prompts) == 2
    assert "does not use 70" in prompts[1]


def test_answer_question_meaningStillWrong_holdsAndLogsWouldBeAnswer(ask_cohort):
    # Arrange: both replies drop the "not"
    wrong = _reply(filters=[_cond("died_1y", "is", values=["yes"])])

    # Act
    with structlog.testing.capture_logs() as logs:
        answer, prompts = ask_cohort("How many patients did not die within a year?", wrong, wrong)

    # Assert: held with the reason and the reading; the would-be number logged, no values
    assert answer.status == "refused" and len(prompts) == 2
    assert answer.reason.startswith("I could not be sure I read the 'not'")
    held = [e for e in logs if e["event"] == "model_query_held_answer"]
    assert (
        held
        and held[0]["ran"] is True
        and held[0]["numbers"] == [float(tables.cohort().filter(tables.pl.col("died_1y") == "yes").height)]
    )


def test_answer_question_categoryVariesOverVisits_asksWhichVisit(ask_clinic):
    # Arrange
    reply = _reply(filters=[_cond("wheeze", "is", values=["yes"])])

    # Act
    with structlog.testing.capture_logs() as logs:
        answer, _prompts = ask_clinic("How many patients wheeze?", reply)

    # Assert
    assert answer.status == "clarify"
    assert [label for label, _q in answer.choices] == [
        "At any shot visit (ever)",
        "At every shot visit",
        "At the first shot visit",
        "At the latest shot visit",
    ]
    assert any(e["event"] == "model_query_held_answer" and e["why"] == "which_visit" for e in logs)


def test_answer_question_choiceReAsked_isAnswered(ask_clinic):
    # Arrange
    reply = _reply(filters=[_cond("wheeze", "is", values=["yes"])])

    # Act
    answer, _prompts = ask_clinic("How many patients wheeze? at the latest shot visit", reply)

    # Assert
    assert answer.status == "answered" and answer.reply["which_visit"] == "last"


def test_answer_question_unknownValue_fedBackThenAnswered(ask_cohort):
    # Arrange
    bad = _reply(filters=[_cond("stage", "is", values=["IV"])])
    good = _reply(filters=[_cond("stage", "is", values=["III"])])

    # Act
    answer, prompts = ask_cohort("How many patients have stage III?", bad, good)

    # Assert
    assert answer.status == "answered" and "not a value of stage" in prompts[1]


@pytest.mark.parametrize(
    ("question", "replies", "status", "reason"),
    [
        pytest.param("List the patients who died", (), "refused", LISTING_REASON, id="listing_words"),
        pytest.param(
            "Which ones died?", (_reply(answer="list_patients"),), "refused", LISTING_REASON, id="listing_reply"
        ),
        pytest.param("How many patients?", (), "unavailable", MODEL_UNAVAILABLE_REASON, id="no_model"),
    ],
)
def test_answer_question_refusals_sayWhy(ask_cohort, question, replies, status, reason):
    # Act
    answer, _prompts = ask_cohort(question, *replies)

    # Assert
    assert (answer.status, answer.reason) == (status, reason)


def test_answer_question_followUp_promptCarriesEarlierTurn(ask_cohort):
    # Arrange
    earlier = _reply(group_by="stage")
    reply = _reply(group_by="stage", filters=[_cond("sex", "is", values=["female"])])

    # Act
    answer, prompts = ask_cohort("Now only the women", reply, previous=(("How many patients by stage?", earlier),))

    # Assert
    assert answer.status == "answered" and "How many patients by stage?" in prompts[0]


def test_answer_question_statisticalTest_routedToStatisticsCode(ask_cohort):
    # Arrange
    reply = _reply(answer="odds_ratio", numerator=[_cond("died_1y", "is", values=["yes"])], group_by="sex")

    # Act
    answer, _prompts = ask_cohort("What is the odds ratio of dying within a year for sex?", reply)

    # Assert: computed by analysis.compute, never statistics in SQL
    assert answer.status == "answered" and answer.analysis["type"] == "predictor"
    assert "AVG" not in answer.sql
