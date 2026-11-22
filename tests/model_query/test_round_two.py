# SPDX-License-Identifier: MIT
"""Quantifier wording, stated limits, flag links, time operations, stated bins, row nouns and
shares decided by their own limit, on a made-up anticoagulation clinic (one row per INR draw)."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.model_query.context import value_labels
from clinical_analytics.model_query.engine import answer_question
from clinical_analytics.model_query.plan import ModelReply, compile_plan
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables


def _plan(answer: str, **fields: Any) -> dict[str, Any]:
    base = {"thought": "t", "mapping": [], "answer": answer, "where": None, "group_by": None}
    return base | {"numerator": None, "number": None, "outcome": None, "per_patient": None, "order_by": None} | fields


@pytest.fixture
def inr(make_query_table):
    return make_query_table(tables.inr_draws(), doc_context=tables.INR_DICTIONARY, id_column="pt_number")


@pytest.fixture
def ask_inr(inr):
    """Factory: answer ``question`` on the anticoagulation clinic with the replies given, in order."""

    def _ask(question: str, *replies: dict[str, Any]):
        catalog, context = inr
        queue = list(replies)
        prompts: list[str] = []

        def model(_system: str, user: str, _schema: dict[str, Any]) -> Any:
            prompts.append(user)
            return queue.pop(0) if queue else None

        answer = answer_question(
            question, context, model=model, open_sandbox=lambda: SandboxConnection(catalog), examples=[]
        )
        return answer, prompts

    return _ask


@pytest.mark.parametrize(
    "question",
    [
        "How many patients had an INR of 2 or more at every single draw?",
        "How many patients had an INR of 2 or more with every draw?",
        "How many patients had an INR of 2 or more at each of their draws?",
    ],
)
def test_answer_question_everyWording_isAPerPatientAggregate(ask_inr, question):
    # Act: B and D only (A had 1.8 at its first draw, C had 2.1 to 3.8 but one draw not run)
    answer, _prompts = ask_inr(question, _plan("count", where="inr >= 2"))

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 3  # C's unrun draw is unknown, not a failure


def test_answer_question_firstDrawLimitNotStated_isNeverApplied(ask_inr):
    # Arrange: "pre-draw" says when the INR was taken; it does not mean draw 1
    reply = _plan("mean", number="inr", where="draw_no = 1")

    # Act
    answer, _prompts = ask_inr("What is the mean pre-draw INR?", reply, reply)

    # Assert
    assert answer.status != "answered" and answer.analysis is None


def test_answer_question_namedTreatmentLinkedToAFlagBySimilarityOnly_isNotAFilter(ask_inr):
    # Arrange: "anticoagulation patients in the registry" read as the warfarin flag, then dropped
    first = _plan(
        "count",
        where="on_warfarin = 'Y'",
        mapping=[{"phrase": "anticoagulation", "role": "filter", "column": "on_warfarin"}],
    )

    # Act
    answer, prompts = ask_inr("How many anticoagulation patients are in the registry?", first, _plan("count"))

    # Assert
    assert "nothing in the question refers to it" in prompts[1]
    assert answer.status == "answered" and answer.analysis["total_count"] == 4


def test_answer_question_baselineWordDropped_isSentBackThenHeld(ask_inr):
    # Arrange: the mean over every draw, twice
    reply = _plan("mean", number="inr")

    # Act
    answer, prompts = ask_inr("What is the mean baseline INR?", reply, reply)

    # Assert
    assert "per_patient to first" in prompts[1]
    assert answer.status != "answered"


def test_answer_question_baselineAsFirstPerPatient_isAnswered(ask_inr):
    # Act: first INRs 1.8, 2.2, 3.4, 2.5
    answer, _prompts = ask_inr(
        "What is the mean baseline INR?", _plan("mean", number="inr", per_patient="first", order_by="draw_date")
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["mean"] == pytest.approx((1.8 + 2.2 + 3.4 + 2.5) / 4)


@pytest.mark.parametrize("fields", [{"group_by": "age"}, {"where": "age < 50 OR age >= 70"}])
def test_answer_question_statedAgeBands_areTheBreakdown(ask_inr, fields):
    # Act
    answer, _prompts = ask_inr(
        "How many patients by age group: under 50, 50 to 69, 70 and over?", _plan("count", **fields)
    )

    # Assert: A and D under 50, B 50 to 69, C 70 and over
    assert answer.status == "answered", answer.reason
    counts = {row["age_group"]: row["count"] for row in answer.analysis["group_counts"]}
    assert counts == {"under 50": 2, "50 to 69": 1, "70 and over": 1}


def test_compile_plan_numberColumnWithManyValuesAndNoBands_isFedBack(make_query_table):
    # Arrange: a number column with more values than a table can show
    frame = tables.inr_draws().with_columns(tables.pl.int_range(0, 11).alias("weight_g"))
    _catalog, context = make_query_table(frame, id_column="pt_number")

    # Act / Assert
    with pytest.raises(ValueError, match="too many to show"):
        compile_plan(
            ModelReply(answer="count", group_by="weight_g"),
            "How many draws by weight?",
            context.__class__(
                **{
                    **context.__dict__,
                    "columns": tuple(
                        c.__class__(**{**c.__dict__, "distinct": 50}) if c.name == "weight_g" else c
                        for c in context.columns
                    ),
                }
            ),
        )


def test_answer_question_resultsNoun_countsRowsNotPatients(ask_inr):
    # Act
    answer, _prompts = ask_inr("How many INR results are in the file?", _plan("count"))

    # Assert: results of the INR, not the 4 patients
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 10  # the 10 draws with an INR recorded


def test_answer_question_shareWhoseLimitDecidesIt_isSentBack(ask_inr):
    # Arrange: "recorded" written as inr > 0, which already decides most of "above 3"
    first = _plan("share", where="inr > 0", numerator="inr > 3")
    second = _plan("share", where="inr IS NOT NULL", numerator="inr > 3")

    # Act
    answer, prompts = ask_inr("Of the draws where INR was recorded, what percent were above 3?", first, second)

    # Assert: 2 of 10 recorded draws
    assert "a condition on the column the share counts" in prompts[1]
    assert answer.status == "answered" and answer.analysis["proportion"] == pytest.approx(2 / 10)


@pytest.mark.parametrize(
    ("text", "values", "expected"),
    [
        (
            "1 = planned (elective, booked), 2 = unplanned / urgent, 3 = other (see notes)",
            (1, 2, 3),
            (("1", "planned (elective, booked)"), ("2", "unplanned / urgent"), ("3", "other (see notes)")),
        ),
        (
            "Grade: 1 = BMI = 30 or more, 2 = under 30. Blank if unknown.",
            (1, 2),
            (("1", "BMI = 30 or more"), ("2", "under 30")),
        ),
        ("0=No; 1=Yes; 9 = not asked", (0, 1), (("0", "No"), ("1", "Yes"))),
    ],
)
def test_value_labels_labelsWithParenthesesCommasSlashesAndEquals_keptWhole(text, values, expected):
    # Act / Assert
    assert value_labels(text, values) == expected


@pytest.mark.parametrize(
    ("question", "reply", "expected"),
    [
        ("How many draw records are in this file?", _plan("count"), 11),
        ("How many patients had three or more blood draws?", _plan("count", where="draw_no >= 3"), 2),
        ("How many patients had an INR of 2 or more on every blood draw?", _plan("count", where="inr >= 2"), 3),
    ],
)
def test_answer_question_theDictionarysRowNoun_countsAndQuantifiesLikeVisits(
    make_query_table, question, reply, expected
):
    # Arrange: the dictionary says one row per "blood sample"; questions say draw or sample
    doc = tables.INR_DICTIONARY.replace("one row per INR blood draw", "one row per blood draw")
    catalog, context = make_query_table(tables.inr_draws(), doc_context=doc, id_column="pt_number")
    queue = [reply]

    # Act
    answer = answer_question(
        question,
        context,
        model=lambda *_a: queue.pop(0) if queue else None,
        open_sandbox=lambda: SandboxConnection(catalog),
        examples=[],
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == expected


def test_answer_question_countByThePatientKey_countsThePatients(ask_inr):
    # Act: "three or more draws" written as a breakdown by the patient key
    answer, _prompts = ask_inr(
        "How many patients had three or more draws?", _plan("count", where="draw_no >= 3", group_by="pt_number")
    )

    # Assert: A and C, never a list of patients
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 2


@pytest.mark.parametrize("question", ["How many draws were done for referred patients aged 60 or over?"])
def test_check_wordFormsOfAColumnName_meet(make_query_table, question):
    # Arrange: "referred" and a column named referral_source
    from clinical_analytics.model_query import coverage

    frame = tables.inr_draws().with_columns(tables.pl.lit("gp").alias("referral_source"))
    _catalog, context = make_query_table(frame, id_column="pt_number")

    # Act
    found = coverage.check(question, context, used=["referral_source", "age"], values=["gp"], links=[])

    # Assert
    assert found.uncovered == ()
