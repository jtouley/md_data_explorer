# SPDX-License-Identifier: MIT
"""Structure the plan compiler and the checks must get right on visit-level and one-row-per-stay
tables: sequence numbers, topic words, per-patient quantifiers and summaries, dates by year,
the counting unit on one-row-per-record tables, and coverage of ordinary words.

Made-up glaucoma clinic and rehabilitation unit (tests/model_query/tables.py).
"""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.model_query.engine import answer_question
from clinical_analytics.model_query.plan import ModelReply, compile_plan
from clinical_analytics.model_query.units import UnitError, resolve_unit
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables


def _plan(answer: str, **fields: Any) -> dict[str, Any]:
    base = {"thought": "t", "mapping": [], "answer": answer, "where": None, "group_by": None}
    return base | {"numerator": None, "number": None, "outcome": None, "per_patient": None, "order_by": None} | fields


@pytest.fixture
def eye(make_query_table):
    return make_query_table(tables.eye_visits(), doc_context=tables.EYE_DICTIONARY, id_column="pat_no")


@pytest.fixture
def ask_eye(eye):
    """Factory: answer ``question`` on the glaucoma clinic with the replies given, in order."""

    def _ask(question: str, *replies: dict[str, Any]):
        catalog, context = eye
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


# ---------------------------------------------------------------------------- sequence numbers


def test_build_context_visitNumberWithinPatient_isASequenceNotAKey(eye):
    # Act
    context = eye[1]

    # Assert: visit_no repeats across patients and never within one
    column = context.column("visit_no")
    assert column is not None and column.is_sequence
    assert "visit_no" not in context.encounter_keys


@pytest.mark.parametrize(
    ("question", "reply", "expected"),
    [
        ("How many visits are there?", _plan("count"), 11),
        ("How many visits had drops in use?", _plan("count", where="drops_used = 'Y'"), 6),
    ],
)
def test_answer_question_visitCount_countsRowsNotSequenceValues(ask_eye, question, reply, expected):
    # Act
    answer, _prompts = ask_eye(question, reply)

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis is not None and answer.analysis["total_count"] == expected


# ---------------------------------------------------------------------------- dates by year


def test_answer_question_visitsPerYearOnADate_groupsByTheYear(ask_eye):
    # Act
    answer, _prompts = ask_eye("How many visits per year?", _plan("count", group_by="visit_date"))

    # Assert
    assert answer.status == "answered", answer.reason
    counts = {row["visit_date_year"]: row["count"] for row in answer.analysis["group_counts"]}
    assert counts == {2022: 5, 2023: 4, 2024: 2}


def test_compile_plan_breakdownByADateWithoutAYearOrMonth_isFedBack(eye):
    # Act / Assert
    with pytest.raises(ValueError, match="is a date: group by YEAR"):
        compile_plan(ModelReply(answer="count", group_by="visit_date"), "How many visits by visit date?", eye[1])


# ---------------------------------------------------------------------------- topic words


def test_answer_question_fileTopicWordBecameAFlagFilter_isNeverApplied(ask_eye):
    # Arrange: "glaucoma" names the whole file; the model reads it as the drops flag, then drops it
    first = _plan(
        "count", where="drops_used = 'Y'", mapping=[{"phrase": "glaucoma", "role": "filter", "column": "drops_used"}]
    )

    # Act
    answer, prompts = ask_eye("How many glaucoma visits are in this file?", first, _plan("count"))

    # Assert: the flag goes back to the model; the answer counts every visit
    assert "nothing in the question refers to it" in prompts[1]
    assert answer.status == "answered" and answer.analysis is not None and answer.analysis["total_count"] == 11


# ---------------------------------------------------------------------------- per-patient quantifiers


@pytest.mark.parametrize("where", ["drops_used = 'N'", "drops_used = 'Y'"])
def test_answer_question_neverOnAnyVisit_isAPerPatientAggregate(ask_eye, where):
    # Act: G2 and G4 never had drops in use; G1 had none at its first visit only
    answer, _prompts = ask_eye("How many patients never had drops in use on any visit?", _plan("count", where=where))

    # Assert: whichever way the model wrote the visit condition
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 2


def test_answer_question_underAThresholdOnEveryVisit_isAPerPatientAggregate(ask_eye):
    # Act: pressure under 21 at every visit: G2 and G4 only
    answer, _prompts = ask_eye(
        "How many patients had a pressure under 21 on every visit?", _plan("count", where="iop_mmhg < 21")
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 2


def test_answer_question_thresholdOutsideTheColumnsRange_isNotApplied(ask_eye):
    # Arrange: "under 200" put on a percentage that runs 0 to 100
    reply = _plan("count", where="adherence_pct < 200")

    # Act
    answer, _prompts = ask_eye("How many patients were under 200 on every visit?", reply, reply)

    # Assert
    assert answer.status != "answered" and answer.analysis is None


def test_answer_question_shareThatNeverHad_isRolledUpPerPatient(ask_eye):
    # Act: 2 of 4 patients never had drops in use
    answer, _prompts = ask_eye(
        "What share of patients never had drops in use?", _plan("share", numerator="drops_used = 'Y'")
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["proportion"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------- per-patient summaries


def test_answer_question_averageVisitsPerPatient_isNotAListing(ask_eye):
    # Act: the model groups by the patient key; 11 visits over 4 patients
    answer, _prompts = ask_eye("What is the average number of visits per patient?", _plan("count", group_by="pat_no"))

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["mean"] == pytest.approx(11 / 4)


def test_answer_question_daysBetweenConsecutiveVisits_isAWindowOverEachPatient(ask_eye):
    # Act
    answer, _prompts = ask_eye(
        "What is the median number of days between consecutive visits for the same patient?",
        _plan("median", per_patient="gap_days", number="visit_date"),
    )

    # Assert: 7 gaps (one fewer than visits, per patient)
    assert answer.status == "answered", answer.reason
    assert answer.analysis["non_null_count"] == 7


def test_answer_question_baselineValuePerPatient_isTheFirstVisit(ask_eye):
    # Act: first pressures 25, 18, 30, 20
    answer, _prompts = ask_eye(
        "What is the mean baseline pressure?", _plan("mean", per_patient="first", number="iop_mmhg")
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["mean"] == pytest.approx((25 + 18 + 30 + 20) / 4)


# ---------------------------------------------------------------------------- one row per record


def test_resolve_unit_eventNounOnOneRowPerStay_countsRows(make_query_table):
    # Arrange
    _catalog, context = make_query_table(tables.rehab_stays())

    # Act
    unit = resolve_unit("How many stays were there by year?", context)

    # Assert
    assert (unit.key, unit.noun) == (None, "stay")


def test_resolve_unit_nounCountedByAColumn_isAsked(make_query_table):
    # Arrange: therapy_sessions holds a number of sessions per stay
    _catalog, context = make_query_table(tables.rehab_stays())

    # Act / Assert
    with pytest.raises(UnitError, match="therapy_sessions holds a number of sessions"):
        resolve_unit("How many sessions were there?", context)


# ---------------------------------------------------------------------------- values and coverage


def test_answer_question_emptyStringOnANumberColumn_readsAsRecorded(ask_eye):
    # Act: "among visits that were graded" written as grade != ''
    answer, _prompts = ask_eye(
        "Among visits that were graded, what share had drops in use?",
        _plan("share", where="grade != ''", numerator="drops_used = 'Y'"),
    )

    # Assert: 7 graded visits, 4 with drops in use
    assert answer.status == "answered", answer.reason
    assert answer.analysis["proportion"] == pytest.approx(5 / 7)


@pytest.mark.parametrize(
    "question",
    [
        "How many patients scored a grade of 3?",
        "How many patients enrolled had a grade of 3?",
        "How many kids had a grade of 3?",
    ],
)
def test_answer_question_ordinaryWords_needNoColumn(ask_eye, question):
    # Act
    answer, _prompts = ask_eye(question, _plan("count", where="grade = 3"))

    # Assert
    assert answer.status == "answered", answer.reason


def test_answer_question_daysFromFirstToLastVisit_isASpanPerPatient(ask_eye):
    # Act: spans G1 360, G2 365, G3 760, G4 365 days
    answer, _prompts = ask_eye(
        "What is the median number of days from each patient's first to last visit?",
        _plan("median", per_patient="span_days", number="visit_date"),
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["median"] == pytest.approx(365.0)


@pytest.mark.parametrize(
    ("question", "where", "message"),
    [
        ("How many patients never had drops on every visit?", "drops_used = 'Y'", "both 'never' and 'every time'"),
        ("How many patients never had a pressure of 21?", "iop_mmhg >= 21", "without saying above or below"),
    ],
)
def test_answer_question_quantifierThatCannotBeRead_isAskedNotGuessed(ask_eye, question, where, message):
    # Act
    answer, _prompts = ask_eye(question, _plan("count", where=where))

    # Assert
    assert answer.status == "refused" and message in (answer.reason or "")


def test_answer_question_testWhoseOutcomeIsPinnedByWhere_comparesBothValues(ask_eye):
    # Act: "drops_used = 'Y'" in where would leave one outcome value to compare
    answer, _prompts = ask_eye(
        "Is drop use different between women and men?",
        _plan("compare_groups", outcome="drops_used", group_by="sex", where="drops_used = 'Y'"),
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.reading == "Compared drops_used between the groups of sex."


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many patients had more than one visit?", 4),
        ("How many patients had at least three visits?", 2),
        ("How many patients had 3 or more visits?", 2),
    ],
)
def test_answer_question_patientsWithSeveralVisits_countsPerPatientRows(ask_eye, question, expected):
    # Act
    answer, _prompts = ask_eye(question, _plan("count"))

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == expected


def test_answer_question_visitsWithinDaysOfThePreviousOne_useTheSamePatientsPreviousDate(ask_eye):
    # Act: G1's second visit (59 days) and G3's second (30 days)
    answer, _prompts = ask_eye("How many visits were within 60 days of the patient's previous visit?", _plan("count"))

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 2


def test_answer_question_mostRecentMonthInTheData_limitsToThatMonth(ask_eye):
    # Act: the last visit is in September 2024, the only one that month
    answer, _prompts = ask_eye("How many visits were in the most recent month in the data?", _plan("count"))

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 1


def test_answer_question_perPatientSaidAsABreakdownByThePatientKey_isAcrossPatients(ask_eye):
    # Act: last pressures 19, 17, 19, 19
    answer, _prompts = ask_eye(
        "What is each patient's last pressure, on average?",
        _plan("mean", per_patient="last", number="iop_mmhg", group_by="pat_no", order_by="visit_date"),
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["mean"] == pytest.approx(18.5)


def test_answer_question_amongSubsetWithoutNumerator_splitsPopulationAndEvent(ask_eye):
    # Act: "among graded visits" keeps the population; "had drops in use" is counted
    answer, _prompts = ask_eye(
        "Among graded visits, what share had drops in use?",
        _plan("share", where="grade IS NOT NULL AND drops_used = 'Y'"),
    )

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["proportion"] == pytest.approx(5 / 7)


def test_answer_question_replyThatIsNotJson_isFedBackNotUnavailable(ask_eye):
    # Act: an empty object is what the model call gives for text it could not parse
    answer, prompts = ask_eye("How many visits are there?", {}, _plan("count"))

    # Assert
    assert answer.status == "answered" and len(prompts) == 2


def test_compile_plan_averageOfACountColumn_isAMeanNotAUnitQuestion(make_query_table):
    # Arrange: "sessions" is held by therapy_sessions, which a mean reads directly
    _catalog, context = make_query_table(tables.rehab_stays())

    # Act
    compiled = compile_plan(
        ModelReply(answer="mean", number="therapy_sessions"),
        "What is the average number of sessions per stay?",
        context,
    )

    # Assert
    assert compiled.kind == "statistic"


@pytest.mark.parametrize(
    ("question", "where", "expected"),
    [
        ("How many patients came to more than one visit?", "visit_no > 1", 4),
        ("How many patients had a pressure under 21 at every visit?", "iop_mmhg < 21", 2),
    ],
)
def test_answer_question_countWithAPerPatientField_isStillACountOfPatients(ask_eye, question, where, expected):
    # Act: the model adds per_patient count to a count of patients
    answer, _prompts = ask_eye(question, _plan("count", where=where, per_patient="count"))

    # Assert: never turned into a mean of rows per patient
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == expected


@pytest.fixture
def ask_speech(make_query_table):
    """Factory: answer ``question`` on the speech therapy sessions with the replies given."""
    catalog, context = make_query_table(tables.speech_sessions(), id_column="patient_no")

    def _ask(question: str, *replies: dict[str, Any]):
        queue = list(replies)
        answer = answer_question(
            question,
            context,
            model=lambda *_args: queue.pop(0) if queue else None,
            open_sandbox=lambda: SandboxConnection(catalog),
            examples=[],
        )
        return answer, context

    return _ask


def test_build_context_counterNamedColumnStartingAtOne_isASequence(ask_speech):
    # Act
    _answer, context = ask_speech("How many sessions?", _plan("count"))

    # Assert: session_number is not named as an ID, but numbers each patient's sessions
    column = context.column("session_number")
    assert column is not None and column.is_sequence


def test_answer_question_atLeastNSessionsWrittenOnTheCounter_countsPatientsRows(ask_speech):
    # Act: "session_number >= 3" restates "at least three sessions"
    answer, _context = ask_speech(
        "How many patients attended at least three sessions?", _plan("count", where="session_number >= 3")
    )

    # Assert: S1 (4) and S3 (3)
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 2


@pytest.mark.parametrize("where", ["homework = 'done'", "homework = 'not done'"])
def test_answer_question_alwaysOnANegatedPairColumn_isAFlag(ask_speech, where):
    # Act: only S2 did the homework every session; "not done" is the flag's other value
    question = (
        "How many patients always did their homework?"
        if "not" not in where
        else ("How many patients never did their homework?")
    )
    # A count of 0 is checked once with the model, so the reply is given twice.
    answer, _context = ask_speech(question, _plan("count", where=where), _plan("count", where=where))

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == (1 if "not" not in where else 0)


@pytest.mark.parametrize(
    ("question", "where", "expected"),
    [
        ("How many donors are blood group O?", "blood_group = 'O'", 2),
        ("How many donors gave blood for the first time in 2020?", "first_year = 2020", 3),
    ],
)
def test_answer_question_oneLetterValueAndTopicWordOfANamePart_areAnswered(make_query_table, question, where, expected):
    # Arrange: "blood" is the panel's topic (part of blood_group's name, not a value)
    catalog, context = make_query_table(
        tables.donor_panel(), doc_context=tables.DONOR_PANEL_DICTIONARY, id_column="donor_id"
    )
    queue = [_plan("count", where=where)]

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
