# SPDX-License-Identifier: MIT
"""The checks that run on every compiled plan before a number is shown: the counting unit from the
question's head noun, dictionary codes, coverage and links, measure and per-patient ambiguity.

All on a made-up allergy clinic table (tests/model_query/tables.py), one row per visit.
"""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.model_query import coverage
from clinical_analytics.model_query.context import dictionary_entries, value_labels
from clinical_analytics.model_query.engine import answer_question
from clinical_analytics.model_query.plan import ModelReply, compile_plan
from clinical_analytics.model_query.units import UnitError, head_noun, plural, resolve_unit, singular
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables


def _plan(answer: str, **fields: Any) -> dict[str, Any]:
    base = {"thought": "t", "mapping": [], "answer": answer, "where": None, "group_by": None}
    return base | {"numerator": None, "number": None, "outcome": None} | fields


@pytest.fixture
def clinic(make_query_table):
    """(catalog, context) of the clinic table with its dictionary; pt_id is the patient key."""
    return make_query_table(tables.clinic_visits(), doc_context=tables.CLINIC_DICTIONARY, id_column="pt_id")


@pytest.fixture
def ask_clinic(clinic):
    """Factory: answer ``question`` on the clinic table with the replies given, in order."""

    def _ask(question: str, *replies: dict[str, Any]):
        catalog, context = clinic
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


# ---------------------------------------------------------------------------- A. the counting unit


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many patients had shots?", ("patients", "patients")),
        ("How many unique patients had surgery?", ("patients", "patients")),
        ("What % of visits had shots?", ("events", "visits")),
        ("Number of clinic visits each year?", ("events", "visits")),
        ("How many surgeries were done in 2024?", ("events", "surgeries")),
        ("Average age of our pts?", ("patients", "pts")),
        ("How many were positive?", None),
    ],
)
def test_head_noun_question_givesTheCountedNoun(question, expected):
    # Act / Assert
    assert head_noun(question) == expected


@pytest.mark.parametrize(("noun", "one"), [("surgeries", "surgery"), ("visits", "visit"), ("cases", "case")])
def test_singular_plural_roundTrip(noun, one):
    # Act / Assert
    assert (singular(noun), plural(one)) == (one, noun)


@pytest.mark.parametrize(
    ("question", "key", "word"),
    [
        ("How many patients had shots?", "pt_id", "patients"),
        ("What share of visits had shots?", None, "visits"),
        ("How many clinic visits were there each year?", None, "visits"),
    ],
)
def test_resolve_unit_visitRows_countsTheNounAsked(clinic, question, key, word):
    # Act
    unit = resolve_unit(question, clinic[1])

    # Assert: visits are the table's (distinct) rows; patients their key, counted distinct
    assert (unit.key, unit.word) == (key, word)


def test_answer_question_shareOfVisits_countsVisitsNotPatients(ask_clinic):
    # Act: 6 of 13 visits had shots (4 of 6 patients ever did)
    answer, _prompts = ask_clinic("What share of visits had shots?", _plan("share", numerator="shots = 'yes'"))

    # Assert
    assert answer.status == "answered" and answer.analysis is not None
    assert answer.analysis["proportion"] == pytest.approx(6 / 13)


def test_resolve_unit_patientsOnATableWithoutAPatientKey_isAsked(make_query_table):
    # Arrange: no column names a patient
    _catalog, context = make_query_table(tables.clinic_visits().drop("pt_id"), id_column=None)

    # Act / Assert
    with pytest.raises(UnitError, match="No column of this table identifies patients"):
        resolve_unit("How many patients had shots?", context)


# ---------------------------------------------------------------------------- E. dictionary codes


def test_value_labels_dictionaryCodes_areReadForHeldCodesOnly():
    # Arrange
    entries = dictionary_entries(tables.CLINIC_DICTIONARY, ["gender", "shots"])

    # Act / Assert
    assert value_labels(entries["gender"], (1, 2)) == (("1", "male"), ("2", "female"))
    assert value_labels(entries["gender"], (1,)) == (("1", "male"),)


def test_build_context_codedColumn_showsTheLabels(clinic):
    # Act
    gender = clinic[1].column("gender")

    # Assert
    assert gender is not None and gender.labels == (("1", "male"), ("2", "female"))


@pytest.mark.parametrize("where", ["gender = 'female'", "gender = 'women'", "gender = 2"])
def test_answer_question_womenOnACodedColumn_filtersTheCode(ask_clinic, where):
    # Act: 3 of the 6 patients are coded 2 (female)
    answer, _prompts = ask_clinic("How many women are in the clinic?", _plan("count", where=where))

    # Assert
    assert answer.status == "answered", answer.reason
    assert '"gender" = 2' in " ".join((answer.sql or "").split())
    assert answer.analysis is not None and answer.analysis["total_count"] == 3


# ---------------------------------------------------------------------------- B, C. coverage and links


def test_answer_question_subgroupNamedButNotFiltered_isNeverAnswered(ask_clinic):
    # Arrange: "women" twice counted with no filter
    reply = _plan("count")

    # Act
    answer, prompts = ask_clinic("How many women are in the clinic?", reply, reply)

    # Assert
    assert answer.status != "answered" and answer.analysis is None
    assert "The question says 'women'" in prompts[1]


def test_answer_question_conceptWithNoColumn_isHeldNamingIt(ask_clinic):
    # Arrange: the clinic has no smoking column; the model substitutes two unrelated columns
    reply = _plan("count", where="shots = 'yes' AND ige > 100")

    # Act
    answer, _prompts = ask_clinic("How many smokers are there?", reply, reply)

    # Assert: held, saying what is missing; never answered with other columns
    assert answer.status in ("clarify", "refused") and answer.analysis is None
    assert "smokers" in (answer.reason or "") or "shots" in (answer.reason or "")


def test_answer_question_breakdownDescribedInOtherWords_isAnsweredThroughTheDictionary(ask_clinic):
    # Arrange: "sex" is the gender column (generic synonym); "the year they came" its visit_year
    reply = _plan("count", group_by="gender")

    # Act
    answer, _prompts = ask_clinic("Break down the patients by sex.", reply)

    # Assert
    assert answer.status == "answered", answer.reason


def test_check_mappedPhrase_linksOnlyWhenTheColumnIsClose(clinic):
    # Arrange
    _catalog, context = clinic
    close = [coverage.Link("allergy shots", "filter", "shots")]

    # Act
    found = coverage.check("How many got allergy shots?", context, used=["shots"], values=["yes"], links=close)

    # Assert
    assert found.uncovered == () and found.unlinked == ()


@pytest.mark.parametrize(
    ("question", "uncovered"),
    [
        ("How many women had shots?", ("women",)),
        ("How many had shots? Leave out 'women'.", ()),
        ("Please, how many patients in our clinic had shots?", ()),
    ],
)
def test_check_contentRuns_mustReachThePlan(clinic, question, uncovered):
    # Act
    found = coverage.check(question, clinic[1], used=["shots"], values=["yes"], links=[])

    # Assert
    assert found.uncovered == uncovered


# ---------------------------------------------------------------------------- D. ambiguity


def test_answer_question_measureTakenTwice_asksWhichColumn(ask_clinic):
    # Act: eosinophils exist at enrolment and at six months
    answer, _prompts = ask_clinic("What is the average eosinophil count?", _plan("mean", number="eos_baseline"))

    # Assert: asked, never picked
    assert answer.status == "clarify"
    assert [label for label, _q in answer.choices] == ["eos_baseline", "eos_month6"]


@pytest.mark.parametrize(
    "question", ["What is the average eosinophil count at baseline?", "What is the average eos_baseline?"]
)
def test_answer_question_measureTimeNamed_isAnswered(ask_clinic, question):
    # Act
    answer, _prompts = ask_clinic(question, _plan("mean", number="eos_baseline"))

    # Assert
    assert answer.status == "answered", answer.reason


def test_answer_question_perPatientValueThatVaries_asksVisitOrPatient(ask_clinic):
    # Act: IgE differs between a patient's visits
    answer, _prompts = ask_clinic("What is the mean IgE of our patients?", _plan("mean", number="ige"))

    # Assert
    assert answer.status == "clarify"
    assert [label for label, _q in answer.choices] == ["Every visit counts", "One value per patient (their average)"]


def test_answer_question_perPatientValueThatIsConstant_isOneValuePerPatient(ask_clinic):
    # Act: height is the same on every visit of a patient
    answer, _prompts = ask_clinic("What is the average height of our patients?", _plan("mean", number="height_cm"))

    # Assert: the mean over the 6 patients, not over the 13 visits
    assert answer.status == "answered" and answer.analysis is not None
    assert answer.analysis["mean"] == pytest.approx((170 + 160 + 150 + 180 + 165 + 175) / 6)
    assert answer.reading is not None and "one value per patient" in answer.reading


@pytest.mark.parametrize(
    ("choice", "mean"),
    [
        ("Count every visit.", (100 + 140 + 80 + 80 + 60 + 300 + 320 + 50 + 70 + 210 + 230 + 90 + 95) / 13),
        ("Use one value per patient (their average).", (120 + 220 / 3 + 310 + 60 + 220 + 92.5) / 6),
    ],
)
def test_answer_question_perPatientChoice_isAnsweredAsChosen(ask_clinic, choice, mean):
    # Act
    answer, _prompts = ask_clinic(f"What is the mean IgE of our patients? {choice}", _plan("mean", number="ige"))

    # Assert
    assert answer.status == "answered" and answer.analysis is not None
    assert answer.analysis["mean"] == pytest.approx(mean)


def test_compile_plan_meanOverEventsNoun_isOverRows(clinic):
    # Act
    compiled = compile_plan(ModelReply(answer="mean", number="ige"), "What is the mean IgE per visit?", clinic[1])

    # Assert
    assert compiled.per_patient_varies_sql is None and "over visits (rows)" in compiled.reading
