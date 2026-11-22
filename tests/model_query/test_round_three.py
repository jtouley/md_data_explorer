# SPDX-License-Identifier: MIT
"""Dictionary labels in every format, stated ranges as breakdowns and test exposures, the row
noun and row-person keys, dates never averaged, percent changes, and the asks for a group that
changes over visits and a rate with no denominator, on made-up school, sleep and wound tables."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.model_query.context import dictionary_entries, value_labels
from clinical_analytics.model_query.plan import ModelReply, compile_test, stated_bins
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables


def _plan(answer: str, **fields: Any) -> dict[str, Any]:
    base = {"thought": "t", "mapping": [], "answer": answer, "where": None, "group_by": None}
    return base | {"numerator": None, "number": None, "outcome": None, "per_patient": None, "order_by": None} | fields


@pytest.fixture
def school(make_query_table):
    return make_query_table(tables.school_screening(), doc_context=tables.SCHOOL_DICTIONARY, id_column=None)


@pytest.fixture
def sleep(make_query_table):
    return make_query_table(tables.sleep_studies(), doc_context=tables.SLEEP_DICTIONARY, id_column="pt_id")


@pytest.fixture
def wound(make_query_table):
    return make_query_table(tables.wound_clinic(), doc_context=tables.WOUND_DICTIONARY)


# ---------------------------------------------------------------------------- dictionary labels


@pytest.mark.parametrize(
    ("text", "values", "expected"),
    [
        pytest.param(
            "1 = None, 2 = Mechanical valve (mitral, aortic, or both), 3 = Other (clot, rare syndrome, etc.)",
            (1, 2, 3),
            (
                ("1", "None"),
                ("2", "Mechanical valve (mitral, aortic, or both)"),
                ("3", "Other (clot, rare syndrome, etc.)"),
            ),
            id="long-labels-with-parentheses",
        ),
        pytest.param(
            "1=Walked, with aid\n2=Wheelchair, transfers alone\n3=Bedbound",
            (1, 2, 3),
            (("1", "Walked, with aid"), ("2", "Wheelchair, transfers alone"), ("3", "Bedbound")),
            id="label-with-comma",
        ),
        pytest.param(
            "A: Admitted\nB: Discharged home. Most common.\nC: Transferred",
            ("A", "B", "C"),
            (("A", "Admitted"), ("B", "Discharged home"), ("C", "Transferred")),
            id="colon-lines",
        ),
        pytest.param(
            "0\tNot screened\n1\tScreened, negative\n2\tScreened, positive (referred)",
            (0, 1, 2),
            (("0", "Not screened"), ("1", "Screened, negative"), ("2", "Screened, positive (referred)")),
            id="tab-separated",
        ),
        pytest.param(
            "1.0 = mild; 2.0 = moderate", (1.0, 2.0), (("1.0", "mild"), ("2.0", "moderate")), id="decimal-codes"
        ),
        pytest.param("", (1, 2), (), id="empty"),
        pytest.param("1 = yes", None, (), id="no-values"),
    ],
)
def test_value_labels_everyDictionaryFormat_givesWholeLabels(text, values, expected):
    # Act
    labels = value_labels(text, values)

    # Assert
    assert labels == expected


@pytest.mark.parametrize(
    ("frame", "doc", "id_column"),
    [
        pytest.param(tables.school_screening, tables.SCHOOL_DICTIONARY, None, id="school-tab-codes"),
        pytest.param(tables.sleep_studies, tables.SLEEP_DICTIONARY, "pt_id", id="sleep-inline-codes"),
        pytest.param(tables.clinic_visits, tables.CLINIC_DICTIONARY, "pt_id", id="clinic"),
        pytest.param(tables.bleed_outcomes, tables.BLEED_DICTIONARY, "patient_id", id="bleed"),
    ],
)
def test_build_context_everyCodeTheDictionaryLists_decodes(make_query_table, frame, doc, id_column):
    # Arrange
    _catalog, context = make_query_table(frame(), doc_context=doc, id_column=id_column)
    entries = dictionary_entries(doc, [c.name for c in context.columns])

    # Act / Assert: a code the column holds and its entry lists with a label is never shown raw
    for column in context.columns:
        if column.values is None or column.name not in entries:
            continue
        listed = {code for code, _label in value_labels(entries[column.name], column.values)}
        assert {code for code, _label in column.labels} == listed, column.name


def test_build_context_tabCodedLongLabels_decodeEveryHeldCode(school):
    # Arrange
    _catalog, context = school

    # Act
    vision = context.column("vision")

    # Assert
    assert vision is not None
    assert dict(vision.labels) == {
        "1": "Passed (both eyes, with or without glasses)",
        "2": "Referred, one eye (left or right, either test)",
        "3": "Referred, both eyes (or could not be tested, e.g. uncooperative, etc.)",
    }


# ---------------------------------------------------------------------------- stated ranges


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param(
            "Mean area by age: under 50, 50 to under 70, 70 or older",
            [(None, 50.0, False, False), (50.0, 70.0, True, False), (70.0, None, True, False)],
            id="to-under",
        ),
        pytest.param(
            "How many by age: under 50 years, 50 to 59 years, 60 to 69 years, and 70 years or older?",
            [(None, 50.0, False, False), (50.0, 59.0, True, True), (60.0, 69.0, True, True), (70.0, None, True, False)],
            id="four-with-units",
        ),
        pytest.param(
            "Count by weight: 40 to 50 kg, 50 to 60 kg, 60 kg and over",
            [(40.0, 50.0, True, False), (50.0, 60.0, True, False), (60.0, None, True, False)],
            id="shared-ends-half-open",
        ),
        pytest.param(
            "between 20 and 29, between 30 and 39, at least 40",
            [(20.0, 29.0, True, True), (30.0, 39.0, True, True), (40.0, None, True, False)],
            id="between-at-least",
        ),
        pytest.param("aged 70+ vs younger", [(None, 70.0, False, False), (70.0, None, True, False)], id="split"),
        pytest.param("under 50, over 50", [], id="boundary-in-neither"),
        pytest.param("How many visits over 3 days in 2020?", [], id="one-limit"),
        pytest.param("", [], id="empty"),
    ],
)
def test_stated_bins_rangeWordings_areHalfOpenBreakdowns(question, expected):
    # Act
    bins = stated_bins(question)

    # Assert
    assert [b[:4] for b in bins] == expected


@pytest.mark.parametrize(
    ("question", "reply", "expected"),
    [
        pytest.param(
            "How many patients by age: under 50, 50 to under 70, 70 or older?",
            _plan("count", group_by="CASE WHEN age < 50 THEN 'a' WHEN age BETWEEN 50 AND 70 THEN 'b' ELSE 'c' END"),
            {"under 50": 2, "50 to under 70": 3, "70 or older": 3},
            id="model-case-replaced",
        ),
        pytest.param(
            "How many patients are under 50, 50 to 59, 60 to 69, or 70 or older?",
            _plan("count", where="age < 50 AND age >= 50 AND age <= 69 AND age >= 70"),
            {"under 50": 2, "50 to 59": 1, "60 to 69": 2, "70 or older": 3},
            id="limits-become-breakdown",
        ),
    ],
)
def test_answer_question_statedRanges_areABreakdownWithoutNulls(make_model_ask, wound, question, reply, expected):
    # Arrange
    ask = make_model_ask(wound)

    # Act
    answer, _prompts = ask(question, reply)

    # Assert: W09 (no age) is in no group
    assert answer.status == "answered", answer.reason
    counts = {row["age_group"]: row["count"] for row in answer.analysis["group_counts"]}
    assert counts == expected


def test_compile_test_oddsRatioWithStatedSplit_isTheExposureNotALimit(wound):
    # Arrange: the model wrote the split as a limit and the exposure as the raw age
    catalog, context = wound
    reply = ModelReply(answer="odds_ratio", where="age >= 70", group_by="age", outcome="healed = 'yes'")

    # Act
    call = compile_test(reply, context, "What are the odds of healing for patients aged 70+ vs younger?")
    with SandboxConnection(catalog) as sandbox:
        rows = sandbox.run_model_sql(call.sql)

    # Assert: 3 patients 70 or older (exposed, 1), 5 younger (0); W09 has no age and is in neither
    assert call.group == "age_70"
    assert "70+ (against younger)" in call.reading and "where" not in call.reading
    assert rows.height == 8
    assert sorted(rows.get_column("age_70").to_list()) == [0, 0, 0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------- row nouns and keys


@pytest.mark.parametrize("with_patient", [True, False], ids=["patient-key", "no-patient-key"])
def test_answer_question_rowNounEpisodes_countsRowsNotTheEpisodeNumber(make_model_ask, make_query_table, with_patient):
    # Arrange: episode_no runs 1..3 for each patient
    table = make_query_table(
        tables.sleep_studies(with_patient=with_patient),
        doc_context=tables.SLEEP_DICTIONARY,
        id_column="pt_id" if with_patient else None,
    )
    ask = make_model_ask(table)

    # Act
    answer, _prompts = ask("How many sleep study episodes are there?", _plan("count"))

    # Assert
    assert answer.status == "answered", answer.reason
    assert answer.analysis["total_count"] == 8
    assert "episode_no" not in (answer.sql or "")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("How many students were screened?", 6, id="rows-are-students"),
        pytest.param("How many guardians are there?", 4, id="other-role-distinct"),
        pytest.param("How many patients were screened?", 6, id="patients-are-rows"),
    ],
)
def test_answer_question_onePersonPerRowNamedByRole_countsWithoutRefusing(make_model_ask, school, question, expected):
    # Arrange
    ask = make_model_ask(school)

    # Act
    answer, prompts = ask(question, _plan("count"))

    # Assert
    assert answer.status == "answered", prompts[-1][:700]
    assert answer.analysis["total_count"] == expected


def test_build_context_uniqueRoleIdAndRepeatingOtherRole_rowsArePersons(school):
    # Arrange
    _catalog, context = school

    # Act / Assert
    assert context.patient_key == "student_id"
    assert context.rows_are_patients


# ---------------------------------------------------------------------------- dates and percents


def test_answer_question_medianOfARawDate_becomesDaysBetweenConsecutiveStudies(make_model_ask, sleep):
    # Arrange: the model asked for the median of the date column itself
    ask = make_model_ask(sleep)

    # Act
    answer, _prompts = ask(
        "What is the median number of days between consecutive sleep studies for the same patient?",
        _plan("median", number="study_date"),
    )

    # Assert: gaps 60, 90, 20, 30
    assert answer.status == "answered", answer.reason
    assert answer.analysis["median"] == pytest.approx(45.0)


def test_answer_question_meanOfARawDateWithoutGapWords_isSentBack(make_model_ask, sleep):
    # Arrange
    ask = make_model_ask(sleep)
    reply = _plan("mean", number="study_date")

    # Act
    answer, prompts = ask("What is the mean study date?", reply, reply)

    # Assert
    assert "is a date" in prompts[1]
    assert answer.status != "answered"


def test_answer_question_percentChangeAsADifference_isSentBackThenDividedByBaseline(make_model_ask, wound):
    # Arrange: first the change in cm2, then the change over the baseline
    ask = make_model_ask(wound)
    first = _plan("mean", number="area_baseline_cm2 - area_week12_cm2")
    second = _plan("mean", number="100.0 * (area_baseline_cm2 - area_week12_cm2) / area_baseline_cm2")

    # Act
    answer, prompts = ask("What is the mean percent reduction in wound area?", first, second)

    # Assert: 50, 100, 25, 100, 25, 100, 75, 50 (W08 not measured)
    assert "percent change" in prompts[1]
    assert answer.status == "answered", prompts[-1][:900]
    assert answer.analysis["mean"] == pytest.approx(525 / 8)


def test_answer_question_percentChangeNeverDivided_isNotAnswered(make_model_ask, wound):
    # Arrange
    ask = make_model_ask(wound)
    reply = _plan("mean", number="area_baseline_cm2 - area_week12_cm2")

    # Act
    answer, _prompts = ask("What is the mean percent reduction in wound area?", reply, reply)

    # Assert
    assert answer.status != "answered"


# ---------------------------------------------------------------------------- asks


def test_answer_question_groupThatChangesOverVisits_asksWhichValueCounts(make_model_ask, sleep):
    # Arrange
    ask = make_model_ask(sleep)

    # Act
    answer, _prompts = ask("How many patients are in each OSA class?", _plan("count", group_by="osa_class"))

    # Assert: A and D change class
    assert answer.status == "clarify"
    assert answer.reason.startswith("2 patients have more than one osa_class")
    assert [label for label, _q in answer.choices] == [
        "Every osa_class a patient had",
        "Each patient's first osa_class",
        "Each patient's last osa_class",
    ]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        pytest.param(1, {"1": 3, "3": 1}, id="first"),
        pytest.param(2, {"1": 1, "2": 2, "3": 1}, id="last"),
        pytest.param(0, {"1": 3, "2": 2, "3": 1}, id="every"),
    ],
)
def test_answer_question_groupChoice_isAnsweredAsChosen(make_model_ask, sleep, choice, expected):
    # Arrange
    ask = make_model_ask(sleep)
    reply = _plan("count", group_by="osa_class")
    asked, _prompts = ask("How many patients are in each OSA class?", reply)

    # Act
    answer, _prompts = ask(asked.choices[choice][1], reply)

    # Assert
    assert answer.status == "answered", answer.reason
    counts = {str(row["osa_class"]): row["count"] for row in answer.analysis["group_counts"]}
    assert {k: v for k, v in counts.items()} == {_label(k): v for k, v in expected.items()}


def _label(code: str) -> str:
    return {"1": "mild", "2": "moderate", "3": "severe"}[code]


def test_answer_question_rateWithNoDenominator_asksPatientsOrEpisodes(make_model_ask, sleep):
    # Arrange
    ask = make_model_ask(sleep)

    # Act
    answer, _prompts = ask("What is the CPAP start rate?", _plan("share", numerator="cpap_started = 'yes'"))

    # Assert
    assert answer.status == "clarify"
    assert [label for label, _q in answer.choices] == ["Out of patients", "Out of episodes"]


def test_answer_question_rateOutOfEpisodesChosen_isAnswered(make_model_ask, sleep):
    # Arrange
    ask = make_model_ask(sleep)
    reply = _plan("share", numerator="cpap_started = 'yes'")
    asked, _prompts = ask("What is the CPAP start rate?", reply)

    # Act
    answer, _prompts = ask(asked.choices[1][1], reply)

    # Assert: 4 of 8 episodes
    assert answer.status == "answered", answer.reason
    assert answer.analysis["proportion"] == pytest.approx(4 / 8)


def test_answer_question_rateOutOfPatientsChosen_asksEverOrAlways(make_model_ask, sleep):
    # Arrange: A started CPAP after its second study only
    ask = make_model_ask(sleep)
    reply = _plan("share", numerator="cpap_started = 'yes'")
    asked, _prompts = ask("What is the CPAP start rate?", reply)

    # Act
    answer, _prompts = ask(asked.choices[0][1], reply)

    # Assert
    assert answer.status == "clarify"
    assert "ever true" in answer.reason


# ---------------------------------------------------------------------------- after held-out set 6


@pytest.fixture
def nursery(make_query_table):
    """The school table as one row per child, age in months, and a tutor ID that repeats."""
    frame = tables.school_screening().rename(
        {"student_id": "child_id", "guardian_id": "tutor_id", "age_years": "age_months"}
    )
    doc = (
        tables.SCHOOL_DICTIONARY.replace("one row per student", "one row per child")
        .replace("student_id   Student identifier.", "child_id     Child identifier.")
        .replace("guardian_id  Identifier of the student's guardian (siblings share one).", "tutor_id     Tutor.")
        .replace("age_years    Age at screening, years.", "age_months   Age at screening, months.")
    )
    return make_query_table(frame.with_columns(tables.pl.col("age_months") * 3), doc_context=doc, id_column=None)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("How many children were screened?", 6, id="row-person-noun-irregular-plural"),
        pytest.param("How many tutors are there?", 4, id="role-outside-the-person-list"),
    ],
)
def test_answer_question_rowPersonsAndOtherRoles_areCounted(make_model_ask, nursery, question, expected):
    # Arrange
    ask = make_model_ask(nursery)

    # Act
    answer, prompts = ask(question, _plan("count"))

    # Assert
    assert answer.status == "answered", prompts[-1][:700]
    assert answer.analysis["total_count"] == expected


@pytest.mark.parametrize(
    ("question", "number"),
    [
        pytest.param(
            "What is the mean percent reduction in wound area?",
            "100.0 * (area_week12_cm2 - area_baseline_cm2) / area_baseline_cm2",
            id="reduction-written-as-a-rise",
        ),
        pytest.param(
            "What is the mean percent improvement in wound area?",
            "100.0 * (area_baseline_cm2 - area_week12_cm2) / area_baseline_cm2",
            id="improvement-direction-unstated",
        ),
    ],
)
def test_answer_question_percentChangeOfUnknownOrWrongSign_isNotAnswered(make_model_ask, wound, question, number):
    # Arrange
    ask = make_model_ask(wound)
    reply = _plan("mean", number=number)

    # Act
    answer, _prompts = ask(question, reply, reply)

    # Assert
    assert answer.status != "answered"


def test_answer_question_improvementWhereLowerIsBetter_isTheFallOverBaseline(make_model_ask, make_query_table):
    # Arrange: the dictionary says a smaller wound is better
    doc = tables.WOUND_DICTIONARY.replace("square centimetres.", "square centimetres (lower is better).", 1)
    ask = make_model_ask(make_query_table(tables.wound_clinic(), doc_context=doc))
    rise = _plan("mean", number="100.0 * (area_week12_cm2 - area_baseline_cm2) / area_baseline_cm2")
    fall = _plan("mean", number="100.0 * (area_baseline_cm2 - area_week12_cm2) / area_baseline_cm2")

    # Act
    answer, prompts = ask("What is the mean percent improvement in wound area?", rise, fall)

    # Assert
    assert "positive percent" in prompts[1]
    assert answer.status == "answered", prompts[-1][:700]
    assert answer.analysis["mean"] == pytest.approx(525 / 8)


def test_answer_question_measureTimeSaidInItsDescription_isNotAsked(make_model_ask, wound):
    # Arrange
    ask = make_model_ask(wound)

    # Act
    answer, prompts = ask(
        "What is the median wound area at the first visit?", _plan("median", number="area_baseline_cm2")
    )

    # Assert: 4, 5, 6, 8, 10, 10, 12, 16, 20
    assert answer.status == "answered", prompts[-1][:700]
    assert answer.analysis["median"] == pytest.approx(10.0)
