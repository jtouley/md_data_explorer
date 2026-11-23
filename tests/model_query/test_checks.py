# SPDX-License-Identifier: MIT
"""Meaning checks on the compiled plan: what the question states is applied, what it contradicts
is caught, what it leaves open is asked. Never read from the model's own description."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.model_query import checks
from clinical_analytics.model_query.plan import Plan, compile_plan, parse_plan
from tests.model_query import tables


def _plan(**fields: Any) -> Plan:
    return parse_plan(tables.reply(**fields))


_cond = tables.cond


@pytest.fixture
def clinic(make_query_table):
    return make_query_table(tables.clinic(), doc_context=tables.CLINIC_DICTIONARY)[1]


@pytest.fixture
def cohort(make_query_table):
    return make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY)[1]


@pytest.mark.parametrize(
    ("question", "op_in", "op_out"),
    [
        pytest.param("How many patients are under 65?", ">=", "<", id="under_inverted"),
        pytest.param("How many patients aged 65 or older?", ">", ">=", id="or_older"),
        pytest.param("How many patients over 65?", ">=", ">", id="over"),
        pytest.param("How many patients aged at least 65?", "<", ">=", id="at_least"),
        pytest.param("Patients 65+ : how many?", "<=", ">=", id="plus"),
        pytest.param("How many patients aged 65?", ">", ">", id="no_comparison_word_left_as_is"),
    ],
)
def test_apply_stated_comparisonWithNumber_setsTheStatedOperator(cohort, question, op_in, op_out):
    # Arrange
    plan = _plan(filters=[_cond("age", op_in, 65)])

    # Act
    fixed, changed = checks.apply_stated(plan, question, cohort)

    # Assert
    assert fixed.filters[0].op == op_out
    assert bool(changed) == (op_in != op_out)


@pytest.mark.parametrize(
    ("question", "model_says", "expected"),
    [
        pytest.param("How many patients smoked at their latest visit?", "any", "last", id="latest"),
        pytest.param("How many patients smoked at baseline?", "last", "first", id="baseline"),
        pytest.param("How many patients ever smoked?", "last", "any", id="ever"),
        pytest.param("How many patients smoked at every visit?", "any", "every", id="every"),
        pytest.param("How many patients smoked?", "last", "any", id="unstated_is_not_assumed"),
    ],
)
def test_apply_stated_timepointWords_setWhichVisit(clinic, question, model_says, expected):
    # Arrange
    plan = _plan(filters=[_cond("smoker", "is", values=["yes"])], which_visit=model_says)

    # Act
    fixed, _changed = checks.apply_stated(plan, question, clinic)

    # Assert
    assert fixed.which_visit == expected


def test_apply_stated_neverOnVisits_readsEveryVisit(clinic):
    # Arrange: "never smoked" = not a smoker at every visit, not "a visit without smoking"
    plan = _plan(filters=[_cond("smoker", "is", values=["no"])], which_visit="any")

    # Act
    fixed, _changed = checks.apply_stated(plan, "How many patients never smoked?", clinic)

    # Assert
    assert fixed.which_visit == "every"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("How many clinic visits had a smoker?", "clinic visits", id="records_noun"),
        pytest.param("How many women smoked?", "patients", id="person_noun"),
    ],
)
def test_apply_stated_countedNoun_setsTheUnit(clinic, question, expected):
    # Arrange
    other = "patients" if expected != "patients" else "clinic visits"
    plan = _plan(count_of=other, filters=[_cond("smoker", "is", values=["yes"])])

    # Act
    fixed, _changed = checks.apply_stated(plan, question, clinic)

    # Assert
    assert fixed.count_of == expected


@pytest.mark.parametrize(
    ("question", "fields", "kind"),
    [
        pytest.param("How many patients are over 70?", {}, "number", id="number_dropped"),
        pytest.param(
            "How many patients have a low FEV1?",
            {"filters": [_cond("fev_pct", "<", 60)]},
            "invented_number",
            id="threshold_invented",
        ),
        pytest.param(
            "How many patients did not die within a year?",
            {"filters": [_cond("died_1y", "is", values=["yes"])]},
            "negation",
            id="not_dropped",
        ),
        pytest.param(
            "How many patients died within a year?",
            {"filters": [_cond("died_1y", "is not", values=["yes"])]},
            "negation",
            id="negation_invented",
        ),
        pytest.param("How many patients have stage III?", {}, "value", id="named_value_unused"),
    ],
)
def test_problems_meaningNotComputed_isFound(cohort, question, fields, kind):
    # Arrange
    compiled = compile_plan(_plan(**fields), cohort)

    # Act
    found = checks.problems(compiled, question, cohort)

    # Assert
    assert kind in [p.kind for p in found]


@pytest.mark.parametrize(
    ("question", "fields"),
    [
        pytest.param("How many patients are over 70?", {"filters": [_cond("age", ">", 70)]}, id="number_used"),
        pytest.param(
            "How many patients did not die within a year?",
            {"filters": [_cond("died_1y", "is", values=["no"])]},
            id="no_value_is_negation",
        ),
        pytest.param(
            "How many patients have stage III?", {"filters": [_cond("stage", "is", values=["III"])]}, id="value"
        ),
        pytest.param(
            "How many patients aged 50 to 60?",
            {"filters": [_cond("age", ">=", 50), _cond("age", "<=", 60)]},
            id="range",
        ),
        pytest.param(
            "How many patients have no FEV1 recorded?", {"filters": [_cond("fev_pct", "is missing")]}, id="missing"
        ),
    ],
)
def test_problems_planComputesTheQuestion_findsNothing(cohort, question, fields):
    # Arrange
    compiled = compile_plan(_plan(**fields), cohort)

    # Act / Assert
    assert checks.problems(compiled, question, cohort) == []


def test_needs_visit_choice_unstatedVisit_onVaryingColumn(clinic):
    # Arrange
    compiled = compile_plan(_plan(filters=[_cond("smoker", "is", values=["yes"])]), clinic)

    # Act / Assert: no timepoint word, a visit-level table: which visit counts is asked
    assert checks.needs_visit_choice(compiled, "How many patients smoke?", clinic) is True
    assert checks.needs_visit_choice(compiled, "How many patients smoked at their latest visit?", clinic) is False


def test_visit_ask_choicesReAskWithATimepointTheChecksRead(clinic):
    # Arrange
    compiled = compile_plan(_plan(filters=[_cond("smoker", "is", values=["yes"])]), clinic)

    # Act
    ask = checks.visit_ask(compiled, clinic, varying=5)

    # Assert: each choice, appended to the question, states a timepoint the checks then apply
    assert ask.kind == "which_visit"
    for _label, words in ask.choices:
        question = f"How many patients smoke? {words}"
        assert not checks.needs_visit_choice(compiled, question, clinic)


def test_ask_unit_visitTableNoNoun_asksPatientsOrVisits(clinic):
    # Arrange
    compiled = compile_plan(_plan(answer="share", numerator=[_cond("smoker", "is", values=["yes"])]), clinic)

    # Act
    ask = checks.ask_unit(compiled, "What share smoked?", clinic)

    # Assert
    assert ask is not None and [label for label, _w in ask.choices] == ["Patients", "Clinic visits"]


def test_ask_open_unansweredQuestionOnUsedColumn_isAsked(make_query_table):
    # Arrange: the contract holds an open question about a column the plan reads
    from clinical_analytics.contracts.semantics import OpenQuestion

    question = OpenQuestion(id="q1", kind="date_order", question="Does 03/04/21 mean 3 April?", column="visit_on")
    _catalog, context = make_query_table(
        tables.clinic(), doc_context=tables.CLINIC_DICTIONARY, semantics_update={"open_questions": (question,)}
    )
    compiled = compile_plan(_plan(filters=[_cond("visit_on", "before", "2023-01-01")]), context)

    # Act
    ask = checks.ask_open(compiled, context)

    # Assert
    assert ask is not None and ask.reason.startswith("Does 03/04/21 mean 3 April?")


@pytest.mark.parametrize(
    ("question", "used", "expected"),
    [
        pytest.param("How many patients by sex?", frozenset({"sex"}), True, id="simple"),
        pytest.param("How many patients are over 70?", frozenset(), False, id="number"),
        pytest.param("How many patients have stage III?", frozenset(), False, id="value_not_in_plan"),
        pytest.param("How many patients did not die?", frozenset({"died_1y"}), False, id="negation"),
    ],
)
def test_matcher_may_answer_onlyQuestionsItFullyReads(cohort, question, used, expected):
    # Act / Assert
    assert checks.matcher_may_answer(question, cohort, used) is expected


def test_matcher_may_answer_visitTable_neverAnswersFirst(clinic):
    # Act / Assert: a visit-level table needs a per-patient choice the matcher cannot make
    assert checks.matcher_may_answer("How many patients by sex?", clinic, frozenset({"sex"})) is False


def test_apply_stated_reAskedChoiceTimepoint_keepsTheUnit(clinic):
    # Arrange: a choice's words name the records' noun as a time, not as what is counted
    plan = _plan(answer="share", numerator=[_cond("smoker", "is", values=["yes"])])

    # Act
    fixed, _changed = checks.apply_stated(plan, "What share of patients smoke? at the latest clinic visit", clinic)

    # Assert
    assert (fixed.count_of, fixed.which_visit) == ("patients", "last")


def test_needs_visit_choice_overAllRecordsInContractNoun_poolsWithoutAsking(clinic):
    # Arrange: a summary over every visit, said with the contract's own noun for a record
    compiled = compile_plan(_plan(answer="median", measure="systolic_mmhg"), clinic)

    # Act / Assert
    assert checks.needs_visit_choice(compiled, "Median systolic pressure over all clinic visits", clinic) is False
    assert checks.needs_visit_choice(compiled, "Median systolic pressure", clinic) is True


def test_apply_stated_countWithNumerator_numeratorIsALimit(cohort):
    # Arrange: a count has no numerator; the model's condition there must not be dropped
    plan = _plan(filters=[_cond("age", "<", 70)], numerator=[_cond("died_1y", "is", values=["yes"])])

    # Act
    fixed, changed = checks.apply_stated(plan, "How many patients under 70 died within a year?", cohort)

    # Assert
    assert [c.column for c in fixed.filters] == ["age", "died_1y"] and fixed.numerator == ()
    assert "numerator->filters" in changed


def test_apply_stated_shareRepeatingItsFilter_keepsItOnlyAsNumerator(cohort):
    # Arrange: the same condition as filter and numerator would make the share 100%
    died = _cond("died_1y", "is", values=["yes"])
    plan = _plan(answer="share", filters=[died], numerator=[died])

    # Act
    fixed, _changed = checks.apply_stated(plan, "What share of patients died within a year?", cohort)

    # Assert
    assert fixed.filters == () and [c.column for c in fixed.numerator] == ["died_1y"]
