# SPDX-License-Identifier: MIT
"""Review probes of the model-written query path (synthetic tables, recorded replies): dropped or
added conditions, negation with a breakdown, totals of a breakdown, timepoint words, implicit
columns' open questions, per-patient facts, statistic words, denominators, ties, the missing
group, sums on tables of groups, small-number logging, wide tables, negated comparisons, zero
groups, units and ranges, and the contract's non-response codes and unit columns."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts.semantics import AggregateInfo, MissingCode, OpenQuestion
from clinical_analytics.model_query import checks, engine
from clinical_analytics.model_query.plan import compile_plan, parse_plan
from tests.model_query import tables

_cond = tables.cond
_reply = tables.reply
_DIED = _cond("died_1y", "is", values=["yes"])


@pytest.fixture
def cohort(make_query_table):
    return make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY)


@pytest.fixture
def ask_cohort(cohort, make_model_ask):
    return make_model_ask(cohort)


@pytest.fixture
def ask_clinic(make_query_table, make_model_ask):
    return make_model_ask(make_query_table(tables.clinic(), doc_context=tables.CLINIC_DICTIONARY))


def _died() -> pl.DataFrame:
    return tables.cohort().filter(pl.col("died_1y") == "yes")


@pytest.mark.parametrize(
    ("question", "reply"),
    [
        pytest.param("How many patients died within a year?", _reply(), id="B1_dropped_condition"),
        pytest.param(
            "How many patients died within a year?",
            _reply(filters=[_DIED, _cond("sex", "is", values=["female"])]),
            id="B1_added_condition",
        ),
        pytest.param(
            "How many patients did not die within a year, by sex?",
            _reply(filters=[_DIED], group_by="sex"),
            id="B2_negation_with_breakdown",
        ),
        pytest.param("What is the median age?", _reply(answer="mean", measure="age"), id="B7_statistic_word"),
        pytest.param(
            "What share of the female patients died within a year?",
            _reply(answer="share", filters=[_DIED], numerator=[_cond("sex", "is", values=["female"])]),
            id="S1_denominator_swapped",
        ),
    ],
)
def test_answer_question_meaningNotComputed_isFedBackThenHeld(ask_cohort, question, reply):
    # Act: the model sends the same reading twice
    answer, prompts = ask_cohort(question, reply, reply)

    # Assert: held, never a number
    assert (answer.status, answer.analysis, len(prompts)) == ("refused", None, 2)


def test_answer_question_inventedDateLimit_isHeld(ask_clinic):
    # Arrange: no date, year or period in the question
    reply = _reply(count_of="clinic visits", filters=[_cond("visit_on", "on or after", "2022-06-01")])

    # Act
    answer, _prompts = ask_clinic("How many clinic visits were there?", reply, reply)

    # Assert
    assert answer.status == "refused" and answer.analysis is None


def test_answer_question_groupedCount_totalIsTheFilteredTotal(ask_cohort):
    # Act
    answer, _prompts = ask_cohort(
        "How many patients died within a year, by sex?", _reply(filters=[_DIED], group_by="sex")
    )

    # Assert (B3): the total is the patients who died, each group's percent of them
    died = _died()
    assert answer.analysis["total_count"] == died.height
    female = next(g for g in answer.analysis["group_counts"] if g["sex"] == "female")
    assert female["percent"] == pytest.approx(
        round(100 * died.filter(pl.col("sex") == "female").height / died.height, 1)
    )
    assert f"out of {died.height} patients who met the limits" in answer.reading


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("Of all the patients, how many smoked?", frozenset(), id="all_the_patients"),
        pytest.param("How many patients had a clinic visit last year?", frozenset(), id="last_year_is_a_date"),
        pytest.param("How many patients smoked at their last clinic visit?", frozenset({"last"}), id="last_visit"),
        pytest.param("Average systolic pressure at baseline", frozenset({"first"}), id="baseline"),
    ],
)
def test_read_cues_timepointWords_onlyWhenAttachedToARecord(make_query_table, question, expected):
    # Arrange (B4)
    _catalog, context = make_query_table(tables.clinic(), doc_context=tables.CLINIC_DICTIONARY)

    # Act
    cues = checks.read_cues(question, context)

    # Assert
    assert cues.times == expected


def test_compile_plan_firstOrLast_readsTheOrderColumn(make_query_table):
    # Arrange (B5): an open question about the order column holds a latest-value answer
    question = OpenQuestion(id="q", kind="date_order", question="Is visit 2 after visit 1?", column="visit_seq")
    _catalog, context = make_query_table(
        tables.clinic(), doc_context=tables.CLINIC_DICTIONARY, semantics_update={"open_questions": (question,)}
    )

    # Act
    compiled = compile_plan(parse_plan(_reply(answer="mean", measure="systolic_mmhg", which_visit="last")), context)

    # Assert
    assert "visit_seq" in compiled.columns and checks.ask_open(compiled, context) is not None


def test_answer_question_openMissingCodeOnCountColumn_isHeld(make_query_table, make_model_ask):
    # Arrange (B5): a frequency table whose count column has an unanswered "does -1 mean missing?"
    question = OpenQuestion(
        id="q", kind="missing_code", question="Does -1 in injury_cases mean not recorded?", column="injury_cases"
    )
    frame = tables.injuries()
    aggregate = AggregateInfo(kind="frequency", count_column="injury_cases", source="structure")
    ask = make_model_ask(
        make_query_table(frame, semantics_update={"aggregate": aggregate, "open_questions": (question,)})
    )

    # Act
    answer, _prompts = ask("How many injury cases?", _reply(count_of="cases"))

    # Assert
    assert answer.analysis is None and "Does -1 in injury_cases mean not recorded?" in answer.reason


def test_answer_question_perPatientFactOnVisits_readsOneValuePerPatient(make_query_table, make_model_ask):
    # Arrange (B6): an age at entry, the same at every visit
    frame = tables.clinic().with_columns((40 + pl.col("pt_ref").str.slice(1).cast(pl.Int64) % 25).alias("entry_age"))
    ask = make_model_ask(make_query_table(frame, doc_context=tables.CLINIC_DICTIONARY))

    # Act
    answer, _prompts = ask("What is the average entry age of the patients?", _reply(answer="mean", measure="entry_age"))

    # Assert: never weighted by the number of visits
    per_patient = frame.group_by("pt_ref").agg(pl.col("entry_age").first())["entry_age"].mean()
    assert answer.analysis["mean"] == pytest.approx(per_patient)
    assert "each patient's value once" in answer.reading


def test_answer_question_tiedFirstRecords_areNotPicked(make_query_table, make_model_ask):
    # Arrange (S3): one patient has two first visits that disagree
    frame = tables.clinic()
    extra = frame.filter((pl.col("pt_ref") == "H004") & (pl.col("visit_seq") == 1)).with_columns(
        pl.lit("Y").alias("smoker")
        if frame.filter(pl.col("pt_ref") == "H004")["smoker"][0] == "N"
        else pl.lit("N").alias("smoker")
    )
    ask = make_model_ask(make_query_table(pl.concat([frame, extra]), doc_context=tables.CLINIC_DICTIONARY))

    # Act
    answer, _prompts = ask(
        "How many patients smoked at their first clinic visit?",
        _reply(filters=[_cond("smoker", "is", values=["yes"])], which_visit="first"),
    )

    # Assert
    assert answer.analysis is None and "first" in answer.reason and "not clear" in answer.reason


def test_answer_question_topGroup_neverTheMissingGroup(make_query_table, make_model_ask):
    # Arrange (S4): most patients have no ward recorded
    frame = pl.DataFrame(
        {"case_ref": [f"C{i:02d}" for i in range(30)], "ward": [None] * 20 + ["north"] * 6 + ["south"] * 4}
    )
    ask = make_model_ask(make_query_table(frame))

    # Act
    answer, _prompts = ask("Which ward had the most patients?", _reply(answer="top", group_by="ward"))

    # Assert
    assert [g["ward"] for g in answer.analysis["group_counts"]] == ["north"]
    assert "Not in any group: 20" in answer.reading


def test_answer_question_frequencySumOfANonCount_isRefused(make_query_table, make_model_ask):
    # Arrange (S5): a rate beside the count column
    frame = tables.injuries().with_columns((pl.col("injury_cases") * 1.7).alias("rate_per_100k"))
    aggregate = AggregateInfo(kind="frequency", count_column="injury_cases", source="structure")
    ask = make_model_ask(make_query_table(frame, semantics_update={"aggregate": aggregate, "open_questions": ()}))

    # Act
    answer, _prompts = ask(
        "What is the total rate per 100k?", _reply(answer="sum", count_of="cases", measure="rate_per_100k")
    )

    # Assert
    assert answer.status == "refused" and "can't be added up" in answer.reason


def test_answer_question_frequencyLineThatReadsLikeATotal_isAsked(make_query_table, make_model_ask):
    # Arrange (S6): an "All users" line the contract does not mark as a total
    frame = tables.injuries()
    totals = (
        frame.group_by("district", "year")
        .agg(pl.col("injury_cases").sum())
        .with_columns(pl.lit("All users").alias("road_user"))
    )
    frame = pl.concat([frame, totals.select(frame.columns)])
    aggregate = AggregateInfo(kind="frequency", count_column="injury_cases", source="structure")
    ask = make_model_ask(make_query_table(frame, semantics_update={"aggregate": aggregate, "open_questions": ()}))

    # Act
    answer, _prompts = ask("How many injury cases?", _reply(count_of="cases"))

    # Assert
    assert answer.analysis is None and "may be a total" in answer.reason


def test_numbers_fewRecords_neverLogsAStatistic():
    # Act (S7)
    logged = engine._numbers({"type": "descriptive", "statistic": "median", "median": 71.0, "non_null_count": 3})

    # Assert: the count, never a value of so few records
    assert logged == [3.0]


def test_answer_question_tableTooWide_isTrimmedToTheNamedColumns(make_query_table, make_model_ask):
    # Arrange (S8): 200 coded items
    frame = pl.DataFrame(
        {"resp_ref": [f"R{i:03d}" for i in range(60)]}
        | {f"item_{k:03d}": [("a", "b", "c")[(i + k) % 3] for i in range(60)] for k in range(200)}
    )
    catalog, context = make_query_table(frame)
    prompts: list[str] = []
    schemas: list[dict[str, Any]] = []

    def model(system: str, user: str, schema: dict[str, Any]) -> Any:
        prompts.append(system)
        schemas.append(schema)
        return _reply(count_of="records", filters=[_cond("item_007", "is", values=["a"])])

    from clinical_analytics.sql.sandbox import SandboxConnection

    # Act
    engine.answer_question(
        "How many said a to item 007?",
        context,
        model=model,
        open_sandbox=lambda: SandboxConnection(catalog),
        examples=[],
    )

    # Assert: the prompt fits the window, and holds the column the question names
    assert prompts and len(prompts[0]) // 4 < engine.PROMPT_TOKENS and "item_007" in prompts[0]


@pytest.mark.parametrize(
    ("question", "op"),
    [
        pytest.param("How many patients are not under 65?", ">="),
        pytest.param("How many patients are no older than 65?", "<="),
    ],
)
def test_apply_stated_negatedComparison_isTheOppositeComparison(cohort, question, op):
    # Arrange (S10)
    _catalog, context = cohort
    plan = parse_plan(_reply(filters=[_cond("age", "<", 65)]))

    # Act
    fixed, _changed = checks.apply_stated(plan, question, context)
    compiled = compile_plan(fixed, context)

    # Assert: the comparison is the opposite, and no negation problem is left
    assert fixed.filters[0].op == op
    assert "negation" not in [p.kind for p in checks.problems(compiled, question, context)]


def test_answer_question_averageYearlyTotal_countsEmptyYearsAsZero(make_query_table, make_model_ask):
    # Arrange (S11): no injuries at all in 1990
    frame = tables.injuries().filter(pl.col("year") != 1990)
    aggregate = AggregateInfo(kind="frequency", count_column="injury_cases", source="structure")
    ask = make_model_ask(make_query_table(frame, semantics_update={"aggregate": aggregate, "open_questions": ()}))

    # Act
    answer, _prompts = ask(
        "What is the average yearly number of injury cases?", _reply(answer="mean", count_of="cases", totals_by="year")
    )

    # Assert: 60 years, 1990 counted as zero
    assert answer.analysis["non_null_count"] == 60
    assert answer.analysis["mean"] == pytest.approx(frame["injury_cases"].sum() / 60)


@pytest.mark.parametrize(
    ("question", "value"),
    [
        pytest.param("How many patients are older than 600 months?", 600, id="unit_differs"),
        pytest.param("How many patients are older than 150?", 150, id="out_of_range"),
    ],
)
def test_answer_question_unitOrRangeOff_isHeldAtOnce(ask_cohort, question, value):
    # Act (S12)
    answer, prompts = ask_cohort(question, _reply(filters=[_cond("age", ">", value)]))

    # Assert: held at once (the model cannot fix the user's unit), never 0 of 60
    assert answer.analysis is None and len(prompts) == 1


def test_column_display_descriptionThatOnlyListsLabels_usesTheName(make_query_table):
    # Arrange (nit): "female / male" describes the values, not the column
    _catalog, context = make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY)

    # Act / Assert
    assert context.column("sex").display == "sex"


def test_answer_question_nonresponseCodes_leftOutOfTheShare(make_query_table, make_model_ask):
    # Arrange (#155): "9 = don't know" kept as an answer, left out of the base
    frame = pl.DataFrame(
        {"resp_ref": [f"R{i:03d}" for i in range(40)], "smokes": [(1, 2, 9, 2)[i % 4] for i in range(40)]}
    )
    catalog, context = make_query_table(frame)
    columns = dict(context.semantics.columns)
    columns["smokes"] = columns["smokes"].model_copy(
        update={
            "labels": {"1": "yes", "2": "no", "9": "don't know"},
            "nonresponse_codes": (MissingCode(value=9, label="don't know", source="dictionary"),),
        }
    )
    ask = make_model_ask(make_query_table(frame, semantics_update={"columns": columns, "open_questions": ()}))

    # Act
    answer, _prompts = ask(
        "What share smokes?",
        _reply(answer="share", count_of="records", numerator=[_cond("smokes", "is", values=["yes"])]),
    )

    # Assert: 10 of 30 (the 10 "don't know" left out), and the card says so
    assert answer.analysis["proportion"] == pytest.approx(10 / 30)
    assert "don't know" in answer.reading


def test_answer_question_durationWithUnitColumn_comparedInTheStatedUnit(make_query_table, make_model_ask):
    # Arrange (#155): durations in months or years, the unit in its own column
    frame = pl.DataFrame(
        {
            "case_ref": [f"C{i:02d}" for i in range(6)],
            "illness_dur": [18, 3, 30, 1, 2, 40],
            "dur_unit": ["months", "years", "months", "years", "years", "months"],
        }
    )
    catalog, context = make_query_table(frame)
    columns = dict(context.semantics.columns)
    columns["illness_dur"] = columns["illness_dur"].model_copy(update={"unit_column": "dur_unit"})
    ask = make_model_ask(make_query_table(frame, semantics_update={"columns": columns, "open_questions": ()}))

    # Act
    answer, _prompts = ask(
        "How many cases had an illness duration over 2 years?",
        _reply(count_of="records", filters=[_cond("illness_dur", ">", 2)]),
    )

    # Assert: 30 months, 3 years and 40 months are over 2 years
    assert answer.analysis["total_count"] == 3


def test_problems_denominatorPhraseEndsAtItsVerb(cohort):
    # Arrange: "the share of our patients died ...": the verb is the event, not the denominator
    _catalog, context = cohort
    compiled = compile_plan(parse_plan(_reply(answer="share", numerator=[_DIED])), context)

    # Act
    kinds = [p.kind for p in checks.problems(compiled, "What share of our patients died within a year?", context)]

    # Assert
    assert "denominator" not in kinds
