# SPDX-License-Identifier: MIT
"""What the model is told about the table: columns, values, grain and the data dictionary."""

from __future__ import annotations

import polars as pl
import pytest

from clinical_analytics.model_query.context import (
    column_lines,
    context_text,
    dictionary_descriptions,
    grain_text,
)
from clinical_analytics.model_query.examples import load_examples, related_columns, similar_examples
from clinical_analytics.model_query.prompt import system_prompt, user_prompt
from tests.model_query import tables


def test_build_context_visitRows_profilesValuesGrainAndDuplicates(make_query_table):
    # Act
    _catalog, context = make_query_table(tables.visits(), doc_context=tables.VISITS_DICTIONARY)

    # Assert
    assert (context.rows, context.distinct_rows, context.patients) == (252, 240, 150)
    assert (context.patient_key, context.encounter_keys) == ("patient_id", ("encounter_id",))
    assert (context.rows_are_patients, context.row_unit, context.duplicate_rows) == (False, "visits", 12)
    result = context.column("RESULT")
    assert result is not None and result.values == ("negative", "positive")
    assert context.column("patient_id").values is None  # an ID's values are never listed
    assert context.column("age_years").values is None and context.column("age_years").low == 40
    assert context.column("titer").nulls > 0
    assert context.notes.startswith("Synthetic clinic")  # a short dictionary is given whole


def test_build_context_onePatientPerRow_saysRowsArePatients(make_query_table):
    # Act
    _catalog, context = make_query_table(tables.patients())

    # Assert
    assert (context.rows_are_patients, context.row_unit) == (True, "patients")
    assert grain_text(context, writes_sql=False) == "Each row is one patient (12 patients)."
    assert "COUNT(*)" in grain_text(context)


@pytest.mark.parametrize(("confirmed", "unit"), [(False, "rows"), (True, "patients")])
def test_build_context_noPatientId_countsRowsUntilConfirmed(make_query_table, confirmed, unit):
    # Arrange: no ID column in the contract
    frame = tables.patients().drop("patient_id")

    # Act
    _catalog, context = make_query_table(frame, confirmed_patient_rows=confirmed, id_column=None)

    # Assert
    assert (context.patient_key, context.row_unit) == (None, unit)


def test_context_text_visitRows_namesHowEachUnitIsCounted(make_query_table):
    # Arrange
    _catalog, context = make_query_table(tables.visits())

    # Act
    for_sql, for_plan = context_text(context), context_text(context, writes_sql=False)

    # Assert
    assert "- result (VARCHAR); values: 'negative', 'positive'" in for_sql
    assert "- patient_id (VARCHAR); patient ID" in for_sql
    assert 'COUNT(DISTINCT "patient_id"), never COUNT(*)' in for_sql
    assert "COUNT(" not in for_plan and "counts each patient once (patient_id)" in for_plan
    assert "12 rows are exact duplicates" in for_plan


def test_dictionary_descriptions_longDictionary_splitsColumnsFromNotes():
    # Arrange
    doc = (
        "Study of 40 people.\n\n"
        "age_cat: Age in categories.\n\n0: under 50.\n\n1: 50 or older.\n\n"
        "sex  Biological sex.\n    Recorded at entry.\n\n"
        "Evidence of immunity for this project: two doses.\n"
    )

    # Act
    described, notes = dictionary_descriptions(doc, ["age_cat", "sex", "dose"])

    # Assert: code lines stay with their column; a new paragraph of prose is a general note
    assert described == {
        "age_cat": "Age in categories. 0: under 50. 1: 50 or older.",
        "sex": "Biological sex. Recorded at entry.",
    }
    assert notes == "Study of 40 people. Evidence of immunity for this project: two doses."


@pytest.mark.parametrize("doc", [None, ""])
def test_dictionary_descriptions_noDictionary_givesNothing(doc):
    # Act / Assert
    assert dictionary_descriptions(doc, ["a"]) == ({}, "")


def test_column_lines_longDescription_isCutInTheListingAndFullAboveTheQuestion(make_query_table):
    # Arrange: a dictionary too long to give whole, so each column gets its own description
    long = "Whether the patient was admitted. " + "More detail about how this was recorded. " * 4
    doc = f"admitted: {long}\n\n" + "filler line about the study.\n" * 120
    _catalog, context = make_query_table(tables.patients(), doc_context=doc)

    # Act
    listing, full = context_text(context), column_lines(context)["admitted"]

    # Assert
    assert "Whether the patient was admitted." in listing and "…" in listing
    assert full.count("More detail") >= 3


def test_examples_pool_everyExample_hasBothFormatsOrIsATestCall():
    # Act
    pool = load_examples()

    # Assert
    assert len(pool) >= 25
    for example in pool:
        plan, sql = example.reply("plan"), example.reply("sql")
        assert list(plan) == [
            "thought", "answer", "where", "group_by", "numerator", "number", "outcome", "per_patient", "order_by"
        ]  # fmt: skip
        assert list(sql)[:3] == ["thought", "answer", "sql"]
        assert (sql["answer"] == "sql") == (sql["sql"] is not None)


def test_similar_examples_question_returnsTheClosestLast():
    # Act
    found = similar_examples("Odds ratio of death with thrombolysis, with a confidence interval?", top_k=3)

    # Assert: the closest example is the one right above the question
    assert len(found) == 3
    assert found[-1].plan["answer"] == "odds_ratio"


def test_related_columns_narrowTable_addsNothing():
    # Act / Assert: a table short enough to read whole gets no shortlist
    assert related_columns("How many women?", {"sex": "sex", "age": "age"}) == []


def test_related_columns_wideTable_findsTheColumnTheQuestionMeans():
    # Arrange
    described = {f"lab_{i}": f"laboratory value number {i}" for i in range(20)}
    described["current_dmt"] = (
        "current dmt. Whether the person is on a disease-modifying therapy. yes, no, never_treated"
    )

    # Act
    found = related_columns("How many patients are on a disease-modifying therapy?", described, top_k=3)

    # Assert
    assert found[0] == "current_dmt"


def test_prompts_planFormat_carryTableExamplesConversationAndFeedback(make_query_table):
    # Arrange
    _catalog, context = make_query_table(tables.patients())
    example = load_examples()[0]
    previous = [("How many by sex?", {"answer": "count", "group_by": "sex"})]

    # Act
    system = system_prompt(context, "plan")
    user = user_prompt(
        "Now only the admitted.",
        [example],
        "plan",
        previous=previous,
        failed_reply={"answer": "count", "where": "x = 1"},
        problem="There is no column named x.",
        related=["- admitted (VARCHAR); values: 'No', 'Yes'"],
    )

    # Assert
    assert "Table unified (12 rows)" in system and '"numerator"' in system
    assert system_prompt(context, "sql") != system
    for part in (example.question, "How many by sex?", "Now only the admitted.", "There is no column named x.",
                 '"where": "x = 1"', "- admitted (VARCHAR)"):  # fmt: skip
        assert part in user
    assert user_prompt("q", [], "plan").endswith("Reply:")


def test_build_context_emptyTable_hasNoValuesToList(make_query_table):
    # Arrange
    frame = pl.DataFrame({"patient_id": [], "sex": []}, schema={"patient_id": pl.String, "sex": pl.String})

    # Act
    _catalog, context = make_query_table(frame)

    # Assert
    assert (context.rows, context.patients) == (0, 0)
    assert context.column("sex").values is None
