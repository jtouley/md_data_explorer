# SPDX-License-Identifier: MIT
"""Summary tables (each line counts a group), many-valued filters, side-by-side comparisons,
questions the table cannot answer, and a model that takes too long."""

from __future__ import annotations

import polars as pl
import pytest

from clinical_analytics.model_query import card, checks
from clinical_analytics.model_query.engine import COUNT_COLUMN_QUESTION
from clinical_analytics.model_query.llm import ModelTimedOut
from clinical_analytics.model_query.plan import parse_plan
from tests.model_query import tables

_cond = tables.cond


@pytest.fixture
def injuries(make_query_table):
    return make_query_table(tables.injuries())


@pytest.fixture
def ask_injuries(injuries, make_model_ask):
    return make_model_ask(injuries)


def test_answer_question_summaryCount_addsUpTheCountColumn(ask_injuries):
    # Arrange
    reply = tables.reply(count_of="cases", filters=[_cond("road_user", "is", values=["cyclist"])])

    # Act
    answer, _prompts = ask_injuries("How many cyclist injuries were there?", reply)

    # Assert: the total of the counts, never a count of lines
    frame = tables.injuries()
    expected = frame.filter(pl.col("road_user") == "cyclist")["injury_cases"].sum()
    assert answer.status == "answered" and answer.analysis["total_count"] == expected
    assert 'SUM("injury_cases")' in answer.sql and "COUNT(*)" not in answer.sql
    assert answer.reading.startswith("Counted: the total of the case counts")
    assert f"Result: {expected:,} of {frame['injury_cases'].sum():,} cases" in answer.reading


def test_answer_question_summaryShare_isPartOfTheSummedCount(ask_injuries):
    # Arrange
    reply = tables.reply(answer="share", count_of="cases", numerator=[_cond("road_user", "is", values=["pedestrian"])])

    # Act
    answer, _prompts = ask_injuries("What share of injuries were pedestrians?", reply)

    # Assert
    frame = tables.injuries()
    expected = frame.filter(pl.col("road_user") == "pedestrian")["injury_cases"].sum() / frame["injury_cases"].sum()
    assert answer.analysis["proportion"] == pytest.approx(expected)


def test_answer_question_summaryWithoutCountColumn_asksWhichColumnCounts(make_query_table, make_model_ask):
    # Arrange: a summary table whose count column the contract does not name
    table = make_query_table(tables.injuries(), semantics_update={"case_count_column": None})
    ask = make_model_ask(table)

    # Act
    answer, _prompts = ask("How many injuries were there?", tables.reply(count_of="cases"))

    # Assert: asked, never COUNT(*) over the groups
    assert answer.status == "clarify" and answer.reason == COUNT_COLUMN_QUESTION
    assert answer.analysis is None and answer.choices[0][0].startswith("Total of ")


def test_answer_question_filterOnColumnWithManyValues_isAnswered(ask_injuries):
    # Arrange: 60 years; the group limit is for showing groups, never for a filter
    reply = tables.reply(
        count_of="cases", filters=[_cond("year", ">=", 2000), _cond("year", "<=", 2009)], group_by="district"
    )

    # Act
    answer, _prompts = ask_injuries("How many injuries from 2000 to 2009, by district?", reply)

    # Assert
    frame = tables.injuries().filter(pl.col("year").is_between(2000, 2009))
    expected = dict(frame.group_by("district").agg(pl.col("injury_cases").sum()).iter_rows())
    got = {row["district"]: row["count"] for row in answer.analysis["group_counts"]}
    assert answer.status == "answered" and got == expected


def test_answer_question_textColumnWithManyValues_isFilterable(make_query_table, make_model_ask):
    # Arrange: 50 distinct wards, too many to list, still a filter
    frame = pl.DataFrame(
        {"case_ref": [f"C{i:03d}" for i in range(100)], "ward": [f"Ward {i % 50}" for i in range(100)]}
    )
    ask = make_model_ask(make_query_table(frame))

    # Act
    answer, _prompts = ask("How many cases on ward 7?", tables.reply(filters=[_cond("ward", "is", values=["ward 7"])]))

    # Assert: matched whole and regardless of case
    assert answer.status == "answered" and answer.analysis["total_count"] == 2


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("Compare FEV1 between the stages", "mean", id="compare_is_figures"),
        pytest.param("Is FEV1 significantly different between the stages?", "compare_groups", id="test_asked"),
    ],
)
def test_apply_stated_compareWithoutTestWords_givesSideBySideFigures(make_query_table, question, expected):
    # Arrange
    _catalog, context = make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY)
    plan = parse_plan(tables.reply(answer="compare_groups", measure="fev_pct", group_by="stage"))

    # Act
    fixed, _changed = checks.apply_stated(plan, question, context)

    # Assert
    assert fixed.answer == expected


def test_answer_question_notInTable_saysSoPlainly(ask_injuries):
    # Arrange
    reply = tables.reply(answer="not_in_table", count_of="cases", missing="population")

    # Act
    answer, prompts = ask_injuries("What is the injury rate per 100,000 population?", reply)

    # Assert: no choice to drop the word and compute something else, and no retry
    assert answer.status == "refused" and answer.choices == () and len(prompts) == 1
    assert answer.reason == "This table has no column for population, so I can't answer that from this file."


def test_answer_question_modelTookTooLong_saysSo(ask_injuries):
    # Act
    answer, _prompts = ask_injuries("How many injuries?", ModelTimedOut(300.0))

    # Assert
    assert answer.status == "unavailable" and answer.reason.startswith("The local model took too long")


def test_reading_lines_summaryCount_passesLinter(injuries):
    # Arrange
    from clinical_analytics.model_query.plan import compile_plan

    _catalog, context = injuries
    compiled = compile_plan(parse_plan(tables.reply(count_of="cases", group_by="road_user")), context)

    # Act
    lines = card.reading_lines(compiled, context)

    # Assert
    assert card.card_problems(lines, context) == []


def test_answer_question_episodeNoun_countsEachEpisodeOnce(make_query_table, make_model_ask):
    # Arrange: admissions within a patient, recorded by the contract as an episode key
    from clinical_analytics.contracts.semantics import EpisodeKey

    frame = tables.clinic().with_columns(
        (pl.col("pt_ref") + "-A" + ((pl.col("visit_seq") + 1) // 2).cast(pl.Utf8)).alias("adm_ref")
    )
    episode = EpisodeKey(column="adm_ref", within="pt_ref", noun="admission", source="structure")
    table = make_query_table(frame, doc_context=tables.CLINIC_DICTIONARY, semantics_update={"episode_keys": (episode,)})
    ask = make_model_ask(table)

    # Act
    answer, _prompts = ask(
        "How many admissions had a smoker?", tables.reply(filters=[_cond("smoker", "is", values=["yes"])])
    )

    # Assert: the unit comes from the question's noun and the contract's episode key
    expected = frame.filter(pl.col("smoker") == "Y")["adm_ref"].n_unique()
    assert answer.status == "answered" and answer.analysis["total_count"] == expected
    assert answer.reading.startswith("Counted: admissions, each once")
