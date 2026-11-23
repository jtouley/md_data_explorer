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
def discards(make_query_table):
    # The contract's "Does each line count cases?" answered yes, as the upload's user would.
    return make_query_table(tables.vials(), semantics_update=tables.frequency_marks(tables.vials()))


@pytest.fixture
def ask_vials(discards, make_model_ask):
    return make_model_ask(discards)


def test_answer_question_summaryCount_addsUpTheCountColumn(ask_vials):
    # Arrange
    reply = tables.reply(count_of="cases", filters=[_cond("vial_kind", "is", values=["multi"])])

    # Act
    answer, _prompts = ask_vials("How many multi-dose vial discards were there?", reply)

    # Assert: the total of the counts, never a count of lines
    frame = tables.vials()
    expected = frame.filter(pl.col("vial_kind") == "multi")["discard_count"].sum()
    assert answer.status == "answered" and answer.analysis["total_count"] == expected
    assert 'SUM("discard_count")' in answer.sql and "COUNT(*)" not in answer.sql
    assert answer.reading.startswith("Counted: the total of the case counts")
    assert f"Result: {expected:,} of {frame['discard_count'].sum():,} cases" in answer.reading


def test_answer_question_summaryShare_isPartOfTheSummedCount(ask_vials):
    # Arrange
    reply = tables.reply(answer="share", count_of="cases", numerator=[_cond("vial_kind", "is", values=["single"])])

    # Act
    answer, _prompts = ask_vials("What share of discards were single vials?", reply)

    # Assert
    frame = tables.vials()
    expected = frame.filter(pl.col("vial_kind") == "single")["discard_count"].sum() / frame["discard_count"].sum()
    assert answer.analysis["proportion"] == pytest.approx(expected)


def test_answer_question_summaryWithoutCountColumn_asksWhichColumnCounts(make_query_table, make_model_ask):
    # Arrange: a summary table whose count column the contract does not name
    from clinical_analytics.contracts.semantics import AggregateInfo

    unnamed = AggregateInfo(kind="frequency", count_column=None, source="structure")
    table = make_query_table(
        tables.vials(), semantics_update={"case_count_column": None, "aggregate": unnamed, "open_questions": ()}
    )
    ask = make_model_ask(table)

    # Act
    answer, _prompts = ask("How many discards were there?", tables.reply(count_of="cases"))

    # Assert: asked, never COUNT(*) over the groups
    assert answer.status == "clarify" and answer.reason == COUNT_COLUMN_QUESTION
    assert answer.analysis is None and answer.choices[0][0].startswith("Total of ")


def test_answer_question_filterOnColumnWithManyValues_isAnswered(ask_vials):
    # Arrange: 60 rounds; the group limit is for showing groups, never for a filter
    reply = tables.reply(
        count_of="cases", filters=[_cond("round_no", ">=", 36), _cond("round_no", "<=", 45)], group_by="depot"
    )

    # Act
    answer, _prompts = ask_vials("How many discards in rounds 36 to 45, by depot?", reply)

    # Assert
    frame = tables.vials().filter(pl.col("round_no").is_between(36, 45))
    expected = dict(frame.group_by("depot").agg(pl.col("discard_count").sum()).iter_rows())
    got = {row["depot"]: row["count"] for row in answer.analysis["group_counts"]}
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


def test_answer_question_notInTable_saysSoPlainly(ask_vials):
    # Arrange
    reply = tables.reply(answer="not_in_table", count_of="cases", missing="population")

    # Act
    answer, prompts = ask_vials("What is the discard rate per 100,000 population?", reply)

    # Assert: no choice to drop the word and compute something else, and no retry
    assert answer.status == "refused" and answer.choices == () and len(prompts) == 1
    assert answer.reason == "This table has no column for population, so I can't answer that from this file."


def test_answer_question_modelTookTooLong_saysSo(ask_vials):
    # Act
    answer, _prompts = ask_vials("How many discards?", ModelTimedOut(300.0))

    # Assert
    assert answer.status == "unavailable" and answer.reason.startswith("The local model took too long")


def test_reading_lines_summaryCount_passesLinter(discards):
    # Arrange
    from clinical_analytics.model_query.plan import compile_plan

    _catalog, context = discards
    compiled = compile_plan(parse_plan(tables.reply(count_of="cases", group_by="vial_kind")), context)

    # Act
    lines = card.reading_lines(compiled, context)

    # Assert
    assert card.card_problems(lines, context) == []


def test_answer_question_episodeNoun_countsEachEpisodeOnce(make_query_table, make_model_ask):
    # Arrange: admissions within a patient, recorded by the contract as an episode key
    from clinical_analytics.contracts.semantics import EpisodeKey

    frame = tables.shots().with_columns(
        (pl.col("holder_ref") + "-A" + ((pl.col("shot_round") + 1) // 2).cast(pl.Utf8)).alias("adm_ref")
    )
    episode = EpisodeKey(column="adm_ref", within="holder_ref", noun="admission", source="structure")
    table = make_query_table(frame, doc_context=tables.SHOTS_DICTIONARY, semantics_update={"episode_keys": (episode,)})
    ask = make_model_ask(table)

    # Act
    answer, _prompts = ask(
        "How many admissions had a wheeze?", tables.reply(filters=[_cond("wheeze", "is", values=["yes"])])
    )

    # Assert: the unit comes from the question's noun and the contract's episode key
    expected = frame.filter(pl.col("wheeze") == "Y")["adm_ref"].n_unique()
    assert answer.status == "answered" and answer.analysis["total_count"] == expected
    assert answer.reading.startswith("Counted: admissions, each once")
