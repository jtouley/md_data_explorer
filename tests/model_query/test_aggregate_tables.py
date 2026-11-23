# SPDX-License-Identifier: MIT
"""Tables of groups as the contract describes them: frequency tables add up their count column
(repeated strata included), estimate tables are read one record at a time; the plan language
for totals per group, top groups and different values; and checks that never read the whole
table where the filtered one is meant."""

from __future__ import annotations

import polars as pl
import pytest
import structlog

from clinical_analytics.contracts.semantics import Grain
from clinical_analytics.model_query import checks
from clinical_analytics.model_query.engine import answer_question
from clinical_analytics.model_query.plan import ESTIMATES_NEVER_ADDED, LOOKUP_NOT_ONE
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables

_cond = tables.cond


def _frequency(frame: pl.DataFrame) -> dict[str, object]:
    """The contract's marks for a frequency table, as #155's inference writes them."""
    grain = Grain(
        kind="stratum", noun="count of cases", one_row_per_patient=False, rows=frame.height, source="structure"
    )
    return {"grain": grain, "case_count_column": "injury_cases", "patient_key": None}


@pytest.fixture
def repeated_strata() -> pl.DataFrame:
    """The injuries table with some strata written twice (two reports of one district-year-user)."""
    frame = tables.injuries()
    return pl.concat([frame, frame.filter(pl.col("year") % 10 == 0)])


@pytest.fixture
def ask_repeated(make_query_table, make_model_ask, repeated_strata):
    return make_model_ask(make_query_table(repeated_strata, semantics_update=_frequency(repeated_strata)))


def test_answer_question_repeatedStrata_addsEveryLineCount(ask_repeated, repeated_strata):
    # Act
    answer, _prompts = ask_repeated("How many injuries were there?", tables.reply(count_of="cases"))

    # Assert: the sum of the count column over every line, never a count of lines
    assert answer.analysis["total_count"] == repeated_strata["injury_cases"].sum()
    assert answer.analysis["total_count"] != repeated_strata.height


def test_answer_question_top_isTheGroupWithTheLargestTotal(ask_repeated, repeated_strata):
    # Act
    answer, _prompts = ask_repeated(
        "Which district had the most injuries?", tables.reply(answer="top", count_of="cases", group_by="district")
    )

    # Assert
    totals = repeated_strata.group_by("district").agg(pl.col("injury_cases").sum())
    best = totals.filter(pl.col("injury_cases") == pl.col("injury_cases").max())["district"].to_list()
    assert [g["district"] for g in answer.analysis["group_counts"]] == sorted(best)


def test_answer_question_averageYearlyTotal_summarisesPerYearTotals(ask_repeated, repeated_strata):
    # Arrange
    reply = tables.reply(
        answer="mean", count_of="cases", totals_by="year", filters=[_cond("road_user", "is", values=["cyclist"])]
    )

    # Act
    answer, _prompts = ask_repeated("What is the average yearly total of cyclist injuries?", reply)

    # Assert
    yearly = repeated_strata.filter(pl.col("road_user") == "cyclist").group_by("year").agg(pl.col("injury_cases").sum())
    assert answer.analysis["mean"] == pytest.approx(yearly["injury_cases"].mean())
    assert answer.analysis["non_null_count"] == yearly.height


def test_answer_question_countDistinct_countsDifferentValues(ask_repeated):
    # Act
    answer, _prompts = ask_repeated(
        "How many different districts are there?",
        tables.reply(answer="count_distinct", count_of="cases", measure="district"),
    )

    # Assert
    assert answer.analysis["total_count"] == 3


def test_answer_question_groupLevelsCountedAfterFilters_isAnswered(ask_repeated, repeated_strata):
    # Arrange: 60 years in the table, 10 after the filter: a breakdown by year is shown
    reply = tables.reply(
        count_of="cases", filters=[_cond("year", ">=", 2010), _cond("year", "<=", 2019)], group_by="year"
    )

    # Act
    answer, _prompts = ask_repeated("How many injuries each year from 2010 to 2019?", reply)

    # Assert
    assert answer.status == "answered" and len(answer.analysis["group_counts"]) == 10


@pytest.fixture
def estimates_table(make_query_table):
    frame = tables.coverage_estimates()
    grain = Grain(
        kind="estimate", noun="published estimate", one_row_per_patient=False, rows=frame.height, source="structure"
    )
    return make_query_table(frame, semantics_update={"grain": grain, "patient_key": None})


@pytest.mark.parametrize(
    ("reply", "status", "reason"),
    [
        pytest.param(
            tables.reply(answer="mean", count_of="records", measure="coverage_pct",
                         filters=[_cond("region", "is", values=["north"]), _cond("year", "is", values=[2020])]),
            "answered", None, id="one_record_read",
        ),
        pytest.param(
            tables.reply(
                answer="mean",
                count_of="records",
                measure="coverage_pct",
                filters=[_cond("region", "is", values=["north"])],
            ),
            "refused", LOOKUP_NOT_ONE.format(n=10), id="several_records_refused",
        ),
        pytest.param(tables.reply(count_of="records"), "refused", ESTIMATES_NEVER_ADDED, id="never_summed"),
    ],
)  # fmt: skip
def test_answer_question_estimateTable_readsOneRecordNeverSums(make_model_ask, estimates_table, reply, status, reason):
    # Arrange
    ask = make_model_ask(estimates_table)

    # Act: the question names the year only when the reply does
    year = " in 2020" if any(c["column"] == "year" for c in reply["filters"]) else ""
    answer, _prompts = ask(f"What was the coverage in the north{year}?", reply)

    # Assert
    assert answer.status == status
    if reason is not None:
        assert answer.reason == reason
    else:
        expected = tables.coverage_estimates().filter((pl.col("region") == "north") & (pl.col("year") == 2020))
        assert answer.analysis["mean"] == pytest.approx(expected["coverage_pct"].item())


def test_answer_question_textValueNotHeld_asksWithNearestValues(make_query_table, make_model_ask):
    # Arrange
    frame = pl.DataFrame(
        {"case_ref": [f"C{i:03d}" for i in range(100)], "ward": [f"Ward {i % 50}" for i in range(100)]}
    )
    ask = make_model_ask(make_query_table(frame))

    # Act
    answer, _prompts = ask(
        "How many cases on ward 77?", tables.reply(filters=[_cond("ward", "is", values=["ward 77"])])
    )

    # Assert: asked, never a zero from a value the column does not hold
    assert answer.status == "clarify" and answer.analysis is None
    assert answer.reason.startswith("I can't find 'ward 77'")
    assert answer.choices[0][0] == "Ward 7"


def test_read_cues_valueThatIsAPersonWord_neverSetsTheUnit(make_query_table):
    # Arrange: a column whose values are person words
    frame = tables.clinic().with_columns(
        pl.when(pl.col("visit_seq") % 2 == 0).then(pl.lit("women")).otherwise(pl.lit("men")).alias("seen_by")
    )
    _catalog, context = make_query_table(frame, doc_context=tables.CLINIC_DICTIONARY)

    # Act
    cues = checks.read_cues("How many clinic visits were seen by women?", context)

    # Assert
    assert (cues.persons, cues.records) == (False, True)


def test_answer_question_budgetSpent_saysTheModelTookTooLong(make_query_table):
    # Arrange: the first reply needs another attempt, and the question's budget is spent
    catalog, context = make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY)
    replies = [tables.reply(filters=[_cond("stage", "is", values=["IV"])]), tables.reply()]

    # Act
    answer = answer_question(
        "How many patients have stage III?", context, model=lambda *_a: replies.pop(0),
        open_sandbox=lambda: SandboxConnection(catalog), examples=[], budget_s=0.0,
    )  # fmt: skip

    # Assert
    assert answer.status == "unavailable" and answer.reason.startswith("The local model took too long")


def test_answer_question_held_logsPlanShapeWithoutValues(make_query_table, make_model_ask):
    # Arrange
    ask = make_model_ask(make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY))
    wrong = tables.reply(filters=[_cond("died_1y", "is", values=["yes"])])

    # Act
    with structlog.testing.capture_logs() as logs:
        ask("How many patients did not die within a year?", wrong, wrong)

    # Assert
    held = next(e for e in logs if e["event"] == "model_query_held_answer")
    assert held["plan_shape"]["conditions"] == [["died_1y", "is"]]
    assert "yes" not in str(held["plan_shape"])
