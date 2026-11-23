# SPDX-License-Identifier: MIT
"""Tables of groups as the contract describes them: frequency tables add up their count column
(repeated strata included), estimate tables are read one record at a time; the plan language
for totals per group, top groups and different values; and checks that never read the whole
table where the filtered one is meant."""

from __future__ import annotations

import polars as pl
import pytest
import structlog

from clinical_analytics.contracts.semantics import AggregateInfo, Grain, TotalLevel
from clinical_analytics.model_query import checks
from clinical_analytics.model_query.engine import answer_question
from clinical_analytics.model_query.plan import ESTIMATES_NEVER_ADDED, LOOKUP_NOT_ONE
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables

_cond = tables.cond


def _frequency(frame: pl.DataFrame, totals: tuple[TotalLevel, ...] = ()) -> dict[str, object]:
    """The contract's marks for a frequency table (#155's ``aggregate``), as the contract holds them."""
    grain = Grain(
        kind="stratum", noun="count of cases", one_row_per_patient=False, rows=frame.height, source="structure"
    )
    aggregate = AggregateInfo(
        kind="frequency",
        count_column="discard_count",
        strata=("depot", "round_no", "vial_kind"),
        total_levels=totals,
        source="structure",
    )
    return {
        "grain": grain,
        "aggregate": aggregate,
        "case_count_column": "discard_count",
        "patient_key": None,
        "open_questions": (),
    }


@pytest.fixture
def repeated_strata() -> pl.DataFrame:
    """The discards table with some strata written twice (two reports of one depot, round and vial kind)."""
    frame = tables.vials()
    return pl.concat([frame, frame.filter(pl.col("round_no") % 10 == 0)])


@pytest.fixture
def ask_repeated(make_query_table, make_model_ask, repeated_strata):
    return make_model_ask(make_query_table(repeated_strata, semantics_update=_frequency(repeated_strata)))


def test_answer_question_repeatedStrata_addsEveryLineCount(ask_repeated, repeated_strata):
    # Act
    answer, _prompts = ask_repeated("How many discards were there?", tables.reply(count_of="cases"))

    # Assert: the sum of the count column over every line, never a count of lines
    assert answer.analysis["total_count"] == repeated_strata["discard_count"].sum()
    assert answer.analysis["total_count"] != repeated_strata.height


def test_answer_question_top_isTheGroupWithTheLargestTotal(ask_repeated, repeated_strata):
    # Act
    answer, _prompts = ask_repeated(
        "Which depot had the most discards?", tables.reply(answer="top", count_of="cases", group_by="depot")
    )

    # Assert
    totals = repeated_strata.group_by("depot").agg(pl.col("discard_count").sum())
    best = totals.filter(pl.col("discard_count") == pl.col("discard_count").max())["depot"].to_list()
    assert [g["depot"] for g in answer.analysis["group_counts"]] == sorted(best)


def test_answer_question_averageYearlyTotal_summarisesPerYearTotals(ask_repeated, repeated_strata):
    # Arrange
    reply = tables.reply(
        answer="mean", count_of="cases", totals_by="round_no", filters=[_cond("vial_kind", "is", values=["multi"])]
    )

    # Act
    answer, _prompts = ask_repeated("What is the mean total of multi-dose vial discards per round?", reply)

    # Assert
    per_round = (
        repeated_strata.filter(pl.col("vial_kind") == "multi").group_by("round_no").agg(pl.col("discard_count").sum())
    )
    assert answer.analysis["mean"] == pytest.approx(per_round["discard_count"].mean())
    assert answer.analysis["non_null_count"] == per_round.height


def test_answer_question_countDistinct_countsDifferentValues(ask_repeated):
    # Act
    answer, _prompts = ask_repeated(
        "How many different depots are there?",
        tables.reply(answer="count_distinct", count_of="cases", measure="depot"),
    )

    # Assert
    assert answer.analysis["total_count"] == tables.vials()["depot"].n_unique()


def test_answer_question_groupLevelsCountedAfterFilters_isAnswered(ask_repeated, repeated_strata):
    # Arrange: 60 rounds in the table, 10 after the filter: a breakdown by round is shown
    reply = tables.reply(
        count_of="cases", filters=[_cond("round_no", ">=", 46), _cond("round_no", "<=", 55)], group_by="round_no"
    )

    # Act
    answer, _prompts = ask_repeated("How many discards each round, rounds 46 to 55?", reply)

    # Assert
    assert answer.status == "answered" and len(answer.analysis["group_counts"]) == 10


@pytest.fixture
def estimates_table(make_query_table):
    frame = tables.coverage_estimates()
    grain = Grain(
        kind="estimate", noun="published estimate", one_row_per_patient=False, rows=frame.height, source="structure"
    )
    aggregate = AggregateInfo(
        kind="estimate", value_columns=("coverage_pct",), strata=("region", "year"), source="structure"
    )
    return make_query_table(
        frame, semantics_update={"grain": grain, "aggregate": aggregate, "patient_key": None, "open_questions": ()}
    )


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
    frame = tables.shots().with_columns(
        pl.when(pl.col("shot_round") % 2 == 0).then(pl.lit("women")).otherwise(pl.lit("men")).alias("seen_by")
    )
    _catalog, context = make_query_table(frame, doc_context=tables.SHOTS_DICTIONARY)

    # Act
    cues = checks.read_cues("How many shot visits were seen by women?", context)

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


def test_answer_question_frequencyTotalLines_neverAddedToTheirParts(make_query_table, make_model_ask):
    # Arrange: each depot and round has an "all kinds" line holding the total of its vial kinds
    frame = tables.vials()
    totals = (
        frame.group_by("depot", "round_no")
        .agg(pl.col("discard_count").sum())
        .with_columns(pl.lit("all kinds").alias("vial_kind"))
    )
    with_totals = pl.concat([frame, totals.select(frame.columns)])
    marks = _frequency(with_totals, (TotalLevel(column="vial_kind", values=("all kinds",)),))
    ask = make_model_ask(make_query_table(with_totals, semantics_update=marks))

    # Act
    answer, _prompts = ask("How many discards were there?", tables.reply(count_of="cases"))

    # Assert: the sum of the groups, once
    assert answer.analysis["total_count"] == frame["discard_count"].sum()


def test_answer_question_estimateStratumNotNamed_readsItsTotalLine(make_query_table, make_model_ask):
    # Arrange: the region the question does not name is read at its "all regions" line
    frame = tables.coverage_estimates()
    total = frame.filter(pl.col("region") == "north").with_columns(
        pl.lit("all regions").alias("region"), pl.lit(88.8).alias("coverage_pct")
    )
    frame = pl.concat([frame, total])
    grain = Grain(
        kind="estimate", noun="published estimate", one_row_per_patient=False, rows=frame.height, source="structure"
    )
    aggregate = AggregateInfo(
        kind="estimate", strata=("region", "year"), source="structure",
        total_levels=(TotalLevel(column="region", values=("all regions",)),),
    )  # fmt: skip
    ask = make_model_ask(
        make_query_table(
            frame, semantics_update={"grain": grain, "aggregate": aggregate, "patient_key": None, "open_questions": ()}
        )
    )
    reply = tables.reply(
        answer="mean", count_of="records", measure="coverage_pct", filters=[_cond("year", "is", values=[2020])]
    )

    # Act
    answer, _prompts = ask("What was the coverage in 2020?", reply)

    # Assert
    assert answer.status == "answered" and answer.analysis["mean"] == pytest.approx(88.8)
