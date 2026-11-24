# SPDX-License-Identifier: MIT
"""Edges of ``contracts.aggregates.count_cases``, found by mutation testing: which lines a sum
reads, how total rows are left out (and a question naming one), and what an estimate table says
when one line does not match. Invented school-vaccination clinic tables only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
import pytest

from clinical_analytics.contracts.aggregates import count_cases
from clinical_analytics.contracts.semantics import AggregateInfo, TotalLevel

ALL = "All districts"
TOTALS = (TotalLevel(column="district", values=(ALL,)),)


def _doses() -> pl.DataFrame:
    """Doses given by district and term, with a total line per term and one line of no district."""
    return pl.DataFrame(
        {
            "district": ["North", "South", ALL, "North", "South", ALL, None],
            "term": [1, 1, 1, 2, 2, 2, 1],
            "doses": [1000, 2000, 3000, 500, 700, 1200, 400],
        }
    )


def _frequency(repeated: int = 0) -> AggregateInfo:
    return AggregateInfo(
        kind="frequency",
        count_column="doses",
        strata=("district", "term"),
        total_levels=TOTALS,
        repeated_strata_lines=repeated,
        source="structure",
    )


@pytest.mark.parametrize(
    ("equals", "total", "lines"),
    [
        pytest.param({}, 4600, 5, id="every_group_totals_left_out_null_kept"),
        pytest.param({"term": [1]}, 3400, 3, id="one_term"),
        pytest.param({"district": ["North"]}, 1500, 2, id="one_district"),
        pytest.param({"district": [ALL], "term": [2]}, 1200, 2, id="total_named_reads_its_groups"),
        pytest.param({"district": [ALL]}, 4600, 5, id="total_named_alone"),
        pytest.param({"ward": ["W1"]}, 4600, 5, id="column_not_in_table_ignored"),
        pytest.param({"district": ["East"]}, 0, 0, id="no_line_matches_sums_to_zero"),
    ],
)
def test_countCases_frequencyTable_sumLeavesTotalsOut(
    equals: Mapping[str, Sequence[object]], total: int, lines: int
) -> None:
    # Act
    result = count_cases(_doses(), _frequency(), equals)

    # Assert
    assert (result.total, result.lines) == (total, lines)
    assert result.message == (
        f"Each line counts cases for a group, so the answer adds up doses over {lines:,} lines: {total:,}."
    )


def test_countCases_frequencyWithRepeatedGroups_noteSaysTheyAreIncluded() -> None:
    # Act
    result = count_cases(_doses(), _frequency(repeated=1200), {})

    # Assert
    assert result.message.endswith(" 1,200 lines repeat a group; their counts are included.")


@pytest.mark.parametrize("count_column", [None, "doses_given"], ids=["no_count_column", "count_column_absent"])
def test_countCases_noCountColumn_failsLoud(count_column: str | None) -> None:
    # Arrange
    aggregate = _frequency().model_copy(update={"count_column": count_column})

    # Act / Assert
    with pytest.raises(ValueError, match="^The table has no count column$"):
        count_cases(_doses(), aggregate, {})


def _estimate(strata: tuple[str, ...]) -> AggregateInfo:
    return AggregateInfo(kind="estimate", count_column="doses", strata=strata, total_levels=TOTALS, source="structure")


def test_countCases_estimateOneLineMatches_readNeverAdded() -> None:
    # Act
    result = count_cases(_doses(), _estimate(("district", "term")), {"term": [1]})

    # Assert: the open district reads its total line
    assert (result.total, result.lines) == (3000, 1)
    assert result.message == "This table's lines are read, never added: the matching line gives doses = 3,000."


@pytest.mark.parametrize(
    ("frame_columns", "strata", "equals", "why", "lines"),
    [
        pytest.param((), ("district", "term"), {}, "name the term to read one line", 2, id="one_open_group"),
        pytest.param(("sex",), ("district", "term", "sex"), {}, "name the term and sex to read one line", 2, id="two"),
        pytest.param(
            ("sex", "school"),
            ("district", "term", "sex", "school"),
            {},
            "name the term, sex and school to read one line",
            2,
            id="three",
        ),
        pytest.param((), ("district",), {"district": ["North"]}, "the groups named match more than one line", 2,
                     id="named_groups_match_several"),
    ],
)  # fmt: skip
def test_countCases_estimateSeveralLinesMatch_noNumberAndWhy(
    frame_columns: tuple[str, ...],
    strata: tuple[str, ...],
    equals: Mapping[str, Sequence[object]],
    why: str,
    lines: int,
) -> None:
    # Arrange
    frame = _doses().with_columns(pl.lit("x").alias(c) for c in frame_columns)

    # Act
    result = count_cases(frame, _estimate(strata), equals)

    # Assert
    assert (result.total, result.lines) == (None, lines)
    assert result.message == (
        f"This table holds published counts and totals, so its lines are never added together; {why}."
    )
