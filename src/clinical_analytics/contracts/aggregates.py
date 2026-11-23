# SPDX-License-Identifier: MIT
"""How many cases a table of groups holds, read the way its contract says (ADR017, ADR014).

A table of groups is never counted by its lines. A **frequency** table adds up its count column
over the lines that are not totals; an **estimate** table is never added up: the answer is the
one matching line (the total row for any group the question leaves open), or no number at all.
Total rows ("All", "United States", "Total") are named in the contract, so a sum never counts a
case twice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl
import structlog

from clinical_analytics.contracts.semantics import AggregateInfo

logger = structlog.get_logger()

# Layer config key: the table's AggregateInfo (inferred, or as the user declared it).
LAYER_AGGREGATE_KEY = "aggregate"


@dataclass(frozen=True)
class CaseCount:
    """The cases a question asks for, or None with the reason, always with a sentence to show."""

    total: int | None
    lines: int
    message: str


def _words(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def count_cases(frame: pl.DataFrame, aggregate: AggregateInfo, equals: Mapping[str, Sequence[object]]) -> CaseCount:
    """The cases in ``frame`` for the groups ``equals`` names (column -> accepted values).

    Raises:
        ValueError: the table has no count column to read.
    """
    count = aggregate.count_column
    if count is None or count not in frame.columns:
        raise ValueError("The table has no count column")
    selected = frame
    for column, accepted in equals.items():
        if column in selected.columns:
            selected = selected.filter(pl.col(column).cast(pl.String).is_in([str(v) for v in accepted]))
    totals = {t.column: t.values for t in aggregate.total_levels}
    if aggregate.kind == "frequency":
        # Lines that total other lines never add to the sum.
        for column, values in totals.items():
            if column not in equals:
                selected = selected.filter(~pl.col(column).is_in(list(values)) | pl.col(column).is_null())
        total = int(selected.get_column(count).sum() or 0)
        note = (
            f" {aggregate.repeated_strata_lines:,} lines repeat a group; their counts are included."
            if aggregate.repeated_strata_lines
            else ""
        )
        logger.info("aggregate_cases_summed", lines=selected.height)
        return CaseCount(
            total,
            selected.height,
            f"Each line counts cases for a group, so the answer adds up {count} over "
            f"{selected.height:,} lines: {total:,}.{note}",
        )
    # Estimate table: read the matching line, with the total value for every group left open.
    for column, values in totals.items():
        if column not in equals:
            selected = selected.filter(pl.col(column).is_in(list(values)))
    open_groups = [c for c in aggregate.strata if c not in equals and c not in totals]
    if selected.height == 1:
        total = int(selected.get_column(count)[0])
        return CaseCount(
            total, 1, f"This table's lines are read, never added: the matching line gives {count} = {total:,}."
        )
    why = (
        f"name the {_words(open_groups)} to read one line"
        if open_groups
        else "the groups named match more than one line"
    )
    return CaseCount(
        None,
        selected.height,
        f"This table holds published counts and totals, so its lines are never added together; {why}.",
    )
