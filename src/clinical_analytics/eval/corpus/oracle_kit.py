# SPDX-License-Identifier: MIT
"""Building blocks for population (aggregate) oracles, in Polars only.

A population table's rows are already estimates: one row per place x period x stratum x measure.
The true answer to most questions is therefore a *lookup*: filter to the dimensions the question
names and read the value off the row. Pooling rows (adding "Both" to Male and Female, a state to
its country, a CI bound to its estimate) gives a different number, which the scorer grades WRONG.

Every lookup here checks the grain: each answer group must come from exactly one row. A filter
that leaves two rows per group is an oracle bug, raised as ``OracleError`` (the question is graded
ERROR) instead of being silently summed or averaged. Values are read as text and converted here,
with the publishers' missing-value markers ("Not Available", blank, "*") as null, so no type
inference stands between the file and the answer.

Written from the publishers' data dictionaries, not from the app's code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple, TypeAlias

import polars as pl

from clinical_analytics.eval.corpus.score import OracleAnswer, OracleError

Tables: TypeAlias = Mapping[str, pl.LazyFrame]
# Publishers' markers for "no value" (compared lower-cased, after trimming).
MISSING_TOKENS = ("", "na", "n/a", "nan", "none", "null", "not available", "*", "**", "***", "****", ".")
CI_PARTS = ("estimate", "ci_low", "ci_high")


class Query(NamedTuple):
    """One oracle: the lazy query, and how its collected frame becomes the answer. (A NamedTuple,
    not a dataclass: oracle modules are loaded by path, outside ``sys.modules``.)"""

    unit: str
    frame: pl.LazyFrame
    finish: Callable[[pl.DataFrame], OracleAnswer]


Oracle: TypeAlias = Callable[[Tables], Query]


def scan_text(path: Path, **options: object) -> pl.LazyFrame:
    """A CSV with every column as text (``infer_schema=False``): codes such as Namibia's iso2
    "NA" stay values, and no column type is guessed from the first rows."""
    return pl.scan_csv(path, infer_schema=False, **options)  # type: ignore[arg-type]


def number(column: str) -> pl.Expr:
    """The column as Float64: thousands separators removed, missing-value markers null."""
    text = pl.col(column).str.strip_chars()
    missing = text.is_null() | text.str.to_lowercase().is_in(list(MISSING_TOKENS))
    return pl.when(missing).then(None).otherwise(text.str.replace_all(",", "")).cast(pl.Float64, strict=False)


def unreadable(column: str) -> pl.Expr:
    """True where the column holds text that is neither a number nor a missing-value marker."""
    text = pl.col(column).str.strip_chars()
    missing = text.is_null() | text.str.to_lowercase().is_in(list(MISSING_TOKENS))
    return ~missing & number(column).is_null()


def _check_grain(df: pl.DataFrame, what: str) -> None:
    if df.height and int(df["rows"].max()) != 1:  # type: ignore[arg-type]
        worst = int(df["rows"].max())  # type: ignore[arg-type]
        raise OracleError(f"{what}: {worst} rows for one answer cell; the filter does not reach the table's grain")
    if "bad" in df.columns and int(df["bad"].sum()):
        raise OracleError(f"{what}: {int(df['bad'].sum())} value(s) are neither numbers nor missing-value markers")


def _top(value: dict[str, object], top: int | None, order: str, key: Callable[[object], float]) -> dict[str, object]:
    if top is None:
        return value
    ranked = sorted(((k, v) for k, v in value.items() if v is not None), key=lambda kv: key(kv[1]))
    if order == "desc":
        ranked.reverse()
    return dict(ranked[:top])


def lookup_by(
    table: str,
    group: str,
    value: str,
    where: pl.Expr,
    unit: str = "value",
    top: int | None = None,
    order: str = "desc",
) -> Oracle:
    """{group level: the one row's value} over the rows ``where`` selects (``value_by``)."""

    def build(tables: Tables) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            _check_grain(df, f"{value} by {group}")
            found: dict[str, object] = {str(r[group]): r["value"] for r in df.iter_rows(named=True)}
            return OracleAnswer(_top(found, top, order, float), unit)  # type: ignore[arg-type]

        query = (
            tables[table]
            .filter(where & pl.col(group).is_not_null())
            .group_by(group)
            .agg(pl.len().alias("rows"), number(value).first().alias("value"), unreadable(value).sum().alias("bad"))
        )
        return Query(unit, query, finish)

    return build


def lookup_ci_by(
    table: str,
    group: str,
    columns: tuple[str, str, str],
    where: pl.Expr,
    unit: str = "value",
    top: int | None = None,
    order: str = "desc",
) -> Oracle:
    """{group level: {"estimate", "ci_low", "ci_high"}} from one row per level (``ci_by``).
    A level whose estimate is suppressed is None (it must not be shown)."""

    def build(tables: Tables) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            _check_grain(df, f"{columns[0]} by {group}")
            found: dict[str, object] = {}
            for row in df.iter_rows(named=True):
                parts = {name: row[name] for name in CI_PARTS}
                found[str(row[group])] = None if any(v is None for v in parts.values()) else parts
            ranked = _top(found, top, order, lambda v: float(v["estimate"]))  # type: ignore[index]
            return OracleAnswer(ranked, unit)

        bad = pl.sum_horizontal(unreadable(c) for c in columns).sum().alias("bad")
        query = (
            tables[table]
            .filter(where & pl.col(group).is_not_null())
            .group_by(group)
            .agg(
                pl.len().alias("rows"),
                *(number(c).first().alias(name) for name, c in zip(CI_PARTS, columns, strict=True)),
                bad,
            )
        )
        return Query(unit, query, finish)

    return build


def derived_by(
    table: str,
    group: str,
    value: pl.Expr,
    where: pl.Expr,
    unit: str = "value",
    top: int | None = None,
    order: str = "desc",
    rows: int | None = None,
    unique_by: str | None = None,
) -> Oracle:
    """{group level: an aggregate ``value``} where the question itself asks for an aggregate over
    rows that do not overlap (countries summed to a region, weeks to a season peak). ``value`` is
    an aggregation expression; ``where`` must leave only non-overlapping rows. ``rows``: the exact
    number of rows every group must have (the grain check), when it is known; ``unique_by``: a
    column every row of a group must differ in (one row per week of a season, say)."""

    def build(tables: Tables) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            if rows is not None and df.height and set(df["rows"].to_list()) != {rows}:
                raise OracleError(f"{group}: groups have {sorted(set(df['rows'].to_list()))} rows, expected {rows}")
            if unique_by is not None and df.height and (df["rows"] != df["distinct"]).any():
                raise OracleError(f"{group}: a group has two rows for one {unique_by}; the filter misses a dimension")
            found: dict[str, object] = {
                str(r[group]): None if r["value"] is None else float(r["value"]) for r in df.iter_rows(named=True)
            }
            return OracleAnswer(_top(found, top, order, float), unit)  # type: ignore[arg-type]

        query = (
            tables[table]
            .filter(where & pl.col(group).is_not_null())
            .group_by(group)
            .agg(
                pl.len().alias("rows"),
                value.alias("value"),
                pl.col(unique_by or group).n_unique().alias("distinct"),
            )
        )
        return Query(unit, query, finish)

    return build


def derived_values(table: str, values: Mapping[str, pl.Expr], where: pl.Expr, unit: str = "value") -> Oracle:
    """Named aggregates over the rows ``where`` selects (``values``); a null is an OracleError."""

    def build(tables: Tables) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            row = df.row(0, named=True)
            if any(row[name] is None for name in values):
                raise OracleError(f"no value for {[n for n in values if row[n] is None]}")
            return OracleAnswer({name: float(row[name]) for name in values}, unit)

        query = tables[table].filter(where).select(*(expr.alias(name) for name, expr in values.items()))
        return Query(unit, query, finish)

    return build


def lookup_values(table: str, columns: Mapping[str, str], where: pl.Expr, unit: str = "value") -> Oracle:
    """Named values from the single row ``where`` selects (``values``)."""

    def build(tables: Tables) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            if df.height != 1:
                raise OracleError(f"{df.height} rows match; the filter does not reach the table's grain")
            row = df.row(0, named=True)
            if any(row[name] is None for name in columns):
                raise OracleError(f"suppressed or missing: {[n for n in columns if row[n] is None]}")
            return OracleAnswer({name: float(row[name]) for name in columns}, unit)

        query = tables[table].filter(where).select(*(number(c).alias(name) for name, c in columns.items()))
        return Query(unit, query, finish)

    return build


def held(table: str, reason: str, where: pl.Expr | None = None) -> Oracle:
    """A question the app must hold or refuse (the data cannot answer it without pooling, or at
    all). The value is None; the row count only shows the oracle ran on the real file."""

    def build(tables: Tables) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            return OracleAnswer(None, f"held: {reason}")

        frame = tables[table] if where is None else tables[table].filter(where)
        return Query(f"held: {reason}", frame.select(pl.len()), finish)

    return build


def answers(tables: Tables, oracles: Mapping[str, Oracle]) -> dict[str, OracleAnswer]:
    """Every question's answer from one ``collect_all``. A question whose data does not support a
    trustworthy answer gets ``oracle_error`` instead of aborting the others."""
    queries = {question_id: oracle(tables) for question_id, oracle in oracles.items()}
    collected = pl.collect_all([query.frame for query in queries.values()])
    result: dict[str, OracleAnswer] = {}
    for (question_id, query), df in zip(queries.items(), collected, strict=True):
        try:
            result[question_id] = query.finish(df)
        except OracleError as error:
            result[question_id] = OracleAnswer(None, query.unit, oracle_error=str(error))
    return result
