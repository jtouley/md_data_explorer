# SPDX-License-Identifier: MIT
"""Building blocks for patient-level oracles (one or several rows per patient), in Polars only.

An oracle module reads its dataset as the publisher ships it, turns it into a frame of text
columns (``as_text``), and maps each question id to one of the builders here. ``answers`` plans
every question as a lazy query and collects them in one ``pl.collect_all``.

The grain is explicit. ``unit="patients"`` with ``patient=<id column>`` counts distinct patients;
``unit="visits"`` counts rows (a test, an encounter, an infection interval: the app's word for a
row of a table with several rows per patient). The patients behind each shown cell are counted
the same way, so the scorer knows which cells must be hidden; a table with no patient ID counts
rows as patients, since no row can be linked to another.

Missing values follow the publishers' markers, not the app's: blank, NA, NaN, null, "?" and "."
are missing everywhere, and "None" is missing only in a number column (in Diabetes 130's
A1Cresult it is the level "not measured"). A group whose value is missing is left out. A finishing
step that finds the data cannot support the answer raises ``OracleError``; only that question is
graded ERROR.

Written from the publishers' data dictionaries, not from the app's code.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, NamedTuple, TypeAlias

import polars as pl

from clinical_analytics.eval.corpus.score import CellKey, OracleAnswer, OracleError

TEXT_MISSING: tuple[str, ...] = ("", "na", "n/a", "nan", "null", "?", ".")
NUMBER_MISSING: tuple[str, ...] = (*TEXT_MISSING, "none")
Z_95 = 1.959963984540054
ALL_ROWS = pl.lit(True)

Unit: TypeAlias = Literal["patients", "visits"]
Stat: TypeAlias = Literal["mean", "median"]


class Query(NamedTuple):
    """One oracle: the lazy query, and how its collected frame becomes the answer. (A NamedTuple,
    not a dataclass: oracle modules are loaded by path, outside ``sys.modules``.)"""

    unit: str
    frame: pl.LazyFrame
    finish: Callable[[pl.DataFrame], OracleAnswer]


Oracle: TypeAlias = Callable[[pl.LazyFrame], Query]


# ---------------------------------------------------------------------------- reading


def as_text(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Every column as text, as a CSV would store it: a whole float as an integer ("3", not
    "3.0", for a code read from a workbook or SAS file), other numbers as Polars prints them."""
    schema = frame.collect_schema()

    def as_string(name: str) -> pl.Expr:
        column = pl.col(name)
        if schema[name].is_float():
            whole = column.is_finite() & (column.round(0) == column)
            return pl.when(whole).then(column.cast(pl.Int64).cast(pl.String)).otherwise(column.cast(pl.String))
        return column.cast(pl.String)

    return frame.select(as_string(name) for name in schema.names())


def cell_text(value: object) -> str | None:
    """One spreadsheet cell as a CSV would store it: a whole number without ".0", an empty cell
    as None, anything else as its text."""
    if value is None or value == "":
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def frame_from_rows(header: Sequence[object], rows: Iterable[Sequence[object]]) -> pl.LazyFrame:
    """A sheet (header row, then data rows) as a frame of text columns.

    Raises:
        OracleError: a header cell is empty or repeated, or a row is wider than the header.
    """
    read = [cell_text(name) for name in header]
    names = [name for name in read if name is not None]
    if len(names) != len(read) or len(set(names)) != len(names):
        raise OracleError(f"sheet header has an empty or repeated name: {read}")
    data = [[cell_text(value) for value in row] for row in rows]
    if wide := [row for row in data if len(row) > len(names)]:
        raise OracleError(f"{len(wide)} row(s) are wider than the {len(names)}-column header")
    columns = {name: [row[i] if i < len(row) else None for row in data] for i, name in enumerate(names)}
    return pl.DataFrame(columns, schema=dict.fromkeys(columns, pl.String)).lazy()


def _is_missing(column: str, tokens: tuple[str, ...]) -> pl.Expr:
    text = pl.col(column).str.strip_chars()
    return text.is_null() | text.str.to_lowercase().is_in(list(tokens))


def text(column: str) -> pl.Expr:
    """The column's text, stripped; null where it holds a missing-value marker."""
    return pl.when(_is_missing(column, TEXT_MISSING)).then(None).otherwise(pl.col(column).str.strip_chars())


def number(column: str) -> pl.Expr:
    """The column as Float64; null where it holds a missing-value marker (or "None")."""
    missing = _is_missing(column, NUMBER_MISSING)
    return pl.when(missing).then(None).otherwise(pl.col(column).str.strip_chars()).cast(pl.Float64, strict=False)


def unreadable(column: str) -> pl.Expr:
    """True where the column holds text that is neither a number nor a missing-value marker."""
    return ~_is_missing(column, NUMBER_MISSING) & number(column).is_null()


def is_value(column: str, *values: str) -> pl.Expr:
    """The column's text is one of ``values`` (null where the column is missing)."""
    return text(column).is_in(list(values))


# ---------------------------------------------------------------------------- counts


def _counted(unit: Unit, patient: str | None) -> pl.Expr:
    """The count for ``unit``: distinct patients when an ID is given, else rows."""
    if unit == "patients" and patient is not None:
        return pl.col(patient).n_unique()
    return pl.len()


def _patients(patient: str | None) -> pl.Expr:
    return pl.len() if patient is None else pl.col(patient).n_unique()


def count(where: pl.Expr = ALL_ROWS, unit: Unit = "patients", patient: str | None = None) -> Oracle:
    """How many patients (or rows) match ``where``."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            n, people = int(df["n"].item()), int(df["people"].item())
            return OracleAnswer(n, unit, {("total",): people})

        query = frame.filter(where).select(_counted(unit, patient).alias("n"), _patients(patient).alias("people"))
        return Query(unit, query, finish)

    return build


def count_by(group: str, where: pl.Expr = ALL_ROWS, unit: Unit = "patients", patient: str | None = None) -> Oracle:
    """Patients (or rows) in each level of ``group``, among rows matching ``where``."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            rows = list(df.iter_rows(named=True))
            value = {str(row["key"]): int(row["n"]) for row in rows}
            return OracleAnswer(value, unit, {(str(row["key"]),): int(row["people"]) for row in rows})

        query = (
            frame.with_columns(text(group).alias("key"))
            .filter(where & pl.col("key").is_not_null())
            .group_by("key")
            .agg(_counted(unit, patient).alias("n"), _patients(patient).alias("people"))
        )
        return Query(unit, query, finish)

    return build


# ---------------------------------------------------------------------------- statistics


def _one_row_per_patient(frame: pl.LazyFrame, patient: str | None, *columns: str) -> pl.LazyFrame:
    """One row per patient with ``columns``, which must not vary within a patient (checked in
    the finishing step through the ``varies`` count)."""
    if patient is None:
        return frame.with_columns(pl.lit(0, dtype=pl.UInt32).alias("varies"))
    return frame.group_by(patient).agg(
        *(pl.col(column).first() for column in columns),
        pl.max_horizontal(*(pl.col(column).n_unique() for column in columns)).gt(1).cast(pl.UInt32).alias("varies"),
    )


def stat_by(stat: Stat, metric: str, group: str, where: pl.Expr = ALL_ROWS, patient: str | None = None) -> Oracle:
    """The mean or median of ``metric`` per level of ``group``. With ``patient``, over one value
    per patient (a patient attribute repeated on each of a patient's rows counts once)."""
    unit = f"{stat} {metric}"

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            if bad := int(df["bad"].sum()):
                raise OracleError(f"{metric}: {bad} value(s) are neither numbers nor missing-value markers")
            if varies := int(df["varies"].sum()):
                raise OracleError(f"{metric} or {group} differs between the rows of {varies} patient(s)")
            rows = list(df.iter_rows(named=True))
            value = {str(row["key"]): None if row["value"] is None else float(row["value"]) for row in rows}
            return OracleAnswer(value, unit, {(str(row["key"]),): int(row["n"]) for row in rows})

        per_patient = _one_row_per_patient(
            frame.filter(where).select(
                *([pl.col(patient)] if patient else []), text(group).alias("key"), pl.col(metric)
            ),
            patient,
            "key",
            metric,
        )
        values = number(metric)
        reducer = values.median() if stat == "median" else values.mean()
        query = (
            per_patient.filter(pl.col("key").is_not_null())
            .group_by("key")
            .agg(
                reducer.alias("value"),
                values.is_not_null().sum().alias("n"),
                unreadable(metric).sum().alias("bad"),
                pl.col("varies").sum(),
            )
        )
        return Query(unit, query, finish)

    return build


def rates(event: pl.Expr, group: str, where: pl.Expr = ALL_ROWS, patient: str | None = None) -> Oracle:
    """Share with ``event`` per level of ``group``, over rows where ``event`` is known. With
    ``patient``, over patients: a patient has the event when any of their rows has it."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            cells: dict[CellKey, int] = {}
            value: dict[str, float] = {}
            for row in df.iter_rows(named=True):
                key, n, events = str(row["key"]), int(row["n"]), int(row["events"])
                value[key] = events / n
                cells |= {(key,): n, (key, "events"): events, (key, "non_events"): n - events}
            return OracleAnswer(value, "proportion", cells)

        known = frame.with_columns(text(group).alias("key"), event.alias("event")).filter(
            where & pl.col("key").is_not_null() & pl.col("event").is_not_null()
        )
        if patient is not None:
            known = known.group_by(patient, "key").agg(pl.col("event").any())
        query = known.group_by("key").agg(pl.len().alias("n"), pl.col("event").sum().alias("events"))
        return Query("proportion", query, finish)

    return build


def odds_ratio(exposure: pl.Expr, outcome: pl.Expr, where: pl.Expr = ALL_ROWS) -> Oracle:
    """Sample odds ratio and Woolf (Wald) 95% CI of ``outcome`` for ``exposure`` (both boolean
    expressions; rows where either is null are left out); equal to an unadjusted logistic
    regression. Undefined when a cell of the 2x2 table is zero."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            table = {(bool(r["exposed"]), bool(r["outcome"])): int(r["n"]) for r in df.iter_rows(named=True)}
            a, b = table.get((True, True), 0), table.get((True, False), 0)
            c, d = table.get((False, True), 0), table.get((False, False), 0)
            cells: dict[CellKey, int] = {
                ("exposed", "events"): a,
                ("exposed", "non_events"): b,
                ("unexposed", "events"): c,
                ("unexposed", "non_events"): d,
            }
            if 0 in (a, b, c, d):
                return OracleAnswer(None, "odds ratio", cells, undefined="a cell of the 2x2 table is zero")
            log_or = math.log((a * d) / (b * c))
            se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            value = {
                "or": math.exp(log_or),
                "ci_low": math.exp(log_or - Z_95 * se),
                "ci_high": math.exp(log_or + Z_95 * se),
            }
            return OracleAnswer(value, "odds ratio", cells)

        query = (
            frame.with_columns(exposure.alias("exposed"), outcome.alias("outcome"))
            .filter(where & pl.col("exposed").is_not_null() & pl.col("outcome").is_not_null())
            .group_by("exposed", "outcome")
            .agg(pl.len().alias("n"))
        )
        return Query("odds ratio", query, finish)

    return build


def held(unit: str = "held", where: pl.Expr = ALL_ROWS, patient: str | None = None) -> Oracle:
    """A question the app must hold or refuse (a listing, no dates, an unlinkable person). The
    value is None; the patients ``where`` selects are recorded for the leak check."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            return OracleAnswer(None, unit, {("total",): int(df.item())})

        return Query(unit, frame.filter(where).select(_patients(patient)), finish)

    return build


# ---------------------------------------------------------------------------- answers


def answers(frame: pl.LazyFrame, oracles: Mapping[str, Oracle]) -> dict[str, OracleAnswer]:
    """Every question's answer, from one ``collect_all`` over ``frame``. A question whose data
    does not support an answer gets ``oracle_error`` instead of stopping the others."""
    queries = {question_id: oracle(frame) for question_id, oracle in oracles.items()}
    collected = pl.collect_all([query.frame for query in queries.values()])
    result: dict[str, OracleAnswer] = {}
    for (question_id, query), df in zip(queries.items(), collected, strict=True):
        try:
            result[question_id] = query.finish(df)
        except OracleError as error:
            result[question_id] = OracleAnswer(None, query.unit, oracle_error=str(error))
    return result
