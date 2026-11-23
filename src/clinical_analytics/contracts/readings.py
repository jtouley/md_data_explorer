# SPDX-License-Identifier: MIT
"""What the user confirmed about values, applied by every reader of a version's data (ADR017).

The typed Parquet of a version is written once, before any question is answered. A confirmed
"999 means not recorded" or "this column is day first" is therefore applied where data is read:
the semantic layer's view (``core.semantic``), governed SQL's view (``sql.catalog``) and any
reader of the raw frame (the aggregate path) through ``apply_declared``. One definition here;
each reader renders it in its own engine.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from clinical_analytics.contracts import config
from clinical_analytics.contracts.declarations import Declarations
from clinical_analytics.contracts.semantics import DatasetSemantics


@dataclass(frozen=True)
class DeclaredReads:
    """column -> values that mean not recorded; column -> (format, century) for text dates."""

    missing: dict[str, tuple[int | float | str, ...]] = field(default_factory=dict)
    dates: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    # column -> total-row values of a frequency table: those lines are left out of every read, so
    # no reader adds a total to the groups it totals (a sum of cases counts each case once).
    totals: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # value column -> (its unit column, the one unit it is read in): durations in mixed units read
    # in one unit; a line without a recognised unit reads as missing (asked about until answered).
    units: dict[str, tuple[str, str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.missing or self.dates or self.totals or self.units)


def declared_reads(
    declarations: Declarations, dataset_version: str | None, semantics: DatasetSemantics | None = None
) -> DeclaredReads:
    """What every reader applies: confirmed "not recorded" values, date readings, and from the
    semantics (``semantics``) a coded item's non-response codes ("7 = Refused") and range codes
    the user chose to leave out: left out of denominators and averages, kept in the stored data."""
    missing: dict[str, list[int | float | str]] = {}
    for answer in declarations.missing_codes_for(dataset_version):
        if answer.missing:
            missing.setdefault(answer.column, []).append(answer.value)
    if semantics is not None:
        for name, column in semantics.columns.items():
            for code in column.nonresponse_codes:
                missing.setdefault(name, []).append(code.value)
        left_out = {a.question_id for a in declarations.answers_for(dataset_version) if a.answer.startswith("no")}
        for question in semantics.open_questions:
            if question.kind == "censored_value" and question.id in left_out and question.column:
                missing.setdefault(question.column, []).append(question.value if question.value is not None else "")
    dates: dict[str, tuple[str, str | None]] = {
        d.column: (d.format, d.century) for d in declarations.date_readings_for(dataset_version)
    }
    units = {
        name: (col.unit_column, col.unit.value)
        for name, col in (semantics.columns.items() if semantics is not None else ())
        if col.unit_column and col.unit is not None and col.unit.value in config.TIME_UNIT_DAYS
    }
    return DeclaredReads({c: tuple(v) for c, v in missing.items()}, dates, total_rows(semantics), units)


def unit_factors(target: str) -> dict[str, float]:
    """unit spelling (lowercase) -> the factor that reads a value in it in ``target``."""
    return {k: config.TIME_UNIT_DAYS[v] / config.TIME_UNIT_DAYS[target] for k, v in config.TIME_UNIT_NAMES.items()}


def total_rows(semantics: DatasetSemantics | None) -> dict[str, tuple[str, ...]]:
    """The total rows every reader leaves out (column -> values): those of a frequency table, as
    applied (label and sums agree) or confirmed by the user. An estimate table keeps its totals,
    which are the lines a question reads."""
    aggregate = semantics.aggregate if semantics is not None else None
    if aggregate is None or aggregate.kind != "frequency":
        return {}
    return {t.column: tuple(t.values) for t in aggregate.total_levels if t.values}


def _as_number(value: object) -> float | None:
    try:
        return float(str(value))
    except ValueError:
        return None


def missing_expr(column: str, dtype: pl.DataType, values: Iterable[object]) -> pl.Expr:
    """``column`` with the confirmed values as missing, compared as numbers on a number column."""
    col = pl.col(column)
    if dtype.is_numeric():
        numbers = [n for n in (_as_number(v) for v in values) if n is not None]
        matched = col.cast(pl.Float64).is_in(numbers)
    else:
        matched = col.cast(pl.String).is_in([str(v) for v in values])
    return pl.when(matched).then(None).otherwise(col).alias(column)


def date_expr(column: str, choice_format: str, century: str | None) -> pl.Expr:
    """A text column read with the confirmed format; a two-digit year set to the confirmed century."""
    parsed = pl.col(column).cast(pl.String).str.strip_chars().str.strptime(pl.Date, choice_format, strict=False)
    if century == "1900s":
        parsed = pl.when(parsed.dt.year() >= 2000).then(parsed.dt.offset_by("-100y")).otherwise(parsed)
    elif century == "2000s":
        parsed = pl.when(parsed.dt.year() < 2000).then(parsed.dt.offset_by("100y")).otherwise(parsed)
    return parsed.alias(column)


def apply_declared(frame: pl.DataFrame, reads: DeclaredReads) -> pl.DataFrame:
    """``frame`` as every answer reads it once the user's confirmations are applied."""
    if not reads:
        return frame
    lazy = frame.lazy()
    for column, values in reads.totals.items():
        if column in frame.columns:
            lazy = lazy.filter(~pl.col(column).cast(pl.String).is_in(list(values)) | pl.col(column).is_null())
    exprs = [missing_expr(c, frame.schema[c], v) for c, v in reads.missing.items() if c in frame.columns]
    exprs += [date_expr(c, f, century) for c, (f, century) in reads.dates.items() if c in frame.columns]
    for column, (unit_column, target) in reads.units.items():
        if column in frame.columns and unit_column in frame.columns:
            unit = pl.col(unit_column).cast(pl.String).str.strip_chars().str.to_lowercase()
            factor = unit.replace_strict(unit_factors(target), default=None, return_dtype=pl.Float64)
            exprs.append((pl.col(column).cast(pl.Float64) * factor).alias(column))
    return lazy.with_columns(exprs).collect()


def ibis_reads(raw: Any, reads_config: dict[str, Any]) -> dict[str, Any]:
    """The same reads as ibis expressions over ``raw`` (the semantic layer's source table)."""
    import ibis

    mutations: dict[str, Any] = {}
    for column, values in (reads_config.get("missing") or {}).items():
        if column not in raw.columns or not values:
            continue
        ref = raw[column]
        if ref.type().is_numeric():
            numbers = [n for n in (_as_number(v) for v in values) if n is not None]
            matched = ref.cast("float64").isin(numbers)
        else:
            matched = ref.cast("string").isin([str(v) for v in values])
        mutations[column] = matched.ifelse(ibis.null().cast(ref.type()), ref)
    for column, (choice_format, century) in (reads_config.get("dates") or {}).items():
        if column not in raw.columns:
            continue
        parsed = raw[column].cast("string").strip().as_date(choice_format)
        if century == "1900s":
            parsed = (parsed.year() >= 2000).ifelse(parsed - ibis.interval(years=100), parsed)
        elif century == "2000s":
            parsed = (parsed.year() < 2000).ifelse(parsed + ibis.interval(years=100), parsed)
        mutations[column] = parsed
    for column, (unit_column, target) in (reads_config.get("units") or {}).items():
        if column not in raw.columns or unit_column not in raw.columns:
            continue
        unit = raw[unit_column].cast("string").strip().lower()
        factor = unit.substitute(unit_factors(target), else_=ibis.null().cast("float64"))
        mutations[column] = raw[column].cast("float64") * factor
    return mutations


def ibis_total_filter(raw: Any, reads_config: dict[str, Any]) -> Any:
    """The ibis predicate that keeps every line but a frequency table's total rows, or None."""
    predicate = None
    for column, values in (reads_config.get("totals") or {}).items():
        if column not in raw.columns or not values:
            continue
        ref = raw[column]
        keep = ~ref.cast("string").isin([str(v) for v in values]) | ref.isnull()
        predicate = keep if predicate is None else predicate & keep
    return predicate


def reads_config(reads: DeclaredReads) -> dict[str, Any]:
    """The layer config form of ``reads`` (JSON-friendly)."""
    return {
        "missing": {c: list(v) for c, v in reads.missing.items()},
        "dates": {c: list(d) for c, d in reads.dates.items()},
        "totals": {c: list(v) for c, v in reads.totals.items()},
        "units": {c: list(u) for c, u in reads.units.items()},
    }
