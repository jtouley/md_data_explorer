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

from clinical_analytics.contracts.declarations import Declarations


@dataclass(frozen=True)
class DeclaredReads:
    """column -> values that mean not recorded; column -> (format, century) for text dates."""

    missing: dict[str, tuple[int | float | str, ...]] = field(default_factory=dict)
    dates: dict[str, tuple[str, str | None]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.missing or self.dates)


def declared_reads(declarations: Declarations, dataset_version: str | None) -> DeclaredReads:
    missing: dict[str, list[int | float | str]] = {}
    for answer in declarations.missing_codes_for(dataset_version):
        if answer.missing:
            missing.setdefault(answer.column, []).append(answer.value)
    dates: dict[str, tuple[str, str | None]] = {
        d.column: (d.format, d.century) for d in declarations.date_readings_for(dataset_version)
    }
    return DeclaredReads({c: tuple(v) for c, v in missing.items()}, dates)


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
    exprs = [missing_expr(c, frame.schema[c], v) for c, v in reads.missing.items() if c in frame.columns]
    exprs += [date_expr(c, f, century) for c, (f, century) in reads.dates.items() if c in frame.columns]
    return frame.lazy().with_columns(exprs).collect() if exprs else frame


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
    return mutations


def reads_config(reads: DeclaredReads) -> dict[str, Any]:
    """The layer config form of ``reads`` (JSON-friendly)."""
    return {
        "missing": {c: list(v) for c, v in reads.missing.items()},
        "dates": {c: list(d) for c, d in reads.dates.items()},
    }
