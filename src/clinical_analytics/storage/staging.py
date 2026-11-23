# SPDX-License-Identifier: MIT
"""Typed staging: read an upload once as text, then give every column one explicit type.

Types are decided over the whole file, column by column, never from a sample:

* ID-named columns (``ID_COLUMN_NAMES``) and digit codes with a leading zero (zip ``02139``,
  ward ``01``) stay text.
* ``true``/``false`` become Boolean; whole numbers Int64 (nulls stay null, not Float64); decimal
  numbers Float64. ``1,234`` is not a number. Whole numbers that do not fit Int64 stay text
  (``int_overflow``); they never become Float64, which would lose digits.
* Missing-value markers (``MISSING_VALUE_TOKENS``: ``NA``, ``.``, ``#N/A``, ``NaN``, ...) become
  null only in a column whose other values are all numbers or all dates, and the count is
  reported (``missing_tokens``). A column that stays text keeps every value: ``None`` or ``-``
  can be a real category level.
* Dates are read with ONE format per column, the one that reads every value with a plausible
  year (``contracts.dates``; never a different format per row). Day-first and month-first dates
  are only used when some value settles the order (a day above 12); otherwise the column stays
  text (``ambiguous_dates``) and the contract asks which order the file uses. Mostly dates that
  no one format reads keep the whole column as text (``not_dates``), the same rule as for
  numbers: nothing is silently turned into a missing value.
* A column that cannot take a type stays text, and only that column: a note says which column
  and why (``ColumnTypeNote``). Nothing falls back to "every column is text".

A first row that is a units sub-header (``years``, ``mg/dL`` above numeric columns, with no ID)
is kept as unit metadata and is not a patient; the report says that row was removed. Trailing
lines with no values (``,,,`` written by exporters) are dropped; any other row with no ID and no
values fails the upload.

Formats: delimited text (``.csv``, ``.tsv``, ``.psv``; ``DELIMITERS``), Excel (``.xlsx`` with
openpyxl; legacy ``.xls`` BIFF workbooks, recognised by their bytes, with xlrd) and SAS
transport (``.xpt``). Excel and delimited text are staged as text; SAS transport files carry
their own types (numbers or text), and only whole-number columns change, to Int64.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Final, Literal

import polars as pl
import structlog

from clinical_analytics.contracts.config import (
    MISSING_VALUE_TOKENS,
    SOURCE_NOT_APPLICABLE_TOKENS,
    SOURCE_SUPPRESSED_PATTERN,
    SOURCE_SUPPRESSED_WORDS,
    SOURCE_UNRELIABLE_PATTERN,
    SUPPRESSED_FLAG_SUFFIX,
    UNRELIABLE_FLAG_SUFFIX,
)
from clinical_analytics.contracts.dates import DateAmbiguity, DateChoice, choose_text_format, date_coverage

logger = structlog.get_logger()

# Columns stored as text whatever they hold (avoids integer overflow, keeps joins consistent).
ID_COLUMN_NAMES: Final = frozenset({"patient_id", "patientid", "subject_id", "subjectid", "id", "mrn", "study_id"})

# A column is reported as "numeric with some text" when at least this share is numeric.
NUMERIC_NOTE_SHARE: Final = 0.5

# "date_values_unparsed" is no longer produced; it stays readable in notes stored by earlier versions.
NoteReason = Literal[
    "leading_zero",
    "not_numeric",
    "date_values_unparsed",
    "ambiguous_dates",
    "not_dates",
    "int_overflow",
    "missing_tokens",
    "shared_column_type",
    "from_file_names",
]

# CDC WONDER exports put a notes block under the data: a line "---" starts it, and every line
# from there is one quoted field (dataset, query, citation, footnotes). Split off only in that
# shape (first header "Notes", no separator in any notes line), so a "---" line in any other
# file never hides the rows below it.
_EXPORT_NOTES_START: Final = re.compile(rb'(?m)^[ \t]*"?---"?[ \t]*\r?$')
# CDC WONDER's first column: "Notes". Its "Total" rows stay in the data: aggregate mode reads them
# as total rows (core/estimates.py).
EXPORT_NOTES_COLUMN: Final = "notes"
# StagedTable.source_format (and upload metadata SOURCE_FORMAT_KEY) for a CDC WONDER export.
CDC_WONDER_FORMAT: Final = "cdc_wonder"

# Column separator by file extension (``.gz`` stripped first). Text tables in a ZIP use the same.
DELIMITERS: Final[dict[str, str]] = {".csv": ",", ".tsv": "\t", ".psv": "|"}

# An .xlsx is a ZIP. A workbook that is not one is read as legacy .xls (BIFF, raw or in OLE2).
_ZIP_MAGIC: Final = b"PK\x03\x04"

# A units-row cell must be one of these (case-insensitive) or a ratio such as "mg/dL" or
# "x10^9/L" (``_UNIT_RATIO_RE``), unless the file has an ID-named column whose cell is empty.
UNIT_VOCABULARY: Final = frozenset(
    {
        "%",
        "years",
        "year",
        "yrs",
        "yr",
        "y",
        "months",
        "month",
        "mo",
        "weeks",
        "week",
        "wk",
        "days",
        "day",
        "d",
        "hours",
        "hour",
        "h",
        "hr",
        "min",
        "minutes",
        "s",
        "kg",
        "g",
        "mg",
        "mcg",
        "ug",
        "µg",
        "lb",
        "lbs",
        "cm",
        "mm",
        "m",
        "in",
        "ml",
        "l",
        "mmhg",
        "bpm",
        "°c",
        "°f",
        "degc",
        "degf",
        "iu",
        "u",
        "meq",
        "mmol",
        "µmol",
        "umol",
        "mol",
        "date",
        "yyyy-mm-dd",
        "dd/mm/yyyy",
        "mm/dd/yyyy",
        "score",
        "points",
        "ratio",
        "index",
        "cells",
        "count",
        "n",
    }
)
_UNIT_RATIO_RE: Final = re.compile(r"^[^\d\s/]*[a-zµ%°][^/]*/[^/]*[a-zµ0-9²][^/]*$", re.IGNORECASE)

_INT_RE: Final = r"^[+-]?\d+$"
_FLOAT_RE: Final = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
_LEADING_ZERO_RE: Final = r"^0\d+$"
_DIGITS_RE: Final = r"^\d+$"
# A censored lab value ("<5", ">100", "≤0.1") is data, never a unit label.
_CENSORED_RE: Final = re.compile(r"^\s*[<>≤≥]=?\s*[+-]?(\d+\.?\d*|\.\d+)\s*$")
_NUMBER_RE: Final = re.compile(_FLOAT_RE)


class StagingError(ValueError):
    """The upload cannot be staged as given (a structural problem the user must fix)."""


@dataclass(frozen=True)
class ColumnTypeNote:
    """Why a column did not get the type its values mostly suggest, and how many values failed."""

    column: str
    stored_type: str
    reason: NoteReason
    failed_values: int = 0
    checked_values: int = 0  # non-empty values in the column

    def as_dict(self) -> dict[str, str | int]:
        return {
            "column": self.column,
            "stored_type": self.stored_type,
            "reason": self.reason,
            "failed_values": self.failed_values,
            "checked_values": self.checked_values,
        }


@dataclass(frozen=True)
class ExcelSheets:
    """What was read from a workbook: the first sheet only, and the rows of it that are notes.

    ``skipped`` names the other sheets that hold data; ``empty`` counts the other sheets with no
    values. ``footnote_rows`` counts the rows below the data, after a blank row, that were read
    as notes ("Source: ..."), not data.
    """

    read: str
    skipped: tuple[str, ...] = ()
    empty: int = 0
    footnote_rows: int = 0

    def as_dict(self) -> dict[str, str | int | list[str]]:
        return {
            "read": self.read,
            "skipped": list(self.skipped),
            "empty": self.empty,
            "footnote_rows": self.footnote_rows,
        }


@dataclass(frozen=True)
class StagedTable:
    """One typed table plus what staging decided about it."""

    data: pl.DataFrame
    notes: tuple[ColumnTypeNote, ...] = ()
    units: dict[str, str] = field(default_factory=dict)
    sheets: ExcelSheets | None = None  # Excel only: the sheet read and the sheets skipped
    # Text exports with a notes block (CDC WONDER): the block as documentation text, and
    # sentences for the user about rows left out of the data (subtotal rows).
    export_notes: str = ""
    table_notes: tuple[str, ...] = ()
    # The published export the file was read as ("cdc_wonder"), or "" for any other file.
    source_format: str = ""
    # The one format each date column was read with (``contracts.dates``), for the contract.
    dates: tuple[DateChoice, ...] = ()


def is_id_column_name(name: str) -> bool:
    return name.strip().lower() in ID_COLUMN_NAMES


# --- per-column typing ------------------------------------------------------------------------


@dataclass(frozen=True)
class _ColumnPlan:
    expr: pl.Expr
    dtype: pl.DataType
    notes: tuple[ColumnTypeNote, ...] = ()
    date: DateChoice | None = None  # the one format a date column was read with


def _clean(col: str) -> pl.Expr:
    """Trimmed text, with blank cells as null."""
    trimmed = pl.col(col).str.strip_chars()
    return pl.when(trimmed == "").then(None).otherwise(trimmed)


def _is_missing_token(expr: pl.Expr) -> pl.Expr:
    """True where trimmed text is a missing-value marker (``MISSING_VALUE_TOKENS``)."""
    return expr.str.strip_chars().str.to_lowercase().is_in(list(MISSING_VALUE_TOKENS))


def _token_mask(s: pl.Series) -> pl.Series:
    """``_is_missing_token`` for a series of trimmed text."""
    return s.str.strip_chars().str.to_lowercase().is_in(list(MISSING_VALUE_TOKENS))


def _clean_tokens(col: str) -> pl.Expr:
    """``_clean`` with missing-value markers as null (only for columns typed as numbers or dates)."""
    cleaned = _clean(col)
    return pl.when(_is_missing_token(cleaned)).then(None).otherwise(cleaned)


def _parse_dates(s: pl.Series) -> tuple[pl.Series | None, NoteReason | None, DateChoice | None]:
    """Read a trimmed text series as dates with ONE format chosen over every value
    (``contracts.dates.choose_text_format``), or say why not (None, None, None: not dates)."""
    found = choose_text_format(s)
    if isinstance(found, DateAmbiguity):
        return None, "ambiguous_dates", None
    if found is None:
        return None, None, None
    dtype: pl.Datetime | pl.Date = pl.Datetime("us") if found.has_time else pl.Date()
    return s.str.strip_chars().str.strptime(dtype, found.format, strict=False), None, found


def _plan_column(df: pl.DataFrame, col: str) -> _ColumnPlan:
    s = df.select(_clean(col)).to_series().alias(col)
    non_null = s.drop_nulls()
    n = non_null.len()
    text = _ColumnPlan(pl.col(col), pl.String())
    if n == 0 or is_id_column_name(col):
        return text

    def text_with(reason: NoteReason, failed: int) -> _ColumnPlan:
        return _ColumnPlan(pl.col(col), pl.String(), (ColumnTypeNote(col, "String", reason, failed, n),))

    # Missing-value markers are judged apart from the values: they become null only if the
    # column types as numbers or dates; a text column keeps them as they are.
    token_mask = _token_mask(non_null)
    tokens = int(token_mask.sum())
    values = non_null.filter(~token_mask)
    if values.len() == 0:
        return text

    def typed(expr: pl.Expr, dtype: pl.DataType) -> _ColumnPlan:
        notes = (ColumnTypeNote(col, str(dtype), "missing_tokens", tokens, n),) if tokens else ()
        return _ColumnPlan(expr, dtype, notes)

    is_digits = values.str.contains(_DIGITS_RE)
    if bool(is_digits.all()) and bool(values.str.contains(_LEADING_ZERO_RE).any()):
        return text_with("leading_zero", 0)

    if not tokens and bool(non_null.str.to_lowercase().is_in(["true", "false"]).all()):
        expr = _clean(col).str.to_lowercase().replace_strict({"true": True, "false": False}, return_dtype=pl.Boolean)
        return _ColumnPlan(expr, pl.Boolean())

    if bool(values.str.contains(_INT_RE).all()):
        overflow = int(values.cast(pl.Int64, strict=False).null_count())
        if overflow:
            # Never Float64: digits past 2**53 would be lost without a word.
            return text_with("int_overflow", overflow)
        return typed(_clean_tokens(col).cast(pl.Int64), pl.Int64())

    is_float = values.str.contains(_FLOAT_RE)
    if bool(is_float.all()):
        return typed(_clean_tokens(col).cast(pl.Float64), pl.Float64())

    parsed, why_not, choice = _parse_dates(s.set(_token_mask(s).fill_null(False), None) if tokens else s)
    if why_not is not None:
        return text_with(why_not, 0)
    if parsed is not None:
        plan = typed(pl.lit(parsed).alias(col), parsed.dtype)
        return _ColumnPlan(plan.expr, plan.dtype, plan.notes, choice)
    covered = date_coverage(values)
    if covered and covered * 2 >= values.len():
        # Mostly dates, but no one format reads them all: kept as text, so no value is silently
        # stored as missing (and the original text stays queryable).
        return text_with("not_dates", values.len() - covered)

    numeric = int(is_float.sum())
    if numeric / n >= NUMERIC_NOTE_SHARE:
        return text_with("not_numeric", values.len() - numeric)
    return text


def _type_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, list[ColumnTypeNote], tuple[DateChoice, ...]]:
    plans = {col: _plan_column(df, col) for col in df.columns}
    typed = df.with_columns([plan.expr.alias(col) for col, plan in plans.items()])
    dates = tuple(plan.date for plan in plans.values() if plan.date is not None)
    return typed, [note for plan in plans.values() for note in plan.notes], dates


def settle_shared_column_types(
    frames: dict[str, pl.DataFrame],
) -> tuple[dict[str, pl.DataFrame], list[tuple[str, ColumnTypeNote]]]:
    """Give a column name shared by several tables one type in all of them (join keys).

    Each table is typed on its own, so ``ward`` can be text in a table where a value starts
    with ``0`` and Int64 in another. Mixed whole and decimal numbers become Float64; any other
    mix becomes text (which loses nothing). Returns the frames and one note per changed column.
    """
    dtypes: dict[str, set[pl.DataType]] = {}
    for df in frames.values():
        for col, dtype in df.schema.items():
            dtypes.setdefault(col, set()).add(dtype)
    target: dict[str, pl.DataType] = {}
    for col, kinds in dtypes.items():
        if len(kinds) < 2:
            continue
        target[col] = pl.Float64() if kinds <= {pl.Int64(), pl.Float64()} else pl.String()
    settled: dict[str, pl.DataFrame] = {}
    notes: list[tuple[str, ColumnTypeNote]] = []
    for name, df in frames.items():
        changes = [c for c in df.columns if c in target and df.schema[c] != target[c]]
        settled[name] = df.with_columns(pl.col(c).cast(target[c]) for c in changes) if changes else df
        notes.extend((name, ColumnTypeNote(c, str(target[c]), "shared_column_type", 0, df[c].count())) for c in changes)
    if notes:
        logger.info("staging_shared_column_types_settled", columns=len(notes))
    return settled, notes


# --- units row and empty rows -----------------------------------------------------------------


def _is_label(value: str) -> bool:
    """Text that is not a number and not a censored value: a unit label like 'mg/dL' or '%'."""
    return not _NUMBER_RE.match(value) and not _CENSORED_RE.match(value)


def _is_unit(value: str) -> bool:
    """A known unit word (``UNIT_VOCABULARY``) or a ratio unit such as 'mg/dL'."""
    return value.strip().lower() in UNIT_VOCABULARY or bool(_UNIT_RATIO_RE.match(value.strip()))


def _units_row(df: pl.DataFrame) -> dict[str, str] | None:
    """Units by column if the first row is a units sub-header, else None.

    The row is units when it has no ID, every filled cell is a label (not a number, not "<5",
    not a missing-value marker), every filled cell sits above a column whose remaining rows are
    numeric or dates, and either the file has an ID-named column (empty in this row) or some
    cell is a known unit. Without that, an ID-less file's first patient could be dropped.
    """
    if df.height < 2:
        return None
    first = df.row(0, named=True)
    if any(is_id_column_name(c) and (first[c] or "").strip() for c in df.columns):
        return None
    filled = {
        c: v.strip()
        for c, v in first.items()
        if v is not None and v.strip() and v.strip().lower() not in MISSING_VALUE_TOKENS
    }
    if not filled or not all(_is_label(v) for v in filled.values()):
        return None
    has_id_column = any(is_id_column_name(c) for c in df.columns)
    if not has_id_column and not any(_is_unit(v) for v in filled.values()):
        return None
    rest, _, _ = _type_columns(df.slice(1))
    typed_cols = {c for c, t in rest.schema.items() if t.is_numeric() or t.is_temporal()}
    if not set(filled) <= typed_cols:
        return None
    return filled


def _drop_trailing_blank_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Drop trailing rows with no values (``,,,`` lines exporters write after the data)."""
    if df.width == 0 or df.height == 0:
        return df
    blank = df.select(pl.all_horizontal([_clean(c).is_null() for c in df.columns])).to_series()
    filled = blank.not_().arg_true()
    keep = int(filled[-1]) + 1 if filled.len() else 0
    if keep < df.height:
        logger.info("staging_trailing_blank_rows_dropped", rows=df.height - keep)
    return df.head(keep)


def _reject_empty_rows(df: pl.DataFrame, first_row_number: int) -> None:
    """Fail on any row with no ID and no values (``first_row_number`` = file row of data row 0)."""
    if df.width == 0 or df.height == 0:
        return
    blank = pl.all_horizontal([_clean(c).is_null() for c in df.columns])
    idx = df.with_row_index("_row").filter(blank)["_row"]
    if idx.len():
        row = int(idx[0]) + first_row_number
        raise StagingError(
            f"The file has an empty line at row {row} (no patient ID and no values). "
            "Remove empty lines and upload again."
        )


# --- entry points ---------------------------------------------------------------------------


# "401.63 (390.33, 412.62)": an estimate and its interval packed in one cell (IHME Excel tables).
_NUM = r"(-?\d[\d,]*\.?\d*)"
PACKED_INTERVAL = rf"^\s*{_NUM}\s*\(\s*{_NUM}\s*(?:,|to|–|—|;)\s*{_NUM}\s*\)\s*$"
_YEAR_IN_HEADER = re.compile(r"(?<!\d)(1[89]\d\d|20\d\d|2100)(?!\d)")
MIN_YEAR_COLUMNS = 3  # a table with years as columns has at least this many


def _packed(df: pl.DataFrame, col: str) -> bool:
    """Every non-empty cell of a text column is a value with its interval in brackets."""
    cleaned = df.select(_clean_tokens(col).alias(col)).get_column(col).drop_nulls()
    return cleaned.len() > 0 and bool(cleaned.str.contains(PACKED_INTERVAL).all())


def _header_year(col: str) -> tuple[str, int] | None:
    """(header without its year, the year) when the header names exactly one year."""
    years = _YEAR_IN_HEADER.findall(col)
    if len(years) != 1:
        return None
    stem = re.sub(r"[\s,;:*()\-–]+$", "", _YEAR_IN_HEADER.sub("", col)).strip(" ,;:*()-–")
    return stem, int(years[0])


def _years_to_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Unpivot packed estimate columns that are one per year ("Mortality Rate, 1980*") into rows:
    the other columns, then ``year`` and one value column per header stem (several stems: a
    ``measure`` column and ``value``). Returns ``df`` unchanged when there are not enough."""
    by_year = {c: found for c in df.columns if (found := _header_year(c)) is not None and _packed(df, c)}
    if len(by_year) < MIN_YEAR_COLUMNS:
        return df
    ids = [c for c in df.columns if c not in by_year]
    if {"year", "measure", "value"} & set(ids):
        # The unpivot would add a column the file already has: leave the years as columns.
        logger.info("staging_years_unpivot_skipped", reason="name_taken", year_columns=len(by_year))
        return df
    stems = sorted({stem for stem, _year in by_year.values()})
    value = stems[0] if len(stems) == 1 and stems[0] else "value"
    long = df.unpivot(on=list(by_year), index=ids, variable_name="__column", value_name=value)
    mapping = pl.DataFrame(
        {
            "__column": list(by_year),
            "measure": [stem for stem, _ in by_year.values()],
            "year": [str(year) for _, year in by_year.values()],
        }
    )
    long = long.join(mapping, on="__column", how="left").drop("__column")
    keep = [*ids, *(["measure"] if len(stems) > 1 else []), "year", value]
    logger.info("staging_years_unpivoted", year_columns=len(by_year), measures=len(stems))
    return long.select(keep)


# "45.1 to 48.3" under a header that names an interval (FluVaxView's "95% CI (%)"): the bounds
# packed in one cell without the estimate.
RANGE_CELL = rf"^\s*{_NUM}\s*(?:to|–|—)\s*{_NUM}\s*$"
_INTERVAL_HEADER = re.compile(r"(\bci\b|confidence|interval)", re.IGNORECASE)


def _range_column(df: pl.DataFrame, col: str) -> bool:
    if df.schema[col] != pl.String or not _INTERVAL_HEADER.search(col):
        return False
    cleaned = df.select(_clean_tokens(col).alias(col)).get_column(col).drop_nulls()
    return cleaned.len() > 0 and bool(cleaned.str.contains(RANGE_CELL).all())


def split_interval_ranges(df: pl.DataFrame) -> pl.DataFrame:
    """Columns named for an interval whose every non-empty cell is "lo to hi" become ``<col> lower``
    and ``<col> upper``. Others are left as they are."""
    ranges = [c for c in df.columns if _range_column(df, c)]
    if not ranges:
        return df
    parts: list[pl.Expr] = []
    order: list[str] = []
    for col in df.columns:
        if col not in ranges:
            order.append(col)
            continue
        groups = _clean_tokens(col).str.extract_groups(RANGE_CELL)
        for index, name in enumerate((f"{col} lower", f"{col} upper"), start=1):
            parts.append(groups.struct.field(str(index)).str.replace_all(",", "").alias(name))
            order.append(name)
    logger.info("staging_interval_ranges_split", columns=len(ranges))
    return df.with_columns(parts).select(order)


def unpack_intervals(df: pl.DataFrame) -> pl.DataFrame:
    """Text frame with packed "value (lower, upper)" cells split into three columns (``x``,
    ``x lower``, ``x upper``), after any years laid out as columns are turned into rows. Only a
    column whose every non-empty cell is packed is split; others are left as they are."""
    df = split_interval_ranges(_years_to_rows(df))
    packed = [c for c in df.columns if df.schema[c] == pl.String and _packed(df, c)]
    if not packed:
        return df
    parts: list[pl.Expr] = []
    for col in packed:
        groups = _clean_tokens(col).str.extract_groups(PACKED_INTERVAL)
        for index, name in enumerate((col, f"{col} lower", f"{col} upper"), start=1):
            parts.append(groups.struct.field(str(index)).str.replace_all(",", "").alias(name))
    order: list[str] = []
    for col in df.columns:
        order += [col, f"{col} lower", f"{col} upper"] if col in packed else [col]
    logger.info("staging_intervals_unpacked", columns=len(packed))
    return df.with_columns(parts).select(order)


def _flag_masks(
    cleaned: pl.Series, public: bool, source_words: bool = False
) -> tuple[pl.Series, pl.Series, pl.Series] | None:
    """(suppressed, unreliable, not-applicable) masks for a column that holds numbers plus source
    markers only, or None when any other text is present (the column is left to the typing rules)."""
    # Fail closed (ADR014 §7): a source's markers are read only in a table declared public and
    # accepted, or in a CDC WONDER-shaped export (``source_words``). Any other file is never
    # reinterpreted: a patient file's "<5", "Suppressed" or "Unreliable" stays the text it is.
    if not (public or source_words):
        return None
    folded = cleaned.str.strip_chars().str.to_lowercase()
    # A public table's own markers ("<10", "(D)", "*") are suppression. In a WONDER export that is
    # not declared, only the unambiguous words count ("<5" could be a censored value).
    pattern = SOURCE_SUPPRESSED_PATTERN if public else SOURCE_SUPPRESSED_WORDS
    suppressed = folded.str.contains(pattern).fill_null(False)
    unreliable = folded.str.contains(SOURCE_UNRELIABLE_PATTERN).fill_null(False)
    not_applicable = folded.is_in(list(SOURCE_NOT_APPLICABLE_TOKENS)).fill_null(False)
    if not bool(suppressed.any() or unreliable.any()):
        return None
    numbers = folded.str.replace(SOURCE_UNRELIABLE_PATTERN, "").str.strip_chars()
    rest = numbers.filter(~(suppressed | not_applicable) & numbers.is_not_null())
    rest = rest.filter(~rest.is_in(list(MISSING_VALUE_TOKENS)))
    if rest.len() == 0 or not bool(rest.str.contains(_FLOAT_RE).all()):
        return None
    return suppressed, unreliable, not_applicable


def split_source_flags(df: pl.DataFrame, *, public: bool = False, source_words: bool = False) -> pl.DataFrame:
    """Text frame with each source's own suppression markers moved out of its numbers (ADR014 §10).

    CDC WONDER writes "Suppressed" for deaths of 1-9 (and for rates built on them), "12.3
    (Unreliable)" for a rate on fewer than 20 deaths and "Not Applicable" where there is no rate.
    A column whose other cells are all numbers keeps its numbers; a "Suppressed" cell becomes empty
    and ``<column> suppressed`` is true there, and an unreliable value keeps its number and
    ``<column> unreliable`` is true. A suppressed cell is never a zero and never a missing value
    the answer could fill in: answers name it as suppressed at the source.
    """
    added: list[pl.Expr] = []
    order: list[str] = []
    for col in df.columns:
        order.append(col)
        if df.schema[col] != pl.String or is_id_column_name(col):
            continue
        cleaned = df.select(_clean(col)).to_series()
        masks = _flag_masks(cleaned, public, source_words)
        if masks is None:
            continue
        suppressed, unreliable, not_applicable = masks
        flag_names = [f"{col}{suffix}" for suffix in (SUPPRESSED_FLAG_SUFFIX, UNRELIABLE_FLAG_SUFFIX)]
        if any(name in df.columns for name in flag_names):
            # A flag column of that name exists already: splitting would lose which cells were
            # suppressed, so the column stays text (fail closed: no number is read from it).
            logger.info("staging_source_flags_skipped", column_index=df.columns.index(col), reason="name_taken")
            continue
        value = cleaned.str.replace(SOURCE_UNRELIABLE_PATTERN, "").str.strip_chars()
        value = value.set(suppressed | not_applicable, None)
        added.append(pl.lit(value).alias(col))
        for suffix, mask in ((SUPPRESSED_FLAG_SUFFIX, suppressed), (UNRELIABLE_FLAG_SUFFIX, unreliable)):
            name = f"{col}{suffix}"
            if bool(mask.any()):
                added.append(pl.lit(pl.Series(name, mask.cast(pl.String).str.to_lowercase())).alias(name))
                order.append(name)
        logger.info(
            "staging_source_flags_split",
            column_index=df.columns.index(col),
            suppressed=int(suppressed.sum()),
            unreliable=int(unreliable.sum()),
        )
    return df.with_columns(added).select(order) if added else df


def stage_text_frame(
    df: pl.DataFrame, *, header_row_number: int = 1, public: bool = False, source_words: bool = False
) -> StagedTable:
    """Type an all-text frame (one row per data line). ``header_row_number``: the header's file row.

    Raises:
        TypeError: a column is not text (callers stage raw text only).
        StagingError: a row has no ID and no values.
    """
    not_text = [c for c, t in df.schema.items() if t != pl.String]
    if not_text:
        raise TypeError(f"stage_text_frame needs text columns; got non-text {not_text}")
    df = _name_blank_headers(df)
    _reject_case_twin_headers(df)

    units = _units_row(df)
    body = df.slice(1) if units else df
    first_data_row = header_row_number + (2 if units else 1)
    _reject_empty_rows(body, first_data_row)
    typed, notes, dates = _type_columns(
        split_source_flags(unpack_intervals(body), public=public, source_words=source_words)
    )
    log = logger.bind(rows=typed.height, columns=typed.width)
    for note in notes:
        log.info("staging_column_note", column_index=df.columns.index(note.column), reason=note.reason)
    if units:
        log.info("staging_units_row_kept", unit_columns=len(units))
    return StagedTable(data=typed, notes=tuple(notes), units=units or {}, dates=dates)


def _name_blank_headers(df: pl.DataFrame) -> pl.DataFrame:
    """A blank header (R's ``write.csv`` row-name column, ``""``) becomes ``column_<i>``, as in Excel.

    A column with no name cannot be queried (the query view needs a name for every column).
    """
    renames = {c: f"column_{i}" for i, c in enumerate(df.columns) if not c.strip()}
    taken = {c.casefold() for c in df.columns if c not in renames}  # the query engine ignores case
    if any(name.casefold() in taken for name in renames.values()):
        renames = {c: f"column_{i}_unnamed" for i, c in enumerate(df.columns) if not c.strip()}
    return df.rename(renames) if renames else df


def _reject_case_twin_headers(df: pl.DataFrame) -> None:
    """Refuse headers that differ only in letter case (``Age`` and ``age``).

    The query engine (DuckDB) reads column names without regard to case, so it would rename one
    of the two and its answers could read the wrong column. Header names are not cell values, so
    the message may quote them.

    Raises:
        StagingError: two or more headers are the same apart from case.
    """
    by_folded: dict[str, list[str]] = {}
    for name in df.columns:
        by_folded.setdefault(name.casefold(), []).append(name)
    twins = [names for names in by_folded.values() if len(names) > 1]
    if twins:
        listed = "; ".join(" and ".join(f"'{n}'" for n in names) for names in twins)
        raise StagingError(
            f"Columns {listed} differ only in upper and lower case, which this app reads as one name. "
            "Rename one of them and upload the file again."
        )


def read_csv_typed(file_bytes: bytes, separator: str = ",", *, public: bool = False) -> StagedTable:
    """Read delimited text (CSV; ``separator`` for TSV/PSV) as text over the whole file, then stage.

    A CDC WONDER-shaped export (first header "Notes"): its notes block under the data (from the
    "---" line) is kept as documentation, never rows, and its source words ("Suppressed",
    "12.3 (Unreliable)") become flag columns (``split_source_flags``). ``public``: the upload was
    declared public estimates and accepted (``stage_declared``). Any other file is read as on
    main: a patient file's "<5" or "Suppressed" stays the text it is.

    Raises:
        StagingError: the file cannot be parsed, or a row has no ID and no values.
    """
    wonder = is_notes_export(file_bytes, separator)
    data_bytes, export_notes = split_export_notes(file_bytes, separator)
    try:
        text = pl.read_csv(io.BytesIO(data_bytes), separator=separator, infer_schema=False)
    except (pl.exceptions.ComputeError, pl.exceptions.NoDataError) as e:
        raise StagingError(f"The {_delimited_kind(separator)} file could not be read: {type(e).__name__}") from e
    staged = stage_text_frame(_drop_trailing_blank_rows(text), public=public, source_words=wonder)
    staged = replace(staged, source_format=CDC_WONDER_FORMAT if wonder else "")
    if not export_notes.strip():
        return staged
    lines = sum(1 for line in export_notes.splitlines() if line.strip().strip('"').strip() not in ("", "---"))
    note = (
        f"The notes under the data ({lines} line{'s' if lines != 1 else ''}: dataset, query, citation) "
        "were kept as the dataset's documentation, not as rows."
    )
    return replace(staged, export_notes=export_notes, table_notes=(note,))


def is_notes_export(file_bytes: bytes, separator: str) -> bool:
    """True when the first header is "Notes" (CDC WONDER's layout): read the file on its own, so
    its subtotal rows, notes block and suppressed values are handled (never as a numbered set)."""
    first_header = file_bytes.split(b"\n", 1)[0].split(separator.encode(), 1)[0].strip().strip(b'"').strip()
    return first_header.lower() == EXPORT_NOTES_COLUMN.encode()


def split_export_notes(file_bytes: bytes, separator: str = "\t") -> tuple[bytes, str]:
    """(the data lines, the notes block as text) of a CDC WONDER-shaped export; (``file_bytes``,
    "") for any other file. WONDER-shaped: the first header is "Notes", a "---" line follows
    the data, and no line from there on holds the separator (each is one quoted note)."""
    if not is_notes_export(file_bytes, separator):
        return file_bytes, ""
    match = _EXPORT_NOTES_START.search(file_bytes)
    if match is None:
        return file_bytes, ""
    block = file_bytes[match.start() :]
    if separator.encode() in block:
        return file_bytes, ""  # data-shaped lines below "---": not a notes block, never hidden
    return file_bytes[: match.start()], block.decode("utf-8", errors="replace")


def _delimited_kind(separator: str) -> str:
    return {",": "CSV", "\t": "tab-separated", "|": "pipe-separated"}.get(separator, "text")


def read_xport_typed(file_bytes: bytes) -> StagedTable:
    """Read a SAS transport file (``.xpt``, as NHANES publishes), keeping its own column types.

    SAS stores every number as a float, so a column of whole numbers (codes such as 1/2, IDs,
    counts) becomes Int64 when every value fits; other numbers stay Float64. Empty text cells
    become null. Dates SAS formats as dates arrive as dates.

    Raises:
        StagingError: the file is not a readable SAS transport file.
    """
    import pyreadstat

    try:
        df, _meta = pyreadstat.read_xport(io.BytesIO(file_bytes), output_format="polars")
    except (pyreadstat.ReadstatError, pyreadstat.PyreadstatError) as e:
        raise StagingError(f"The SAS transport (.xpt) file could not be read: {type(e).__name__}") from e
    if not isinstance(df, pl.DataFrame):  # output_format="polars" always gives one
        raise StagingError("The SAS transport (.xpt) file could not be read as a table")
    return StagedTable(data=_whole_floats_as_int(df))


def _whole_floats_as_int(df: pl.DataFrame) -> pl.DataFrame:
    """Float columns whose values are all whole and fit Int64 become Int64; blank text is null.

    NaN counts as missing (it is how a reader may store a SAS missing value). A column with any
    fraction, or a value too large for Int64, stays Float64 so no digit is lost. A column with no
    value at all (read as type Null) becomes text, as an empty CSV column does.
    """
    limit = float(2**53)  # every whole float below this converts to Int64 exactly
    casts: list[pl.Expr] = []
    for col, dtype in df.schema.items():
        if dtype == pl.String:
            casts.append(_clean(col).alias(col))
        elif dtype == pl.Null:  # a column with no value at all: text, as a CSV's would be
            casts.append(pl.col(col).cast(pl.String))
        elif dtype.is_float():
            values = df.get_column(col).fill_nan(None).drop_nulls()
            whole = values.len() > 0 and bool(((values == values.floor()) & (values.abs() < limit)).all())
            filled = pl.col(col).fill_nan(None)
            casts.append((filled.cast(pl.Int64) if whole else filled).alias(col))
    return df.with_columns(casts) if casts else df


def _cell_text(value: object) -> str | None:
    """Excel cell as the text a CSV export would hold (dates ISO, booleans true/false)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        if value.hour == value.minute == value.second == value.microsecond == 0:
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _unique_headers(raw: list[object]) -> list[str]:
    """Header names: blank cells become column_<i>, repeats get .1, .2 (as pandas did)."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, cell in enumerate(raw):
        name = _cell_text(cell)
        name = name.strip() if name and name.strip() else f"column_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


HEADER_SCAN_ROWS: Final = 5
_HEADER_MIN_SCORE: Final = 0.3


def _header_score(row: list[object]) -> float:
    """How much a row looks like column headers: filled, textual, short, distinct (0..1)."""
    values = [str(v).strip() for v in row if v is not None and str(v).strip()]
    if not values:
        return 0.0
    score = 0.4 * len(values) / len(row)
    counted = [v for v in values if not (_NUMBER_RE.match(v) and len(v) > 3)]
    if not counted:
        return score
    strings = sum(1 for v in counted if not _NUMBER_RE.match(v))
    avg_len = sum(len(v) for v in counted) // len(counted)
    score += 0.3 * strings / len(counted)
    if 5 <= avg_len <= 50:
        score += 0.15
    elif avg_len < 5:
        score += 0.15 * avg_len / 5
    else:
        score += 0.15 * 50 / avg_len
    score += 0.15 * min(len(set(values)) / len(counted), 1.0)
    return score


def detect_header_row(rows: list[list[object]], max_rows: int = HEADER_SCAN_ROWS) -> int:
    """0-based index of the row among the first ``max_rows`` that best looks like headers (else 0).

    Skips title rows above the real header (a merged "Study cohort" banner).
    """
    scanned = rows[:max_rows]
    # Score each row against the widest one: a sheet can store a one-cell title banner as a
    # one-cell row, which is "fully filled" only relative to its own length.
    width = max((len(r) for r in scanned), default=0)
    scores = [_header_score(list(r) + [None] * (width - len(r))) for r in scanned]
    if not scores or max(scores) <= _HEADER_MIN_SCORE:
        return 0
    return scores.index(max(scores))


def read_excel_rows(file_bytes: bytes) -> list[list[object]]:
    """Cell values of the first sheet, row by row (formulas as their cached values).

    Raises:
        StagingError: the workbook cannot be read.
    """
    return _read_first_sheet(file_bytes)[0]


def _read_first_sheet(file_bytes: bytes) -> tuple[list[list[object]], ExcelSheets]:
    """The first sheet's rows, plus its name and the names of the other sheets that hold data.

    Only the first sheet is loaded. The others are named so the upload can say they were not
    read: a workbook whose data spans sheets must never look fully loaded. Empty sheets (an
    unused "Sheet2") are not named.

    Raises:
        StagingError: the workbook cannot be read or has no sheet.
    """
    if not file_bytes.startswith(_ZIP_MAGIC):
        return _read_first_xls_sheet(file_bytes)  # .xlsx is a ZIP; anything else may be BIFF

    import zipfile

    import openpyxl
    from openpyxl.utils.exceptions import InvalidFileException

    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    except (InvalidFileException, zipfile.BadZipFile, KeyError, OSError) as e:
        raise StagingError(f"The Excel file could not be read: {type(e).__name__}") from e
    try:
        if not wb.worksheets:
            raise StagingError("The Excel file has no sheets")
        first, *others = wb.worksheets
        # Read-only mode trusts each sheet's declared size, and some exporters declare A1:A1
        # (IHME's workbooks do): every row would come back one cell wide, title only.
        for sheet in wb.worksheets:
            sheet.reset_dimensions()
        rows = [list(r) for r in first.iter_rows(values_only=True)]
        skipped = tuple(sheet.title for sheet in others if _sheet_has_values(sheet))
        return rows, ExcelSheets(read=first.title, skipped=skipped, empty=len(others) - len(skipped))
    finally:
        wb.close()


# What openpyxl raises on a cell it cannot parse (a date serial out of range, a bad number).
_SHEET_CELL_ERRORS: Final = (ValueError, OverflowError, KeyError, TypeError)


def _read_first_xls_sheet(file_bytes: bytes) -> tuple[list[list[object]], ExcelSheets]:
    """``_read_first_sheet`` for a legacy .xls workbook (BIFF2-8, raw or in an OLE2 file), via xlrd.

    Cells come back as the values openpyxl gives: numbers, text, booleans, datetimes for cells
    formatted as dates (``_cell_text`` writes them as ISO text), None for empty cells. An error
    cell holds its Excel text (``#N/A``, a missing-value marker; ``#DIV/0!``).

    Raises:
        StagingError: the workbook cannot be read (encrypted, damaged, or not a workbook).
    """
    import struct

    import xlrd
    from xlrd.compdoc import CompDocError

    if not file_bytes:
        raise StagingError("The Excel (.xls) file could not be read: it is empty")
    try:
        # xlrd writes format warnings to a log file (stdout by default): keep them out of output.
        wb = xlrd.open_workbook(file_contents=file_bytes, on_demand=True, logfile=io.StringIO())
    except (xlrd.XLRDError, CompDocError, struct.error, IndexError, UnicodeDecodeError, ValueError) as e:
        raise StagingError(f"The Excel (.xls) file could not be read: {type(e).__name__}") from e
    try:
        names = wb.sheet_names()
        if not names:
            raise StagingError("The Excel file has no sheets")
        first = wb.sheet_by_index(0)
        rows = [[_xls_value(first.cell(r, c), wb.datemode) for c in range(first.ncols)] for r in range(first.nrows)]
        skipped = tuple(
            name for i, name in enumerate(names[1:], start=1) if _xls_sheet_has_values(wb.sheet_by_index(i))
        )
    except (xlrd.XLRDError, struct.error, IndexError, UnicodeDecodeError, ValueError) as e:
        raise StagingError(f"The Excel (.xls) file could not be read: {type(e).__name__}") from e
    finally:
        wb.release_resources()
    return rows, ExcelSheets(read=names[0], skipped=skipped, empty=len(names) - 1 - len(skipped))


def _xls_sheet_has_values(sheet: Any) -> bool:
    """``_sheet_has_values`` for an unread .xls sheet: one that cannot be scanned counts as
    holding data (named as not read), never failing the upload of the sheet that is read."""
    import struct

    import xlrd

    try:
        return _rows_have_values((_xls_value(cell, 0) for cell in row) for row in sheet.get_rows())
    except (xlrd.XLRDError, struct.error, IndexError, UnicodeDecodeError, *_SHEET_CELL_ERRORS) as e:
        logger.warning("staging_excel_other_sheet_unscannable", error_type=type(e).__name__)
        return True


def _xls_value(cell: Any, datemode: int) -> object:
    """One xlrd cell as openpyxl would give it (see ``_read_first_xls_sheet``)."""
    import xlrd

    if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
        return None
    if cell.ctype == xlrd.XL_CELL_TEXT:
        return cell.value if str(cell.value).strip() else None
    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            moment = xlrd.xldate.xldate_as_datetime(cell.value, datemode)
            # A serial below 1 is a time of day with no date (openpyxl gives a time, not 1899-12-31).
            return moment.time() if 0 <= cell.value < 1 else moment
        except (xlrd.xldate.XLDateError, OverflowError, ValueError):
            return cell.value  # a serial no date can hold: keep the number, never drop it
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return bool(cell.value)
    if cell.ctype == xlrd.XL_CELL_ERROR:
        return xlrd.error_text_from_code.get(cell.value, "#N/A")
    if isinstance(cell.value, float) and cell.value.is_integer():
        return int(cell.value)  # BIFF stores every number as a float; openpyxl gives whole ones as int
    return cell.value


def _rows_have_values(rows: Iterable[Iterable[object]]) -> bool:
    """True when any cell holds a value (stops at the first one)."""
    return any(cell is not None and str(cell).strip() for row in rows for cell in row)


def _sheet_has_values(sheet: Any) -> bool:
    """True when any cell of the sheet holds a value (stops at the first one).

    Only the first sheet is read. Another sheet that cannot even be scanned counts as holding
    data: it is named as not read, and it never fails the upload of the sheet that is read.
    """
    try:
        return _rows_have_values(sheet.iter_rows(values_only=True))
    except _SHEET_CELL_ERRORS as e:
        logger.warning("staging_excel_other_sheet_unscannable", error_type=type(e).__name__)
        return True


def read_excel_typed(file_bytes: bytes, header_row: int | None = None, *, public: bool = False) -> StagedTable:
    """Read the first sheet of an .xlsx or .xls workbook as text, then stage.

    Args:
        file_bytes: Workbook bytes.
        header_row: 0-based header row; detected (``detect_header_row``) when None.

    Raises:
        StagingError: the workbook cannot be read or has no header row.
    """
    rows, sheets = _read_first_sheet(file_bytes)
    if header_row is None:
        header_row = detect_header_row(rows)
    if len(rows) <= header_row:
        raise StagingError("The Excel sheet has no header row")
    # Each row comes back as long as its last stored cell (reset_dimensions), so the table is as
    # wide as its widest row: a column with data under a blank header cell is kept as
    # column_N, never cut off. Footnote rows are split off first, so a note written further
    # right never adds a column.
    span = max(len(row) for row in rows[header_row:])
    body = [[_cell_text(v) for v in (r + [None] * span)[:span]] for r in rows[header_row + 1 :]]
    body, footnote_rows = _split_footnotes(body)
    width = max([len(rows[header_row])] + [_last_filled(row) for row in body])
    headers = _unique_headers((rows[header_row] + [None] * width)[:width])
    body = [row[:width] for row in body]
    if footnote_rows:
        logger.info("staging_excel_footnote_rows_dropped", rows=footnote_rows)
    # Trailing all-empty rows are sheet formatting, not data lines (unlike in a CSV).
    text = _drop_trailing_blank_rows(pl.DataFrame(body, schema=dict.fromkeys(headers, pl.String), orient="row"))
    staged = stage_text_frame(text, header_row_number=header_row + 1, public=public)
    return replace(staged, sheets=replace(sheets, footnote_rows=footnote_rows))


def _filled(row: list[str | None]) -> list[str]:
    return [value for value in row if value is not None and value.strip()]


def _last_filled(row: list[str | None]) -> int:
    """How far into the row its last filled cell is (0 for a blank row)."""
    return max((at + 1 for at, value in enumerate(row) if value is not None and value.strip()), default=0)


def _is_footnote(row: list[str | None]) -> bool:
    """One filled cell, in a table of two or more columns, that reads as prose: two or more
    words, and either two of them with letters or a sentence ending ("Source: ...", "Note:
    rates per 100,000", "Generated 2026.").

    In a one-column table every row has one filled cell, so none is a footnote there. An ID
    such as "MRN 0012" has one word with letters and no sentence ending, and a number has no
    space, so a row that names a patient is never read as a footnote. A footnote row leaves
    the data, so the upload always reports how many rows it read as notes.
    """
    filled = _filled(row)
    if len(row) < 2 or len(filled) != 1:
        return False
    text = filled[0].strip()
    words = text.split()
    if len(words) < 2:
        return False
    lettered = sum(1 for word in words if any(char.isalpha() for char in word))
    return lettered >= 2 or (lettered >= 1 and text[-1] in ".:;")


def _split_footnotes(body: list[list[str | None]]) -> tuple[list[list[str | None]], int]:
    """The rows above any footnotes at the end of a sheet, and how many footnote rows there were.

    A footnote block is one or more footnote rows (``_is_footnote``) at the end of the sheet with
    a blank row above it. Without that blank row the rows are data-adjacent and stay data (and a
    row with no ID and no values still fails). Blank rows around footnotes go with them.
    """
    end = len(body)
    footnotes = 0
    while True:
        last = end
        while last > 0 and not _filled(body[last - 1]):
            last -= 1
        first = last
        while first > 0 and _is_footnote(body[first - 1]):
            first -= 1
        # The block must stand below a blank row, with data above that.
        if first == last or first == 0 or _filled(body[first - 1]):
            break
        footnotes += last - first
        end = first
    return body[:end], footnotes


# --- the report shown to the user -------------------------------------------------------------


_NOTE_TEXT: Final[dict[str, str]] = {
    "leading_zero": "kept as text: codes with leading zeros (like 02139) must not become numbers",
    "not_numeric": "kept as text: {n} values are not plain numbers (like '<5' or '1,234'), so it is not averaged",
    "date_values_unparsed": "read as dates; {n} values matched no date format and count as missing",
    "int_overflow": "kept as text: {n} whole numbers are too large to store as numbers without losing digits",
    "missing_tokens": "{n} values were missing-value markers (like NA, . or #N/A) and are stored as missing",
    "shared_column_type": "type changed to match the same column in another table, so the tables join",
    "ambiguous_dates": "kept as text: dates could be day/month or month/day and nothing in the file settles which",
    "not_dates": "kept as text: {n} values are not dates",
    "from_file_names": "added from the file names: {n} files with the same columns were read as one table",
}


def describe_sheets(sheets: Mapping[str, object] | None) -> str:
    """The upload note for a workbook, or "" when nothing in it was left out.

    Only the first sheet is read. When another sheet holds data, the note says which sheet was
    read out of how many and names every sheet that was not, so the user never takes one sheet's
    answers for the whole workbook's. Footnote rows below the data are counted too.
    """
    if not sheets:
        return ""
    raw_skipped = sheets.get("skipped")
    skipped = [str(name) for name in raw_skipped] if isinstance(raw_skipped, list) else []
    empty = _count(sheets.get("empty"))
    footnotes = _count(sheets.get("footnote_rows"))
    lines: list[str] = []
    if skipped:
        unread = [f"“{name}”" for name in skipped]
        if empty:
            unread.append("1 empty sheet" if empty == 1 else f"{empty} empty sheets")
        names = unread[0] if len(unread) == 1 else ", ".join(unread[:-1]) + f" and {unread[-1]}"
        lines.append(
            f"Read 1 of {1 + len(skipped) + empty} sheets: “{sheets.get('read', '')}”. Not read: {names}. "
            "To use another sheet, save it as its own file and upload it."
        )
    if footnotes == 1:
        lines.append("1 row below the data, after a blank row, was read as a note and left out of the data.")
    elif footnotes > 1:
        lines.append(
            f"{footnotes} rows below the data, after a blank row, were read as notes and left out of the data."
        )
    return " ".join(lines)


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def describe_column_types(notes: list[dict[str, str | int]], units: dict[str, str]) -> list[str]:
    """One sentence per staging decision the user should know about, with its count as it is."""
    lines = []
    for note in notes:
        text = _NOTE_TEXT[str(note["reason"])].format(n=int(note.get("failed_values", 0)))
        lines.append(f"{note['column']}: {text}.")
    if units:
        lines.append("The row under the header was read as units, not a patient, and is not in the data.")
    lines.extend(f"{column}: unit {unit} (from the units row under the header)." for column, unit in units.items())
    return lines
