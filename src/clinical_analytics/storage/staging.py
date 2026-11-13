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
* Dates are parsed per value against a fixed list of formats. Day-first and month-first dates
  are only used when some value settles the order (a day above 12), and the same order must
  hold for every separator; otherwise the column stays text. A value that matches no format
  keeps the whole column as text (``not_dates``), the same rule as for numbers: nothing is
  silently turned into a missing value.
* A column that cannot take a type stays text, and only that column: a note says which column
  and why (``ColumnTypeNote``). Nothing falls back to "every column is text".

A first row that is a units sub-header (``years``, ``mg/dL`` above numeric columns, with no ID)
is kept as unit metadata and is not a patient; the report says that row was removed. Trailing
lines with no values (``,,,`` written by exporters) are dropped; any other row with no ID and no
values fails the upload.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Final, Literal

import polars as pl
import structlog

from clinical_analytics.contracts.config import MISSING_VALUE_TOKENS

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
]

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

# Unambiguous formats, tried first, in this order.
_ISO_DATE_FORMATS: Final = ("%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y", "%d.%m.%Y")
_DATETIME_FORMATS: Final = (
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
)
# Order-ambiguous families: (day-first, month-first).
_AMBIGUOUS_FAMILIES: Final = (("%d/%m/%Y", "%m/%d/%Y"), ("%d-%m-%Y", "%m-%d-%Y"))


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
    as notes ("Source: ..."), not patients.
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


def is_id_column_name(name: str) -> bool:
    return name.strip().lower() in ID_COLUMN_NAMES


# --- per-column typing ------------------------------------------------------------------------


@dataclass(frozen=True)
class _ColumnPlan:
    expr: pl.Expr
    dtype: pl.DataType
    notes: tuple[ColumnTypeNote, ...] = ()


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


def _parse_dates(s: pl.Series) -> tuple[pl.Series | None, NoteReason | None]:
    """Parse a trimmed text series as Date/Datetime, or say why not (None when nothing parses)."""
    parsed_dt = [s.str.strptime(pl.Datetime("us"), fmt, strict=False) for fmt in _DATETIME_FORMATS]
    parsed_date = [s.str.strptime(pl.Date, fmt, strict=False) for fmt in _ISO_DATE_FORMATS]

    settled: set[str] = set()  # orders that some separator family settled ("dmy", "mdy")
    for day_first, month_first in _AMBIGUOUS_FAMILIES:
        dmy = s.str.strptime(pl.Date, day_first, strict=False)
        mdy = s.str.strptime(pl.Date, month_first, strict=False)
        only_dmy = int((dmy.is_not_null() & mdy.is_null()).sum())
        only_mdy = int((mdy.is_not_null() & dmy.is_null()).sum())
        disagree = int((dmy.is_not_null() & mdy.is_not_null() & (dmy != mdy)).sum())
        if only_dmy and only_mdy:
            return None, "ambiguous_dates"  # both orders are in the column
        if only_dmy:
            parsed_date.append(dmy)
            settled.add("dmy")
        elif only_mdy:
            parsed_date.append(mdy)
            settled.add("mdy")
        elif disagree:
            return None, "ambiguous_dates"  # nothing settles the order
        else:
            parsed_date.append(dmy)  # every value reads the same either way
    if len(settled) > 1:
        return None, "ambiguous_dates"  # "/" reads day-first, "-" month-first: two conventions

    as_date = pl.select(pl.coalesce(parsed_date)).to_series()
    as_dt = pl.select(pl.coalesce(parsed_dt)).to_series()
    if as_dt.is_not_null().any():
        combined = pl.select(pl.coalesce([as_dt, as_date.cast(pl.Datetime("us"))])).to_series()
    else:
        combined = as_date
    if not combined.is_not_null().any():
        return None, None
    return combined, None


def _plan_column(df: pl.DataFrame, col: str) -> _ColumnPlan:
    s = df.select(_clean(col)).to_series()
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

    parsed, why_not = _parse_dates(s.set(_token_mask(s).fill_null(False), None) if tokens else s)
    if why_not is not None:
        return text_with(why_not, 0)
    if parsed is not None:
        failed = values.len() - int(parsed.is_not_null().sum())
        if failed:
            # Same rule as numbers: an unparsed value keeps the column as text, so no value is
            # silently stored as missing (and the original text stays queryable).
            return text_with("not_dates", failed)
        return typed(pl.lit(parsed).alias(col), parsed.dtype)

    numeric = int(is_float.sum())
    if numeric / n >= NUMERIC_NOTE_SHARE:
        return text_with("not_numeric", values.len() - numeric)
    return text


def _type_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, list[ColumnTypeNote]]:
    plans = {col: _plan_column(df, col) for col in df.columns}
    typed = df.with_columns([plan.expr.alias(col) for col, plan in plans.items()])
    return typed, [note for plan in plans.values() for note in plan.notes]


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
    rest, _ = _type_columns(df.slice(1))
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


def stage_text_frame(df: pl.DataFrame, *, header_row_number: int = 1) -> StagedTable:
    """Type an all-text frame (one row per data line). ``header_row_number``: the header's file row.

    Raises:
        TypeError: a column is not text (callers stage raw text only).
        StagingError: a row has no ID and no values.
    """
    not_text = [c for c, t in df.schema.items() if t != pl.String]
    if not_text:
        raise TypeError(f"stage_text_frame needs text columns; got non-text {not_text}")

    units = _units_row(df)
    body = df.slice(1) if units else df
    first_data_row = header_row_number + (2 if units else 1)
    _reject_empty_rows(body, first_data_row)
    typed, notes = _type_columns(body)
    log = logger.bind(rows=typed.height, columns=typed.width)
    for note in notes:
        log.info("staging_column_note", column_index=df.columns.index(note.column), reason=note.reason)
    if units:
        log.info("staging_units_row_kept", unit_columns=len(units))
    return StagedTable(data=typed, notes=tuple(notes), units=units or {})


def read_csv_typed(file_bytes: bytes) -> StagedTable:
    """Read CSV bytes as text over the whole file, then stage.

    Raises:
        StagingError: the CSV cannot be parsed, or a row has no ID and no values.
    """
    try:
        text = pl.read_csv(io.BytesIO(file_bytes), infer_schema=False)
    except (pl.exceptions.ComputeError, pl.exceptions.NoDataError) as e:
        raise StagingError(f"The CSV file could not be read: {type(e).__name__}") from e
    return stage_text_frame(_drop_trailing_blank_rows(text))


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


def _sheet_has_values(sheet: Any) -> bool:
    """True when any cell of the sheet holds a value (stops at the first one)."""
    rows = sheet.iter_rows(values_only=True)
    return any(cell is not None and str(cell).strip() for row in rows for cell in row)


def read_excel_typed(file_bytes: bytes, header_row: int | None = None) -> StagedTable:
    """Read the first sheet of an .xlsx as text, then stage.

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
    headers = _unique_headers(rows[header_row])
    width = len(headers)
    body = [[_cell_text(v) for v in (r + [None] * width)[:width]] for r in rows[header_row + 1 :]]
    body, footnote_rows = _split_footnotes(body)
    if footnote_rows:
        logger.info("staging_excel_footnote_rows_dropped", rows=footnote_rows)
    # Trailing all-empty rows are sheet formatting, not data lines (unlike in a CSV).
    text = _drop_trailing_blank_rows(pl.DataFrame(body, schema=dict.fromkeys(headers, pl.String), orient="row"))
    staged = stage_text_frame(text, header_row_number=header_row + 1)
    return replace(staged, sheets=replace(sheets, footnote_rows=footnote_rows))


def _filled(row: list[str | None]) -> list[str]:
    return [value for value in row if value is not None and value.strip()]


def _is_footnote(row: list[str | None]) -> bool:
    """One filled cell holding several words ("Source: ...", "Note: rates per 100,000").

    An ID or a number has no space, so a row that names a patient is never read as a footnote.
    A footnote row leaves the data, so the upload always reports how many rows it read as notes.
    """
    filled = _filled(row)
    return len(filled) == 1 and len(filled[0].split()) > 1


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


def _shown_count(part: int, total: int, limit: int) -> str:
    """A value count that never reveals 1..limit-1, directly or as the rest of the column."""
    if 0 < part < limit:
        return f"fewer than {limit}"
    if 0 < total - part < limit:
        return f"all but fewer than {limit}"
    return str(part)


_NOTE_TEXT: Final[dict[str, str]] = {
    "leading_zero": "kept as text: codes with leading zeros (like 02139) must not become numbers",
    "not_numeric": "kept as text: {n} values are not plain numbers (like '<5' or '1,234'), so it is not averaged",
    "date_values_unparsed": "read as dates; {n} values matched no date format and count as missing",
    "int_overflow": "kept as text: {n} whole numbers are too large to store as numbers without losing digits",
    "missing_tokens": "{n} values were missing-value markers (like NA, . or #N/A) and are stored as missing",
    "shared_column_type": "type changed to match the same column in another table, so the tables join",
    "ambiguous_dates": "kept as text: dates could be day/month or month/day and nothing in the file settles which",
    "not_dates": "kept as text: {n} values are not dates",
}


def describe_sheets(sheets: dict[str, object] | None) -> str:
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
        lines.append(
            "1 row below the data, after a blank row, was read as a note, not a patient, and is not in the data."
        )
    elif footnotes > 1:
        lines.append(
            f"{footnotes} rows below the data, after a blank row, were read as notes, not patients, "
            "and are not in the data."
        )
    return " ".join(lines)


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def describe_column_types(notes: list[dict[str, str | int]], units: dict[str, str], small_cell_limit: int) -> list[str]:
    """One sentence per staging decision the user should know about (counts small-cell safe)."""
    lines = []
    for note in notes:
        failed, checked = int(note.get("failed_values", 0)), int(note.get("checked_values", 0))
        text = _NOTE_TEXT[str(note["reason"])].format(n=_shown_count(failed, checked, small_cell_limit))
        lines.append(f"{note['column']}: {text}.")
    if units:
        lines.append("The row under the header was read as units, not a patient, and is not in the data.")
    lines.extend(f"{column}: unit {unit} (from the units row under the header)." for column, unit in units.items())
    return lines
