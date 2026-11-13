# SPDX-License-Identifier: MIT
"""Typed staging: read an upload once as text, then give every column one explicit type.

Types are decided over the whole file, column by column, never from a sample:

* ID-named columns (``ID_COLUMN_NAMES``) and digit codes with a leading zero (zip ``02139``,
  ward ``01``) stay text.
* ``true``/``false`` become Boolean; whole numbers Int64 (nulls stay null, not Float64); decimal
  numbers Float64. ``NaN``/``inf`` words and ``1,234`` are not numbers.
* Dates are parsed per value against a fixed list of formats. Day-first and month-first slash
  dates are only used when some value settles the order (a day above 12); otherwise the column
  stays text. Values that match no format become null, and the count is reported.
* A column that cannot take a type stays text, and only that column: a note says which column
  and why (``ColumnTypeNote``). Nothing falls back to "every column is text".

A first row that is a units sub-header (``years``, ``mg/dL`` above numeric columns, with no ID)
is kept as unit metadata and is not a patient. A row with no ID and no values fails the upload.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Final, Literal

import polars as pl
import structlog

logger = structlog.get_logger()

# Columns stored as text whatever they hold (avoids integer overflow, keeps joins consistent).
ID_COLUMN_NAMES: Final = frozenset({"patient_id", "patientid", "subject_id", "subjectid", "id", "mrn", "study_id"})

# A date column needs this share of its non-null values to parse; below it, the column is text.
DATE_MIN_PARSED_SHARE: Final = 0.9
# A column is reported as "numeric with some text" when at least this share is numeric.
NUMERIC_NOTE_SHARE: Final = 0.5

NoteReason = Literal["leading_zero", "not_numeric", "date_values_unparsed", "ambiguous_dates", "not_dates"]

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
class StagedTable:
    """One typed table plus what staging decided about it."""

    data: pl.DataFrame
    notes: tuple[ColumnTypeNote, ...] = ()
    units: dict[str, str] = field(default_factory=dict)


def is_id_column_name(name: str) -> bool:
    return name.strip().lower() in ID_COLUMN_NAMES


# --- per-column typing ------------------------------------------------------------------------


@dataclass(frozen=True)
class _ColumnPlan:
    expr: pl.Expr
    dtype: pl.DataType
    note: ColumnTypeNote | None = None


def _clean(col: str) -> pl.Expr:
    """Trimmed text, with blank cells as null."""
    trimmed = pl.col(col).str.strip_chars()
    return pl.when(trimmed == "").then(None).otherwise(trimmed)


def _parse_dates(s: pl.Series) -> tuple[pl.Series | None, NoteReason | None]:
    """Parse a trimmed text series as Date/Datetime, or say why not (None when nothing parses)."""
    parsed_dt = [s.str.strptime(pl.Datetime("us"), fmt, strict=False) for fmt in _DATETIME_FORMATS]
    parsed_date = [s.str.strptime(pl.Date, fmt, strict=False) for fmt in _ISO_DATE_FORMATS]

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
        elif only_mdy:
            parsed_date.append(mdy)
        elif disagree:
            return None, "ambiguous_dates"  # nothing settles the order
        else:
            parsed_date.append(dmy)  # every value reads the same either way

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

    is_digits = non_null.str.contains(_DIGITS_RE)
    if bool(is_digits.all()) and bool(non_null.str.contains(_LEADING_ZERO_RE).any()):
        return _ColumnPlan(pl.col(col), pl.String(), ColumnTypeNote(col, "String", "leading_zero", 0, n))

    lowered = non_null.str.to_lowercase()
    if bool(lowered.is_in(["true", "false"]).all()):
        expr = _clean(col).str.to_lowercase().replace_strict({"true": True, "false": False}, return_dtype=pl.Boolean)
        return _ColumnPlan(expr, pl.Boolean())

    is_int = non_null.str.contains(_INT_RE)
    if bool(is_int.all()) and non_null.cast(pl.Int64, strict=False).null_count() == 0:
        return _ColumnPlan(_clean(col).cast(pl.Int64), pl.Int64())

    is_float = non_null.str.contains(_FLOAT_RE)
    if bool(is_float.all()):
        return _ColumnPlan(_clean(col).cast(pl.Float64), pl.Float64())

    parsed, why_not = _parse_dates(s)
    if why_not is not None:
        return _ColumnPlan(pl.col(col), pl.String(), ColumnTypeNote(col, "String", why_not, 0, n))
    if parsed is not None:
        ok = int(parsed.is_not_null().sum())
        failed = n - ok
        if ok / n >= DATE_MIN_PARSED_SHARE:
            note = ColumnTypeNote(col, str(parsed.dtype), "date_values_unparsed", failed, n) if failed else None
            return _ColumnPlan(pl.lit(parsed).alias(col), parsed.dtype, note)
        return _ColumnPlan(pl.col(col), pl.String(), ColumnTypeNote(col, "String", "not_dates", failed, n))

    numeric = int(is_float.sum())
    if numeric / n >= NUMERIC_NOTE_SHARE:
        return _ColumnPlan(pl.col(col), pl.String(), ColumnTypeNote(col, "String", "not_numeric", n - numeric, n))
    return text


def _type_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, list[ColumnTypeNote]]:
    plans = {col: _plan_column(df, col) for col in df.columns}
    typed = df.with_columns([plan.expr.alias(col) for col, plan in plans.items()])
    return typed, [plan.note for plan in plans.values() if plan.note is not None]


# --- units row and empty rows -----------------------------------------------------------------


def _is_label(value: str) -> bool:
    """Text that is not a number and not a censored value: a unit label like 'mg/dL' or '%'."""
    return not _NUMBER_RE.match(value) and not _CENSORED_RE.match(value)


def _units_row(df: pl.DataFrame) -> dict[str, str] | None:
    """Units by column if the first row is a units sub-header, else None.

    The row is units when it has no ID, every filled cell is a label (not a number, not "<5"),
    and every filled cell sits above a column whose remaining rows are numeric or dates.
    """
    if df.height < 2:
        return None
    first = df.row(0, named=True)
    if any(is_id_column_name(c) and (first[c] or "").strip() for c in df.columns):
        return None
    filled = {c: v.strip() for c, v in first.items() if v is not None and v.strip()}
    if not filled or not all(_is_label(v) for v in filled.values()):
        return None
    rest, _ = _type_columns(df.slice(1))
    typed_cols = {c for c, t in rest.schema.items() if t.is_numeric() or t.is_temporal()}
    if not set(filled) <= typed_cols:
        return None
    return filled


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
    return stage_text_frame(text)


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
    scores = [_header_score(r) for r in rows[:max_rows]]
    if not scores or max(scores) <= _HEADER_MIN_SCORE:
        return 0
    return scores.index(max(scores))


def read_excel_rows(file_bytes: bytes) -> list[list[object]]:
    """Cell values of the first sheet, row by row (formulas as their cached values).

    Raises:
        StagingError: the workbook cannot be read.
    """
    import zipfile

    import openpyxl
    from openpyxl.utils.exceptions import InvalidFileException

    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    except (InvalidFileException, zipfile.BadZipFile, KeyError, OSError) as e:
        raise StagingError(f"The Excel file could not be read: {type(e).__name__}") from e
    try:
        return [list(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    finally:
        wb.close()


def read_excel_typed(file_bytes: bytes, header_row: int | None = None) -> StagedTable:
    """Read the first sheet of an .xlsx as text, then stage.

    Args:
        file_bytes: Workbook bytes.
        header_row: 0-based header row; detected (``detect_header_row``) when None.

    Raises:
        StagingError: the workbook cannot be read or has no header row.
    """
    rows = read_excel_rows(file_bytes)
    if header_row is None:
        header_row = detect_header_row(rows)
    if len(rows) <= header_row:
        raise StagingError("The Excel sheet has no header row")
    headers = _unique_headers(rows[header_row])
    width = len(headers)
    body = [[_cell_text(v) for v in (r + [None] * width)[:width]] for r in rows[header_row + 1 :]]
    # Trailing all-empty rows are sheet formatting, not data lines (unlike in a CSV).
    while body and all(v is None or not v.strip() for v in body[-1]):
        body.pop()
    text = pl.DataFrame(body, schema=dict.fromkeys(headers, pl.String), orient="row")
    return stage_text_frame(text, header_row_number=header_row + 1)


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
    "ambiguous_dates": "kept as text: dates could be day/month or month/day and nothing in the file settles which",
    "not_dates": "kept as text: {n} values are not dates",
}


def describe_column_types(notes: list[dict[str, str | int]], units: dict[str, str], small_cell_limit: int) -> list[str]:
    """One sentence per staging decision the user should know about (counts small-cell safe)."""
    lines = []
    for note in notes:
        failed, checked = int(note.get("failed_values", 0)), int(note.get("checked_values", 0))
        text = _NOTE_TEXT[str(note["reason"])].format(n=_shown_count(failed, checked, small_cell_limit))
        lines.append(f"{note['column']}: {text}.")
    lines.extend(f"{column}: unit {unit} (from the units row under the header)." for column, unit in units.items())
    return lines
