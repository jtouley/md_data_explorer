# SPDX-License-Identifier: MIT
"""One date format per column: chosen over every value, recorded with a worked example.

A column is read as dates only when ONE format reads every non-empty value with a plausible
year (``config.DATE_YEAR_MIN``..``config.DATE_YEAR_MAX``). Values are never parsed row by row
against a list of formats: a column where ``03/04/21`` and ``2021-04-03`` sit side by side has
no single format and stays text.

Day-first and month-first readings are settled by values: a value with a first part above 12
can only be day-first, a second part above 12 only month-first. When both readings read every
value and disagree somewhere, the order is open (``DateAmbiguity``): the column stays text and
the contract asks the clinician which order the file uses. A dictionary that states the format
("DD/MM/YYYY") settles it (``format_from_text``).

Numbers can be dates too, but only with evidence beyond the digits (a date-named column, a
dictionary statement): eight-digit ``YYYYMMDD`` integers and Excel serial day numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final, Literal, TypeAlias

import polars as pl

from clinical_analytics.contracts import config

# Formats tried in this order; each reads the whole column or is not the column's format.
_DATE_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    # %B reads full and abbreviated month names ("March", "Mar"), in any letter case.
    "%d-%B-%Y",
    "%d %B %Y",
    "%d-%B-%y",
    "%d %B %y",
    "%B %d, %Y",
    "%B %d %Y",
    "%d.%m.%Y",
    "%d.%m.%y",
)
_DATETIME_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
)
# Day/month order families: the same separator, year width and time part, read both ways.
_ORDER_SEPARATORS: Final = ("/", "-")
_YEAR_PARTS: Final = ("%Y", "%y")
_TIME_PARTS: Final = ("", " %H:%M", " %H:%M:%S")
YYYYMMDD: Final = "%Y%m%d"
# Recorded instead of a strptime pattern for values that are not text.
EXCEL_SERIAL: Final = "excel_serial"
NATIVE: Final = "native"
_EXCEL_EPOCH: Final = date(1899, 12, 30)
# Excel serials accepted as dates: 1954-10-03 .. 2119-01-08. A small whole number (a count, a
# score) is never read as a day number.
_EXCEL_SERIAL_RANGE: Final = (20_000, 80_000)

Order: TypeAlias = Literal["day_first", "month_first"]


@dataclass(frozen=True)
class DateChoice:
    """The one format a column is read with, and a worked example from the column itself."""

    column: str
    format: str  # strptime pattern, YYYYMMDD, EXCEL_SERIAL or NATIVE
    example_in: str
    example_out: str
    has_time: bool = False
    order: Order | None = None  # settled day/month order, for order formats
    evidence: str = ""


@dataclass(frozen=True)
class DateAmbiguity:
    """What the values can't settle, so the column stays text and the clinician is asked.

    * ``order``: both day/month orders read every value and disagree, or too few values settle
      the order (one 13/01 among a hundred dates that read either way);
    * ``century``: a two-digit year that reads as a future date (``03/15/45`` as 2045).

    ``format``: the reading asked about (the day-first one for an order question), for the
    question's worked example in the column's own style.
    """

    column: str
    example: str
    day_first: str
    month_first: str
    kind: Literal["order", "century"] = "order"
    format: str = "%d/%m/%y"


def spoken(value: date | datetime) -> str:
    """'3 April 2021', or '3 April 2021, 14:05' for a time of day."""
    text = f"{value.day} {value:%B} {value.year}"
    if isinstance(value, datetime) and (value.hour or value.minute or value.second):
        text += f", {value:%H:%M}"
    return text


def _plausible(parsed: pl.Series) -> bool:
    years = parsed.dt.year().drop_nulls()
    if years.len() == 0:
        return False
    low, high = years.min(), years.max()
    return (
        isinstance(low, int) and isinstance(high, int) and low >= config.DATE_YEAR_MIN and high <= config.DATE_YEAR_MAX
    )


def _parse(values: pl.Series, fmt: str) -> pl.Series:
    dtype: pl.Datetime | pl.Date = pl.Datetime("us") if "%H" in fmt else pl.Date()
    return values.str.strptime(dtype, fmt, strict=False)


def _reads_all(values: pl.Series, fmt: str) -> pl.Series | None:
    """``values`` (no nulls) parsed with ``fmt`` when every value parses with a plausible year."""
    parsed = _parse(values, fmt)
    if parsed.null_count() or not _plausible(parsed):
        return None
    return parsed


def _order_pair(sep: str, year: str, time: str) -> tuple[str, str]:
    return f"%d{sep}%m{sep}{year}{time}", f"%m{sep}%d{sep}{year}{time}"


def _example(values: pl.Series, parsed: pl.Series, prefer: pl.Series | None = None) -> tuple[str, str]:
    """A worked example: the first value whose reading needed the chosen order, else the first."""
    index = 0
    if prefer is not None and bool(prefer.any()):
        index = int(prefer.arg_true()[0])
    return str(values[index]), spoken(parsed[index])


def choose_text_format(series: pl.Series, *, stated: str | None = None) -> DateChoice | DateAmbiguity | None:
    """The single format that reads every value of a text column, or why there is none.

    ``stated``: a strptime pattern a dictionary gives for the column; it is used when it reads
    every value, and settles day/month order. Returns None when the column is not dates. A
    two-digit-year format that would put a date in the future is a ``century`` ambiguity.
    """
    found = _choose_text_format(series, stated=stated)
    if isinstance(found, DateChoice) and "%y" in found.format and life_event_named(series.name):
        # A two-digit birth or death year never settles its century: 03/15/21 is a baby or a
        # centenarian. Always asked.
        return DateAmbiguity(series.name, "", "15 March 1945", "15 March 2045", "century", found.format)
    if isinstance(found, DateChoice) and "%y" in found.format:
        parsed = _parse(_present(series), found.format)
        latest = parsed.max()
        today = date.today()
        if isinstance(latest, date) and (latest.date() if isinstance(latest, datetime) else latest) > today:
            later = parsed.filter(parsed.cast(pl.Date) > today)[0]
            century = later.replace(year=later.year - 100)
            return DateAmbiguity(series.name, "", spoken(century), spoken(later), "century", found.format)
    return found


def _choose_text_format(series: pl.Series, *, stated: str | None = None) -> DateChoice | DateAmbiguity | None:
    if series.dtype != pl.String:
        return None
    values = _present(series)
    if values.len() == 0:
        return None
    name = series.name
    if stated is not None:
        parsed = _reads_all(values, stated)
        if parsed is not None:
            ex_in, ex_out = _example(values, parsed)
            return DateChoice(name, stated, ex_in, ex_out, "%H" in stated, None, "the data dictionary states it")
    for fmt in (*_DATETIME_FORMATS, *_DATE_FORMATS):
        parsed = _reads_all(values, fmt)
        if parsed is not None:
            ex_in, ex_out = _example(values, parsed)
            return DateChoice(name, fmt, ex_in, ex_out, "%H" in fmt, None, "every value reads with this format")
    for sep in _ORDER_SEPARATORS:
        for year in _YEAR_PARTS:
            for time in _TIME_PARTS:
                found = _settle_order(values, name, *_order_pair(sep, year, time))
                if found is not None:
                    return found
    if bool(values.str.contains(r"^\d{8}$").all()):
        parsed = _reads_all(values, YYYYMMDD)
        if parsed is not None and date_named(name):
            ex_in, ex_out = _example(values, parsed)
            return DateChoice(name, YYYYMMDD, ex_in, ex_out, False, None, "eight digits as year, month, day")
    return None


def date_coverage(series: pl.Series) -> int:
    """The most values of a text column that any one format reads (0 when none): tells a column
    of dates with a few strays from a column of words, for the staging note."""
    values = _present(series)
    if values.len() == 0:
        return 0
    formats = [*_DATETIME_FORMATS, *_DATE_FORMATS]
    for sep in _ORDER_SEPARATORS:
        for year in _YEAR_PARTS:
            for time in _TIME_PARTS:
                formats.extend(_order_pair(sep, year, time))
    return max(int(_parse(values, fmt).is_not_null().sum()) for fmt in formats)


def _present(series: pl.Series) -> pl.Series:
    trimmed = series.str.strip_chars().drop_nulls()
    return trimmed.filter(trimmed != "")


def _settle_order(values: pl.Series, name: str, day_first: str, month_first: str) -> DateChoice | DateAmbiguity | None:
    dmy = _reads_all(values, day_first)
    mdy = _reads_all(values, month_first)
    if dmy is None and mdy is None:
        return None
    has_time = "%H" in day_first
    if dmy is not None and mdy is not None:
        differ = (dmy != mdy).fill_null(False)
        if bool(differ.any()):
            index = int(differ.arg_true()[0])
            return DateAmbiguity(name, str(values[index]), spoken(dmy[index]), spoken(mdy[index]), "order", day_first)
        ex_in, ex_out = _example(values, dmy)
        return DateChoice(
            name, day_first, ex_in, ex_out, has_time, "day_first", "every value reads the same either way"
        )
    # Several values must settle the order, and a fair share of them: one 13/01/2021 among a
    # hundred dates that read either way could be a typing slip, not the file's convention.
    settled_by = int(_parse(values, month_first if dmy is not None else day_first).null_count())
    if (
        settled_by < min(config.DATE_ORDER_MIN_VALUES, max(1, values.len() // 2))  # a 2-value column: 1
        or settled_by < config.DATE_ORDER_MIN_SHARE * values.len()
    ):
        chosen = dmy if dmy is not None else mdy
        assert chosen is not None
        return DateAmbiguity(name, str(values[0]), spoken(chosen[0]), spoken(chosen[0]), "order", day_first)
    if dmy is not None:
        ex_in, ex_out = _example(values, dmy, dmy.dt.day() <= 12)
        return DateChoice(name, day_first, ex_in, ex_out, has_time, "day_first", "many first numbers are above 12")
    assert mdy is not None
    ex_in, ex_out = _example(values, mdy, mdy.dt.day() <= 12)
    return DateChoice(name, month_first, ex_in, ex_out, has_time, "month_first", "many second numbers are above 12")


_DATE_NAME_TOKENS: Final = frozenset({"date", "dt", "dttm", "dob", "dod", "datetime", "timestamp", "dte"})
_TOKEN_SPLIT: Final = re.compile(r"[^a-z0-9]+")


_LIFE_EVENT_TOKENS: Final = frozenset({"dob", "dod", "birth", "born", "bday", "birthday", "death", "died", "deceased"})


def life_event_named(column: str) -> bool:
    """A birth or death date column ("dob", "date_of_birth", "deathDate")."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column).lower()
    tokens = {t for t in _TOKEN_SPLIT.split(spaced) if t}
    return bool(tokens & _LIFE_EVENT_TOKENS) or "birth" in spaced or "death" in spaced


def swap_order(choice_format: str) -> str:
    """'%d/%m/%y' <-> '%m/%d/%y': the other day/month reading of an order format."""
    return choice_format.replace("%d", "\0").replace("%m", "%d").replace("\0", "%m")


def date_named(column: str) -> bool:
    """'onset_date', 'DOB', 'admitDt' (case boundary split)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column).lower()
    return any(t in _DATE_NAME_TOKENS for t in _TOKEN_SPLIT.split(spaced) if t)


def choose_number_format(
    series: pl.Series, *, from_excel: bool = False, stated: str | None = None, date_hint: bool = False
) -> DateChoice | None:
    """Numbers that are dates: ``YYYYMMDD`` integers, or Excel serial day numbers from a workbook.

    Only for a date-named column or one the dictionary says is a date (``stated``): an ID or a
    count is never read as a date because its digits happen to fit.
    """
    if not (series.dtype.is_integer() or series.dtype.is_float()):
        return None
    if not (date_named(series.name) or stated is not None or date_hint):
        return None
    values = series.drop_nulls()
    if values.len() == 0:
        return None
    whole = bool((values == values.floor()).all()) if values.dtype.is_float() else True
    if whole and stated in (None, YYYYMMDD):
        as_text = values.cast(pl.Int64).cast(pl.String)
        if bool(as_text.str.contains(r"^\d{8}$").all()):
            parsed = _reads_all(as_text, YYYYMMDD)
            if parsed is not None:
                return DateChoice(
                    series.name,
                    YYYYMMDD,
                    str(as_text[0]),
                    spoken(parsed[0]),
                    False,
                    None,
                    "eight-digit numbers read as year, month, day",
                )
    low, high = _EXCEL_SERIAL_RANGE
    if from_excel and bool(values.is_between(low, high).all()):
        first = excel_serial_value(float(values[0]))
        return DateChoice(
            series.name,
            EXCEL_SERIAL,
            f"{float(values[0]):g}",
            spoken(first),
            not whole,
            None,
            "Excel day numbers in a workbook",
        )
    return None


def excel_serial_value(serial: float) -> datetime:
    return datetime.combine(_EXCEL_EPOCH, datetime.min.time()) + timedelta(days=serial)


def parse_expr(column: str, choice_format: str, has_time: bool, dtype: pl.DataType) -> pl.Expr:
    """The expression that reads ``column`` with its one chosen format (no per-row fallback)."""
    col = pl.col(column)
    target: pl.Datetime | pl.Date = pl.Datetime("us") if has_time else pl.Date()
    if choice_format == EXCEL_SERIAL:
        days = col.cast(pl.Float64)
        epoch = pl.lit(datetime.combine(_EXCEL_EPOCH, datetime.min.time()))
        moment = epoch + pl.duration(milliseconds=(days * 86_400_000).round(0).cast(pl.Int64))
        return (moment if has_time else moment.dt.date()).alias(column)
    text = col.cast(pl.Int64).cast(pl.String) if dtype.is_numeric() else col.str.strip_chars()
    text = pl.when(text == "").then(None).otherwise(text)
    return text.str.strptime(target, choice_format, strict=True).alias(column)


# Format words a dictionary uses, mapped to the strptime pattern they state.
_STATED_FORMATS: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(pattern, re.IGNORECASE), fmt)
    for pattern, fmt in (
        (r"\bMM/DD/YYYY\s+HH:MM\b", "%m/%d/%Y %H:%M"),
        (r"\bDD/MM/YYYY\s+HH:MM\b", "%d/%m/%Y %H:%M"),
        (r"\bMM/DD/YYYY\b", "%m/%d/%Y"),
        (r"\bDD/MM/YYYY\b", "%d/%m/%Y"),
        (r"\bMM/DD/YY\b", "%m/%d/%y"),
        (r"\bDD/MM/YY\b", "%d/%m/%y"),
        (r"\bMM-DD-YYYY\b", "%m-%d-%Y"),
        (r"\bDD-MM-YYYY\b", "%d-%m-%Y"),
        (r"\bDD\.MM\.YYYY\b", "%d.%m.%Y"),
        (r"\bDD-Mon-YYYY\b", "%d-%B-%Y"),
        (r"\bDD Mon YYYY\b", "%d %B %Y"),
        (r"\bYYYY-MM-DD\b", "%Y-%m-%d"),
        (r"\bYYYY/MM/DD\b", "%Y/%m/%d"),
        (r"\bYYYYMMDD\b", YYYYMMDD),
    )
)


def format_from_text(text: str) -> str | None:
    """The date format a dictionary entry states ("ED arrival, MM/DD/YYYY HH:MM"), or None."""
    return next((fmt for pattern, fmt in _STATED_FORMATS if pattern.search(text)), None)


def describe_format(choice_format: str) -> str:
    """The format as a person writes it: '%d/%m/%y' -> 'DD/MM/YY'."""
    if choice_format == EXCEL_SERIAL:
        return "Excel day numbers"
    if choice_format == NATIVE:
        return "stored as dates"
    words = {"%Y": "YYYY", "%y": "YY", "%m": "MM", "%d": "DD", "%B": "Mon", "%H": "HH", "%M": "MM", "%S": "SS"}
    text = choice_format.replace("%.f", ".fff")
    for code, word in words.items():
        text = text.replace(code, word)
    return text
