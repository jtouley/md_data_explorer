# SPDX-License-Identifier: MIT
"""One rule for numbers stored as text, used by every analysis path and the quality report.

* Missing-value tokens (``contracts.config.MISSING_VALUE_TOKENS``: "n/a", ".", ...) are missing.
* A value recorded as a limit ("<5", ">100", "< 0.5 mg/L") or a censoring word
  (``CENSORED_TEXT_WORDS``: "BLQ", "ND", "<LOD") is censored: it is not a number and never
  becomes its limit. It is counted (``n_censored``) so an answer can refuse or say so.
* A number may be signed, start with a dot or carry an exponent ("+5", ".5", "5.", "1e-3").
* A column whose present values are mostly numbers (``NEAR_NUMERIC_SHARE``) but not all is still
  numbers: the entries that are neither a number, a limit nor a missing token ("pending") are
  counted and named (``unreadable``), and the answer refuses naming them. Below that share the
  column is text ("Grade 2", "1/2"), not numbers.
* A comma that could be a decimal or a thousands separator ("1,234" alone) raises
  ``AmbiguousNumberFormatError`` instead of guessing.

``statistic_refusal`` is the one refusal (and wording) for a statistic over such a column.
Polars expressions only (no per-value Python).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from clinical_analytics.contracts.config import (
    CENSORED_TEXT_PATTERN,
    CENSORED_TEXT_WORDS,
    MISSING_VALUE_TOKENS,
    NAMED_UNREADABLE_MAX,
    NEAR_NUMERIC_SHARE,
)

_NUMBER = r"[+-]?(\d[\d,]*(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
_PLAIN_NUMBER = rf"^{_NUMBER}$"
_THOUSANDS = r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$"
_DECIMAL_COMMA = r"^[+-]?\d+,\d+$"


class AmbiguousNumberFormatError(ValueError):
    """Text numbers whose comma could be a decimal or a thousands separator."""


@dataclass(frozen=True)
class NumericText:
    """A text column read as numbers.

    ``values``: Float64, null where the text was missing, a missing token, censored or unreadable.
    ``n_censored``: values recorded as a limit ("<5", "BLQ"), never substituted.
    ``n_missing_tokens``: values that were missing tokens ("n/a", "."), not true nulls.
    ``n_unreadable``: values that are none of the above ("pending"); ``unreadable`` names up to
    ``NAMED_UNREADABLE_MAX`` of them that hold no digit (a word, never a mistyped value).
    """

    values: pl.Series
    n_censored: int
    n_missing_tokens: int
    n_unreadable: int = 0
    unreadable: tuple[str, ...] = ()


def missing_token_expr(column: str) -> pl.Expr:
    """True where a text value is a missing token (not where it is null)."""
    return pl.col(column).str.strip_chars().str.to_lowercase().is_in(list(MISSING_VALUE_TOKENS))


def censored_expr(column: str) -> pl.Expr:
    """True where a text value is recorded as a limit ("<5", ">100") or a censoring word ("BLQ")."""
    text = pl.col(column).str.strip_chars()
    return text.str.contains(CENSORED_TEXT_PATTERN) | text.str.to_lowercase().is_in(list(CENSORED_TEXT_WORDS))


def _to_float(text: pl.Series, name: str) -> pl.Series:
    """Plain number text (no limits, no tokens) to Float64 under the comma rule."""
    present = text.drop_nulls()
    with_comma = present.filter(present.str.contains(","))
    if with_comma.len() == 0:
        return text.cast(pl.Float64, strict=False)
    if with_comma.str.contains(_THOUSANDS).all():
        unambiguous = (
            with_comma.str.contains(r"\.").any()
            or with_comma.str.contains(r",\d{3},").any()
            or present.str.contains(r"^[+-]?\d+\.\d+$").any()
        )
        if not unambiguous:
            raise AmbiguousNumberFormatError(
                f"{name} has values like '{with_comma[0]}' where the comma could mean thousands or a "
                "decimal point. Store the column as numbers, or use a dot for decimals."
            )
        return text.str.replace_all(",", "").cast(pl.Float64, strict=False)
    if with_comma.str.contains(_DECIMAL_COMMA).all() and not present.str.contains(r"\.").any():
        return text.str.replace(",", ".").cast(pl.Float64, strict=False)
    raise AmbiguousNumberFormatError(
        f"{name} mixes comma formats (e.g. '{with_comma[0]}'), so its numbers cannot be read reliably."
    )


def parse_numeric_text(series: pl.Series, weights: pl.Series | None = None) -> NumericText | None:
    """``series`` (text) read as numbers, or None when it is text, not numbers: fewer than
    ``NEAR_NUMERIC_SHARE`` of its present values (after limits and missing tokens) are numbers,
    or none is.

    ``weights``: how many rows hold each value, when ``series`` is a column's distinct values
    (a DuckDB value count); the counts are then weighted and ``values`` stays one per value.

    Raises:
        AmbiguousNumberFormatError: a comma whose meaning cannot be told.
    """
    if series.dtype != pl.String:
        return None
    name = series.name
    frame = series.to_frame().select(
        pl.col(name).str.strip_chars().alias("text"),
        missing_token_expr(name).fill_null(False).alias("token"),
        censored_expr(name).fill_null(False).alias("censored"),
        (pl.lit(1, dtype=pl.Int64) if weights is None else pl.lit(weights.cast(pl.Int64))).alias("w"),
    )
    frame = frame.with_columns(
        pl.when(pl.col("token") | pl.col("censored")).then(None).otherwise(pl.col("text")).alias("plain")
    ).with_columns(pl.col("plain").str.contains(_PLAIN_NUMBER).alias("number"))
    numbers = int(frame.filter(pl.col("number")).get_column("w").sum())
    other = frame.filter(pl.col("plain").is_not_null() & ~pl.col("number"))
    n_other = int(other.get_column("w").sum())
    if numbers == 0 or numbers < NEAR_NUMERIC_SHARE * (numbers + n_other):
        return None
    plain = frame.select(pl.when(pl.col("number")).then(pl.col("plain")).otherwise(None)).to_series()
    words = other.filter(~pl.col("plain").str.contains(r"\d")).get_column("plain").unique(maintain_order=True)
    return NumericText(
        values=_to_float(plain, name).alias(name),
        n_censored=int(frame.filter(pl.col("censored")).get_column("w").sum()),
        n_missing_tokens=int(frame.filter(pl.col("token")).get_column("w").sum()),
        n_unreadable=n_other,
        unreadable=tuple(sorted(words.to_list())[:NAMED_UNREADABLE_MAX]),
    )


def is_numeric_text(series: pl.Series, weights: pl.Series | None = None) -> bool:
    """Whether ``series`` is numbers stored as text (numbers, limits or missing tokens).

    A comma whose meaning cannot be told is still numeric text: its missing tokens are missing.
    ``weights`` as for ``parse_numeric_text``.
    """
    try:
        return parse_numeric_text(series, weights) is not None
    except AmbiguousNumberFormatError:
        return True


def missing_token_sql(quoted: str) -> str:
    """DuckDB form of ``missing_token_expr``: true where the text in ``quoted`` (an identifier
    already quoted) is a missing token once trimmed of whitespace and lowercased."""
    tokens = ", ".join("'" + token.replace("'", "''") + "'" for token in MISSING_VALUE_TOKENS)
    whitespace = "' ' || chr(9) || chr(10) || chr(11) || chr(12) || chr(13)"
    return f"lower(trim({quoted}, {whitespace})) IN ({tokens})"


def all_missing_message(column: str, what: str) -> str:
    """The one sentence for a statistic over a column with no values."""
    from clinical_analytics.analysis.variable_types import short_name

    return f"Every value of {short_name(column)} is missing, so there is nothing to {what}."


def statistic_refusal(
    series: pl.Series, what: str, weights: pl.Series | None = None, rows: int | None = None
) -> tuple[str | None, NumericText | None]:
    """(why a statistic of ``series`` would be wrong, the numbers it holds) for every path.

    The refusal is the one wording each path uses: a column with no values, a comma format that
    cannot be read, values recorded as a limit, or entries that are not numbers (named when they
    are words). The second item is the column read as numbers, None when it is not numeric text.
    ``what`` is the verb ("describe", "compare"). ``rows``: rows behind ``series`` when it holds
    distinct values with ``weights``.
    """
    from clinical_analytics.analysis.variable_types import short_name
    from clinical_analytics.core.result_presentation import line_count_text

    name = short_name(series.name)
    total = rows if rows is not None else series.len()
    nulls = int(weights.filter(series.is_null()).sum()) if weights is not None else series.null_count()
    if nulls == total:
        return all_missing_message(series.name, what), None
    if series.dtype != pl.String:
        return None, None
    try:
        parsed = parse_numeric_text(series, weights)
    except AmbiguousNumberFormatError as exc:
        return str(exc), None
    if parsed is None:
        return None, None
    numbers = total - nulls - parsed.n_missing_tokens - parsed.n_censored - parsed.n_unreadable
    if numbers <= 0 and not parsed.n_censored and not parsed.n_unreadable:
        return all_missing_message(series.name, what), parsed
    if parsed.n_censored:
        amount = line_count_text(parsed.n_censored, [total - nulls - parsed.n_missing_tokens - parsed.n_censored])
        return (
            f"{name} has {amount} values recorded as a limit, such as '<5', so a statistic of it would be "
            f"wrong. Replace the limits with measured values in the file, or ask for a count by {name}."
        ), parsed
    if parsed.n_unreadable:
        named = ", ".join(f"'{word}'" for word in parsed.unreadable)
        like = f"values like {named}" if named else "some values"
        return (
            f"{name} has {like} that are not numbers, so a statistic of it would be wrong. "
            f"Replace them with measured values or leave them empty in the file."
        ), parsed
    return None, parsed
