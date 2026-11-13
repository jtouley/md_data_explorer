# SPDX-License-Identifier: MIT
"""One rule for numbers stored as text, used by every analysis path and the quality report.

* Missing-value tokens (``contracts.config.MISSING_VALUE_TOKENS``: "n/a", ".", ...) are missing.
* A value recorded as a limit ("<5", ">100") is censored: it is not a number and never becomes
  its limit. It is counted (``n_censored``) so an answer can refuse or say so.
* Everything else must be a plain number ("12", "-3.5", "1,234.5"); one value that is not
  ("Grade 2", "1/2") means the column is text, not numbers.
* A comma that could be a decimal or a thousands separator ("1,234" alone) raises
  ``AmbiguousNumberFormatError`` instead of guessing.

Polars expressions only (no per-value Python).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from clinical_analytics.contracts.config import CENSORED_TEXT_PATTERN, MISSING_VALUE_TOKENS

_PLAIN_NUMBER = r"^-?\d[\d,]*(\.\d+)?$"
_THOUSANDS = r"^-?\d{1,3}(,\d{3})+(\.\d+)?$"
_DECIMAL_COMMA = r"^-?\d+,\d+$"


class AmbiguousNumberFormatError(ValueError):
    """Text numbers whose comma could be a decimal or a thousands separator."""


@dataclass(frozen=True)
class NumericText:
    """A text column read as numbers.

    ``values``: Float64, null where the text was missing, a missing token or censored.
    ``n_censored``: values recorded as a limit ("<5"), never substituted.
    ``n_missing_tokens``: values that were missing tokens ("n/a", "."), not true nulls.
    """

    values: pl.Series
    n_censored: int
    n_missing_tokens: int


def missing_token_expr(column: str) -> pl.Expr:
    """True where a text value is a missing token (not where it is null)."""
    return pl.col(column).str.strip_chars().str.to_lowercase().is_in(list(MISSING_VALUE_TOKENS))


def censored_expr(column: str) -> pl.Expr:
    """True where a text value is recorded as a limit ("<5", ">100")."""
    return pl.col(column).str.contains(CENSORED_TEXT_PATTERN)


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
            or present.str.contains(r"^-?\d+\.\d+$").any()
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


def parse_numeric_text(series: pl.Series) -> NumericText | None:
    """``series`` (text) read as numbers, or None when any value is neither a number, a limit nor
    a missing token, or when no value is a number.

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
    )
    plain = frame.select(
        pl.when(pl.col("token") | pl.col("censored")).then(None).otherwise(pl.col("text")).alias("plain")
    ).to_series()
    present = plain.drop_nulls()
    if present.len() == 0 or not present.str.contains(_PLAIN_NUMBER).all():
        return None
    return NumericText(
        values=_to_float(plain, name).alias(name),
        n_censored=int(frame.get_column("censored").sum()),
        n_missing_tokens=int(frame.get_column("token").sum()),
    )
