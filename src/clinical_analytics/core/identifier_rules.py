# SPDX-License-Identifier: MIT
"""Which columns may identify a patient, and which columns form a value with its interval.

One place for the rules the upload path (``VariableTypeDetector.ensure_patient_id``), schema
inference and the data contract share. A patient ID is never built from a measurement: a float
column, a column with fractional values, or a column named like an estimate or its bounds (``mx``,
``upper``, ``lower``, rates). Counting such a column's distinct values would report "patients"
that are numbers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import polars as pl

# The row ID added when a dataset has no column that identifies a patient. Its name holds none of
# the patient-key names (core.schema_inference.PATIENT_KEY_NAMES), so no answer counts its values
# as patients: n is "rows".
ROW_NUMBER_COLUMN: Final = "row_number"

# Name tokens (lower case, split on anything but letters and digits) of a measured value: an
# estimate, its bounds, a rate or a share. A column with one of them is never a patient ID,
# whatever its uniqueness ("mx" in a mortality table is unique to 6 decimals).
MEASUREMENT_NAME_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "mx",
        "val",
        "value",
        "mean",
        "median",
        "estimate",
        "rate",
        "upper",
        "lower",
        "lci",
        "uci",
        "ci",
        "se",
        "sd",
        "pct",
        "percent",
        "prop",
        "proportion",
        "ratio",
        "age",
        "deaths",
        "cases",
        "population",
    }
)

# Bound names of an interval ("upper"/"lower", "mx_upper"/"mx_lower", "lower_mx"/"upper_mx").
INTERVAL_LOWER: Final = "lower"
INTERVAL_UPPER: Final = "upper"
# With bare "lower"/"upper" columns, the value is the first of these names present, else the one
# other float column.
INTERVAL_VALUE_NAMES: Final[tuple[str, ...]] = ("val", "value", "mean", "estimate", "mx")

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_BOUND_FORMS = re.compile(
    rf"^(?:(?P<pre>{INTERVAL_LOWER}|{INTERVAL_UPPER})[^a-z0-9]+(?P<pbase>.+)|(?P<sbase>.+?)[^a-z0-9]+(?P<suf>{INTERVAL_LOWER}|{INTERVAL_UPPER}))$"
)


def name_tokens(column: str) -> list[str]:
    """'Upper CI (95%)' -> ['upper', 'ci', '95']."""
    return [t for t in _TOKEN_SPLIT.split(column.lower()) if t]


def is_measurement_name(column: str) -> bool:
    """A column named for a measured value, an interval bound or a rate."""
    return any(token in MEASUREMENT_NAME_TOKENS for token in name_tokens(column))


def has_fractional_values(series: pl.Series) -> bool:
    """Floats, or text that reads as numbers with a fractional part ("150.185474")."""
    if series.dtype.is_float() or isinstance(series.dtype, pl.Decimal):
        return True
    if series.dtype != pl.String:
        return False
    numbers = series.str.strip_chars().cast(pl.Float64, strict=False)
    present = series.drop_nulls().len()
    parsed = numbers.drop_nulls()
    if present == 0 or parsed.len() < present:
        return False  # not all numbers: text, so codes like "P-001" stay eligible
    return bool((parsed != parsed.floor()).any())


def can_identify_patients(series: pl.Series) -> bool:
    """Whether the column's values may name patients: integers or text, no fractional values,
    and not named as a measurement. Uniqueness is the caller's test."""
    dtype = series.dtype
    if not (dtype.is_integer() or dtype in (pl.String, pl.Categorical)):
        return False
    if is_measurement_name(series.name):
        return False
    return not has_fractional_values(series)


@dataclass(frozen=True)
class Interval:
    """A value column with its lower and upper bound columns (lower <= value <= upper)."""

    value: str
    lower: str
    upper: str


def _bound_parts(column: str) -> tuple[str, str] | None:
    """('lower', 'mx') for 'mx_lower' or 'lower_mx'; ('lower', '') for a bare 'lower'."""
    low = column.lower().strip()
    if low in (INTERVAL_LOWER, INTERVAL_UPPER):
        return low, ""
    match = _BOUND_FORMS.match(low)
    if match is None:
        return None
    if match.group("pre"):
        return match.group("pre"), match.group("pbase")
    return match.group("suf"), match.group("sbase")


def find_intervals(schema: Mapping[str, pl.DataType]) -> list[Interval]:
    """Every value with both bound columns, all three numeric, in column order of the value.

    'mx_lower'/'mx_upper' bound 'mx'; bare 'lower'/'upper' bound the column named in
    ``INTERVAL_VALUE_NAMES``, else the only other float column. Ambiguous cases find nothing.
    """
    numeric = {c for c, dtype in schema.items() if dtype.is_numeric() and dtype != pl.Boolean}
    by_lower = {c.lower(): c for c in schema}
    bounds: dict[str, dict[str, str]] = {}
    for column in schema:
        parts = _bound_parts(column)
        if parts is not None and column in numeric:
            side, base = parts
            bounds.setdefault(base, {}).setdefault(side, column)
    found: list[Interval] = []
    for base, sides in bounds.items():
        if set(sides) != {INTERVAL_LOWER, INTERVAL_UPPER}:
            continue
        if base:
            value = by_lower.get(base)
        else:
            named = [by_lower[n] for n in INTERVAL_VALUE_NAMES if n in by_lower]
            others = [
                c
                for c, dtype in schema.items()
                if dtype.is_float() and c not in sides.values() and _bound_parts(c) is None
            ]
            value = named[0] if named else (others[0] if len(others) == 1 else None)
        if value is not None and value in numeric:
            found.append(Interval(value=value, lower=sides[INTERVAL_LOWER], upper=sides[INTERVAL_UPPER]))
    order = list(schema)
    return sorted(found, key=lambda i: order.index(i.value))


# Semantic-layer config flag: the dataset's patient_id was made up by the upload (a hash of a
# combination of columns, or a copy of a column not named as a patient ID), so it keys rows, not
# patients. ``core.answer_checks.patient_key`` then finds no patient key and every n counts rows.
ROW_KEY_CONFIG_FLAG: Final = "patient_id_is_row_key"


def patient_id_is_row_key(metadata: Mapping[str, object]) -> bool:
    """Whether the upload synthesized patient_id (``synthetic_id_metadata.patient_id``).

    'composite' and 'row_index' are always row keys. 'single_column' (a unique column copied to
    patient_id) is a patient key only when that column is named as one (``patient_key_by_name``):
    a unique FIPS code or record number names a row. 'existing' (the file has patient_id) and
    uploads without the metadata are patient keys.
    """
    from clinical_analytics.core.schema_inference import patient_key_by_name

    synthetic = metadata.get("synthetic_id_metadata")
    info = synthetic.get("patient_id") if isinstance(synthetic, Mapping) else None
    if not isinstance(info, Mapping):
        return False
    source = info.get("patient_id_source")
    if source in ("composite", "row_index"):
        return True
    if source == "single_column":
        columns = info.get("patient_id_columns")
        names = [str(c) for c in columns] if isinstance(columns, list) else []
        return patient_key_by_name(names) is None
    return False
