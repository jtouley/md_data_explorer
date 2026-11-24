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
from typing import TYPE_CHECKING, Final

import polars as pl

if TYPE_CHECKING:
    from clinical_analytics.contracts.semantics import DatasetSemantics

# The row ID added when a dataset has no column that identifies a patient. It is recorded as an
# app column in the contract's semantics, so no answer counts its values as patients: n is "rows".
ROW_NUMBER_COLUMN: Final = "row_number"

# Upload metadata: the published export a file was read as ("cdc_wonder"; staging.CDC_WONDER_FORMAT).
# Such a file is never asked "Is each row one patient?" (rows_may_be_patients).
SOURCE_FORMAT_KEY: Final = "source_format"

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
    and not named as a measurement. Uniqueness is the caller's test.

    A float column qualifies only when it is named as a patient ID and every value is a whole
    number: SPSS stores every number as a float, so its "ID" of 1.0, 2.0, ... is still an ID. An
    unnamed float is a measurement even when rounded (a rate rounded to integers).
    """
    from clinical_analytics.contracts.inference import id_named

    dtype = series.dtype
    if is_measurement_name(series.name):
        return False
    if dtype.is_float():
        present = series.drop_nulls()
        return (
            id_named(series.name)
            and present.len() > 0
            and bool(present.is_finite().all())
            and bool((present == present.floor()).all())
        )
    if not (dtype.is_integer() or dtype in (pl.String, pl.Categorical)):
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


# Semantic-layer config flag: no column names a patient (``DatasetSemantics.patient_key`` is
# None), so every n counts rows. Set from the contract's semantics (ADR017), never guessed here.
ROW_KEY_CONFIG_FLAG: Final = "patient_id_is_row_key"


def patient_id_is_row_key(semantics: DatasetSemantics) -> bool:
    """Whether answers count rows: the semantics name no patient key (a row key, a combination of
    columns or a row number tells rows apart, and only the user can say each row is a patient)."""
    return semantics.patient_key is None


def _quoted_list(names: list[str]) -> str:
    quoted = [f"“{name}”" for name in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def row_key_note(semantics: DatasetSemantics) -> str:
    """The upload note when answers count rows, not patients, or "" when a column names patients.

    Said at upload, in plain words, whenever the semantics name no patient key: what tells the
    rows apart (a row number, a combination of columns, or a column not named as a patient ID),
    that every count is a count of rows, and how to get patient counts. Column names only.
    """
    if semantics.patient_key is not None or semantics.population_estimates:
        return ""
    row = semantics.row_key
    columns = list(row.columns)
    if row.kind == "composite" and columns:
        found = f"No column in this file identifies a patient, so rows are told apart by {_quoted_list(columns)}."
    elif row.kind == "column" and columns:
        found = f"Rows are told apart by {_quoted_list(columns)}, which is not named as a patient ID."
    else:
        numbered = next((a.column for a in semantics.app_columns if a.kind == "row_number"), ROW_NUMBER_COLUMN)
        found = f"No column in this file identifies a patient, so each row was numbered (“{numbered}”)."
    if any(q.kind == "rows_are_patients" and not q.answered for q in semantics.open_questions):
        # Counts wait for the answer (contracts.holds): nothing is counted until it is given.
        return (
            f"{found} Counts wait for your answer to “Is each line one patient?”. To count patients by ID "
            "instead, add a column with one ID for each patient, named patient_id, and upload the file again."
        )
    return (
        f"{found} Answers count rows, not patients. To count patients, add a column with one ID for each "
        "patient, named patient_id, and upload the file again."
    )


def added_id_columns(semantics: DatasetSemantics) -> list[str]:
    """The ID columns the app added to the file's own columns (``patient_id`` copied or hashed,
    or the row number), from the semantics' app columns. Reported with the dataset, so a check
    of the file's shape compares the file's own columns."""
    return [a.column for a in semantics.app_columns]


def added_id_columns_from_metadata(metadata: Mapping[str, object]) -> list[str]:
    """``added_id_columns`` for a version stored before its contract held semantics: read from
    the upload's ``synthetic_id_metadata`` record of what it added (not a guess)."""
    synthetic = metadata.get("synthetic_id_metadata")
    info = synthetic.get("patient_id") if isinstance(synthetic, Mapping) else None
    source = info.get("patient_id_source") if isinstance(info, Mapping) else None
    if source in ("single_column", "composite"):
        return ["patient_id"]
    if source == "row_index" and isinstance(info, Mapping):
        return [str(info.get("row_number_column") or ROW_NUMBER_COLUMN)]
    return []


# --------------------------------------------------------------------- "Is each row one patient?"
#
# Asked once at upload only when the contract's semantics hold the open question
# ``rows_are_patients`` (contracts.inference): no column names a patient, and no ID-like column
# repeats (a repeat means rows are visits or tests of someone).

# Upload metadata (``synthetic_id_metadata.patient_id``): the ID-like columns whose values repeat
# (the semantics' grouping columns), kept for uploads stored before ADR017.
REPEATED_ID_COLUMNS_KEY: Final = "repeated_id_columns"
# Semantic-layer config flag: ``rows_may_be_patients`` for the dataset. A stored one-row-per-patient
# confirmation makes answers count patients only where it is set (``answer_checks``), so a
# confirmation given elsewhere (governed SQL) never turns visit rows into patients.
PATIENT_ROWS_ASKABLE_CONFIG_FLAG: Final = "rows_may_be_patients"


def rows_may_be_patients(semantics: DatasetSemantics) -> bool:
    """Whether the contract asks "Is each line one patient?" (answered or not)."""
    return semantics.question("rows_are_patients") is not None
