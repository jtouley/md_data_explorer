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
    and not named as a measurement. Uniqueness is the caller's test.

    A float column qualifies only when it is named as a patient ID and every value is a whole
    number: SPSS stores every number as a float, so its "ID" of 1.0, 2.0, ... is still an ID. An
    unnamed float is a measurement even when rounded (a rate rounded to integers).
    """
    from clinical_analytics.core.schema_inference import patient_key_by_name

    dtype = series.dtype
    if is_measurement_name(series.name):
        return False
    if dtype.is_float():
        present = series.drop_nulls()
        return (
            patient_key_by_name([series.name]) is not None
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


def _quoted_list(names: list[str]) -> str:
    quoted = [f"“{name}”" for name in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def row_key_note(metadata: Mapping[str, object]) -> str:
    """The upload note when answers count rows, not patients, or "" when patient_id names patients.

    Said at upload, in plain words, whenever ``patient_id_is_row_key``: what tells the rows apart
    (a row number, a combination of columns, or a column not named as a patient ID), that every
    count is a count of rows, and how to get patient counts. Column names only, never values.
    """
    if not patient_id_is_row_key(metadata):
        return ""
    synthetic = metadata.get("synthetic_id_metadata")
    info = synthetic.get("patient_id") if isinstance(synthetic, Mapping) else None
    source = info.get("patient_id_source") if isinstance(info, Mapping) else None
    raw_columns = info.get("patient_id_columns") if isinstance(info, Mapping) else None
    columns = [str(c) for c in raw_columns] if isinstance(raw_columns, list) else []
    if source == "composite" and columns:
        found = f"No column in this file identifies a patient, so rows are told apart by {_quoted_list(columns)}."
    elif source == "single_column" and columns:
        found = f"Rows are told apart by {_quoted_list(columns)}, which is not named as a patient ID."
    else:
        found = f"No column in this file identifies a patient, so each row was numbered (“{ROW_NUMBER_COLUMN}”)."
    return (
        f"{found} Answers count rows, not patients. To count patients, add a column with one ID for each "
        "patient, named patient_id, and upload the file again."
    )


def added_id_columns(metadata: Mapping[str, object]) -> list[str]:
    """The ID column the upload added to the file's own columns: ``patient_id`` when it was
    copied from a column or hashed from several, the row number when rows were numbered, none
    when the file had ``patient_id``. Reported with the dataset, so a check of the file's shape
    (the acceptance upload check) compares the file's own columns."""
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
# Asked once at upload when the stored patient_id is a row key (``patient_id_is_row_key``) built
# from one column unique on every row, or when nothing identifies a row (a row number). A "yes" on
# visit data would turn visit counts into "patients", so the question is never offered when a
# column that looks like an ID repeats: those rows are visits or tests of someone.

# Upload metadata (``synthetic_id_metadata.patient_id``): the ID-like columns whose values repeat,
# recorded by ``ensure_patient_id`` for the sources that may be asked. Absent: never asked.
REPEATED_ID_COLUMNS_KEY: Final = "repeated_id_columns"
# Semantic-layer config flag: ``rows_may_be_patients`` for the dataset. A stored one-row-per-patient
# confirmation makes answers count patients only where it is set (``answer_checks``), so a
# confirmation given elsewhere (governed SQL) never turns visit rows into patients.
PATIENT_ROWS_ASKABLE_CONFIG_FLAG: Final = "rows_may_be_patients"

# Name tokens of an identifier ("person_id", "caseId", "Hosp No", "Record Number"): split on
# anything but letters and digits and at a lower-to-upper case change. Broad on purpose: a column
# matched by mistake that repeats ("number_of_children", "site_id") only means the question is not
# asked; a repeating ID that is missed means it is asked on visit rows.
ID_NAME_TOKENS: Final[frozenset[str]] = frozenset(
    {"id", "ids", "identifier", "mrn", "mr", "uuid", "guid", "no", "nr", "num", "number", "ref"}
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def is_id_name(column: str) -> bool:
    """A column named as an identifier of someone or something (``ID_NAME_TOKENS``), or named for
    the patient-key role (``schema_inference.PATIENT_KEY_NAMES``)."""
    from clinical_analytics.core.schema_inference import PATIENT_KEY_NAMES

    if re.sub(r"[^a-z0-9]", "", column.lower()) in PATIENT_KEY_NAMES:
        return True
    return any(token in ID_NAME_TOKENS for token in name_tokens(_CAMEL_BOUNDARY.sub("_", column)))


def repeated_id_columns(df: pl.DataFrame, identifier_columns: list[str]) -> list[str]:
    """The columns that look like IDs and hold a value on more than one row: named as an ID
    (``is_id_name``, whatever the type: SPSS stores an ID of 1, 2, ... as floats) or typed as one
    (``identifier_columns``, over 95% unique), with more than one distinct value (a study code on
    every row identifies no one). One select over the frame.
    """
    candidates = [
        column
        for column in df.columns
        if column != ROW_NUMBER_COLUMN and (column in identifier_columns or is_id_name(column))
    ]
    if not candidates or df.height < 2:
        return []
    found = df.select(
        [
            ((pl.col(c).drop_nulls().n_unique() > 1) & pl.col(c).drop_nulls().is_duplicated().any()).alias(c)
            for c in candidates
        ]
    ).row(0)
    return [column for column, repeats in zip(candidates, found, strict=True) if repeats]


def rows_may_be_patients(metadata: Mapping[str, object]) -> bool:
    """Whether the upload may ask "Is each row one patient?": patient_id is a row key copied from
    one column unique on every row, or rows were numbered because nothing identifies them, and
    ``ensure_patient_id`` recorded that no ID-like column repeats. Composite keys (rows told apart
    only by a combination, so single columns repeat) and uploads without the record are never
    asked: the answer would rest on data that may be visits.
    """
    if not patient_id_is_row_key(metadata):
        return False
    synthetic = metadata.get("synthetic_id_metadata")
    info = synthetic.get("patient_id") if isinstance(synthetic, Mapping) else None
    if not isinstance(info, Mapping) or info.get("patient_id_source") not in ("single_column", "row_index"):
        return False
    if info.get("row_number_column"):
        return False  # a declared public estimate table (ADR014): its rows are groups, never patients
    repeated = info.get(REPEATED_ID_COLUMNS_KEY)
    return isinstance(repeated, list) and not repeated
