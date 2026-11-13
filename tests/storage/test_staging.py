# SPDX-License-Identifier: MIT
"""Typed staging of uploads: one explicit type per column, decided over the whole file."""

from __future__ import annotations

import datetime as dt
import io

import openpyxl
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.contracts.config import MISSING_VALUE_TOKENS
from clinical_analytics.storage.staging import (
    ID_COLUMN_NAMES,
    StagingError,
    describe_column_types,
    read_csv_typed,
    read_excel_typed,
    settle_shared_column_types,
    stage_text_frame,
)


def _csv(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode()


def _xlsx(rows: list[tuple[object, ...]]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(list(row))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _note(staged, column: str):
    return next(n for n in staged.notes if n.column == column)


# --- integers, dates, text -----------------------------------------------------------------


def test_stage_int_with_nulls_staysInt64():
    # Arrange
    data = _csv(["patient_id,age", "P1,40", "P2,", "P3,42"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    expected = pl.DataFrame(
        {"patient_id": ["P1", "P2", "P3"], "age": [40, None, 42]}, schema_overrides={"age": pl.Int64}
    )
    assert_frame_equal(staged.data, expected)


def test_stage_iso_dates_with_nulls_stayDate():
    # Arrange
    data = _csv(["patient_id,visit", "P1,2021-03-01", "P2,", "P3,2021-03-03"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["visit"] == pl.Date
    assert staged.data["visit"].to_list() == [dt.date(2021, 3, 1), None, dt.date(2021, 3, 3)]


def test_stage_iso_datetimes_stayDatetime():
    # Arrange
    data = _csv(["patient_id,seen", "P1,2021-03-01 08:30:00", "P2,2021-03-02 09:00:00"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["seen"] == pl.Datetime("us")


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("zip", ["02139", "10001", "02140"]),
        ("ward", ["01", "02", "11"]),
        ("site_code", ["007", "120", "450"]),
    ],
    ids=["zip", "ward", "code"],
)
def test_stage_leading_zero_codes_stayText(column, values):
    # Arrange
    data = _csv([f"patient_id,{column}", *(f"P{i},{v}" for i, v in enumerate(values))])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema[column] == pl.String
    assert staged.data[column].to_list() == values
    assert _note(staged, column).reason == "leading_zero"


@pytest.mark.parametrize("name", sorted(ID_COLUMN_NAMES))
def test_stage_id_named_integer_column_staysText(name):
    # Arrange
    data = _csv([f"{name},age", "1,40", "2,41"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema[name] == pl.String


def test_stage_late_censored_value_fallsBackOnlyThatColumn():
    # Arrange: a "<5" after row 10,000 used to turn every column into text
    lines = ["patient_id,age,creat"] + [f"Q{i},50,{80 + i % 20}" for i in range(12_000)] + ["Q12000,51,<5"]

    # Act
    staged = read_csv_typed(_csv(lines))

    # Assert
    assert staged.data.schema["age"] == pl.Int64
    assert staged.data.schema["creat"] == pl.String
    note = _note(staged, "creat")
    assert note.reason == "not_numeric"
    assert note.failed_values == 1


def test_stage_thousands_separator_isNotReadAsDecimal():
    # Arrange
    data = _csv(["patient_id,cost", 'P1,"1,234"', "P2,500", "P3,700"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["cost"] == pl.String
    assert staged.data["cost"].to_list() == ["1,234", "500", "700"]


@pytest.mark.parametrize("token", ["NaN", "inf", "-Infinity"])
def test_stage_float_words_inNumericColumn_areMissing(token):
    # Arrange
    data = _csv(["patient_id,bmi", "P1,22.5", f"P2,{token}", "P3,24.1"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["bmi"] == pl.Float64
    assert staged.data["bmi"].to_list() == [22.5, None, 24.1]


# --- missing-value tokens ------------------------------------------------------------------


@pytest.mark.parametrize("token", [t for t in MISSING_VALUE_TOKENS if t] + ["NA", "N/A", "#N/A", "NULL", " None "])
def test_stage_missing_token_inIntColumn_isNull_andCounted(token):
    # Arrange
    data = _csv(["patient_id,age", "P1,34", f"P2,{token}", "P3,51"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["age"] == pl.Int64
    assert staged.data["age"].to_list() == [34, None, 51]
    note = _note(staged, "age")
    assert (note.reason, note.failed_values, note.checked_values) == ("missing_tokens", 1, 3)


def test_stage_missing_tokens_inNumericColumns_keepNumbers():
    # Arrange: the review's example
    data = _csv(["patient_id,age,crp", "p1,34,NA", "p2,NA,5.1", "p3,51,3.2"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    expected = pl.DataFrame(
        {"patient_id": ["p1", "p2", "p3"], "age": [34, None, 51], "crp": [None, 5.1, 3.2]},
        schema={"patient_id": pl.String, "age": pl.Int64, "crp": pl.Float64},
    )
    assert_frame_equal(staged.data, expected)


def test_stage_missing_token_inDateColumn_isNull():
    # Arrange
    data = _csv(["patient_id,visit", "P1,2021-03-01", "P2,.", "P3,2021-03-03"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data["visit"].to_list() == [dt.date(2021, 3, 1), None, dt.date(2021, 3, 3)]
    assert _note(staged, "visit").reason == "missing_tokens"


@pytest.mark.parametrize("level", ["None", "-", "?", "na", "NA"])
def test_stage_text_column_keepsTokenLikeCategoryLevel(level):
    # Arrange: smoking = "None" is a real answer, not a missing value
    data = _csv(["patient_id,smoking", "P1,Current", f"P2,{level}", "P3,Former"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["smoking"] == pl.String
    assert staged.data["smoking"].to_list() == ["Current", level, "Former"]
    assert [n for n in staged.notes if n.column == "smoking"] == []


def test_stage_leading_zero_codes_withToken_stayText_unchanged():
    # Arrange
    data = _csv(["patient_id,zip", "P1,02139", "P2,NA", "P3,10001"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data["zip"].to_list() == ["02139", "NA", "10001"]
    assert _note(staged, "zip").reason == "leading_zero"


# --- integers that do not fit Int64 --------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "dtype", "reason"),
    [
        (str(2**63 - 1), pl.Int64, None),
        (str(-(2**63)), pl.Int64, None),
        (str(2**63), pl.String, "int_overflow"),
        ("12345678901234567890123", pl.String, "int_overflow"),
    ],
    ids=["int64_max", "int64_min", "int64_max_plus_one", "very_long"],
)
def test_stage_int64_boundary_neverBecomesFloat(value, dtype, reason):
    # Arrange
    data = _csv(["patient_id,accession", f"P1,{value}", "P2,1"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["accession"] == dtype
    notes = [n.reason for n in staged.notes if n.column == "accession"]
    assert notes == ([reason] if reason else [])
    if dtype == pl.String:
        assert staged.data["accession"].to_list() == [value, "1"]


def test_stage_all_null_column_isText_withoutNote():
    # Arrange
    data = _csv(["patient_id,empty", "P1,", "P2,"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["empty"] == pl.String
    assert [n for n in staged.notes if n.column == "empty"] == []


def test_stage_true_false_column_isBoolean():
    # Arrange
    data = _csv(["patient_id,flag", "P1,true", "P2,False", "P3,"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data["flag"].to_list() == [True, False, None]


# --- mixed date formats --------------------------------------------------------------------


def test_stage_mixed_date_formats_parsedPerValue():
    # Arrange: ISO and day-first slashes (31/01 proves day-first)
    rows = [f"P{i},2023-01-15,50" for i in range(20)] + [f"Q{i},31/01/2023,51" for i in range(20)]
    data = _csv(["patient_id,admit_date,age", *rows])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["admit_date"] == pl.Date
    assert staged.data["admit_date"].unique().sort().to_list() == [dt.date(2023, 1, 15), dt.date(2023, 1, 31)]


def test_stage_unparsed_date_value_keepsColumnText_neverNulled():
    # Arrange: one value matches no format; it must not silently become missing
    rows = [f"P{i},2023-01-15,50" for i in range(40)] + ["R0,soon,52"]
    data = _csv(["patient_id,admit_date,age", *rows])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["admit_date"] == pl.String
    assert staged.data["admit_date"].to_list()[-1] == "soon"
    note = _note(staged, "admit_date")
    assert (note.reason, note.failed_values) == ("not_dates", 1)


def test_stage_day_first_and_month_first_families_isAmbiguous():
    # Arrange: "/" settles day-first (13/01), "-" settles month-first (01-13)
    data = _csv(["patient_id,seen", "P1,13/01/2024", "P2,01-13-2024", "P3,02/03/2024"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["seen"] == pl.String
    assert _note(staged, "seen").reason == "ambiguous_dates"


def test_stage_ambiguous_day_month_order_staysText_withNote():
    # Arrange: 03/04 and 04/03 can be read either way and nothing settles which
    data = _csv(["patient_id,seen", "P1,03/04/2023", "P2,04/03/2023", "P3,05/06/2023"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["seen"] == pl.String
    assert _note(staged, "seen").reason == "ambiguous_dates"


def test_stage_month_first_slashes_settledByDayAbove12():
    # Arrange
    data = _csv(["patient_id,seen", "P1,03/04/2023", "P2,12/25/2023"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data["seen"].to_list() == [dt.date(2023, 3, 4), dt.date(2023, 12, 25)]


def test_stage_mostly_text_column_withSomeDates_staysText():
    # Arrange
    data = _csv(["patient_id,comment", "P1,2023-01-01", "P2,see notes", "P3,unknown", "P4,n/a"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["comment"] == pl.String


# --- units rows and empty rows -------------------------------------------------------------


def test_stage_excel_units_row_isMetadata_notAPatient():
    # Arrange
    rows = [("patient_id", "age", "hba1c", "ldl"), (None, "years", "%", "mg/dL")]
    rows += [(f"P{i:03d}", 30 + i % 40, 5 + (i % 10) / 10, 100 + i) for i in range(200)]

    # Act
    staged = read_excel_typed(_xlsx(rows))

    # Assert
    assert staged.data.height == 200
    assert staged.units == {"age": "years", "hba1c": "%", "ldl": "mg/dL"}
    assert staged.data.schema["age"] == pl.Int64
    assert staged.data.schema["hba1c"] == pl.Float64


def test_stage_csv_units_row_isMetadata_notAPatient():
    # Arrange
    data = _csv(["patient_id,age,ldl", ",years,mg/dL", "P1,40,100", "P2,41,101"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.height == 2
    assert staged.units == {"age": "years", "ldl": "mg/dL"}


@pytest.mark.parametrize(
    "first_row",
    [
        "P0,<5,<5",  # a patient with censored values, not units
        ",40,100",  # a patient with a missing ID but real values
        "P0,years,100",  # an ID is present: a patient row with a typo, not units
    ],
    ids=["censored", "numeric", "has_id"],
)
def test_stage_first_row_that_is_data_isKept(first_row):
    # Arrange
    data = _csv(["patient_id,age,ldl", first_row, "P1,40,100", "P2,41,101"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.height == 3
    assert staged.units == {}


@pytest.mark.parametrize(
    "first_row",
    ["unknown,missing", "NA,NA", "pending,later"],
    ids=["labels", "null_tokens", "words"],
)
def test_stage_idless_first_row_withoutUnitWords_isAPatient(first_row):
    # Arrange: no ID-named column and no unit word: a real first patient, not units
    data = _csv(["age,weight", first_row, "34,70", "51,80"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.height == 3
    assert staged.units == {}


def test_stage_idless_units_row_withUnitWords_isRemoved_andReported():
    # Arrange
    data = _csv(["age,ldl", "years,mg/dL", "34,100", "51,120"])

    # Act
    staged = read_csv_typed(data)
    lines = describe_column_types([], staged.units, small_cell_limit=11)

    # Assert
    assert staged.data.height == 2
    assert staged.units == {"age": "years", "ldl": "mg/dL"}
    assert lines[0].startswith("The row under the header was read as units")


@pytest.mark.parametrize(
    "tail",
    [[","], [",", ","], [" , "], [",", ""]],
    ids=["one", "two", "spaces", "then_blank"],
)
def test_stage_trailing_comma_only_lines_areDropped(tail):
    # Arrange: Excel and other exporters write trailing ",,," lines
    data = _csv(["patient_id,age", "1,30", "2,31", *tail])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.height == 2


@pytest.mark.parametrize("blank", [",,", " , , "], ids=["empty", "spaces"])
def test_stage_row_without_id_or_values_failsLoudly(blank):
    # Arrange
    data = _csv(["patient_id,age,ldl", "P1,40,100", blank, "P2,41,101"])

    # Act / Assert
    with pytest.raises(StagingError, match="row 3"):
        read_csv_typed(data)


def test_stage_excel_merged_two_level_header_keepsColumns():
    # Arrange: a title row above the real header, as in merged.xlsx
    rows = [("Study cohort", None, None), ("patient_id", "age", "sex")]
    rows += [(f"P{i}", 40 + i, "F" if i % 2 else "M") for i in range(50)]

    # Act
    staged = read_excel_typed(_xlsx(rows))

    # Assert
    assert staged.data.columns == ["patient_id", "age", "sex"]
    assert staged.data.height == 50


def test_stage_text_frame_requires_text_columns():
    # Arrange
    df = pl.DataFrame({"a": [1, 2]})

    # Act / Assert
    with pytest.raises(TypeError, match="text"):
        stage_text_frame(df)


# --- the report shown to the user ----------------------------------------------------------


@pytest.mark.parametrize(
    ("failed", "checked", "shown"),
    [
        (1, 200, "fewer than 11"),
        (10, 200, "fewer than 11"),
        (11, 200, "11"),
        (50, 200, "50"),
        (195, 200, "all but fewer than 11"),
        (189, 200, "189"),
    ],
    ids=["one", "ten", "eleven", "fifty", "complement_small", "complement_eleven"],
)
def test_describeColumnTypes_counts_neverRevealSmallCells(failed, checked, shown):
    # Arrange
    notes = [{"column": "creat", "reason": "not_numeric", "failed_values": failed, "checked_values": checked}]

    # Act
    lines = describe_column_types(notes, {}, small_cell_limit=11)

    # Assert
    assert lines == [
        f"creat: kept as text: {shown} values are not plain numbers (like '<5' or '1,234'), so it is not averaged."
    ]


def test_describeColumnTypes_units_listedPerColumn():
    # Arrange / Act
    lines = describe_column_types([], {"ldl": "mg/dL"}, small_cell_limit=11)

    # Assert
    assert lines == [
        "The row under the header was read as units, not a patient, and is not in the data.",
        "ldl: unit mg/dL (from the units row under the header).",
    ]


# --- column types shared across tables -----------------------------------------------------


def test_settleSharedColumnTypes_textInOneTable_makesKeyTextEverywhere():
    # Arrange: "ward" has a leading zero in one table only
    frames = {
        "patients": pl.DataFrame({"patient_id": ["P1", "P2"], "ward": ["01", "12"]}),
        "visits": pl.DataFrame({"patient_id": ["P1", "P2"], "ward": [12, 13]}),
    }

    # Act
    settled, notes = settle_shared_column_types(frames)

    # Assert
    assert settled["visits"].schema["ward"] == pl.String
    assert settled["visits"]["ward"].to_list() == ["12", "13"]
    assert settled["patients"]["ward"].to_list() == ["01", "12"]
    assert [(t, n.column, n.reason) for t, n in notes] == [("visits", "ward", "shared_column_type")]


def test_settleSharedColumnTypes_intAndFloat_becomeFloat():
    # Arrange
    frames = {"a": pl.DataFrame({"k": [1, 2]}), "b": pl.DataFrame({"k": [1.5, 2.0]})}

    # Act
    settled, _ = settle_shared_column_types(frames)

    # Assert
    assert settled["a"].schema["k"] == pl.Float64


def test_settleSharedColumnTypes_sameTypes_unchanged():
    # Arrange
    frames = {"a": pl.DataFrame({"k": ["x"]}), "b": pl.DataFrame({"k": ["y"]})}

    # Act
    settled, notes = settle_shared_column_types(frames)

    # Assert
    assert notes == []
    assert_frame_equal(settled["a"], frames["a"])
