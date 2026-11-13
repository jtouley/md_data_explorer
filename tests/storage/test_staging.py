# SPDX-License-Identifier: MIT
"""Typed staging of uploads: one explicit type per column, decided over the whole file."""

from __future__ import annotations

import datetime as dt
import io

import openpyxl
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.storage.staging import (
    ID_COLUMN_NAMES,
    StagingError,
    describe_column_types,
    read_csv_typed,
    read_excel_typed,
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
def test_stage_float_words_areNotFloats(token):
    # Arrange
    data = _csv(["patient_id,bmi", "P1,22.5", f"P2,{token}", "P3,24.1"])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["bmi"] == pl.String


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


def test_stage_mixed_date_formats_parsedPerValue_andFailuresCounted():
    # Arrange: ISO and day-first slashes (31/01 proves day-first), plus one unparseable value
    rows = [f"P{i},2023-01-15,50" for i in range(20)] + [f"Q{i},31/01/2023,51" for i in range(20)] + ["R0,soon,52"]
    data = _csv(["patient_id,admit_date,age", *rows])

    # Act
    staged = read_csv_typed(data)

    # Assert
    assert staged.data.schema["admit_date"] == pl.Date
    assert staged.data.schema["age"] == pl.Int64
    assert staged.data["admit_date"].drop_nulls().unique().sort().to_list() == [
        dt.date(2023, 1, 15),
        dt.date(2023, 1, 31),
    ]
    note = _note(staged, "admit_date")
    assert note.reason == "date_values_unparsed"
    assert note.failed_values == 1


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
    assert lines == ["ldl: unit mg/dL (from the units row under the header)."]
