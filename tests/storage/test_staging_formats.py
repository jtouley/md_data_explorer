# SPDX-License-Identifier: MIT
"""Staging of the formats public datasets ship in: legacy .xls, SAS transport .xpt, PSV/TSV.

Every fixture is synthetic and built in the test (xlwt for .xls, pyreadstat for .xpt).
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import polars as pl
import pyreadstat
import pytest
import xlwt
from polars.testing import assert_frame_equal

from clinical_analytics.storage.staging import (
    ExcelSheets,
    StagingError,
    read_csv_typed,
    read_excel_typed,
    read_xport_typed,
    stage_text_frame,
)

_DATE = xlwt.easyxf(num_format_str="YYYY-MM-DD")


def _xls(sheets: dict[str, list[list[object]]], *, raw_biff: bool = False) -> bytes:
    """A synthetic .xls workbook; ``raw_biff`` gives the bare BIFF stream (no OLE2 wrapper),
    as some old exporters write (hbiostat's prostate.xls is one)."""
    wb = xlwt.Workbook()
    for title, rows in sheets.items():
        ws = wb.add_sheet(title)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if value is None:
                    continue
                if isinstance(value, dt.date):
                    ws.write(r, c, value, _DATE)
                else:
                    ws.write(r, c, value)
    if raw_biff:
        return bytes(wb.get_biff_data())
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


_HEADER = ["patient_id", "visit", "age", "hb", "arm", "smoker"]


def _rows(n: int) -> list[list[object]]:
    return [
        [f"P{i:03d}", dt.date(2024, 1, 1 + i % 28), 40 + i % 30, 11.5 + i % 4 / 2, "A" if i % 2 else "B", i % 3 == 0]
        for i in range(n)
    ]


def _expected(n: int) -> pl.DataFrame:
    rows = _rows(n)
    return pl.DataFrame(
        {
            "patient_id": [r[0] for r in rows],
            "visit": [r[1] for r in rows],
            "age": [r[2] for r in rows],
            "hb": [r[3] for r in rows],
            "arm": [r[4] for r in rows],
            "smoker": [r[5] for r in rows],
        },
        schema={
            "patient_id": pl.String,
            "visit": pl.Date,
            "age": pl.Int64,
            "hb": pl.Float64,
            "arm": pl.String,
            "smoker": pl.Boolean,
        },
    )


# --- legacy .xls ------------------------------------------------------------------------------


@pytest.mark.parametrize("raw_biff", [False, True], ids=["ole2_container", "raw_biff_stream"])
def test_readExcelTyped_legacyXls_typesEveryColumnLikeXlsx(raw_biff: bool) -> None:
    # Arrange: dates, whole and decimal numbers, text and booleans
    data = _xls({"Data": [_HEADER, *_rows(24)]}, raw_biff=raw_biff)

    # Act
    staged = read_excel_typed(data)

    # Assert
    assert_frame_equal(staged.data, _expected(24))
    assert staged.sheets == ExcelSheets("Data")


def test_readExcelTyped_legacyXlsSheets_namesSkippedAndCountsEmpty() -> None:
    # Arrange: a codebook sheet with values, and an unused sheet
    data = _xls({"Data": [_HEADER, *_rows(12)], "Codebook": [["code", "label"], [1, "male"]], "Sheet3": []})

    # Act
    staged = read_excel_typed(data)

    # Assert
    assert staged.sheets == ExcelSheets("Data", skipped=("Codebook",), empty=1)
    assert staged.data.height == 12


def test_readExcelTyped_legacyXlsTimeOfDay_staysTimeLikeXlsx() -> None:
    # Arrange: a time-only cell (serial below 1) used to become a datetime on 1899-12-31
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Data")
    ws.write(0, 0, "patient_id")
    ws.write(0, 1, "admit_time")
    for r in range(1, 13):
        ws.write(r, 0, f"P{r:02d}")
        ws.write(r, 1, dt.time(8, r), xlwt.easyxf(num_format_str="HH:MM"))
    buf = io.BytesIO()
    wb.save(buf)

    # Act
    staged = read_excel_typed(buf.getvalue())

    # Assert
    assert staged.data["admit_time"].dtype != pl.Datetime
    assert "1899" not in str(staged.data["admit_time"].cast(pl.String).to_list())
    assert str(staged.data["admit_time"][0]).startswith("08:01")


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600, id="ole2_header_only"),
        pytest.param(b"not a workbook at all, just text\n" * 40, id="text_named_xls"),
        pytest.param(b"", id="empty"),
    ],
)
def test_readExcelTyped_unreadableXls_raisesStagingError(data: bytes) -> None:
    # Arrange: bytes above

    # Act / Assert
    with pytest.raises(StagingError, match="could not be read"):
        read_excel_typed(data)


# --- SAS transport (.xpt) ---------------------------------------------------------------------


def _xpt(tmp_path: Path, df: pl.DataFrame) -> bytes:
    path = tmp_path / "table.xpt"
    pyreadstat.write_xport(df, str(path), file_format_version=5, table_name="TABLE")
    return path.read_bytes()


def test_readXportTyped_sasNumbers_wholeBecomeInt64_fractionsStayFloat(tmp_path: Path) -> None:
    # Arrange: SAS stores every number as a float; codes and IDs are whole
    df = pl.DataFrame(
        {
            "SEQN": [93703.0, 93704.0, 93705.0],
            "RIAGENDR": [2.0, None, 1.0],
            "LBXSATSI": [21.0, 14.5, None],
            "NAME": ["a", "", "c"],
        }
    )

    # Act
    staged = read_xport_typed(_xpt(tmp_path, df))

    # Assert
    expected = pl.DataFrame(
        {
            "SEQN": [93703, 93704, 93705],
            "RIAGENDR": [2, None, 1],
            "LBXSATSI": [21.0, 14.5, None],
            "NAME": ["a", None, "c"],
        },
        schema={"SEQN": pl.Int64, "RIAGENDR": pl.Int64, "LBXSATSI": pl.Float64, "NAME": pl.String},
    )
    assert_frame_equal(staged.data, expected)
    assert staged.notes == ()


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        pytest.param([None, None, None], pl.String, id="all_missing_isText"),
        pytest.param([1.0, None, 3.0], pl.Int64, id="nulls_stayNull_inInt64"),
        pytest.param([float(2**53 - 1), 1.0, 2.0], pl.Int64, id="largest_exact_whole_isInt64"),
        pytest.param([float(2**53), 1.0, 2.0], pl.Float64, id="beyond_exact_staysFloat"),
        pytest.param([0.0, -1.0, 5.397605346934028e-79], pl.Float64, id="sas_tiny_value_staysFloat"),
    ],
)
def test_readXportTyped_numericColumn_boundaries(
    tmp_path: Path, values: list[float | None], dtype: pl.DataType
) -> None:
    # Arrange
    df = pl.DataFrame({"SEQN": [1.0, 2.0, 3.0], "X": values}, schema={"SEQN": pl.Float64, "X": pl.Float64})

    # Act
    staged = read_xport_typed(_xpt(tmp_path, df))

    # Assert
    assert staged.data.schema["X"] == dtype
    assert staged.data["X"].null_count() == sum(v is None for v in values)


def test_readXportTyped_notTransportFile_raisesStagingError() -> None:
    # Arrange
    data = b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!" + b"0" * 200

    # Act / Assert
    with pytest.raises(StagingError, match=r"SAS transport \(\.xpt\) file could not be read"):
        read_xport_typed(data)


# --- delimited text ---------------------------------------------------------------------------


@pytest.mark.parametrize("separator", ["|", "\t"], ids=["psv", "tsv"])
def test_readCsvTyped_otherSeparator_readsAndTypes(separator: str) -> None:
    # Arrange: PhysioNet's "NaN" marker in a numeric column
    lines = [separator.join(["HR", "Gender"]), separator.join(["80.5", "0"]), separator.join(["NaN", "1"])]
    data = ("\n".join(lines) + "\n").encode()

    # Act
    staged = read_csv_typed(data, separator=separator)

    # Assert
    assert staged.data.schema == pl.Schema({"HR": pl.Float64, "Gender": pl.Int64})
    assert staged.data["HR"].null_count() == 1


def test_stageTextFrame_blankHeader_isNamedLikeExcel() -> None:
    # Arrange: R's write.csv puts row names under a "" header (UCI's hcvdat0.csv)
    text = pl.DataFrame({"": ["1", "2"], "ALB": ["38.5", "41"]})

    # Act
    staged = stage_text_frame(text)

    # Assert
    assert staged.data.columns == ["column_0", "ALB"]


def test_stageTextFrame_blankHeaderBesideColumn0_getsUnclashedName() -> None:
    # Arrange
    text = pl.DataFrame({"": ["1", "2"], "column_0": ["a", "b"]})

    # Act
    staged = stage_text_frame(text)

    # Assert
    assert staged.data.columns == ["column_0_unnamed", "column_0"]


def test_stageTextFrame_blankHeaderBesideUpperCaseColumn0_getsUnclashedName() -> None:
    # Arrange: the query engine ignores case, so "COLUMN_0" already takes "column_0"
    text = pl.DataFrame({"": ["1", "2"], "COLUMN_0": ["a", "b"]})

    # Act
    staged = stage_text_frame(text)

    # Assert
    assert staged.data.columns == ["column_0_unnamed", "COLUMN_0"]


@pytest.mark.parametrize(
    ("headers", "named"),
    [
        pytest.param(["Age", "age"], "'Age' and 'age'", id="two"),
        pytest.param(["ID", "x", "Id", "id"], "'ID' and 'Id' and 'id'", id="three"),
        pytest.param(["ALB", "alb", "Sex", "SEX"], "'ALB' and 'alb'; 'Sex' and 'SEX'", id="two_pairs"),
    ],
)
def test_stageTextFrame_headersDifferOnlyInCase_failsNamingThem(headers: list[str], named: str) -> None:
    # Arrange
    text = pl.DataFrame({h: ["1", "2"] for h in headers})

    # Act
    with pytest.raises(StagingError) as raised:
        stage_text_frame(text)

    # Assert
    assert f"Columns {named} differ only in upper and lower case" in str(raised.value)


# --- CDC WONDER exports: tab-separated, a notes block, suppressed values (synthetic) ------------


def test_readCsvTyped_wonderExport_notesBlockIsDocsTotalsLeftOutSuppressedMissing(make_wonder_export) -> None:
    # Arrange: 40 states; every 7th suppressed (deaths and rate), every 5th rate unreliable
    data = make_wonder_export(states=40)

    # Act
    staged = read_csv_typed(data, separator="\t")

    # Assert: data rows only; the "Total" subtotal and the notes block are not rows
    frame = staged.data
    assert frame.height == 40
    assert "Notes" not in frame.columns
    assert frame.schema["Deaths"] == pl.Int64
    assert frame.schema["Crude Rate"] == pl.Float64
    # Suppressed is missing, never 0
    suppressed_rows = frame.filter(pl.col("State Code").cast(pl.Int64) % 7 == 0)
    assert suppressed_rows["Deaths"].null_count() == suppressed_rows.height == 5
    assert frame["Deaths"].min() > 0
    assert frame["Crude Rate"].null_count() == 5 + 7  # 5 suppressed, 7 more unreliable (5, 10, ..., 40)
    reasons = {(n.column, n.reason): n.failed_values for n in staged.notes}
    assert reasons[("Deaths", "source_suppressed")] == 5
    assert reasons[("Crude Rate", "source_suppressed")] == 12
    assert "Dataset: Synthetic Underlying Cause of Death" in staged.export_notes
    assert "Query Date" in staged.export_notes
    assert staged.table_notes == ("1 subtotal row (Notes: Total) was left out, so no row is counted twice.",)


@pytest.mark.parametrize(
    ("states", "total_row", "expected_rows", "expected_notes"),
    [
        pytest.param(12, False, 12, (), id="no_total_row"),
        pytest.param(
            1, True, 1, ("1 subtotal row (Notes: Total) was left out, so no row is counted twice.",), id="one_state"
        ),
    ],
)
def test_readCsvTyped_wonderExportShapes_rowsAndNotes(
    make_wonder_export, states: int, total_row: bool, expected_rows: int, expected_notes: tuple[str, ...]
) -> None:
    # Arrange
    data = make_wonder_export(states=states, total_row=total_row)

    # Act
    staged = read_csv_typed(data, separator="\t")

    # Assert
    assert staged.data.height == expected_rows
    assert staged.table_notes == expected_notes


def test_readCsvTyped_unreliableAsCategory_keptAsText() -> None:
    # Arrange: "Unreliable" is one of a text column's categories, not a withheld number
    rows = [f"P{i:03d}\t{'Unreliable' if i % 3 == 0 else 'Reliable'}\t{i}" for i in range(30)]
    data = ("\n".join(["patient_id\treporting\tvisits", *rows]) + "\n").encode()

    # Act
    staged = read_csv_typed(data, separator="\t")

    # Assert
    assert staged.data.schema["reporting"] == pl.String
    assert staged.data["reporting"].null_count() == 0
    assert not [n for n in staged.notes if n.reason == "source_suppressed"]


def test_readCsvTyped_noNotesBlock_exportNotesEmpty() -> None:
    # Arrange: a plain TSV
    data = b"patient_id\tage\nP1\t40\nP2\t50\n"

    # Act
    staged = read_csv_typed(data, separator="\t")

    # Assert
    assert staged.export_notes == ""
    assert staged.table_notes == ()
    assert staged.data.height == 2


def test_describeColumnTypes_sourceSuppressed_smallCountNeverShown() -> None:
    # Arrange: 3 of 40 values withheld; a count from 1 to 10 is never shown
    from clinical_analytics.storage.staging import describe_column_types

    notes = [
        {
            "column": "Deaths",
            "stored_type": "Int64",
            "reason": "source_suppressed",
            "failed_values": 3,
            "checked_values": 40,
        }
    ]

    # Act
    lines = describe_column_types(notes, {}, 11)

    # Assert
    assert lines == [
        "Deaths: fewer than 11 values were withheld or not given by the data source (Suppressed, Unreliable, "
        "Not Applicable) and are stored as missing, never as 0; sums and counts leave them out."
    ]


def test_readCsvTyped_dashLineInOrdinaryCsv_neverHidesRows() -> None:
    # Arrange: a "---" line in a file that is not a WONDER export (review of #123, finding 1)
    data = b"patient_id,age,comment\n1,40,ok\n2,50,x\n---\n3,60,y\n4,70,z\n"

    # Act
    staged = read_csv_typed(data)

    # Assert: every row is data; nothing goes to documentation
    assert staged.data.height == 5
    assert staged.export_notes == ""


def test_readCsvTyped_notesHeaderButDataBelowDashes_neverHidesRows(make_wonder_export) -> None:
    # Arrange: WONDER header, but tab-separated (data-shaped) lines under "---"
    data = make_wonder_export(states=12, total_row=False).replace(
        b'"---"\n"Dataset', b'"---"\n\t"State 99"\t"99"\t5\t1\t1.0\n"Dataset', 1
    )

    # Act
    staged = read_csv_typed(data, separator="\t")

    # Assert: not split; the rows below stay in the file the user sees as data
    assert staged.export_notes == ""


@pytest.mark.parametrize(
    "marker",
    [
        pytest.param("2.1 (Unreliable)", id="flaggedNumber"),
        pytest.param("Not Applicable", id="notApplicable"),
        pytest.param("unreliable", id="lowercase"),
    ],
)
def test_readCsvTyped_wonderMarkerForms_storedMissingNeverText(marker: str) -> None:
    # Arrange
    # WONDER-shaped: first header "Notes" (markers are read only in that shape)
    rows = [f"\tS{i:02d}\t{marker if i % 4 == 0 else f'{10 + i}.5'}" for i in range(24)]
    data = ("\n".join(["Notes\tState\tCrude Rate", *rows]) + "\n").encode()

    # Act
    staged = read_csv_typed(data, separator="\t")

    # Assert
    assert staged.data.schema["Crude Rate"] == pl.Float64
    assert staged.data["Crude Rate"].null_count() == 6
    assert [n.reason for n in staged.notes if n.column == "Crude Rate"] == ["source_suppressed"]


def test_readCsvTyped_suppressedInColumnThatStaysText_markerKept() -> None:
    # Arrange: thousands separators keep the column text, so "Suppressed" must stay as written
    rows = [f"\tS{i:02d}\t{'Suppressed' if i % 5 == 0 else '1,234'}" for i in range(20)]
    data = ("\n".join(["Notes\tState\tDeaths", *rows]) + "\n").encode()

    # Act
    staged = read_csv_typed(data, separator="\t")

    # Assert
    assert staged.data.schema["Deaths"] == pl.String
    assert staged.data["Deaths"].null_count() == 0
    assert not [n for n in staged.notes if n.reason == "source_suppressed"]


@pytest.mark.parametrize("separator", [",", "\t"], ids=["csv", "tsv"])
@pytest.mark.parametrize("marker", ["<5", "Suppressed", "Unreliable", "Total"])
def test_readCsvTyped_patientFileWithMarker_keptAsTextExactlyAsOnMain(separator: str, marker: str) -> None:
    # Arrange: an ordinary patient file (no "Notes" first column); a marker in a number column
    values = [marker if i % 6 == 0 else str(100 + i) for i in range(30)]
    rows = [separator.join([f"P{i:03d}", value, str(30 + i)]) for i, value in enumerate(values)]
    data = (separator.join(["patient_id", "viral_load", "age"]) + "\n" + "\n".join(rows) + "\n").encode()

    # Act
    staged = read_csv_typed(data, separator=separator)

    # Assert: nothing is nulled, dropped or flagged; the column is the text the file holds
    assert staged.data.height == 30
    assert staged.data.schema["viral_load"] == pl.String
    assert staged.data["viral_load"].to_list() == values
    assert not [n for n in staged.notes if n.reason == "source_suppressed"]
    assert staged.table_notes == ()
    assert staged.export_notes == ""
