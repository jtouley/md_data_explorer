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
