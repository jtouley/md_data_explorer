# SPDX-License-Identifier: MIT
"""Upload → query through the API: a dataset reported ready always answers (DQ H1)."""

from __future__ import annotations

import datetime as dt
import io

import openpyxl
import polars as pl
import pytest
from fastapi.testclient import TestClient


def _mixed_dates_csv() -> bytes:
    """The DQ review's dates.csv shape: ISO dates, then day-first slash dates, constant age."""
    rows = [f"P{i:03d},2023-01-15,50" for i in range(70)] + [f"P{i:03d},15/01/2023,50" for i in range(70, 200)]
    return ("\n".join(["patient_id,admit_date,age", *rows]) + "\n").encode()


def test_uploadDataset_mixedDateFormats_readyThenQueryable(test_client: TestClient, isolated_data_home):
    # Arrange
    from clinical_analytics.api import dependencies

    # Act
    response = test_client.post(
        "/api/datasets/upload",
        files={"file": ("dates.csv", _mixed_dates_csv(), "text/csv")},
        data={"dataset_name": "dates"},
    )
    body = response.json()
    with dependencies.lease_semantic_layer(body["upload_id"]) as lease:
        frame = lease.layer.get_base_view().to_polars()

    # Assert
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert body["cohort_notes"] == []  # single table: nothing left out of the cohort
    assert frame.height == 200
    assert frame.schema["age"] == pl.Int64
    assert frame["admit_date"].unique().to_list() == [dt.date(2023, 1, 15)]
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])


def test_uploadDataset_censoredLabValue_typeNoteNamesColumnWithoutSmallCount(
    test_client: TestClient, isolated_data_home
):
    # Arrange: one "<5" in creat; the count (1) must not be shown
    rows = [f"P{i:03d},{40 + i % 30},{80 + i % 9}" for i in range(199)] + ["P199,50,<5"]
    data = ("\n".join(["patient_id,age,creat", *rows]) + "\n").encode()

    # Act
    response = test_client.post(
        "/api/datasets/upload", files={"file": ("labs.csv", data, "text/csv")}, data={"dataset_name": "labs"}
    )

    # Assert
    assert response.status_code == 200, response.text
    notes = response.json()["type_notes"]
    assert len(notes) == 1
    assert notes[0].startswith("creat: kept as text")
    assert "fewer than 11" in notes[0]
    assert " 1 " not in notes[0]


def _xlsx_bytes(sheets: dict[str, list[list[object]]]) -> bytes:
    """A synthetic workbook, sheets in the order given."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _rate_rows(n: int) -> list[list[object]]:
    # An explicit ID column, so the measured columns are never taken for one.
    header: list[object] = ["patient_id", "location_name", "sex", "year", "mx", "upper", "lower"]
    body: list[list[object]] = [
        [f"R{i:04d}", f"Area {i % 5}", ("Male", "Female", "Both")[i % 3], 2000 + i % 8, 120.5 + i, 130.5 + i, 110.5 + i]
        for i in range(n)
    ]
    return [header, *body]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "above",
    [
        pytest.param([["Synthetic mortality rates by area"]], id="title_in_A1"),
        pytest.param([["Synthetic mortality rates"], []], id="title_then_blank"),
    ],
)
def test_uploadDataset_xlsxTitleAboveHeader_readyWithEveryColumn(
    test_client: TestClient, isolated_data_home, above: list[list[object]]
) -> None:
    # Arrange: the triage's title_row.xlsx shape (a title in A1, the header below it)
    from clinical_analytics.api import dependencies

    data = _xlsx_bytes({"Rates": [*above, *_rate_rows(240)]})

    # Act
    response = test_client.post(
        "/api/datasets/upload",
        files={"file": ("title_row.xlsx", data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"dataset_name": "title_row"},
    )
    body = response.json()
    with dependencies.lease_semantic_layer(body["upload_id"]) as lease:
        frame = lease.layer.get_base_view().to_polars()

    # Assert
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert body["sheet_note"] == ""
    assert set(frame.columns) >= {"patient_id", "location_name", "sex", "year", "mx", "upper", "lower"}
    assert frame.height == 240
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])


@pytest.mark.slow
@pytest.mark.integration
def test_uploadDataset_xlsxThreeSheets_noteSaysWhichSheetOfHowManyWasRead(
    test_client: TestClient, isolated_data_home
) -> None:
    # Arrange: the triage's three_sheets.xlsx shape; only the first sheet is loaded
    from clinical_analytics.api import dependencies

    data = _xlsx_bytes({"Deaths": _rate_rows(60), "Population": _rate_rows(30), "Notes": [["Synthetic"]]})

    # Act
    response = test_client.post(
        "/api/datasets/upload",
        files={
            "file": ("three_sheets.xlsx", data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        },
        data={"dataset_name": "three_sheets"},
    )
    body = response.json()
    with dependencies.lease_semantic_layer(body["upload_id"]) as lease:
        frame = lease.layer.get_base_view().to_polars()

    # Assert
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert body["sheet_note"] == (
        "Read 1 of 3 sheets: “Deaths”. Not read: “Population” and “Notes”. "
        "To use another sheet, save it as its own file and upload it."
    )
    assert frame.height == 60
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])


@pytest.mark.slow
@pytest.mark.integration
def test_uploadDataset_xlsxFootnoteBelowBlankRow_readyAndNoteCountsIt(
    test_client: TestClient, isolated_data_home
) -> None:
    # Arrange: the real workbook that failed with "empty line at row ...": a blank row, then a source note
    from clinical_analytics.api import dependencies

    data = _xlsx_bytes({"Rates": [*_rate_rows(120), [], ["Source: synthetic rates generated for tests."]]})

    # Act
    response = test_client.post(
        "/api/datasets/upload",
        files={"file": ("footnote.xlsx", data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"dataset_name": "footnote"},
    )
    body = response.json()
    with dependencies.lease_semantic_layer(body["upload_id"]) as lease:
        frame = lease.layer.get_base_view().to_polars()

    # Assert
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert body["sheet_note"] == (
        "1 row below the data, after a blank row, was read as a note and left out of the data."
    )
    assert frame.height == 120
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])
