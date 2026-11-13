# SPDX-License-Identifier: MIT
"""Upload → query through the API: a dataset reported ready always answers (DQ H1)."""

from __future__ import annotations

import datetime as dt

import polars as pl
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
    layer = dependencies.get_semantic_layer(body["upload_id"])
    frame = layer.get_base_view().to_polars()

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
