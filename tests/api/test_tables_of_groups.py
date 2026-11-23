# SPDX-License-Identifier: MIT
"""Tables of groups through the API (ADR017): a linked frequency table, the third answer to
"Is each row one patient?", date readings and a post-load estimates declaration.

Synthetic data only; the link's download is mocked (no network).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

from clinical_analytics.api.routes import dataset_url
from clinical_analytics.storage.url_fetch import FetchedFile, SourceProvenance

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.usefixtures("isolated_data_home")]

PLACES = [f"County {i:02d}" for i in range(14)]  # more than 12 places
YEARS = list(range(1995, 2018))  # more than 20 years


def _legionella_csv() -> bytes:
    """Legionella notifications by county, year and sex, with three strata repeated."""
    rows = [
        {"County": c, "Year": y, "Sex": s, "Cases": (i * 3 + y) % 25}
        for i, c in enumerate(PLACES)
        for y in YEARS
        for s in ("Female", "Male")
    ]
    rows += rows[:3]
    return pl.DataFrame(rows).write_csv().encode()


def _fetched(url: str, content: bytes, filename: str) -> FetchedFile:
    provenance = SourceProvenance(
        url=url, final_url=url, fetched_at="2026-11-20T10:00:00+00:00", size_bytes=len(content),
        sha256="0" * 64, content_type="text/csv", last_modified=None, filename=filename,
    )  # fmt: skip
    return FetchedFile(content=content, filename=filename, provenance=provenance)


def test_loadFromUrl_undeclaredFrequencyTable_recognisedWithCountColumn(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a data.cdc.gov link; its column metadata is the dictionary
    data_url = "https://data.cdc.gov/api/views/ab12-cd34/rows.csv?accessType=DOWNLOAD"
    metadata = {
        "description": "Counts of legionella cases by county, year and sex.",
        "columns": [
            {"name": "County", "fieldName": "county", "description": "County of residence"},
            {"name": "Cases", "fieldName": "cases", "description": "Number of notified cases"},
        ],
    }
    calls: list[str] = []

    def fake_fetch(url: str, *, accepted_extensions: frozenset[str]) -> FetchedFile:
        calls.append(url)
        if url.endswith(".json"):
            return _fetched(url, json.dumps(metadata).encode(), "ab12-cd34.json")
        return _fetched(url, _legionella_csv(), "legionella.csv")

    monkeypatch.setattr(dataset_url, "fetch_public_file", fake_fetch)

    # Act
    body = test_client.post("/api/datasets/from-url", json={"url": data_url}).json()
    semantics = test_client.get(f"/api/datasets/{body['upload_id']}/semantics").json()["semantics"]

    # Assert
    assert calls == [data_url, "https://data.cdc.gov/api/views/ab12-cd34.json"]
    assert body["status"] == "ready"
    assert body["one_row_per_patient"] is None
    assert (semantics["aggregate"]["kind"], semantics["aggregate"]["count_column"]) == ("frequency", "Cases")
    assert semantics["aggregate"]["repeated_strata_lines"] == 6
    assert body["facts"][0].startswith("Each line counts cases for one county, year and sex")


@pytest.mark.parametrize(
    "url",
    [
        "https://data.cdc.gov.evil.test/api/views/ab12-cd34/rows.csv",
        "https://data.cdc.gov/api/views/AB12-CD34x/rows.csv",
    ],
)
def test_socrataDictionary_otherHostOrBadId_notFetched(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("fetched")

    monkeypatch.setattr(dataset_url, "fetch_public_file", never)

    # Act / Assert
    assert dataset_url.socrata_dictionary(url) is None


def test_oneRowPerPatient_thirdAnswer_countsCasesFromTheCountColumn(
    upload: Callable[..., Any], test_client: TestClient
) -> None:
    # Arrange: lines told apart by a record code, with a count and a measure (asked)
    frame = pl.DataFrame(
        {
            "record_code": [f"R{i:03d}" for i in range(200)],
            "cases": [i % 5 for i in range(200)],
            "ahi": [5.5 + i for i in range(200)],
        }
    )
    body = upload(frame, "tally").json()
    question = body["one_row_per_patient"]
    assert question["count_columns"] == ["cases"]

    # Act
    answer = test_client.post(
        f"/api/datasets/{body['upload_id']}/one-row-per-patient",
        json={"confirmation": question["confirmation"], "one_row_per_patient": False, "counts_cases_in": "cases"},
    )
    semantics = test_client.get(f"/api/datasets/{body['upload_id']}/semantics").json()["semantics"]

    # Assert
    assert answer.json()["counts"] == "cases"
    assert semantics["case_count_column"] == "cases"
    assert semantics["aggregate"]["source"] == "declared"


def test_dateReadings_orderAnswered_columnReadAsDates(upload: Callable[..., Any], test_client: TestClient) -> None:
    # Arrange: newborn screening dates that read either way
    from clinical_analytics.api import dependencies

    frame = pl.DataFrame(
        {
            "infant_id": [f"B{i:03d}" for i in range(200)],
            "screen_date": ["03/04/21", "05/06/21", "07/08/21", "09/10/21"] * 50,
        }
    )
    body = upload(frame, "screen").json()

    # Act
    answer = test_client.post(
        f"/api/datasets/{body['upload_id']}/date-readings", json={"column": "screen_date", "answer": "month_first"}
    )
    with dependencies.lease_semantic_layer(body["upload_id"]) as lease:
        view = lease.layer.get_base_view().to_polars()
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])

    # Assert
    assert answer.status_code == 200, answer.text
    assert str(view.get_column("screen_date")[0]) == "2021-03-04"


def test_populationEstimates_declaredAfterLoading_semanticsSayEstimates(
    upload: Callable[..., Any], test_client: TestClient, make_population_estimates: Callable[..., pl.DataFrame]
) -> None:
    # Arrange
    body = upload(make_population_estimates(locations=3, years=5), "rates_later").json()

    # Act
    response = test_client.post(
        f"/api/datasets/{body['upload_id']}/population-estimates",
        json={"declaration": "public_population_estimates_no_patient_rows"},
    )
    semantics = test_client.get(f"/api/datasets/{body['upload_id']}/semantics").json()["semantics"]

    # Assert
    assert response.status_code == 200, response.text
    assert semantics["population_estimates"] is True
    assert semantics["patient_key"] is None
