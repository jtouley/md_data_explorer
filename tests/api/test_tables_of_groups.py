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
    # Recognised and recorded, then asked: the structure alone never applies it (ADR017 principle A)
    assert body["one_row_per_patient"]["count_columns"] == ["Cases"]
    assert (semantics["aggregate"]["kind"], semantics["aggregate"]["count_column"]) == ("frequency", "Cases")
    assert semantics["aggregate"]["repeated_strata_lines"] == 6
    assert body["facts"][0].startswith("Each line seems to count cases for one county, year and sex")


@pytest.mark.parametrize(
    "url",
    [
        "https://data.cdc.gov.evil.test/api/views/ab12-cd34/rows.csv",
        "https://data.cdc.gov/api/views/AB12-CD34x/rows.csv",
        "https://data.cdc.gov:8443/api/views/ab12-cd34/rows.csv",
        "https://user@data.cdc.gov/api/views/ab12-cd34/rows.csv",
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


@pytest.fixture
def sql_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from clinical_analytics.core import nl_query_config

    monkeypatch.setattr(nl_query_config, "ENABLE_SQL_QUERIES", True)


def test_governedSql_frequencyTable_heldThenCountOfLinesNeverConfirmableAsPatients(
    upload: Callable[..., Any], test_client: TestClient, sql_on: None
) -> None:
    # Arrange: a frequency table with no person ID
    frame = pl.read_csv(_legionella_csv()).unique(subset=["County", "Year", "Sex"], keep="first", maintain_order=True)
    body = upload(frame, "legionella_sql").json()
    upload_id = body["upload_id"]
    sql = {"dataset_id": upload_id, "sql": "SELECT Sex, count(*) AS n FROM unified GROUP BY Sex"}

    # Act
    held = test_client.post("/api/queries/sql", json=sql).json()
    test_client.post(
        f"/api/datasets/{upload_id}/one-row-per-patient",
        json={
            "confirmation": body["one_row_per_patient"]["confirmation"],
            "one_row_per_patient": False,
            "counts_cases_in": "Cases",
        },
    )
    refused = test_client.post("/api/queries/sql", json=sql).json()
    forced = test_client.post("/api/queries/sql", json={**sql, "confirm_one_row_per_patient": "a" * 32}).json()
    summed = test_client.post(
        "/api/queries/sql",
        json={"dataset_id": upload_id, "sql": "SELECT Sex, sum(Cases) AS cases FROM unified GROUP BY Sex"},
    ).json()

    # Assert: waits for "Does each line count cases?"; once declared, never confirmable as patients
    assert held["status"] == "held"
    assert held["open_questions"] == ["aggregate_unsure"]
    assert refused["status"] == "refused"
    assert refused["needs_confirmation"] is None
    assert refused["confirmation"] is None
    assert "SUM(Cases)" in refused["reason"]
    assert forced["status"] == "refused"
    assert forced["needs_confirmation"] is None
    # The sum it suggests is answered, said to be cases
    assert summed["status"] == "answered", summed
    assert sum(r["cases"] for r in summed["rows"]) == frame["Cases"].sum()
    assert summed["notes"] == ["Each line counts cases for a group: sums of Cases are cases, never patients."]


def _troponin() -> pl.DataFrame:
    """Chest-pain unit troponin with an undocumented 999 the upload asks about."""
    return pl.DataFrame(
        {
            "mrn": [f"C{i:03d}" for i in range(200)],
            "trop_ng_l": [999.0 if i % 20 == 0 else 4.0 + i % 30 for i in range(200)],
        }
    )


def test_governedSql_columnWithOpenMissingCodeQuestion_heldUntilAnswered(
    upload: Callable[..., Any], test_client: TestClient, sql_on: None
) -> None:
    # Arrange
    upload_id = upload(_troponin(), "trop_sql").json()["upload_id"]
    sql = {"dataset_id": upload_id, "sql": "SELECT avg(trop_ng_l) AS mean_trop FROM unified"}
    other = {"dataset_id": upload_id, "sql": "SELECT count(*) AS n FROM unified"}

    # Act
    before = test_client.post("/api/queries/sql", json=sql).json()
    unrelated = test_client.post("/api/queries/sql", json=other).json()
    test_client.post(
        f"/api/datasets/{upload_id}/missing-codes", json={"column": "trop_ng_l", "value": "999", "not_recorded": True}
    )
    after = test_client.post("/api/queries/sql", json=sql).json()

    # Assert: held (no number) until answered; a query that does not read the column is answered
    assert before["status"] == "held"
    assert before["rows"] == []
    assert "Does 999 in trop_ng_l mean not recorded?" in before["reason"]
    assert unrelated["status"] == "answered", unrelated
    assert after["status"] == "answered", after
    assert after["rows"][0]["mean_trop"] < 999


def test_qualityReport_missingCodeAnswered_needsYourAnswerLineGone(
    upload: Callable[..., Any], test_client: TestClient
) -> None:
    # Arrange
    upload_id = upload(_troponin(), "trop_quality").json()["upload_id"]
    before = test_client.get(f"/api/datasets/{upload_id}/quality").json()

    # Act
    test_client.post(
        f"/api/datasets/{upload_id}/missing-codes", json={"column": "trop_ng_l", "value": 999, "not_recorded": False}
    )
    after = test_client.get(f"/api/datasets/{upload_id}/quality").json()

    # Assert
    asked = [c for c in before["checks"] if c["metric"] == "openQuestion"]
    assert [c["id"] for c in asked] == ["missing_code:trop_ng_l:999.0"]
    assert [c for c in after["checks"] if c["metric"] == "openQuestion"] == []
    assert after["warnings"] == before["warnings"] - 1


def _with_total(label: str) -> pl.DataFrame:
    """Legionella counts by county, year and sex, plus a ``label`` line per year and sex holding their sum."""
    groups = pl.read_csv(_legionella_csv()).unique(subset=["County", "Year", "Sex"], keep="first", maintain_order=True)
    totals = groups.group_by("Year", "Sex").agg(pl.col("Cases").sum()).with_columns(pl.lit(label).alias("County"))
    return pl.concat([groups, totals.select(groups.columns)])


@pytest.mark.parametrize(
    ("label", "confirm"),
    [
        pytest.param("All States", None, id="label-and-sums"),
        pytest.param("Statewide", "yes, it is a total", id="sum-confirmed"),
    ],
)
def test_threeReaders_frequencyTableTotals_neverAddedToTheirGroups(
    upload: Callable[..., Any], test_client: TestClient, sql_on: None, label: str, confirm: str | None
) -> None:
    # Arrange: declared as counts in Cases; "Statewide" is a total only once confirmed
    from clinical_analytics.api import dependencies
    from clinical_analytics.contracts.declarations import FileDeclarationStore
    from clinical_analytics.contracts.readings import apply_declared, declared_reads
    from clinical_analytics.contracts.semantics import layer_semantics
    from clinical_analytics.core.data_home import uploads_dir

    frame = _with_total(label)
    truth = int(frame.filter((pl.col("Year") == 2002) & (pl.col("County") != label))["Cases"].sum())
    body = upload(frame, f"legionella_{label.replace(' ', '_')}").json()
    upload_id = body["upload_id"]
    test_client.post(
        f"/api/datasets/{upload_id}/one-row-per-patient",
        json={
            "confirmation": body["one_row_per_patient"]["confirmation"],
            "one_row_per_patient": False,
            "counts_cases_in": "Cases",
        },
    )
    if confirm is not None:
        test_client.post(
            f"/api/datasets/{upload_id}/answers", json={"question_id": f"total_row:County:{label}", "answer": confirm}
        )

    # Act
    sql = test_client.post(
        "/api/queries/sql",
        json={"dataset_id": upload_id, "sql": "SELECT sum(Cases) AS cases FROM unified WHERE Year = 2002"},
    ).json()
    with dependencies.lease_semantic_layer(upload_id) as lease:
        view = lease.layer.get_base_view().to_polars()
        semantics = layer_semantics(lease.layer, declared=True)
    dependencies.invalidate_semantic_layer_cache(upload_id)
    declarations = FileDeclarationStore(uploads_dir().parent).load(upload_id)
    polars_read = apply_declared(frame, declared_reads(declarations, semantics.dataset_version, semantics))

    # Assert: governed SQL, the semantic layer and the polars reads all leave the total out
    assert sql["status"] == "answered", sql
    assert sql["rows"][0]["cases"] == truth
    assert int(view.filter(pl.col("Year") == 2002)["Cases"].sum()) == truth
    assert int(polars_read.filter(pl.col("Year") == 2002)["Cases"].sum()) == truth
