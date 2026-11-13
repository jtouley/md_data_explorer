# SPDX-License-Identifier: MIT
"""Upload identity and status through the API (aggregate-data triage #3, #7, #8).

Synthetic long-format population estimates only (``make_population_estimates``): no real data.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.usefixtures("isolated_data_home")]

UPLOAD = "/api/datasets/upload"


def _csv(df: pl.DataFrame) -> bytes:
    return df.write_csv().encode()


@pytest.fixture
def upload(test_client: TestClient) -> Iterator[Callable[..., Any]]:
    """POST a frame as CSV; drops every cached semantic layer it created afterwards."""
    from clinical_analytics.api import dependencies

    created: list[str] = []

    def _upload(df: pl.DataFrame, name: str) -> Any:
        response = test_client.post(
            UPLOAD, files={"file": (f"{name}.csv", _csv(df), "text/csv")}, data={"dataset_name": name}
        )
        if response.status_code == 200:
            created.append(response.json()["upload_id"])
        return response

    yield _upload
    for upload_id in created:
        dependencies.invalidate_semantic_layer_cache(upload_id)


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="unique_float_mx"),
        pytest.param({"rounded": True}, id="rounded_mx_composite_with_repeats"),
        pytest.param({"unique_upper": True}, id="unique_float_upper"),
    ],
)
def test_uploadDataset_populationEstimates_measurementsKeptRowNumberAdded(
    test_client: TestClient,
    upload: Callable[..., Any],
    make_population_estimates: Callable[..., pl.DataFrame],
    options: dict[str, Any],
) -> None:
    # Arrange
    df = make_population_estimates(**options)

    # Act
    response = upload(df, "estimates")
    body = response.json()
    detail = test_client.get(f"/api/datasets/{body['upload_id']}").json()

    # Assert: on main mx (or upper) was renamed to patient_id and the error check failed
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert body["failed_checks"] == []
    columns = detail["tables"][0]["columns"]
    assert {"mx", "upper", "lower"} <= set(columns)
    assert "patient_id" not in columns
    assert "row_number" in columns
    assert detail["tables"][0]["row_count"] == df.height


def test_uploadDataset_uniqueTextId_copiedAndSourceKept(test_client: TestClient, upload: Callable[..., Any]) -> None:
    # Arrange
    df = pl.DataFrame({"record_code": [f"R{i:03d}" for i in range(200)], "sbp": [110.5 + i for i in range(200)]})

    # Act
    body = upload(df, "records").json()
    assert "upload_id" in body, body
    columns = test_client.get(f"/api/datasets/{body['upload_id']}").json()["tables"][0]["columns"]

    # Assert
    assert {"record_code", "patient_id", "sbp"} <= set(columns)


def test_getDataset_populationEstimates_reportsStoredColumnTypes(
    test_client: TestClient, upload: Callable[..., Any], make_population_estimates: Callable[..., pl.DataFrame]
) -> None:
    # Arrange
    body = upload(make_population_estimates(), "typed").json()

    # Act
    schema = test_client.get(f"/api/datasets/{body['upload_id']}").json()["schema"]

    # Assert: on main every column was "unknown"
    assert schema == {
        "row_number": "integer",
        "measure": "string",
        "location_name": "string",
        "FIPS": "integer",
        "cause_name": "string",
        "sex": "string",
        "age": "string",
        "year": "integer",
        "metric": "string",
        "mx": "number",
        "upper": "number",
        "lower": "number",
    }


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("count by sex", id="count"),
        pytest.param("average mx by cause_name", id="mean"),
    ],
)
def test_question_keylessUpload_countsRowsNeverPatients(
    upload: Callable[..., Any], make_population_estimates: Callable[..., pl.DataFrame], question: str
) -> None:
    # Arrange
    from clinical_analytics.api import dependencies
    from clinical_analytics.core.query_service import QueryService

    upload_id = upload(make_population_estimates(unique_upper=True), "rates").json()["upload_id"]

    # Act
    with dependencies.lease_semantic_layer(upload_id) as lease:
        service = QueryService(lease.layer)
        result = service.ask(question, dataset_id=upload_id, dataset_version="v1")
        if result.requires_confirmation and result.plan_hash:
            result = service.ask(
                question, dataset_id=upload_id, dataset_version="v1", confirmed=True, confirm_plan_hash=result.plan_hash
            )

    # Assert: on main the answer said "n=... patients"
    assert result.analysis is not None, result.failure_reason
    assert result.analysis.get("unit") == "rows"
    assert "rows" in (result.claim or "")
    assert not re.search(r"\bpatients?\b", result.claim or "")


@pytest.mark.parametrize(
    ("ids", "failed", "step"),
    [
        pytest.param(
            [f"P{i % 150:03d}" for i in range(200)],
            ["patient_id_no_duplicates"],
            "confirm that rows are visits",
            id="repeated_ids",
        ),
        pytest.param(
            [None if i % 10 == 0 else f"P{i:03d}" for i in range(200)],
            ["patient_id_not_null"],
            "Fill in the missing IDs",
            id="missing_ids",
        ),
    ],
)
def test_uploadDataset_errorCheckFails_needsReviewNamesCheckAndNextStep(
    upload: Callable[..., Any], ids: list[str | None], failed: list[str], step: str
) -> None:
    # Arrange
    df = pl.DataFrame({"patient_id": ids, "sbp": [110.0 + i % 40 for i in range(200)]})

    # Act
    response = upload(df, "visits")
    body = response.json()

    # Assert: on main this was "ready" with quality.failed=1
    assert response.status_code == 200, body
    assert body["status"] == "needs_review"
    assert body["failed_checks"] == failed
    assert body["message"].startswith("Uploaded, but a data check failed: patient_id: ")
    assert step in body["message"]


def test_uploadDataset_cleanPatients_ready(upload: Callable[..., Any]) -> None:
    # Arrange
    df = pl.DataFrame({"patient_id": [f"P{i:03d}" for i in range(200)], "sbp": [110.0 + i % 40 for i in range(200)]})

    # Act
    body = upload(df, "clean").json()

    # Assert
    assert (body["status"], body["failed_checks"]) == ("ready", [])


def test_uploadDataset_nameTaken_409WithoutOverwriteHint(
    upload: Callable[..., Any], make_population_estimates: Callable[..., pl.DataFrame]
) -> None:
    # Arrange
    df = make_population_estimates(locations=2, years=5)
    first = upload(df, "small")

    # Act
    second = upload(df, "small")

    # Assert: on main 500 "... or enable overwrite to update" (the route has no overwrite)
    assert first.status_code == 200
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail.startswith("A dataset named 'small' already exists.")
    assert "delete the existing dataset" in detail
    assert "overwrite" not in detail.lower()


def test_uploadDataset_intervalBroken_warningNamesColumns(
    upload: Callable[..., Any], make_population_estimates: Callable[..., pl.DataFrame]
) -> None:
    # Arrange: upper below mx on a third of rows, mx empty on a fifth
    df = make_population_estimates(inverted_every=3, null_every=5)
    expected = df.filter(pl.all_horizontal(pl.col("lower", "mx", "upper").is_not_null())).filter(
        (pl.col("lower") > pl.col("mx")) | (pl.col("mx") > pl.col("upper"))
    )

    # Act
    body = upload(df, "intervals").json()
    checks = {c["id"]: c for c in body["quality"]["checks"]}

    # Assert: on main 0 checks failed; a warning never blocks ("ready")
    assert body["status"] == "ready"
    assert checks["mx_interval"]["severity"] == "warning"
    assert checks["mx_interval"]["status"] == "warning"
    assert checks["mx_interval"]["count"] == expected.height
    assert checks["mx_interval"]["message"] == f"mx: {expected.height} rows where lower ≤ mx ≤ upper does not hold"
    assert checks["mx_missing"]["status"] == "warning"
    assert checks["mx_missing"]["message"] == "mx: 20.0% missing"


def test_uploadDataset_zipExtension_refused400(test_client: TestClient) -> None:
    # Act
    response = test_client.post(UPLOAD, files={"file": ("small.zip", b"PK\x03\x04", "application/zip")})

    # Assert: the docs no longer list ZIP
    assert response.status_code == 400
    assert ".zip" in response.json()["detail"]


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        pytest.param("broken.xlsx", b"PK\x03\x04" + b"\x00" * 2048, id="not_a_workbook"),
        pytest.param("noheader.xlsx", None, id="empty_sheet"),
    ],
)
def test_uploadDataset_unreadableFile_400WithPlainMessage(
    test_client: TestClient, filename: str, content: bytes | None
) -> None:
    # Arrange: an empty sheet is a real workbook with nothing in it
    if content is None:
        import io

        from openpyxl import Workbook

        buffer = io.BytesIO()
        workbook = Workbook()
        workbook.active.title = "blank"
        workbook.save(buffer)
        content = buffer.getvalue()

    # Act
    response = test_client.post(UPLOAD, files={"file": (filename, content, "application/octet-stream")})

    # Assert: on main a StagingError was a 500 "Error saving upload: ..."
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail.startswith("The Excel ")
    assert "Error saving upload" not in detail


def test_question_modelStepRaises_refusedWithVisibleReason(
    upload: Callable[..., Any],
    make_population_estimates: Callable[..., pl.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: the language-model tier raises (stubbed directly: no model is needed)
    from clinical_analytics.api import dependencies
    from clinical_analytics.core.nl_query_config import MODEL_STEP_FAILED_REASON
    from clinical_analytics.core.nl_query_engine import NLQueryEngine
    from clinical_analytics.core.query_service import QueryService

    def broken_client(self: NLQueryEngine) -> Any:
        raise RuntimeError("model step broke")

    def llm_only(self: NLQueryEngine, query: str, **_: Any) -> Any:
        return self._llm_parse(query)

    monkeypatch.setattr(NLQueryEngine, "_get_ollama_client", broken_client)
    monkeypatch.setattr(NLQueryEngine, "parse_query", llm_only)
    upload_id = upload(make_population_estimates(), "model").json()["upload_id"]

    # Act
    with dependencies.lease_semantic_layer(upload_id) as lease:
        result = QueryService(lease.layer).ask("which places look unusual", dataset_id=upload_id, dataset_version="v1")

    # Assert: on main a 0.3 DESCRIBE stub went on as a normal parse
    assert result.failure_reason == MODEL_STEP_FAILED_REASON
    assert result.result is None and result.analysis is None
    assert result.requires_confirmation is False
    assert [i["severity"] for i in result.issues] == ["error"]


def test_question_compositeRowKey_countsRowsNeverPatients(
    test_client: TestClient, upload: Callable[..., Any], make_population_estimates: Callable[..., pl.DataFrame]
) -> None:
    # Arrange: location x cause x sex x year is unique, so patient_id is hashed from it
    from clinical_analytics.api import dependencies
    from clinical_analytics.core.query_service import QueryService

    upload_id = upload(make_population_estimates(causes=3), "cells").json()["upload_id"]
    metadata = test_client.get(f"/api/datasets/{upload_id}").json()["metadata"]

    # Act
    with dependencies.lease_semantic_layer(upload_id) as lease:
        result = QueryService(lease.layer).ask("count by sex", dataset_id=upload_id, dataset_version="v1")

    # Assert: a synthesized ID keys rows, not patients
    assert "patient_id" in metadata["columns"] and "mx" in metadata["columns"]
    assert result.analysis is not None, result.failure_reason
    assert result.analysis.get("unit") == "rows"
    assert not re.search(r"\bpatients?\b", result.claim or "")
