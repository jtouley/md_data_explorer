# SPDX-License-Identifier: MIT
""" "Is each row one patient?" through the API: asked at upload, answered for that version.

Synthetic frames only: no real data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

from clinical_analytics.contracts.declarations import FileDeclarationStore, confirmation_token
from clinical_analytics.core.data_home import uploads_dir

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.usefixtures("isolated_data_home")]

N = 200


def _answer_url(upload_id: str) -> str:
    return f"/api/datasets/{upload_id}/one-row-per-patient"


def _one_row_per_record(n: int = N) -> pl.DataFrame:
    """Shaped like COVID-MS: a record code unique on every row, not named as a patient ID."""
    return pl.DataFrame(
        {
            "record_code": [f"R{i:04d}" for i in range(n)],
            "sex": ["female" if i % 2 else "male" for i in range(n)],
            "ward": [("A", "B", "C", "D")[i % 4] for i in range(n)],
        }
    )


def _visit_rows(n: int = N) -> pl.DataFrame:
    """Visit-level: a unique visit code, and an ID for another role (a ward) that repeats; nothing
    names the patient (a person_id would: ADR017 reads it as the patient key)."""
    return pl.DataFrame(
        {
            "visit_code": [f"V{i:04d}" for i in range(n)],
            "ward_no": [f"W{i % (n // 2):04d}" for i in range(n)],
            "sex": ["female" if (i % (n // 2)) % 2 else "male" for i in range(n)],
        }
    )


def _uploaded(upload: Callable[..., Any], df: pl.DataFrame, name: str) -> dict[str, Any]:
    """The upload response body (``upload`` is the API conftest factory); the upload must work."""
    response = upload(df, name)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _version(upload_id: str) -> str:
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    return str((UserDatasetStorage().get_upload_metadata(upload_id) or {})["dataset_version"])


def _count_unit(upload_id: str) -> str | None:
    """The unit of "how many patients?" asked as the API asks it (declarations on disk)."""
    from clinical_analytics.api import dependencies
    from clinical_analytics.core.query_service import QueryService

    data_dir = uploads_dir().parent
    with dependencies.lease_semantic_layer(upload_id) as lease:
        service = QueryService(lease.layer, declarations=FileDeclarationStore(data_dir), data_dir=data_dir)
        result = service.ask("how many patients?", dataset_id=upload_id, dataset_version=lease.layer.loaded_version)
    if result.analysis is None and (result.failure_reason or "").startswith("This answer waits"):
        return "held"  # "Is each line one patient?" is still open (contracts.holds)
    assert result.analysis is not None, result.failure_reason
    return result.analysis.get("unit")


def test_upload_uniqueUnnamedId_questionPendingThenYesCountsPatients(
    test_client: TestClient, upload: Callable[..., Any]
) -> None:
    # Arrange
    body = _uploaded(upload, _one_row_per_record(), "records")
    upload_id = body["upload_id"]
    question = body["one_row_per_patient"]
    before = _count_unit(upload_id)

    # Act
    response = test_client.post(
        _answer_url(upload_id), json={"confirmation": question["confirmation"], "one_row_per_patient": True}
    )
    after = _count_unit(upload_id)

    # Assert: counts wait until answered, then count patients
    assert question["status"] == "pending"
    assert "Counts wait for your answer" in body["unit_note"]
    assert before == "held"
    assert response.status_code == 200, response.text
    assert response.json() == {"dataset_id": upload_id, "counts": "patients"}
    assert after == "patients"


def test_upload_answeredNo_rowsKeptNothingStored(test_client: TestClient, upload: Callable[..., Any]) -> None:
    # Arrange
    body = _uploaded(upload, _one_row_per_record(), "records_no")
    upload_id = body["upload_id"]

    # Act
    response = test_client.post(
        _answer_url(upload_id),
        json={"confirmation": body["one_row_per_patient"]["confirmation"], "one_row_per_patient": False},
    )

    # Assert
    assert response.status_code == 200, response.text
    assert response.json()["counts"] == "rows"
    assert _count_unit(upload_id) == "rows"
    assert FileDeclarationStore(uploads_dir().parent).load(upload_id).grains == ()


@pytest.mark.parametrize(
    "make_token",
    [
        pytest.param(lambda version: confirmation_token("v0", None, N), id="earlier_version"),
        pytest.param(lambda version: confirmation_token(version, None, N - 1), id="other_row_count"),
        pytest.param(lambda version: confirmation_token(version, "record_code", N), id="other_id_column"),
        pytest.param(lambda version: "0" * 32, id="forged"),
    ],
)
def test_answer_staleOrForgedToken_409NothingStored(
    test_client: TestClient, upload: Callable[..., Any], make_token: Callable[[str], str]
) -> None:
    # Arrange
    upload_id = _uploaded(upload, _one_row_per_record(), "records_stale")["upload_id"]
    version = _version(upload_id)

    # Act
    response = test_client.post(
        _answer_url(upload_id), json={"confirmation": make_token(version), "one_row_per_patient": True}
    )

    # Assert
    assert response.status_code == 409
    assert "changed after you were asked" in response.json()["detail"]
    assert FileDeclarationStore(uploads_dir().parent).load(upload_id).grains == ()
    assert _count_unit(upload_id) == "held"


def test_upload_visitLevelIdsRepeat_neverAskedAnswerRefused(
    test_client: TestClient, upload: Callable[..., Any]
) -> None:
    # Arrange
    body = _uploaded(upload, _visit_rows(), "visits")
    upload_id = body["upload_id"]
    version = _version(upload_id)

    # Act: even a client that computes the token is refused
    response = test_client.post(
        _answer_url(upload_id),
        json={"confirmation": confirmation_token(version, None, N), "one_row_per_patient": True},
    )

    # Assert
    assert body["one_row_per_patient"] is None
    assert response.status_code == 409
    assert "not asked" in response.json()["detail"]
    assert FileDeclarationStore(uploads_dir().parent).load(upload_id).grains == ()


def test_upload_patientIdColumn_neverAsked(upload: Callable[..., Any]) -> None:
    # Arrange
    df = _one_row_per_record().rename({"record_code": "patient_id"})

    # Act
    body = _uploaded(upload, df, "patients")

    # Assert
    assert body["one_row_per_patient"] is None
    assert body["unit_note"] == ""


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"confirmation": "z" * 32, "one_row_per_patient": True}, id="not_hex"),
        pytest.param({"confirmation": "0" * 31, "one_row_per_patient": True}, id="too_short"),
        pytest.param({"confirmation": "0" * 32, "one_row_per_patient": "yes"}, id="answer_not_bool"),
        pytest.param({"confirmation": "0" * 32}, id="no_answer"),
        pytest.param({"confirmation": "0" * 32, "one_row_per_patient": True, "rows": 5}, id="extra_field"),
    ],
)
def test_answer_invalidBody_422(test_client: TestClient, upload: Callable[..., Any], payload: dict[str, Any]) -> None:
    # Arrange
    upload_id = _uploaded(upload, _one_row_per_record(), "records_bad")["upload_id"]

    # Act
    response = test_client.post(_answer_url(upload_id), json=payload)

    # Assert
    assert response.status_code == 422
    assert FileDeclarationStore(uploads_dir().parent).load(upload_id).grains == ()


def test_answer_unknownDataset_404(test_client: TestClient) -> None:
    # Arrange
    payload = {"confirmation": "0" * 32, "one_row_per_patient": True}

    # Act
    response = test_client.post(_answer_url("user_upload_missing"), json=payload)

    # Assert
    assert response.status_code == 404


def test_answerYes_governedSqlReadsSameConfirmation(
    test_client: TestClient, upload: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    from clinical_analytics.core import nl_query_config

    monkeypatch.setattr(nl_query_config, "ENABLE_SQL_QUERIES", True)
    body = _uploaded(upload, _one_row_per_record(), "records_sql")
    upload_id = body["upload_id"]
    sql = {"dataset_id": upload_id, "sql": "SELECT ward, count(*) AS n FROM unified GROUP BY ward"}
    refused = test_client.post("/api/queries/sql", json=sql).json()

    # Act
    test_client.post(
        _answer_url(upload_id),
        json={"confirmation": body["one_row_per_patient"]["confirmation"], "one_row_per_patient": True},
    )
    answered = test_client.post("/api/queries/sql", json=sql).json()

    # Assert: the upload's question and SQL's refusal ask about the same thing, with one token
    assert refused["needs_confirmation"] == "one_row_per_patient"
    assert refused["confirmation"] == body["one_row_per_patient"]["confirmation"]
    assert answered["status"] == "answered", answered
    assert answered["needs_confirmation"] is None


def test_sqlConfirmationOnVisitRows_questionsStillCountRows(
    test_client: TestClient, upload: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: visit rows the upload never asks about; the user confirms in governed SQL anyway
    from clinical_analytics.core import nl_query_config

    monkeypatch.setattr(nl_query_config, "ENABLE_SQL_QUERIES", True)
    upload_id = _uploaded(upload, _visit_rows(), "visits_sql")["upload_id"]
    sql = {"dataset_id": upload_id, "sql": "SELECT sex, count(*) AS n FROM unified GROUP BY sex"}
    refused = test_client.post("/api/queries/sql", json=sql).json()
    test_client.post("/api/queries/sql", json={**sql, "confirm_one_row_per_patient": refused["confirmation"]})

    # Act
    unit = _count_unit(upload_id)

    # Assert: the SQL confirmation is stored, but never turns visit rows into patients
    assert FileDeclarationStore(uploads_dir().parent).load(upload_id).grains != ()
    assert unit == "rows"
