# SPDX-License-Identifier: MIT
"""A dataset id in a JSON body is checked before it names any file (422, nothing touched)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

BAD_IDS = [
    "../../../../outside_datahome/evil",
    "../evil",
    "a/b",
    "a\\b",
    "/etc/passwd",
    "..",
    ".hidden",
    "",
    "id with space",
    "x" * 200,
    "ok\n",
]


@pytest.mark.parametrize("dataset_id", BAD_IDS)
def test_post_sessions_traversalDatasetId_returns422(test_client, dataset_id):
    # Arrange
    body = {"dataset_id": dataset_id}

    # Act
    response = test_client.post("/api/sessions", json=body)

    # Assert
    assert response.status_code == 422


@pytest.mark.parametrize("dataset_id", BAD_IDS)
def test_post_queries_traversalDatasetId_returns422BeforeLeasing(test_client, monkeypatch, dataset_id):
    # Arrange: leasing a layer would read metadata and Parquet; it must never be reached
    from clinical_analytics.api.routes import queries

    def fail_lease(_dataset_id: str) -> None:
        raise AssertionError("layer leased for an unchecked dataset id")

    monkeypatch.setattr(queries, "lease_semantic_layer", fail_lease)
    body = {"session_id": "s1", "dataset_id": dataset_id, "query_text": "how many patients"}

    # Act
    response = test_client.post("/api/queries", json=body)

    # Assert
    assert response.status_code == 422


@pytest.mark.parametrize("dataset_id", ["../../outside/evil", "a/b", ""])
def test_lease_semantic_layer_invalidId_raises422WithoutBuilding(monkeypatch, dataset_id):
    # Arrange
    from clinical_analytics.api import dependencies

    def fail_build(_dataset_id: str) -> None:
        raise AssertionError("layer built for an unchecked dataset id")

    monkeypatch.setattr(dependencies._layer_cache, "lease", fail_build)

    # Act
    with pytest.raises(HTTPException) as caught:
        dependencies.lease_semantic_layer(dataset_id)

    # Assert
    assert caught.value.status_code == 422


@pytest.mark.parametrize("dataset_id", ["user_upload_" + "a" * 32, "covid_ms", "Dataset-1"])
def test_session_create_plainDatasetId_isAccepted(dataset_id):
    # Arrange
    from clinical_analytics.api.models.schemas import QueryRequest, SessionCreate

    # Act
    session = SessionCreate(dataset_id=dataset_id)
    query = QueryRequest(session_id="s1", dataset_id=dataset_id, query_text="q")

    # Assert
    assert session.dataset_id == dataset_id
    assert query.dataset_id == dataset_id
