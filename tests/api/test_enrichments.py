"""
Tests for Enrichment API routes (ADR011 metadata enrichment).

Following TDD: Red phase - tests written before implementation.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from clinical_analytics.api.main import app
from fastapi import status
from fastapi.testclient import TestClient


@pytest.fixture
def mock_enrichment_service():
    """Create mock EnrichmentService."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_overlay_store():
    """Create mock OverlayStore."""
    mock = MagicMock()
    return mock


@pytest.fixture
def test_client(mock_enrichment_service, mock_overlay_store):
    """Create test client with mocked services."""
    from clinical_analytics.api.routes.enrichments import (
        get_enrichment_service,
        get_overlay_store,
    )

    app.dependency_overrides[get_enrichment_service] = lambda: mock_enrichment_service
    app.dependency_overrides[get_overlay_store] = lambda: mock_overlay_store
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestEnrichmentsGetPending:
    """Tests for GET /api/datasets/{dataset_id}/enrichments/pending endpoint."""

    def test_enrichments_get_pending_returns_suggestions(self, test_client, mock_enrichment_service):
        """Pending suggestions are returned as a list."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        mock_patch = MetadataPatch(
            patch_id="patch_001",
            operation=PatchOperation.SET_LABEL,
            column="age",
            value="Patient Age",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
            model_id="llama3.2",
            confidence=0.95,
        )
        mock_enrichment_service.get_pending_suggestions.return_value = [mock_patch]

        response = test_client.get("/api/datasets/upload_test/enrichments/pending")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["patch_id"] == "patch_001"
        assert data["suggestions"][0]["operation"] == "SET_LABEL"
        assert data["suggestions"][0]["column"] == "age"
        assert data["suggestions"][0]["suggested_value"] == "Patient Age"

    def test_enrichments_get_pending_empty_returns_empty_list(self, test_client, mock_enrichment_service):
        """No pending suggestions returns empty list."""
        mock_enrichment_service.get_pending_suggestions.return_value = []

        response = test_client.get("/api/datasets/upload_test/enrichments/pending")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["suggestions"] == []
        assert data["total"] == 0


class TestEnrichmentsAccept:
    """Tests for POST /api/datasets/{dataset_id}/enrichments/{patch_id}/accept endpoint."""

    def test_enrichments_post_accept_applies_patch(self, test_client, mock_enrichment_service):
        """Accept endpoint applies the patch and returns success."""
        response = test_client.post(
            "/api/datasets/upload_test/enrichments/patch_001/accept",
            json={},
        )

        assert response.status_code == status.HTTP_200_OK
        mock_enrichment_service.accept_suggestion.assert_called_once()

    def test_enrichments_post_accept_with_custom_user(self, test_client, mock_enrichment_service):
        """Accept endpoint supports custom accepted_by user."""
        response = test_client.post(
            "/api/datasets/upload_test/enrichments/patch_001/accept",
            json={"accepted_by": "test_user"},
        )

        assert response.status_code == status.HTTP_200_OK
        call_args = mock_enrichment_service.accept_suggestion.call_args
        assert call_args.kwargs.get("accepted_by") == "test_user"


class TestEnrichmentsReject:
    """Tests for POST /api/datasets/{dataset_id}/enrichments/{patch_id}/reject endpoint."""

    def test_enrichments_post_reject_marks_rejected(self, test_client, mock_enrichment_service):
        """Reject endpoint marks patch as rejected."""
        response = test_client.post(
            "/api/datasets/upload_test/enrichments/patch_001/reject",
            json={},
        )

        assert response.status_code == status.HTTP_200_OK
        mock_enrichment_service.reject_suggestion.assert_called_once()

    def test_enrichments_post_reject_with_reason(self, test_client, mock_enrichment_service):
        """Reject endpoint accepts custom reason."""
        response = test_client.post(
            "/api/datasets/upload_test/enrichments/patch_001/reject",
            json={"reason": "Incorrect label for this column"},
        )

        assert response.status_code == status.HTTP_200_OK
        call_args = mock_enrichment_service.reject_suggestion.call_args
        assert call_args.kwargs.get("reason") == "Incorrect label for this column"


class TestEnrichmentsHistory:
    """Tests for GET /api/datasets/{dataset_id}/enrichments/history endpoint."""

    def test_enrichments_get_history_returns_patches(self, test_client, mock_overlay_store):
        """History endpoint returns applied patches."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        mock_patch = MetadataPatch(
            patch_id="patch_001",
            operation=PatchOperation.SET_LABEL,
            column="age",
            value="Patient Age",
            status=PatchStatus.ACCEPTED,
            created_at=datetime.now(UTC),
            provenance="llm",
            model_id="llama3.2",
            confidence=0.95,
            accepted_at=datetime.now(UTC),
            accepted_by="user",
        )
        mock_overlay_store.load_patches.return_value = [mock_patch]

        response = test_client.get("/api/datasets/upload_test/enrichments/history")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "patches" in data
        assert len(data["patches"]) == 1
        assert data["patches"][0]["patch_id"] == "patch_001"
        assert data["patches"][0]["status"] == "ACCEPTED"


class TestEnrichmentsGenerate:
    """Tests for POST /api/datasets/{dataset_id}/enrichments/generate endpoint."""

    def test_enrichments_post_generate_returns_accepted(self, test_client, mock_enrichment_service):
        """Generate endpoint returns 202 Accepted status."""
        response = test_client.post(
            "/api/datasets/upload_test/enrichments/generate",
            json={},
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "message" in data

    def test_enrichments_post_generate_with_force(self, test_client, mock_enrichment_service):
        """Generate endpoint supports force regeneration flag."""
        response = test_client.post(
            "/api/datasets/upload_test/enrichments/generate",
            json={"force_regenerate": True},
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
