"""
Tests for Enrichment API routes (ADR011 metadata enrichment).

Following TDD: Red phase - tests written before implementation.
"""

import json
from datetime import UTC, datetime
from unittest.mock import create_autospec

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from clinical_analytics.api.main import app
from clinical_analytics.core.overlay_store import OverlayStore, PatchNotFoundError, PatchStateError
from clinical_analytics.ui.components.enrichment_integration import EnrichmentService

# Exception text a failing store might carry; it must never reach the HTTP response.
_INTERNAL_ERROR_TEXT = "/private/overlays/upload_test/v1/pending.json"


@pytest.fixture
def mock_enrichment_service():
    """EnrichmentService mock whose methods enforce the real signatures."""
    return create_autospec(EnrichmentService, instance=True)


@pytest.fixture
def mock_overlay_store():
    """OverlayStore mock whose methods enforce the real signatures."""
    return create_autospec(OverlayStore, instance=True)


@pytest.fixture
def test_client(mock_enrichment_service, mock_overlay_store):
    """Create test client with mocked services."""
    from clinical_analytics.api.routes.enrichments import (
        get_dataset_version,
        get_enrichment_service,
        get_overlay_store,
    )

    app.dependency_overrides[get_enrichment_service] = lambda: mock_enrichment_service
    app.dependency_overrides[get_overlay_store] = lambda: mock_overlay_store
    app.dependency_overrides[get_dataset_version] = lambda: "v1"
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def real_store_client(tmp_path):
    """Test client wired to a real OverlayStore in tmp_path (real EnrichmentService on top)."""
    from clinical_analytics.api.routes.enrichments import get_dataset_version, get_overlay_store

    store = OverlayStore(base_dir=tmp_path)
    app.dependency_overrides[get_overlay_store] = lambda: store
    app.dependency_overrides[get_dataset_version] = lambda: "v1"
    yield TestClient(app), store
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


class TestEnrichmentsRevert:
    """Tests for POST /api/datasets/{dataset_id}/enrichments/{patch_id}/revert."""

    def test_enrichments_post_revert_calls_service(self, test_client, mock_enrichment_service):
        """Revert endpoint delegates to EnrichmentService.revert_accepted_patch."""
        response = test_client.post(
            "/api/datasets/upload_test/enrichments/patch_001/revert",
            json={},
        )

        assert response.status_code == status.HTTP_200_OK
        mock_enrichment_service.revert_accepted_patch.assert_called_once()

    def test_enrichments_post_revert_with_custom_actor(self, test_client, mock_enrichment_service):
        """Revert endpoint passes reverted_by."""
        response = test_client.post(
            "/api/datasets/upload_test/enrichments/patch_001/revert",
            json={"reverted_by": "auditor_1"},
        )

        assert response.status_code == status.HTTP_200_OK
        call_args = mock_enrichment_service.revert_accepted_patch.call_args
        assert call_args.kwargs.get("reverted_by") == "auditor_1"


class TestEnrichmentsStoreFailures:
    """A failing overlay store is a 4xx/5xx with a plain message, never 200 with success=False."""

    @pytest.mark.parametrize(
        ("action", "service_method"),
        [("accept", "accept_suggestion"), ("reject", "reject_suggestion"), ("revert", "revert_accepted_patch")],
    )
    @pytest.mark.parametrize(
        "error",
        [PermissionError(_INTERNAL_ERROR_TEXT), json.JSONDecodeError(_INTERNAL_ERROR_TEXT, "{", 0)],
        ids=["unwritable", "corrupt"],
    )
    def test_enrichments_action_storeFails_returns500WithPlainDetail(
        self, test_client, mock_enrichment_service, action, service_method, error
    ):
        """Disk and corrupt-file errors surface as 500; the exception text is not echoed."""
        # Arrange
        getattr(mock_enrichment_service, service_method).side_effect = error

        # Act
        response = test_client.post(f"/api/datasets/upload_test/enrichments/patch_001/{action}", json={})

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json() == {"detail": "Could not update the enrichment overlay"}
        assert _INTERNAL_ERROR_TEXT not in response.text

    @pytest.mark.parametrize(
        ("action", "service_method", "conflict_detail"),
        [
            ("accept", "accept_suggestion", "Only a pending patch can be accepted"),
            ("reject", "reject_suggestion", "Only a pending patch can be rejected"),
            ("revert", "revert_accepted_patch", "Only an accepted patch can be reverted"),
        ],
        ids=["accept", "reject", "revert"],
    )
    @pytest.mark.parametrize(
        ("error_cls", "expected_status"),
        [(PatchNotFoundError, status.HTTP_404_NOT_FOUND), (PatchStateError, status.HTTP_409_CONFLICT)],
        ids=["missing", "wrong_state"],
    )
    def test_enrichments_action_patchNotActionable_returnsClientError(
        self,
        test_client,
        mock_enrichment_service,
        action,
        service_method,
        conflict_detail,
        error_cls,
        expected_status,
    ):
        """A missing patch is 404; a patch in the wrong state for the action is 409."""
        # Arrange
        getattr(mock_enrichment_service, service_method).side_effect = error_cls(_INTERNAL_ERROR_TEXT)
        expected_detail = "Patch not found" if error_cls is PatchNotFoundError else conflict_detail

        # Act
        response = test_client.post(f"/api/datasets/upload_test/enrichments/patch_001/{action}", json={})

        # Assert
        assert response.status_code == expected_status
        assert response.json() == {"detail": expected_detail}
        assert _INTERNAL_ERROR_TEXT not in response.text


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


class TestEnrichmentsDecisionsOnRealStore:
    """End to end on a real OverlayStore: a second decision on one patch is a 409."""

    @pytest.mark.parametrize("second_action", ["accept", "reject"])
    def test_enrichments_secondDecision_returns409AndKeepsOneLogRow(self, real_store_client, second_action):
        """The first accept is recorded; any later accept or reject of that patch is refused."""
        # Arrange
        from clinical_analytics.core.metadata_patch import MetadataPatch, PatchOperation, PatchStatus

        client, store = real_store_client
        store.save_pending(
            "upload_twice",
            "v1",
            [
                MetadataPatch(
                    patch_id="patch_twice",
                    operation=PatchOperation.SET_LABEL,
                    column="age",
                    value="Patient Age",
                    status=PatchStatus.PENDING,
                    created_at=datetime.now(UTC),
                    provenance="llm",
                )
            ],
        )
        first = client.post("/api/datasets/upload_twice/enrichments/patch_twice/accept", json={})

        # Act
        second = client.post(f"/api/datasets/upload_twice/enrichments/patch_twice/{second_action}", json={})

        # Assert
        assert first.status_code == status.HTTP_200_OK
        assert second.status_code == status.HTTP_409_CONFLICT
        assert [p.status for p in store.load_patches("upload_twice", "v1")] == [PatchStatus.ACCEPTED]
