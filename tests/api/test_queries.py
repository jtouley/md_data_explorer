"""
Tests for Query API routes with SSE streaming.

Following TDD: Red phase - tests written before implementation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from clinical_analytics.api.main import app
from fastapi import status
from fastapi.testclient import TestClient

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_query_service():
    """Create mock AsyncQueryService."""
    mock = MagicMock()
    return mock


@pytest.fixture
def test_client(mock_query_service):
    """Create test client with mocked query service."""
    from clinical_analytics.api.routes.queries import get_query_service

    app.dependency_overrides[get_query_service] = lambda: mock_query_service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_async_query_service(mock_query_service):
    """Return the mock query service for tests to configure."""
    return mock_query_service


@pytest.fixture
def mock_semantic_layer():
    """Mock SemanticLayer for query execution (not used with dependency override)."""
    return MagicMock()


# ============================================================================
# Test Classes
# ============================================================================


class TestQuerySubmitEndpoint:
    """Tests for POST /api/queries endpoint."""

    def test_queries_post_valid_returns_query_id(self, test_client, mock_async_query_service, mock_semantic_layer):
        """Valid query submission returns query_id and stream URL."""
        # Arrange
        mock_async_query_service.submit_query = AsyncMock(return_value="qry_abc123")

        # Act
        response = test_client.post(
            "/api/queries",
            json={
                "session_id": "sess_test123",
                "dataset_id": "test_dataset",
                "query_text": "What is the average age?",
            },
        )

        # Assert
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "query_id" in data
        assert data["query_id"] == "qry_abc123"
        assert data["status"] == "processing"
        assert "stream_url" in data
        assert "/api/queries/qry_abc123/stream" in data["stream_url"]

    def test_queries_post_invalid_dataset_returns_404(self, test_client, mock_async_query_service, mock_semantic_layer):
        """Invalid dataset ID returns 404 error."""
        # Arrange
        mock_async_query_service.submit_query = AsyncMock(side_effect=ValueError("Dataset 'nonexistent' not found"))

        # Act
        response = test_client.post(
            "/api/queries",
            json={
                "session_id": "sess_test123",
                "dataset_id": "nonexistent",
                "query_text": "What is the average age?",
            },
        )

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_queries_post_empty_query_returns_400(self, test_client, mock_async_query_service, mock_semantic_layer):
        """Empty query text returns 400 validation error."""
        # Act
        response = test_client.post(
            "/api/queries",
            json={
                "session_id": "sess_test123",
                "dataset_id": "test_dataset",
                "query_text": "",
            },
        )

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestQueryStatusEndpoint:
    """Tests for GET /api/queries/{query_id} endpoint."""

    def test_queries_get_pending_returns_status(self, test_client, mock_async_query_service, mock_semantic_layer):
        """Pending query returns processing status."""
        # Arrange
        from clinical_analytics.api.services.query_service import AsyncQueryResult

        mock_async_query_service.get_result = AsyncMock(
            return_value=AsyncQueryResult(
                query_id="qry_abc123",
                status="processing",
            )
        )

        # Act
        response = test_client.get("/api/queries/qry_abc123")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query_id"] == "qry_abc123"
        assert data["status"] == "processing"

    def test_queries_get_completed_returns_result(self, test_client, mock_async_query_service, mock_semantic_layer):
        """Completed query returns full result."""
        # Arrange
        from clinical_analytics.api.services.query_service import AsyncQueryResult

        mock_async_query_service.get_result = AsyncMock(
            return_value=AsyncQueryResult(
                query_id="qry_abc123",
                status="completed",
                intent_type="DESCRIBE",
                result={"mean_age": 45.5, "std_age": 12.3},
                confidence=0.95,
            )
        )

        # Act
        response = test_client.get("/api/queries/qry_abc123")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query_id"] == "qry_abc123"
        assert data["status"] == "completed"
        assert data["intent"] == "DESCRIBE"
        assert data["result_data"] == {"mean_age": 45.5, "std_age": 12.3}
        assert data["confidence"] == 0.95

    def test_queries_get_missing_returns_404(self, test_client, mock_async_query_service, mock_semantic_layer):
        """Missing query ID returns 404."""
        # Arrange
        mock_async_query_service.get_result = AsyncMock(return_value=None)

        # Act
        response = test_client.get("/api/queries/qry_nonexistent")

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestQueryStreamEndpoint:
    """Tests for GET /api/queries/{query_id}/stream SSE endpoint."""

    def test_queries_stream_emits_events(self, test_client, mock_async_query_service, mock_semantic_layer):
        """SSE stream emits progress events."""
        # Arrange
        from datetime import UTC, datetime

        from clinical_analytics.api.models.schemas import SSEEvent
        from clinical_analytics.api.services.query_service import AsyncQueryResult

        # Mock get_result to return a valid result (query exists)
        mock_async_query_service.get_result = AsyncMock(
            return_value=AsyncQueryResult(
                query_id="qry_abc123",
                status="processing",
            )
        )

        async def mock_stream_events(query_id):
            yield SSEEvent(
                event="query_started",
                data={"query_id": query_id},
                timestamp=datetime.now(UTC),
            )
            yield SSEEvent(
                event="query_progress",
                data={"stage": "parsing"},
                timestamp=datetime.now(UTC),
            )
            yield SSEEvent(
                event="query_completed",
                data={"query_id": query_id, "intent_type": "DESCRIBE"},
                timestamp=datetime.now(UTC),
            )

        mock_async_query_service.stream_events = mock_stream_events

        # Act - Use iter_lines for SSE
        with test_client.stream("GET", "/api/queries/qry_abc123/stream") as response:
            lines = list(response.iter_lines())

        # Assert
        assert response.status_code == status.HTTP_200_OK
        # SSE events should be in the response
        content = "\n".join(lines)
        assert "query_started" in content
        assert "query_completed" in content

    def test_queries_stream_missing_returns_404(self, test_client, mock_async_query_service, mock_semantic_layer):
        """SSE stream for missing query returns 404."""
        # Arrange - query doesn't exist
        mock_async_query_service.get_result = AsyncMock(return_value=None)

        # Act
        response = test_client.get("/api/queries/qry_nonexistent/stream")

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
