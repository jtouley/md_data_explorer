"""
Tests for Query API routes with SSE streaming.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from clinical_analytics.api.main import app
from clinical_analytics.api.routes.queries import DataFrameEncoder

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

    async def mock_find_query_result(query_id: str):
        result = await mock_query_service.get_result(query_id)
        if result is not None:
            return (mock_query_service, result)
        return None

    with (
        patch(
            "clinical_analytics.api.routes.queries.get_query_service_for_dataset",
            return_value=mock_query_service,
        ),
        patch(
            "clinical_analytics.api.routes.queries.find_query_result",
            side_effect=mock_find_query_result,
        ),
    ):
        yield TestClient(app)


@pytest.fixture
def mock_async_query_service(mock_query_service):
    """Return the mock query service for tests to configure."""
    return mock_query_service


# ============================================================================
# Test Classes
# ============================================================================


class TestQuerySubmitEndpoint:
    """Tests for POST /api/queries endpoint."""

    def test_queries_post_valid_returns_query_id(self, test_client, mock_async_query_service):
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
        mock_async_query_service.submit_query.assert_awaited_once()
        assert mock_async_query_service.submit_query.await_args.kwargs.get("confirmed") is False

    def test_queries_post_confirmed_true_forwards_to_service(self, test_client, mock_async_query_service):
        """confirmed=true is forwarded so the gate can release a hold."""
        mock_async_query_service.submit_query = AsyncMock(return_value="qry_confirm")

        response = test_client.post(
            "/api/queries",
            json={
                "session_id": "sess_test123",
                "dataset_id": "test_dataset",
                "query_text": "Compare LDL by sex",
                "confirmed": True,
            },
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        assert mock_async_query_service.submit_query.await_args.kwargs.get("confirmed") is True

    def test_queries_post_withContext_forwardsContextAndSession(self, test_client, mock_async_query_service):
        """QueryRequest.context and session_id reach the service instead of being dropped."""
        # Arrange
        mock_async_query_service.submit_query = AsyncMock(return_value="qry_ctx")
        context = {"previous_intent": "COUNT", "previous_group_by": "sex"}

        # Act
        response = test_client.post(
            "/api/queries",
            json={
                "session_id": "sess_ctx",
                "dataset_id": "test_dataset",
                "query_text": "now by site",
                "context": context,
            },
        )

        # Assert
        assert response.status_code == status.HTTP_202_ACCEPTED
        kwargs = mock_async_query_service.submit_query.call_args.kwargs
        assert kwargs["session_id"] == "sess_ctx"
        assert kwargs["context"] == context

    def test_queries_post_invalid_dataset_returns_404(self, test_client, mock_async_query_service):
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

    def test_queries_post_empty_query_returns_400(self, test_client, mock_async_query_service):
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

    def test_queries_get_pending_returns_status(self, test_client, mock_async_query_service):
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

    def test_queries_get_completed_returns_result(self, test_client, mock_async_query_service):
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

    def test_queries_get_missing_returns_404(self, test_client, mock_async_query_service):
        """Missing query ID returns 404."""
        # Arrange
        mock_async_query_service.get_result = AsyncMock(return_value=None)

        # Act
        response = test_client.get("/api/queries/qry_nonexistent")

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestQueryStreamEndpoint:
    """Tests for GET /api/queries/{query_id}/stream SSE endpoint."""

    def test_queries_stream_emits_events(self, test_client, mock_async_query_service):
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

    def test_queries_stream_missing_returns_404(self, test_client, mock_async_query_service):
        """SSE stream for missing query returns 404."""
        # Arrange - query doesn't exist
        mock_async_query_service.get_result = AsyncMock(return_value=None)

        # Act
        response = test_client.get("/api/queries/qry_nonexistent/stream")

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDataFrameEncoder:
    """Tests for DataFrameEncoder JSON serialization."""

    def test_dataframe_encoder_serializes_polars_dataframe(self):
        """DataFrameEncoder converts Polars DataFrame to dict."""
        # Arrange
        df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})

        # Act
        result = json.dumps({"data": df}, cls=DataFrameEncoder)
        parsed = json.loads(result)

        # Assert
        assert "data" in parsed
        assert parsed["data"]["columns"] == ["name", "age"]
        assert len(parsed["data"]["rows"]) == 2
        assert parsed["data"]["rows"][0] == {"name": "Alice", "age": 30}

    def test_dataframe_encoder_handles_nested_dataframe(self):
        """DataFrameEncoder handles DataFrame nested in dict."""
        # Arrange
        df = pl.DataFrame({"value": [1, 2, 3]})
        data = {"summary": "test", "results": df}

        # Act
        result = json.dumps(data, cls=DataFrameEncoder)
        parsed = json.loads(result)

        # Assert
        assert parsed["summary"] == "test"
        assert parsed["results"]["columns"] == ["value"]
        assert len(parsed["results"]["rows"]) == 3

    def test_dataframe_encoder_passes_through_simple_types(self):
        """DataFrameEncoder passes through simple JSON types unchanged."""
        # Arrange
        data = {"string": "test", "number": 42, "list": [1, 2, 3]}

        # Act
        result = json.dumps(data, cls=DataFrameEncoder)
        parsed = json.loads(result)

        # Assert
        assert parsed == data


class TestQueryServiceCache:
    """get_query_service_for_dataset follows the shared semantic layer cache."""

    @pytest.fixture(autouse=True)
    def _clear_query_services(self):
        from clinical_analytics.api.routes import queries

        queries._query_services.clear()
        yield
        queries._query_services.clear()

    def _service_for(self, semantic_layer: MagicMock) -> MagicMock:
        service = MagicMock()
        service.semantic_layer = semantic_layer
        return service

    @pytest.mark.parametrize(
        ("layer_changes", "expected_builds"),
        [(False, 1), (True, 2)],
        ids=["same_layer_reuses_service", "replaced_layer_rebuilds_service"],
    )
    def test_get_query_service_layerIdentity_rebuildsOnlyWhenLayerChanges(self, layer_changes, expected_builds):
        # Arrange
        from clinical_analytics.api.routes import queries

        first_layer, second_layer = MagicMock(), MagicMock()
        layers = [first_layer, second_layer if layer_changes else first_layer]

        # Act
        with (
            patch.object(queries, "get_semantic_layer", side_effect=layers),
            patch.object(queries, "AsyncQueryService", side_effect=self._service_for) as build,
        ):
            first = queries.get_query_service_for_dataset("dataset_1")
            second = queries.get_query_service_for_dataset("dataset_1")

        # Assert
        assert build.call_count == expected_builds
        assert second.semantic_layer is layers[1]
        assert (first is second) is not layer_changes
