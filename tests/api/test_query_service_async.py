"""
Tests for AsyncQueryService - async wrapper around core QueryService.

Following TDD: Red phase - tests written before implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from clinical_analytics.api.services.query_service import AsyncQueryService

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_semantic_layer():
    """Mock SemanticLayer for unit tests."""
    mock = MagicMock()
    mock.get_column_alias_index.return_value = {"age": "age", "outcome": "outcome"}
    return mock


@pytest.fixture
def mock_core_query_service():
    """Mock core QueryService for unit tests."""
    with patch("clinical_analytics.api.services.query_service.CoreQueryService") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def async_query_service(mock_semantic_layer, mock_core_query_service):
    """Create AsyncQueryService with mocked dependencies."""
    return AsyncQueryService(mock_semantic_layer)


# ============================================================================
# Test Classes
# ============================================================================


class TestAsyncQueryServiceDelegation:
    """Tests for AsyncQueryService delegating to core QueryService."""

    @pytest.mark.asyncio
    async def test_async_query_service_delegates_to_core(self, async_query_service, mock_core_query_service):
        """Verify async service delegates query execution to core service."""
        # Arrange
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"test": "data"},
            confidence=0.95,
            run_key="test_run_key",
            context=MagicMock(),
        )

        # Act
        result = await async_query_service.execute_query(
            query="What is the average age?",
            dataset_id="test_dataset",
        )

        # Assert
        mock_core_query_service.ask.assert_called_once()
        call_args = mock_core_query_service.ask.call_args
        assert call_args.kwargs["question"] == "What is the average age?"
        assert result is not None


class TestAsyncQueryServiceQueryId:
    """Tests for query ID generation."""

    @pytest.mark.asyncio
    async def test_async_query_service_returns_query_id(self, async_query_service, mock_core_query_service):
        """Verify submit_query returns a unique query ID."""
        # Arrange
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"test": "data"},
            confidence=0.9,
            run_key="run_key",
            context=MagicMock(),
        )

        # Act
        query_id = await async_query_service.submit_query(
            query="What is the average age?",
            dataset_id="test_dataset",
        )

        # Assert
        assert query_id is not None
        assert isinstance(query_id, str)
        assert query_id.startswith("qry_")


class TestAsyncQueryServiceStreaming:
    """Tests for SSE streaming progress events."""

    @pytest.mark.asyncio
    async def test_async_query_service_streams_progress_events(self, async_query_service, mock_core_query_service):
        """Verify stream_events emits SSE events for query progress."""
        # Arrange
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"test": "data"},
            confidence=0.9,
            run_key="run_key",
            context=MagicMock(),
        )

        query_id = await async_query_service.submit_query(
            query="What is the average age?",
            dataset_id="test_dataset",
        )

        # Act
        events = []
        async for event in async_query_service.stream_events(query_id):
            events.append(event)
            if event.event == "query_completed":
                break

        # Assert
        assert len(events) >= 2  # At least started and completed
        event_types = [e.event for e in events]
        assert "query_started" in event_types
        assert "query_completed" in event_types


class TestAsyncQueryServiceValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_async_query_service_invalid_dataset_raises_error(self, async_query_service, mock_core_query_service):
        """Verify invalid dataset ID raises appropriate error."""
        # Arrange
        mock_core_query_service.ask.side_effect = ValueError("Dataset not found")

        # Act & Assert
        with pytest.raises(ValueError, match="Dataset not found"):
            await async_query_service.execute_query(
                query="What is the average age?",
                dataset_id="nonexistent_dataset",
            )

    @pytest.mark.asyncio
    async def test_async_query_service_empty_query_raises_error(self, async_query_service, mock_core_query_service):
        """Verify empty query raises validation error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await async_query_service.execute_query(
                query="",
                dataset_id="test_dataset",
            )

    @pytest.mark.asyncio
    async def test_async_query_service_whitespace_query_raises_error(
        self, async_query_service, mock_core_query_service
    ):
        """Verify whitespace-only query raises validation error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await async_query_service.execute_query(
                query="   ",
                dataset_id="test_dataset",
            )
