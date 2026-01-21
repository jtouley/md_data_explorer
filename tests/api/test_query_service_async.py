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


class TestAsyncQueryServiceSerialization:
    """Tests for result serialization (DataFrame to JSON-serializable dict)."""

    def test_serialize_result_none_returns_none(self, async_query_service):
        """Verify None result returns None."""
        result = async_query_service._serialize_result(None)
        assert result is None

    def test_serialize_result_dict_returns_dict(self, async_query_service):
        """Verify simple dict is returned as-is."""
        input_dict = {"mean": 45.5, "count": 100}
        result = async_query_service._serialize_result(input_dict)
        assert result == {"mean": 45.5, "count": 100}

    def test_serialize_result_polars_dataframe_converts_to_dict(self, async_query_service):
        """Verify Polars DataFrame is converted to dict with table structure."""
        import polars as pl

        df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        result = async_query_service._serialize_result(df)

        assert "table" in result
        assert "row_count" in result
        assert result["row_count"] == 2
        assert result["table"]["columns"] == ["name", "age"]
        assert len(result["table"]["rows"]) == 2
        assert result["table"]["rows"][0] == {"name": "Alice", "age": 30}

    def test_serialize_result_dict_with_dataframe_value(self, async_query_service):
        """Verify dict containing DataFrame value is serialized correctly."""
        import polars as pl

        df = pl.DataFrame({"value": [1, 2, 3]})
        input_dict = {"summary": "test", "data": df}
        result = async_query_service._serialize_result(input_dict)

        assert result["summary"] == "test"
        assert "table" in result["data"]
        assert result["data"]["row_count"] == 3

    def test_get_result_preview_polars_dataframe_limits_rows(self, async_query_service):
        """Verify result preview limits DataFrame to 10 rows."""
        import polars as pl

        df = pl.DataFrame({"id": list(range(100))})
        result = async_query_service._get_result_preview(df)

        assert result["total_rows"] == 100
        assert len(result["rows"]) == 10

    def test_get_result_preview_none_returns_none(self, async_query_service):
        """Verify None input returns None."""
        result = async_query_service._get_result_preview(None)
        assert result is None

    def test_get_result_preview_dict_returns_first_3_keys(self, async_query_service):
        """Verify dict preview returns first 3 keys."""
        input_dict = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        result = async_query_service._get_result_preview(input_dict)

        assert len(result) == 3
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "d" not in result
