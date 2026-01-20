"""
Tests for AnalysisExecutor - Orchestration for analysis execution pipeline.

Tests verify cache check, execution, enrichment, storage, and history update.
"""

from unittest.mock import MagicMock, patch

from clinical_analytics.core.analysis_executor import AnalysisExecutor
from clinical_analytics.core.analysis_result import AnalysisResult
from clinical_analytics.core.result_cache import ResultCache
from clinical_analytics.core.state_store import InMemoryStateStore


class TestAnalysisExecutor:
    """Test suite for AnalysisExecutor."""

    def test_executor_cached_result_returns_immediately(self):
        """Test that executor returns cached result without re-execution."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)

        # Pre-populate cache
        cached = AnalysisResult(
            type="descriptive",
            payload={"mean": 45.5},
            run_key="run123",
        )
        executor.cache_result(cached, "run123", "test query", "v1")

        # Act
        result = executor.get_cached("run123", "v1")

        # Assert
        assert result is not None
        assert result.type == "descriptive"
        assert result.payload == {"mean": 45.5}

    def test_executor_cache_miss_returns_none(self):
        """Test that executor returns None on cache miss."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)

        # Act
        result = executor.get_cached("nonexistent", "v1")

        # Assert
        assert result is None

    def test_executor_new_result_stores_in_cache(self):
        """Test that executor stores new results in cache."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)
        analysis_result = AnalysisResult(
            type="comparison",
            payload={"p_value": 0.03},
            run_key="run456",
        )

        # Act
        executor.cache_result(analysis_result, "run456", "compare groups", "v2")
        retrieved = executor.get_cached("run456", "v2")

        # Assert
        assert retrieved is not None
        assert retrieved.type == "comparison"
        assert retrieved.payload == {"p_value": 0.03}

    def test_executor_enrich_error_adds_friendly_message(self):
        """Test that executor enriches error results with friendly messages."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)
        error_result = AnalysisResult(
            type="descriptive",
            payload={"error": "ZeroDivisionError: division by zero"},
        )

        # Act
        with patch("clinical_analytics.core.analysis_executor.translate_error_with_llm") as mock_translate:
            mock_translate.return_value = "Unable to calculate due to missing data."
            enriched = executor.enrich_with_error_translation(error_result)

        # Assert
        assert enriched.friendly_error_message == "Unable to calculate due to missing data."
        mock_translate.assert_called_once()

    def test_executor_enrich_success_adds_interpretation(self):
        """Test that executor enriches successful results with LLM interpretation."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)
        success_result = AnalysisResult(
            type="descriptive",
            payload={"mean": 45.5, "std": 12.3},
        )

        # Act
        with patch("clinical_analytics.core.analysis_executor.interpret_result_with_llm") as mock_interpret:
            mock_interpret.return_value = "The average age is 45.5 years."
            enriched = executor.enrich_with_interpretation(success_result, "What is the average age?")

        # Assert
        assert enriched.llm_interpretation == "The average age is 45.5 years."
        mock_interpret.assert_called_once()

    def test_executor_updates_conversation_history(self):
        """Test that executor updates conversation manager with result."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)
        mock_conversation = MagicMock()
        store.set("conversation_manager", mock_conversation)

        result = AnalysisResult(
            type="descriptive",
            payload={"mean": 45.5},
            run_key="run789",
        )

        # Act
        executor.update_conversation_history(result, "What is average?", "run789")

        # Assert
        mock_conversation.add_message.assert_called_once()
        call_args = mock_conversation.add_message.call_args
        assert call_args.kwargs["role"] == "assistant"
        assert call_args.kwargs["run_key"] == "run789"
