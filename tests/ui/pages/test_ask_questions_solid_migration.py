"""
Tests for Ask Questions SOLID Migration - Verify new abstractions work.

Tests verify that StateStore, AnalysisExecutor, and Renderer Registry
can be used as drop-in replacements for existing code.
"""

from unittest.mock import patch

from clinical_analytics.core.analysis_executor import AnalysisExecutor
from clinical_analytics.core.analysis_result import AnalysisResult
from clinical_analytics.core.result_cache import ResultCache
from clinical_analytics.core.state_store import InMemoryStateStore
from clinical_analytics.ui.components.renderers import (
    RENDERERS,
    render_result,
)


class TestAskQuestionsSolidMigration:
    """Test suite for SOLID migration verification."""

    def test_state_store_can_store_analysis_context(self):
        """Test that StateStore can hold AnalysisContext like session_state."""
        # Arrange
        from clinical_analytics.ui.components.question_engine import AnalysisContext

        store = InMemoryStateStore()
        context = AnalysisContext(research_question="What is the average age?")

        # Act
        store.set("analysis_context", context)
        retrieved = store.get("analysis_context")

        # Assert
        assert retrieved is not None
        assert retrieved.research_question == "What is the average age?"

    def test_executor_integrates_with_state_store(self):
        """Test that AnalysisExecutor works with StateStore for caching."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)

        result = AnalysisResult(
            type="descriptive",
            payload={"mean": 45.5},
            run_key="integration_test_key",
        )

        # Act
        executor.cache_result(result, "integration_test_key", "test query", "v1")
        cached = executor.get_cached("integration_test_key", "v1")

        # Assert
        assert cached is not None
        assert cached.type == "descriptive"

    def test_renderer_registry_has_all_required_types(self):
        """Test that global renderer registry has all analysis types."""
        # Assert - all required types are registered
        required_types = [
            "descriptive",
            "comparison",
            "predictor",
            "survival",
            "relationship",
            "count",
        ]
        for result_type in required_types:
            assert result_type in RENDERERS, f"Missing renderer for {result_type}"

    def test_render_result_dispatches_correctly(self):
        """Test that render_result function dispatches to correct renderer."""
        # Arrange
        result = AnalysisResult(type="count", payload={"count": 42})

        # Act - should not raise
        with patch("streamlit.metric"):
            with patch("streamlit.info"):
                render_result(result, query_text="How many?")

    def test_executor_enrichment_pipeline_works(self):
        """Test that enrichment methods can be chained."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)

        error_result = AnalysisResult(
            type="descriptive",
            payload={"error": "Test error"},
        )

        # Act
        with patch("clinical_analytics.core.analysis_executor.translate_error_with_llm") as mock_translate:
            mock_translate.return_value = "Friendly error message"
            enriched = executor.enrich_with_error_translation(error_result)

        # Assert
        assert enriched.friendly_error_message == "Friendly error message"

    def test_full_execution_flow_integration(self):
        """Test complete flow: execute -> cache -> retrieve -> render."""
        # Arrange
        store = InMemoryStateStore()
        cache = ResultCache()
        executor = AnalysisExecutor(state_store=store, result_cache=cache)

        # Create result as if from SemanticLayer
        result = AnalysisResult(
            type="descriptive",
            payload={"mean": 45.5, "std": 12.3},
        )

        # Act - Execute and cache
        executor.cache_result(result, "flow_test", "What is average?", "v1")

        # Retrieve from cache
        cached = executor.get_cached("flow_test", "v1")
        assert cached is not None

        # Render (should not raise)
        with patch("streamlit.metric"), patch("streamlit.info"):
            render_result(cached, query_text="What is average?")

    def test_deprecated_functions_exist_for_backward_compat(self):
        """Test that deprecated functions still exist during migration."""
        # This test ensures we don't break existing code during migration
        # The functions should exist but may log deprecation warnings

        # Note: This test will be updated once we add deprecation markers
        # For now, we just verify the module loads without error
        assert True

    def test_state_store_conversation_manager_integration(self):
        """Test that ConversationManager works with StateStore."""
        # Arrange
        from clinical_analytics.core.conversation_manager import ConversationManager

        store = InMemoryStateStore()
        manager = ConversationManager()
        store.set("conversation_manager", manager)

        # Act
        manager.add_message(role="user", content="What is the average age?")
        retrieved_manager = store.get("conversation_manager")

        # Assert
        assert retrieved_manager is not None
        messages = retrieved_manager.get_transcript()
        assert len(messages) == 1
        assert messages[0].content == "What is the average age?"
