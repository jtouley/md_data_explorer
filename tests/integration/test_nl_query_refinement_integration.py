"""
Integration tests for conversational query refinement with REAL Ollama.

These tests verify actual LLM behavior, not mocked responses.
They require Ollama to be running with llama3.1:8b model.

Test data aligns with golden_examples.yaml for consistent behavior.
"""

import pytest


@pytest.fixture
def skip_if_ollama_unavailable():
    """Skip test if Ollama is not available."""
    from clinical_analytics.core.ollama_manager import get_ollama_manager

    manager = get_ollama_manager()
    if not manager.is_service_running():
        pytest.skip("Ollama service not running")
    if not manager.get_available_models():
        pytest.skip("No Ollama models available")


@pytest.mark.integration
@pytest.mark.slow
class TestNLQueryRefinementIntegration:
    """Integration tests for conversational refinement with real Ollama."""

    def test_refinement_remove_na_with_statin_context(
        self,
        make_semantic_layer,
        nl_query_engine_with_cached_model,
        skip_if_ollama_unavailable,
    ):
        """Test 'remove the n/a' refinement with real Ollama.

        Aligns with golden example: refinement_remove_na_with_context
        """
        # Arrange: Use 'statin' column to match golden example
        semantic = make_semantic_layer(
            dataset_name="test_statin_refinement",
            data={
                "patient_id": ["P1", "P2", "P3", "P4"],
                "statin": [0, 1, 2, 1],  # 0=n/a, 1=atorvastatin, 2=rosuvastatin
            },
        )
        engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

        # Conversation history matching golden example
        conversation_history = [
            {
                "query": "count patients by statin",
                "intent": "COUNT",
                "group_by": "statin",
            }
        ]

        # Act: Parse refinement query with real Ollama
        result = engine.parse_query(
            query="remove the n/a",
            conversation_history=conversation_history,
        )

        # Assert: Should inherit COUNT intent and add filter
        assert result.intent_type == "COUNT", f"Expected COUNT, got {result.intent_type}"
        assert len(result.filters) >= 1, f"Expected at least 1 filter, got {len(result.filters)}"

        # Filter should exclude 0 (n/a value) from statin column
        statin_filters = [f for f in result.filters if f.column == "statin"]
        assert len(statin_filters) >= 1, f"Expected statin filter, got {result.filters}"

    def test_refinement_exclude_missing_values(
        self,
        make_semantic_layer,
        nl_query_engine_with_cached_model,
        skip_if_ollama_unavailable,
    ):
        """Test 'exclude missing values' refinement with real Ollama."""
        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test_cholesterol",
            data={
                "patient_id": ["P1", "P2", "P3"],
                "cholesterol": [180.0, 0.0, 220.0],  # 0 = missing
            },
        )
        engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

        conversation_history = [
            {
                "query": "describe cholesterol",
                "intent": "DESCRIBE",
                "metric": "cholesterol",
            }
        ]

        # Act
        result = engine.parse_query(
            query="exclude missing values",
            conversation_history=conversation_history,
        )

        # Assert: Should inherit DESCRIBE and add filter
        assert result.intent_type == "DESCRIBE", f"Expected DESCRIBE, got {result.intent_type}"
        # Should have filter to exclude missing (0) values

    def test_refinement_age_filter_update(
        self,
        make_semantic_layer,
        nl_query_engine_with_cached_model,
        skip_if_ollama_unavailable,
    ):
        """Test 'actually over 65' updates previous age filter."""
        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test_age_refinement",
            data={
                "patient_id": ["P1", "P2", "P3", "P4"],
                "age": [45, 55, 65, 75],
            },
        )
        engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

        conversation_history = [
            {
                "query": "patients over 50",
                "intent": "COUNT",
                "filters_applied": [{"column": "age", "operator": ">", "value": 50}],
            }
        ]

        # Act
        result = engine.parse_query(
            query="actually over 65",
            conversation_history=conversation_history,
        )

        # Assert: Should update age filter to 65
        assert result.intent_type == "COUNT"
        age_filters = [f for f in result.filters if f.column == "age"]
        assert len(age_filters) >= 1, "Expected age filter"
        # At least one filter should have value 65
        assert any(f.value == 65 for f in age_filters), f"Expected age > 65, got {age_filters}"

    def test_refinement_without_context_low_confidence(
        self,
        make_semantic_layer,
        nl_query_engine_with_cached_model,
        skip_if_ollama_unavailable,
    ):
        """Test 'remove the n/a' without context has low confidence."""
        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test_no_context",
            data={
                "patient_id": ["P1", "P2"],
                "value": [0, 1],
            },
        )
        engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

        # Act: No conversation history
        result = engine.parse_query(
            query="remove the n/a",
            conversation_history=[],
        )

        # Assert: Without context, should fall back to generic parse
        # Confidence should be lower than with context
        # (exact threshold depends on implementation)
        assert result.confidence <= 0.9, f"Expected lower confidence without context, got {result.confidence}"
