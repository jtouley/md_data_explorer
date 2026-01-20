"""
Tests to verify LLM mocking performance optimizations.

These tests verify that:
1. mock_llm_calls fixture patches is_available() to avoid HTTP requests
2. SentenceTransformer model is cached across tests
3. Unit tests run fast (<1s) when properly mocked
"""

import time

import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine


class TestLLMMockPerformance:
    """Test suite for LLM mocking performance optimizations."""

    def test_mock_llm_calls_patches_is_available(self, make_semantic_layer, mock_llm_calls):
        """
        Test that mock_llm_calls fixture patches OllamaClient.is_available().

        This prevents real HTTP requests that cause 30s timeouts when Ollama isn't running.
        """
        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": ["P1", "P2"], "outcome": [0, 1]},
        )
        engine = NLQueryEngine(semantic_layer=semantic)

        # Act: Get Ollama client (should be mocked)
        client = engine._get_ollama_client()

        # Assert: is_available() should return True immediately (mocked, no HTTP request)
        start = time.perf_counter()
        is_available = client.is_available()
        duration = time.perf_counter() - start

        assert is_available is True, "Mock should make is_available() return True"
        assert duration < 0.1, f"is_available() should be instant (<0.1s), took {duration:.3f}s"

    @pytest.mark.slow
    def test_mock_llm_calls_prevents_http_requests(
        self,
        make_semantic_layer,
        mock_llm_calls,
        nl_query_engine_with_cached_model,
    ):
        """
        Test that mock_llm_calls prevents real HTTP requests to Ollama.

        When Ollama isn't running, is_available() would normally timeout after 30s.
        With the mock, it should return immediately.
        """
        # Arrange: Use cached model to avoid SentenceTransformer loading delay
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": ["P1", "P2"], "outcome": [0, 1]},
        )
        engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

        # Act: Try to parse a query that would trigger LLM (refinement query)
        # This should use the mock, not make real HTTP requests
        start = time.perf_counter()
        result = engine.parse_query(
            query="remove the n/a",
            conversation_history=[
                {
                    "query": "count patients",
                    "intent": "COUNT",
                    "group_by": "status",
                }
            ],
        )
        duration = time.perf_counter() - start

        # Assert: Should complete quickly (<1s) with mock + cached model
        assert duration < 1.0, f"Query parsing should be fast with mock (<1s), took {duration:.3f}s"
        assert result is not None, "Should return QueryIntent"
        assert result.intent_type == "COUNT", "Should parse as COUNT intent"

    @pytest.mark.slow
    def test_sentence_transformer_cached_across_tests(
        self,
        make_semantic_layer,
        cached_sentence_transformer,
    ):
        """
        Test that SentenceTransformer model is cached and reused.

        This verifies the session-scoped fixture works correctly.
        """
        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": ["P1", "P2"], "outcome": [0, 1]},
        )
        engine = NLQueryEngine(semantic_layer=semantic)

        # Inject cached encoder
        engine.encoder = cached_sentence_transformer

        # Act: Use semantic matching (should use cached encoder, not reload)
        start = time.perf_counter()
        _ = engine._semantic_match("describe outcome")  # Result not needed, just timing
        duration = time.perf_counter() - start

        # Assert: Should be fast since encoder is pre-loaded
        # Encoding should take <0.5s with cached model
        assert duration < 0.5, f"Semantic match should be fast with cached encoder (<0.5s), took {duration:.3f}s"
        assert engine.encoder is cached_sentence_transformer, "Should use cached encoder"

    def test_nl_query_engine_with_cached_model_fast(
        self,
        nl_query_engine_with_cached_model,
        make_semantic_layer,
        mock_llm_calls,
    ):
        """
        Test that nl_query_engine_with_cached_model fixture provides fast query parsing.

        This combines both optimizations: cached SentenceTransformer + mocked LLM.
        """
        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": ["P1", "P2"], "status": [0, 1]},
        )
        engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

        # Act: Parse query that would trigger Tier 2 (semantic) or Tier 3 (LLM)
        start = time.perf_counter()
        result = engine.parse_query(
            query="remove the n/a",
            conversation_history=[
                {
                    "query": "count patients",
                    "intent": "COUNT",
                    "group_by": "status",
                }
            ],
        )
        duration = time.perf_counter() - start

        # Assert: Should be very fast (<1s) with both optimizations
        assert duration < 1.0, f"Query parsing should be fast with cached model + mock (<1s), took {duration:.3f}s"
        assert result is not None, "Should return QueryIntent"
        assert result.intent_type == "COUNT", "Should parse as COUNT intent"
