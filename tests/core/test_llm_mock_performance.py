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


class TestMockEncoderPerformance:
    """Test suite for mock_encoder fixture that eliminates encoding overhead."""

    def test_mock_encoder_returns_instant_embeddings(
        self,
        make_semantic_layer,
        mock_encoder,
        mock_llm_calls,
    ):
        """
        Test that mock_encoder fixture eliminates SentenceTransformer encoding time.

        With mock_encoder, encode() returns fake embeddings instantly (<10ms)
        instead of real encoding (~100-500ms per call).
        """
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": ["P1", "P2"], "outcome": [0, 1]},
        )
        engine = NLQueryEngine(semantic_layer=semantic)

        # Act: Parse query - should use mock encoder
        start = time.perf_counter()
        result = engine.parse_query("describe outcome")
        duration = time.perf_counter() - start

        # Assert: Should be very fast (<0.5s) with mocked encoder
        assert duration < 0.5, f"Query parsing should be instant with mock_encoder (<0.5s), took {duration:.3f}s"
        assert result is not None, "Should return QueryIntent"

    def test_mock_encoder_encode_called(
        self,
        make_semantic_layer,
        mock_encoder,
        mock_llm_calls,
    ):
        """
        Test that mock_encoder.encode() is actually called during semantic matching.

        Verifies the mock is properly injected and used when Tier 1 (pattern) fails.
        """
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": ["P1", "P2"], "outcome": [0, 1]},
        )
        engine = NLQueryEngine(semantic_layer=semantic)

        # Reset call count
        mock_encoder.encode.reset_mock()

        # Act: Parse query that will FAIL pattern matching (Tier 1) and trigger semantic (Tier 2)
        # Use an unusual phrasing that won't match regex patterns
        engine._semantic_match("tell me about the typical values of outcome")

        # Assert: encode() was called during semantic matching
        assert mock_encoder.encode.called, "Mock encoder should be called during _semantic_match"

    def test_mock_encoder_multiple_queries_fast(
        self,
        make_semantic_layer,
        mock_encoder,
        mock_llm_calls,
    ):
        """
        Test that multiple parse_query calls are all fast with mock_encoder.

        This simulates a test file with many parse_query calls.
        """
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        # Arrange
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": ["P1", "P2"], "outcome": [0, 1], "age": [30, 40]},
        )
        engine = NLQueryEngine(semantic_layer=semantic)

        queries = [
            "describe outcome",
            "compare outcome by age",
            "what predicts outcome",
            "count patients",
            "show correlations",
        ]

        # Act: Parse multiple queries
        start = time.perf_counter()
        for query in queries:
            engine.parse_query(query)
        total_duration = time.perf_counter() - start

        # Assert: All 5 queries should complete in <1s total
        avg_duration = total_duration / len(queries)
        assert total_duration < 1.0, f"5 queries should complete in <1s, took {total_duration:.3f}s"
        assert avg_duration < 0.2, f"Average query time should be <0.2s, was {avg_duration:.3f}s"
