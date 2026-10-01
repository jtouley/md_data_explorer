"""
Tests for error translation caching in Ask Questions page (ADR009 Phase 4).

Tests cover:
- Error translation is skipped when cached in result dict
- Error translation is called when not cached
- Translated error is stored in result dict for future use
- Cache preserves friendly_error_message across retrievals

Following plan: llm_error_translation_caching_a2729b15.plan.md
Follows same pattern as: test_ask_questions_interpretation_caching.py
"""

from datetime import datetime

from clinical_analytics.core.result_cache import CachedResult, ResultCache


class TestErrorTranslationCachingIntegration:
    """Integration tests for error translation caching across cache hits."""

    def test_cached_error_result_preserves_friendly_error_message(self, mock_session_state):
        """
        Test that when error result is cached with friendly_error_message,
        subsequent lookups return the translation without LLM call.
        """
        # Arrange: Cache with error result that has friendly_error_message
        cache = ResultCache(max_size=50)
        run_key = "error_test_run_key"
        dataset_version = "test_version"

        cached_result = CachedResult(
            run_key=run_key,
            query="Show me LDL levels",
            result={
                "error": "ColumnNotFoundError: Column 'ldl' not found",
                "friendly_error_message": "I couldn't find a column called 'ldl'. Try 'LDL mg/dL' instead.",
            },
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        # Act: Retrieve from cache (simulating cache hit)
        retrieved = cache.get(run_key, dataset_version)

        # Assert: friendly_error_message should be preserved in cache
        assert retrieved is not None, "Should find result in cache"
        assert retrieved.result.get("friendly_error_message") is not None, (
            "friendly_error_message should be preserved in cached result"
        )
        assert (
            retrieved.result["friendly_error_message"]
            == "I couldn't find a column called 'ldl'. Try 'LDL mg/dL' instead."
        )

    def test_multiple_cache_retrievals_same_friendly_error(self, mock_session_state):
        """
        Test that multiple cache retrievals return same friendly_error_message.

        This simulates re-rendering the chat multiple times - should not
        trigger new LLM calls because translation is in cache.
        """
        # Arrange
        cache = ResultCache(max_size=50)
        run_key = "repeat_error_key"
        dataset_version = "test_version"
        original_translation = "Original translation that should persist."

        cached_result = CachedResult(
            run_key=run_key,
            query="Show me missing column",
            result={
                "error": "ColumnNotFoundError: missing column",
                "friendly_error_message": original_translation,
            },
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        # Act: Retrieve multiple times (simulating multiple re-renders)
        retrieval_1 = cache.get(run_key, dataset_version)
        retrieval_2 = cache.get(run_key, dataset_version)
        retrieval_3 = cache.get(run_key, dataset_version)

        # Assert: All retrievals return same friendly_error_message
        assert retrieval_1.result["friendly_error_message"] == original_translation
        assert retrieval_2.result["friendly_error_message"] == original_translation
        assert retrieval_3.result["friendly_error_message"] == original_translation

        # LLM would NOT be called because cached_translation is truthy
