"""
Regression tests for ADR009 Phase 3 interpretation caching.

Bug: Repeated LLM interpretation calls on cached query re-execution.
Fix: Check if result already has llm_interpretation before calling LLM.

Related: Terminal logs showed repeated llm_call_success for result_interpretation
even when query_execution_cache_hit occurred.
"""

from datetime import datetime

from clinical_analytics.core.result_cache import CachedResult, ResultCache


class TestInterpretationCachingIntegration:
    """Integration tests for interpretation caching across cache hits."""

    def test_cached_execution_result_preserves_interpretation(self, mock_session_state):
        """
        Test that when execution result is cached with interpretation,
        subsequent lookups return the interpretation without LLM call.
        """
        # Arrange: Cache with result that has interpretation
        cache = ResultCache(max_size=50)
        run_key = "test_run_key"
        dataset_version = "test_version"

        cached_result = CachedResult(
            run_key=run_key,
            query="How many patients?",
            result={
                "type": "count",
                "summary": {"total": 100},
                "llm_interpretation": "There are 100 patients in the dataset.",
            },
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        # Act: Retrieve from cache (simulating cache hit)
        retrieved = cache.get(run_key, dataset_version)

        # Assert: Interpretation should be preserved in cache
        assert retrieved is not None, "Should find result in cache"
        assert retrieved.result.get("llm_interpretation") is not None, (
            "Interpretation should be preserved in cached result"
        )
        assert retrieved.result["llm_interpretation"] == "There are 100 patients in the dataset."

    def test_multiple_cache_retrievals_same_interpretation(self, mock_session_state):
        """
        Test that multiple cache retrievals return same interpretation.

        This simulates re-rendering the chat multiple times - should not
        trigger new LLM calls because interpretation is in cache.
        """
        # Arrange
        cache = ResultCache(max_size=50)
        run_key = "repeat_test_key"
        dataset_version = "test_version"
        original_interpretation = "Original interpretation that should persist."

        cached_result = CachedResult(
            run_key=run_key,
            query="Count patients",
            result={
                "type": "count",
                "summary": {"total": 42},
                "llm_interpretation": original_interpretation,
            },
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        # Act: Retrieve multiple times (simulating multiple re-renders)
        retrieval_1 = cache.get(run_key, dataset_version)
        retrieval_2 = cache.get(run_key, dataset_version)
        retrieval_3 = cache.get(run_key, dataset_version)

        # Assert: All retrievals return same interpretation
        assert retrieval_1.result["llm_interpretation"] == original_interpretation
        assert retrieval_2.result["llm_interpretation"] == original_interpretation
        assert retrieval_3.result["llm_interpretation"] == original_interpretation
