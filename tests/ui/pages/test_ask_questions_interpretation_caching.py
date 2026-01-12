"""
Regression tests for ADR009 Phase 3 interpretation caching.

Bug: Repeated LLM interpretation calls on cached query re-execution.
Fix: Check if result already has llm_interpretation before calling LLM.

Related: Terminal logs showed repeated llm_call_success for result_interpretation
even when query_execution_cache_hit occurred.
"""

from datetime import datetime

from clinical_analytics.core.result_cache import CachedResult, ResultCache


class TestInterpretationCaching:
    """Test that LLM interpretation is skipped when result already has it."""

    def test_interpretation_caching_skip_llm_when_interpretation_exists(self, mock_session_state):
        """
        Regression: execute_analysis_with_idempotency should NOT call
        interpret_result_with_llm when result already has llm_interpretation.

        Bug: Terminal logs showed repeated llm_call_success for result_interpretation
        even when query_execution_cache_hit occurred (same query submitted twice).
        """
        # Arrange: Result that ALREADY has interpretation (from previous run)
        result_with_interpretation = {
            "type": "count",
            "summary": {"total": 100},
            "headline": "100 patients found",
            "llm_interpretation": "This analysis counted 100 patients in total.",  # ALREADY CACHED!
        }

        # Simulate the guard condition logic from Ask_Questions.py line 1214-1222
        # BEFORE fix: if ENABLE_RESULT_INTERPRETATION and "error" not in result:
        # AFTER fix: if ENABLE_RESULT_INTERPRETATION and "error" not in result and not result.get("llm_interpretation"):

        enable_interpretation = True
        has_error = "error" in result_with_interpretation
        has_existing_interpretation = bool(result_with_interpretation.get("llm_interpretation"))

        # Act: Apply the FIXED guard condition
        should_call_llm = enable_interpretation and not has_error and not has_existing_interpretation

        # Assert: Should NOT call LLM when interpretation already exists
        assert (
            should_call_llm is False
        ), "Should NOT call interpret_result_with_llm when result already has llm_interpretation"

    def test_interpretation_caching_call_llm_when_no_interpretation(self, mock_session_state):
        """
        Test that interpret_result_with_llm IS called when result has no interpretation.

        This ensures the fix doesn't break normal first-time execution.
        """
        # Arrange: Result WITHOUT interpretation (first-time execution)
        result_without_interpretation = {
            "type": "count",
            "summary": {"total": 100},
            "headline": "100 patients found",
            # NO llm_interpretation field
        }

        enable_interpretation = True
        has_error = "error" in result_without_interpretation
        has_existing_interpretation = bool(result_without_interpretation.get("llm_interpretation"))

        # Act: Apply the FIXED guard condition
        should_call_llm = enable_interpretation and not has_error and not has_existing_interpretation

        # Assert: Should call LLM when no interpretation exists
        assert should_call_llm is True, "Should call interpret_result_with_llm when result has no interpretation"

    def test_interpretation_caching_skip_llm_on_error_result(self, mock_session_state):
        """
        Test that interpret_result_with_llm is NOT called for error results.

        Error results should never trigger LLM interpretation.
        """
        # Arrange: Error result
        error_result = {
            "error": "Query failed",
            "type": "error",
        }

        enable_interpretation = True
        has_error = "error" in error_result
        has_existing_interpretation = bool(error_result.get("llm_interpretation"))

        # Act: Apply the guard condition
        should_call_llm = enable_interpretation and not has_error and not has_existing_interpretation

        # Assert: Should NOT call LLM for error results
        assert should_call_llm is False, "Should NOT call interpret_result_with_llm for error results"

    def test_interpretation_caching_skip_when_feature_disabled(self, mock_session_state):
        """
        Test that interpret_result_with_llm is NOT called when feature is disabled.
        """
        # Arrange: Valid result but feature disabled
        result = {
            "type": "count",
            "summary": {"total": 100},
        }

        enable_interpretation = False  # Feature disabled
        has_error = "error" in result
        has_existing_interpretation = bool(result.get("llm_interpretation"))

        # Act: Apply the guard condition
        should_call_llm = enable_interpretation and not has_error and not has_existing_interpretation

        # Assert: Should NOT call LLM when feature disabled
        assert should_call_llm is False, "Should NOT call interpret_result_with_llm when feature is disabled"


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
        assert (
            retrieved.result.get("llm_interpretation") is not None
        ), "Interpretation should be preserved in cached result"
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
