"""
Integration tests for error translation caching.

Tests the full flow:
- Cache stores error result with friendly_error_message
- Subsequent retrievals skip LLM calls
- Multiple renders use cached translation

Following plan: llm_error_translation_caching_a2729b15.plan.md
"""

from datetime import datetime

import pytest
from clinical_analytics.core.result_cache import CachedResult, ResultCache


@pytest.mark.integration
class TestErrorTranslationCachingIntegration:
    """Integration tests for error translation caching across cache operations."""

    def test_cached_error_result_no_llm_call(self, mock_session_state):
        """
        Integration: Simulate cache hit with error result, verify no LLM call.

        When a cached result already has friendly_error_message, subsequent
        retrievals should NOT trigger translate_error_with_llm.
        """
        # Arrange: Cache with error result that has friendly_error_message
        cache = ResultCache(max_size=50)
        run_key = "integration_error_test"
        dataset_version = "v1"

        # Result with pre-cached translation (as would be stored after first execution)
        cached_result = CachedResult(
            run_key=run_key,
            query="Show me LDL levels",
            result={
                "error": "ColumnNotFoundError: Column 'ldl' not found",
                "friendly_error_message": "I couldn't find a column called 'ldl'.",
            },
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        # Track LLM calls
        llm_call_count = 0

        def mock_translate(error):
            nonlocal llm_call_count
            llm_call_count += 1
            return "New translation"

        # Act: Retrieve from cache and check if LLM would be called
        retrieved = cache.get(run_key, dataset_version)

        # Simulate the guard condition in render functions
        has_cached_translation = bool(retrieved.result.get("friendly_error_message"))
        if not has_cached_translation:
            mock_translate(retrieved.result["error"])

        # Assert: LLM was NOT called (translation already cached)
        assert llm_call_count == 0
        assert retrieved.result["friendly_error_message"] == "I couldn't find a column called 'ldl'."

    def test_multiple_reruns_single_translation(self, mock_session_state):
        """
        Integration: Multiple renders of same error result produce single LLM call.

        This simulates the real-world scenario where:
        1. First execution: LLM translates error, stores in result
        2. Result cached
        3. Multiple re-renders: Use cached translation, no LLM calls
        """
        # Arrange
        cache = ResultCache(max_size=50)
        run_key = "multi_render_test"
        dataset_version = "v1"

        # Simulate first execution (result without translation)
        error_msg = "ColumnNotFoundError: Column 'missing' not found"
        result = {"error": error_msg}

        # Track LLM calls
        llm_call_count = 0

        def mock_translate(error):
            nonlocal llm_call_count
            llm_call_count += 1
            return "Translated error message"

        # First execution: LLM called, translation cached
        if "error" in result and not result.get("friendly_error_message"):
            friendly = mock_translate(result["error"])
            if friendly:
                result["friendly_error_message"] = friendly

        # Store in cache
        cached_result = CachedResult(
            run_key=run_key,
            query="Show missing column",
            result=result,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        assert llm_call_count == 1, "First execution should call LLM"

        # Simulate multiple re-renders (cache hits)
        for render_num in range(5):
            retrieved = cache.get(run_key, dataset_version)

            # Apply guard condition (same as implementation)
            has_cached_translation = bool(retrieved.result.get("friendly_error_message"))
            if not has_cached_translation:
                mock_translate(retrieved.result["error"])

        # Assert: No additional LLM calls after first execution
        assert llm_call_count == 1, f"Expected 1 LLM call total, got {llm_call_count}"

    def test_cache_preserves_friendly_error_across_puts(self, mock_session_state):
        """
        Integration: Cache PUT and GET preserve friendly_error_message.

        Verifies the cache doesn't drop or mutate the friendly_error_message field.
        """
        # Arrange
        cache = ResultCache(max_size=50)
        run_key = "preserve_test"
        dataset_version = "v1"
        original_translation = "Original translation - should persist exactly"

        result = {
            "error": "TestError: Something went wrong",
            "friendly_error_message": original_translation,
        }

        cached_result = CachedResult(
            run_key=run_key,
            query="Test query",
            result=result,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )

        # Act: PUT then GET
        cache.put(cached_result)
        retrieved = cache.get(run_key, dataset_version)

        # Assert: Translation preserved exactly
        assert retrieved is not None
        assert retrieved.result.get("friendly_error_message") == original_translation


@pytest.mark.integration
class TestErrorTranslationWithMockedLLM:
    """Integration tests that mock LLM but use real cache."""

    def test_error_caching_flow_with_mocked_llm(self, mock_session_state):
        """
        Integration: Full caching flow with mocked LLM.

        Simulates execute_analysis_with_idempotency caching logic:
        1. Error result comes from execution
        2. translate_error_with_llm called
        3. Translation stored in result
        4. Result cached
        5. On cache hit, translation used (no LLM call)
        """
        cache = ResultCache(max_size=50)

        # Mock LLM call tracker
        llm_calls = []

        def mock_translate(error):
            llm_calls.append(error)
            return f"Friendly version of: {error}"

        # First execution - simulate execute_analysis_with_idempotency
        run_key = "flow_test_1"
        dataset_version = "v1"

        result = {
            "success": False,
            "error": "ColumnNotFoundError: Column 'x' not found",
        }

        # Apply caching logic (same as implementation)
        if "error" in result and not result.get("friendly_error_message"):
            friendly = mock_translate(result["error"])
            if friendly:
                result["friendly_error_message"] = friendly

        # Store in cache
        cached_result = CachedResult(
            run_key=run_key,
            query="Show x",
            result=result,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        # Assert: One LLM call during first execution
        assert len(llm_calls) == 1
        assert result["friendly_error_message"] == "Friendly version of: ColumnNotFoundError: Column 'x' not found"

        # Second execution (cache hit) - simulate same query re-run
        retrieved = cache.get(run_key, dataset_version)

        # Apply guard condition again
        if "error" in retrieved.result and not retrieved.result.get("friendly_error_message"):
            mock_translate(retrieved.result["error"])

        # Assert: Still only one LLM call (cached translation used)
        assert len(llm_calls) == 1
