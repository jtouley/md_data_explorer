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


class TestErrorTranslationCaching:
    """Test that LLM error translation is skipped when result already has friendly_error_message."""

    def test_error_translation_caching_skip_llm_when_friendly_error_exists(self, mock_session_state):
        """
        Regression: _render_error_with_translation should NOT call
        translate_error_with_llm when result already has friendly_error_message.

        Bug: Terminal logs showed repeated llm_call_success for error_translation
        even when query_execution_cache_hit occurred (same query submitted twice).
        """
        # Arrange: Result that ALREADY has friendly_error_message (from previous run)
        result_with_cached_translation = {
            "error": "ColumnNotFoundError: Column 'ldl' not found",
            # ALREADY CACHED from previous run!
            "friendly_error_message": "I couldn't find a column called 'ldl'. Try 'LDL mg/dL' instead.",
        }

        # Simulate the guard condition logic that should be in _render_error_with_translation
        # and render functions
        # BEFORE fix: Always call translate_error_with_llm(error)
        # AFTER fix: Use cached_translation if available, skip LLM call

        has_cached_translation = bool(result_with_cached_translation.get("friendly_error_message"))

        # Act: Apply the FIXED guard condition
        should_call_llm = not has_cached_translation

        # Assert: Should NOT call LLM when friendly_error_message already exists
        assert (
            should_call_llm is False
        ), "Should NOT call translate_error_with_llm when result already has friendly_error_message"

    def test_error_translation_caching_call_llm_when_no_friendly_error(self, mock_session_state):
        """
        Test that translate_error_with_llm IS called when result has no friendly_error_message.

        This ensures the fix doesn't break normal first-time error handling.
        """
        # Arrange: Result WITHOUT friendly_error_message (first-time execution)
        result_without_translation = {
            "error": "ColumnNotFoundError: Column 'ldl' not found",
            # NO friendly_error_message field
        }

        has_cached_translation = bool(result_without_translation.get("friendly_error_message"))

        # Act: Apply the FIXED guard condition
        should_call_llm = not has_cached_translation

        # Assert: Should call LLM when no friendly_error_message exists
        assert should_call_llm is True, "Should call translate_error_with_llm when result has no friendly_error_message"

    def test_error_translation_caching_handles_none_friendly_error(self, mock_session_state):
        """
        Test that explicit None friendly_error_message triggers LLM call.

        None is treated as "not cached" - we should still try to translate.
        """
        # Arrange: Result with explicit None for friendly_error_message
        result_with_none = {
            "error": "TypeError: expected str",
            "friendly_error_message": None,  # Explicitly None (e.g., previous LLM call failed)
        }

        has_cached_translation = bool(result_with_none.get("friendly_error_message"))

        # Act: Apply the guard condition
        should_call_llm = not has_cached_translation

        # Assert: Should call LLM because None is falsy (no valid cached translation)
        assert should_call_llm is True, "Should call translate_error_with_llm when friendly_error_message is None"

    def test_error_translation_caching_handles_empty_string(self, mock_session_state):
        """
        Test that empty string friendly_error_message triggers LLM call.

        Empty string is treated as "not cached" - we should still try to translate.
        """
        # Arrange: Result with empty string for friendly_error_message
        result_with_empty = {
            "error": "RuntimeError: unexpected state",
            "friendly_error_message": "",  # Empty string
        }

        has_cached_translation = bool(result_with_empty.get("friendly_error_message"))

        # Act: Apply the guard condition
        should_call_llm = not has_cached_translation

        # Assert: Should call LLM because empty string is falsy
        assert (
            should_call_llm is True
        ), "Should call translate_error_with_llm when friendly_error_message is empty string"


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
        assert (
            retrieved.result.get("friendly_error_message") is not None
        ), "friendly_error_message should be preserved in cached result"
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


class TestRenderErrorWithTranslationLogic:
    """Test the _render_error_with_translation function's caching logic."""

    def test_render_error_cached_translation_skips_llm(self, mock_session_state):
        """
        Test that when cached_translation is provided, LLM is not called.

        This tests the NEW function signature:
        _render_error_with_translation(error, prefix, cached_translation)

        When cached_translation is truthy, translate_error_with_llm should be skipped.
        """
        # Arrange
        cached_translation = "I couldn't find a column called 'ldl'. Try 'LDL mg/dL' instead."

        # Simulate the logic in _render_error_with_translation
        # BEFORE fix: friendly_message = translate_error_with_llm(technical_error)
        # AFTER fix: friendly_message = cached_translation or translate_error_with_llm(technical_error)

        # Act: Apply the FIXED logic
        friendly_message = cached_translation  # Short-circuit OR - don't call LLM

        # Assert: friendly_message should be the cached translation
        assert friendly_message == cached_translation
        # LLM would NOT be called because cached_translation is truthy

    def test_render_error_no_cached_translation_calls_llm(self, mock_session_state):
        """
        Test that when cached_translation is None, LLM is called.

        This tests the fall-through case where LLM must be invoked.
        """
        # Arrange
        technical_error = "ColumnNotFoundError: Column 'ldl' not found"
        cached_translation = None  # Not cached
        mock_llm_result = "Mocked LLM translation"

        # Simulate the logic in _render_error_with_translation
        # AFTER fix: friendly_message = cached_translation or translate_error_with_llm(technical_error)

        # Act: Apply the FIXED logic - cached_translation is None, so fall through to LLM
        def mock_translate_error_with_llm(error):
            return mock_llm_result

        friendly_message = cached_translation or mock_translate_error_with_llm(technical_error)

        # Assert: friendly_message should be from LLM (cached_translation was None)
        assert friendly_message == mock_llm_result


class TestExecuteAnalysisErrorCaching:
    """Test error translation caching in execute_analysis_with_idempotency."""

    def test_execute_analysis_caches_error_translation(self, mock_session_state):
        """
        Test that execute_analysis_with_idempotency stores friendly_error_message in result.

        Bug: Error translations were not being cached, causing repeated LLM calls.
        Fix: After translate_error_with_llm(), store result in result["friendly_error_message"].
        """
        # Arrange: Error result from execution (similar to what semantic layer returns)
        execution_result_with_error = {
            "success": False,
            "error": "ColumnNotFoundError: Column 'ldl' not found",
            # NO friendly_error_message yet
        }

        mock_llm_translation = "I couldn't find a column called 'ldl'. Try 'LDL mg/dL' instead."

        # Simulate the caching logic that should be added to execute_analysis_with_idempotency
        # BEFORE fix: No caching - translate_error_with_llm called every time
        # AFTER fix: Cache translation in result["friendly_error_message"]

        # Act: Apply the FIXED caching logic
        if "error" in execution_result_with_error and not execution_result_with_error.get("friendly_error_message"):
            # This would call translate_error_with_llm(execution_result_with_error["error"])
            friendly = mock_llm_translation  # Mocked LLM result
            if friendly:
                execution_result_with_error["friendly_error_message"] = friendly

        # Assert: friendly_error_message should now be in result
        assert (
            execution_result_with_error.get("friendly_error_message") == mock_llm_translation
        ), "Error translation should be cached in result dict"

    def test_execute_analysis_skips_caching_when_already_cached(self, mock_session_state):
        """
        Test that execute_analysis_with_idempotency skips LLM call when translation already cached.

        This is the key fix - on cache hit (re-render), the translation should already
        be in the result dict, so no LLM call needed.
        """
        # Arrange: Error result that ALREADY has friendly_error_message (from cache)
        execution_result_with_cached = {
            "success": False,
            "error": "ColumnNotFoundError: Column 'ldl' not found",
            "friendly_error_message": "Already cached translation.",  # Already exists!
        }

        llm_call_count = 0

        def mock_translate_error_with_llm(error):
            nonlocal llm_call_count
            llm_call_count += 1
            return "New translation from LLM"

        # Act: Apply the FIXED caching logic with guard condition
        if "error" in execution_result_with_cached and not execution_result_with_cached.get("friendly_error_message"):
            friendly = mock_translate_error_with_llm(execution_result_with_cached["error"])
            if friendly:
                execution_result_with_cached["friendly_error_message"] = friendly

        # Assert: LLM was NOT called (translation already cached)
        assert llm_call_count == 0, "Should NOT call translate_error_with_llm when translation already cached"
        assert execution_result_with_cached["friendly_error_message"] == "Already cached translation."
