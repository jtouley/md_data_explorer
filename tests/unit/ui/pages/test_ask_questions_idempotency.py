"""
Test idempotency guard with result persistence.

Note: run_key generation is now handled by the semantic layer (PR21 refactor).
These tests use pre-defined run_keys since the UI receives run_key from execution results.

Test name follows: test_unit_scenario_expectedBehavior
"""

import hashlib
from unittest.mock import Mock, patch

# mock_session_state, sample_cohort, sample_context moved to conftest.py - use shared fixtures


def _generate_test_run_key(dataset_version: str, query_text: str) -> str:
    """Generate a deterministic test run_key for testing purposes."""
    normalized = " ".join(query_text.lower().split())
    return hashlib.sha256(f"{dataset_version}:{normalized}".encode()).hexdigest()


def test_idempotency_same_query_uses_cached_result(
    mock_session_state, sample_cohort, sample_context, ask_questions_page, monkeypatch
):
    """
    Test that identical query uses cached result (idempotency).

    Test name follows: test_unit_scenario_expectedBehavior
    """
    from datetime import datetime

    from clinical_analytics.core.result_cache import CachedResult, ResultCache

    # Arrange: Set up test data
    dataset_version = "test_dataset_v1"
    query_text = "describe all patients"
    run_key = _generate_test_run_key(dataset_version, query_text)

    # Pre-populate ResultCache with cached result (Milestone A: State Extraction)
    cached_result = {"type": "descriptive", "row_count": 5, "column_count": 4}
    cache = ResultCache(max_size=50)
    cache.put(
        CachedResult(
            run_key=run_key,
            query=query_text,
            result=cached_result,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
    )
    mock_session_state["result_cache"] = cache

    # Mock Streamlit functions - patch st.session_state and st.spinner on the module
    spinner_context = Mock()
    spinner_context.__enter__ = Mock(return_value=spinner_context)
    spinner_context.__exit__ = Mock(return_value=None)
    # Patch session_state directly on the st object in the module
    monkeypatch.setattr(ask_questions_page.st, "session_state", mock_session_state)
    monkeypatch.setattr(ask_questions_page.st, "spinner", Mock(return_value=spinner_context))
    with patch.object(ask_questions_page, "render_analysis_by_type") as mock_render:
        # Act: Execute analysis (with execution_result=None to test cache lookup)
        ask_questions_page.execute_analysis_with_idempotency(
            sample_cohort, sample_context, run_key, dataset_version, query_text, execution_result=None
        )

        # Assert: Used cached result (render called once with cached data)
        # Note: render_analysis_by_type now accepts optional query_text parameter
        mock_render.assert_called_once()
        call_args = mock_render.call_args
        assert call_args[0][0] == cached_result
        assert call_args[0][1] == sample_context.inferred_intent


def test_idempotency_different_query_computes_new_result(
    mock_session_state, sample_cohort, sample_context, ask_questions_page, monkeypatch
):
    """
    Test that different query computes new result (not cached).

    PR21 refactor: Now requires execution_result and semantic_layer parameters.
    The semantic layer handles all execution; this function only formats and caches.

    Test name: test_unit_scenario_expectedBehavior
    """
    from clinical_analytics.core.result_cache import ResultCache

    # Arrange: Set up test data with DIFFERENT query (to ensure new run_key)
    # Clear mock_session_state to ensure clean state (fixture might be shared)
    mock_session_state.clear()
    dataset_version = "test_dataset_v2_unique"
    query_text = "show me patient statistics"
    sample_context.research_question = query_text
    run_key = _generate_test_run_key(dataset_version, query_text)

    # Initialize ResultCache (Milestone A: State Extraction)
    cache = ResultCache(max_size=50)
    mock_session_state["result_cache"] = cache

    # Ensure this run_key is NOT in cache
    assert cache.get(run_key, dataset_version) is None

    # Create mock execution_result (from semantic_layer.execute_query_plan)
    execution_result = {"success": True, "result_df": Mock(), "run_key": run_key}

    # Create mock semantic_layer
    mock_semantic_layer = Mock()
    formatted_result = {
        "type": "descriptive",
        "row_count": 5,
        "column_count": 4,
        "missing_pct": 0.0,
        "summary_stats": [],
        "categorical_summary": {},
    }
    mock_semantic_layer.format_execution_result.return_value = formatted_result

    # Mock Streamlit functions
    spinner_context = Mock()
    spinner_context.__enter__ = Mock(return_value=spinner_context)
    spinner_context.__exit__ = Mock(return_value=None)
    mock_spinner = Mock(return_value=spinner_context)

    with patch.object(ask_questions_page.st, "session_state", mock_session_state):
        with patch.object(ask_questions_page.st, "spinner", mock_spinner):
            with patch.object(ask_questions_page, "render_analysis_by_type") as mock_render:
                # Act: Execute analysis with required parameters
                ask_questions_page.execute_analysis_with_idempotency(
                    sample_cohort,
                    sample_context,
                    run_key,
                    dataset_version,
                    query_text,
                    execution_result=execution_result,
                    semantic_layer=mock_semantic_layer,
                )

                # Assert: Result stored in ResultCache (idempotency works)
                cached_result = cache.get(run_key, dataset_version)
                assert cached_result is not None
                assert cached_result.result["type"] == formatted_result["type"]
                assert cached_result.result["row_count"] == formatted_result["row_count"]
                mock_render.assert_called_once()


def test_idempotency_result_persists_across_reruns(
    mock_session_state, sample_cohort, sample_context, ask_questions_page, monkeypatch
):
    """
    Test that result persists across Streamlit reruns.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Set up test data with UNIQUE query to avoid conflicts
    mock_session_state.clear()
    dataset_version = "test_dataset_v3_unique"
    query_text = "show patient demographics"
    sample_context.research_question = query_text
    run_key = _generate_test_run_key(dataset_version, query_text)

    # Create mock execution_result and semantic_layer
    execution_result = {"success": True, "result_df": Mock(), "run_key": run_key}
    mock_semantic_layer = Mock()
    formatted_result = {
        "type": "descriptive",
        "row_count": 5,
        "column_count": 4,
        "missing_pct": 0.0,
        "summary_stats": [],
        "categorical_summary": {},
    }
    mock_semantic_layer.format_execution_result.return_value = formatted_result

    # Mock Streamlit functions
    spinner_context = Mock()
    spinner_context.__enter__ = Mock(return_value=spinner_context)
    spinner_context.__exit__ = Mock(return_value=None)
    mock_spinner = Mock(return_value=spinner_context)

    with patch.object(ask_questions_page.st, "session_state", mock_session_state):
        with patch.object(ask_questions_page.st, "spinner", mock_spinner):
            with patch.object(ask_questions_page, "render_analysis_by_type") as mock_render:
                with patch.object(ask_questions_page, "remember_run"):
                    # First execution: compute and store
                    ask_questions_page.execute_analysis_with_idempotency(
                        sample_cohort,
                        sample_context,
                        run_key,
                        dataset_version,
                        query_text,
                        execution_result=execution_result,
                        semantic_layer=mock_semantic_layer,
                    )

                    # Verify format_execution_result was called
                    assert mock_semantic_layer.format_execution_result.call_count == 1

                    # Second execution (simulating rerun): should use cache
                    ask_questions_page.execute_analysis_with_idempotency(
                        sample_cohort,
                        sample_context,
                        run_key,
                        dataset_version,
                        query_text,
                        execution_result=execution_result,
                        semantic_layer=mock_semantic_layer,
                    )

                    # Assert: format_execution_result not called again (used cache)
                    assert mock_semantic_layer.format_execution_result.call_count == 1
                    # Render called twice (once per execution)
                    assert mock_render.call_count == 2


def test_idempotency_result_stored_with_dataset_scoped_key(
    mock_session_state, sample_cohort, sample_context, ask_questions_page, monkeypatch
):
    """
    Test that results are stored with dataset-scoped keys.

    Test name: test_unit_scenario_expectedBehavior
    """
    from clinical_analytics.core.result_cache import ResultCache

    # Arrange
    dataset_version = "test_dataset_v3"
    query_text = "analyze patient outcomes"
    sample_context.research_question = query_text
    run_key = _generate_test_run_key(dataset_version, query_text)

    # Initialize ResultCache (Milestone A: State Extraction)
    cache = ResultCache(max_size=50)
    mock_session_state["result_cache"] = cache

    # Create mock execution_result and semantic_layer
    execution_result = {"success": True, "result_df": Mock(), "run_key": run_key}
    mock_semantic_layer = Mock()
    formatted_result = {
        "type": "descriptive",
        "row_count": 5,
        "column_count": 4,
        "missing_pct": 0.0,
        "summary_stats": [],
        "categorical_summary": {},
    }
    mock_semantic_layer.format_execution_result.return_value = formatted_result

    # Mock Streamlit functions
    spinner_context = Mock()
    spinner_context.__enter__ = Mock(return_value=spinner_context)
    spinner_context.__exit__ = Mock(return_value=None)
    mock_spinner = Mock(return_value=spinner_context)

    with patch.object(ask_questions_page.st, "session_state", mock_session_state):
        with patch.object(ask_questions_page.st, "spinner", mock_spinner):
            with patch.object(ask_questions_page, "render_analysis_by_type"):
                # Act: Execute analysis
                ask_questions_page.execute_analysis_with_idempotency(
                    sample_cohort,
                    sample_context,
                    run_key,
                    dataset_version,
                    query_text,
                    execution_result=execution_result,
                    semantic_layer=mock_semantic_layer,
                )

                # Assert: Result stored in ResultCache with dataset-scoped key
                cached_result = cache.get(run_key, dataset_version)
                assert cached_result is not None
                assert cached_result.result["type"] == formatted_result["type"]
                assert cached_result.result["row_count"] == formatted_result["row_count"]
                assert cached_result.dataset_version == dataset_version
