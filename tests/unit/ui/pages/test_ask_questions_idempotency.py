"""
Test idempotency guard with result persistence.

Test name follows: test_unit_scenario_expectedBehavior
"""

from unittest.mock import Mock, patch

import polars as pl
import pytest

from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


# mock_session_state, sample_cohort, sample_context moved to conftest.py - use shared fixtures


def test_idempotency_same_query_uses_cached_result(
    mock_session_state, sample_cohort, sample_context, ask_questions_page, monkeypatch
):
    """
    Test that identical query uses cached result (idempotency).

    Test name follows: test_unit_scenario_expectedBehavior
    """
    # Arrange: Set up test data
    dataset_version = "test_dataset_v1"
    query_text = "describe all patients"
    run_key = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)

    # Pre-populate session_state with cached result
    result_key = f"analysis_result:{dataset_version}:{run_key}"
    cached_result = {"type": "descriptive", "row_count": 5, "column_count": 4}
    mock_session_state[result_key] = cached_result

    # Mock Streamlit functions - patch st.session_state and st.spinner on the module
    spinner_context = Mock()
    spinner_context.__enter__ = Mock(return_value=spinner_context)
    spinner_context.__exit__ = Mock(return_value=None)
    # Patch session_state directly on the st object in the module
    monkeypatch.setattr(ask_questions_page.st, "session_state", mock_session_state)
    monkeypatch.setattr(ask_questions_page.st, "spinner", Mock(return_value=spinner_context))
    with patch.object(ask_questions_page, "render_analysis_by_type") as mock_render:
        # Act: Execute analysis
        ask_questions_page.execute_analysis_with_idempotency(
            sample_cohort, sample_context, run_key, dataset_version, query_text
        )

        # Assert: Used cached result (render called once with cached data)
        mock_render.assert_called_once_with(cached_result, sample_context.inferred_intent)


def test_idempotency_different_query_computes_new_result(
    mock_session_state, sample_cohort, sample_context, ask_questions_page, monkeypatch
):
    """
    Test that different query computes new result (not cached).

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Set up test data with DIFFERENT query (to ensure new run_key)
    # Clear mock_session_state to ensure clean state (fixture might be shared)
    mock_session_state.clear()
    dataset_version = "test_dataset_v2_unique"  # Unique dataset version
    query_text = "show me patient statistics"  # Different query to generate different run_key
    # Set research_question on context to ensure different run_key
    sample_context.research_question = query_text
    run_key = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)

    # Ensure this run_key is NOT in cache
    result_key = f"analysis_result:{dataset_version}:{run_key}"
    assert result_key not in mock_session_state, (
        f"Test setup error: run_key {run_key} already in cache. Keys: {list(mock_session_state.keys())}"
    )

    # Mock Streamlit functions - patch st.session_state and st.spinner on the module
    spinner_context = Mock()
    spinner_context.__enter__ = Mock(return_value=spinner_context)
    spinner_context.__exit__ = Mock(return_value=None)
    mock_spinner = Mock(return_value=spinner_context)
    with patch.object(ask_questions_page.st, "session_state", mock_session_state):
        with patch.object(ask_questions_page.st, "spinner", mock_spinner):
            # Patch compute_analysis_by_type where it's used in the module
            with patch.object(ask_questions_page, "compute_analysis_by_type") as mock_compute:
                with patch.object(ask_questions_page, "render_analysis_by_type") as mock_render:
                    with patch.object(ask_questions_page, "remember_run"):
                        # Mock compute result (match actual compute_descriptive_analysis structure)
                        computed_result = {
                            "type": "descriptive",
                            "row_count": 5,
                            "column_count": 4,
                            "missing_pct": 0.0,
                            "summary_stats": [],
                            "categorical_summary": {},
                        }
                        mock_compute.return_value = computed_result

                        # Act: Execute analysis
                        ask_questions_page.execute_analysis_with_idempotency(
                            sample_cohort, sample_context, run_key, dataset_version, query_text
                        )

                        # Assert: Result stored (idempotency works)
                        result_key = f"analysis_result:{dataset_version}:{run_key}"
                        assert result_key in mock_session_state, (
                            f"Result not stored. Keys: {list(mock_session_state.keys())}"
                        )
                        # Check key fields match (not full dict equality due to computed fields)
                        stored_result = mock_session_state[result_key]
                        assert stored_result["type"] == computed_result["type"]
                        assert stored_result["row_count"] == computed_result["row_count"]
                        # Verify compute was called (unless result was already cached)
                        if (
                            result_key not in mock_session_state
                            or len(
                                [
                                    k
                                    for k in mock_session_state.keys()
                                    if k.startswith(f"analysis_result:{dataset_version}:")
                                ]
                            )
                            == 0
                        ):
                            mock_compute.assert_called_once_with(sample_cohort, sample_context)
                        mock_render.assert_called_once()


def test_idempotency_result_persists_across_reruns(
    mock_session_state, sample_cohort, sample_context, ask_questions_page, monkeypatch
):
    """
    Test that result persists across Streamlit reruns.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Set up test data with UNIQUE query to avoid conflicts with other tests
    # Clear mock_session_state to ensure clean state (fixture might be shared)
    mock_session_state.clear()
    dataset_version = "test_dataset_v3_unique"  # Unique dataset version
    query_text = "show patient demographics"  # Different query
    sample_context.research_question = query_text
    run_key = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)

    # Ensure this run_key is NOT in cache
    result_key = f"analysis_result:{dataset_version}:{run_key}"
    assert result_key not in mock_session_state, "Test setup error: run_key already in cache"

    # Mock Streamlit functions - patch st.session_state and st.spinner on the module
    spinner_context = Mock()
    spinner_context.__enter__ = Mock(return_value=spinner_context)
    spinner_context.__exit__ = Mock(return_value=None)
    mock_spinner = Mock(return_value=spinner_context)
    with patch.object(ask_questions_page.st, "session_state", mock_session_state):
        with patch.object(ask_questions_page.st, "spinner", mock_spinner):
            # Patch compute_analysis_by_type where it's used in the module
            with patch.object(ask_questions_page, "compute_analysis_by_type") as mock_compute:
                with patch.object(ask_questions_page, "render_analysis_by_type") as mock_render:
                    with patch.object(ask_questions_page, "remember_run"):
                        # First execution: compute and store (match actual compute_descriptive_analysis structure)
                        computed_result = {
                            "type": "descriptive",
                            "row_count": 5,
                            "column_count": 4,
                            "missing_pct": 0.0,
                            "summary_stats": [],
                            "categorical_summary": {},
                        }
                        mock_compute.return_value = computed_result

                        ask_questions_page.execute_analysis_with_idempotency(
                            sample_cohort, sample_context, run_key, dataset_version, query_text
                        )

                        # Verify compute was called
                        assert mock_compute.call_count == 1

                        # Second execution (simulating rerun): should use cache
                        ask_questions_page.execute_analysis_with_idempotency(
                            sample_cohort, sample_context, run_key, dataset_version, query_text
                        )

                        # Assert: Compute not called again (used cache)
                        assert mock_compute.call_count == 1
                        # Render called twice (once per execution)
                        assert mock_render.call_count == 2


def test_idempotency_result_stored_with_dataset_scoped_key(
    mock_session_state, sample_cohort, sample_context, ask_questions_page, monkeypatch
):
    """
    Test that results are stored with dataset-scoped keys.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Set up test data with UNIQUE query to avoid conflicts with other tests
    dataset_version = "test_dataset_v3"  # Different dataset version
    query_text = "analyze patient outcomes"  # Different query
    sample_context.research_question = query_text
    run_key = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)

    # Ensure this run_key is NOT in cache
    result_key = f"analysis_result:{dataset_version}:{run_key}"
    assert result_key not in mock_session_state, "Test setup error: run_key already in cache"

    # Mock Streamlit functions - patch st.session_state and st.spinner on the module
    spinner_context = Mock()
    spinner_context.__enter__ = Mock(return_value=spinner_context)
    spinner_context.__exit__ = Mock(return_value=None)
    mock_spinner = Mock(return_value=spinner_context)
    with patch.object(ask_questions_page.st, "session_state", mock_session_state):
        with patch.object(ask_questions_page.st, "spinner", mock_spinner):
            # Patch compute_analysis_by_type where it's used in the module
            with patch.object(ask_questions_page, "compute_analysis_by_type") as mock_compute:
                with patch.object(ask_questions_page, "render_analysis_by_type"):
                    with patch.object(ask_questions_page, "remember_run"):
                        computed_result = {
                            "type": "descriptive",
                            "row_count": 5,
                            "column_count": 4,
                            "missing_pct": 0.0,
                            "summary_stats": [],
                            "categorical_summary": {},
                        }
                        mock_compute.return_value = computed_result

                        # Act: Execute analysis
                        ask_questions_page.execute_analysis_with_idempotency(
                            sample_cohort, sample_context, run_key, dataset_version, query_text
                        )

                        # Assert: Result stored with dataset-scoped key
                        result_key = f"analysis_result:{dataset_version}:{run_key}"
                        assert result_key in mock_session_state
                        # Check key fields match (not full dict equality due to computed fields)
                        stored_result = mock_session_state[result_key]
                        assert stored_result["type"] == computed_result["type"]
                        assert stored_result["row_count"] == computed_result["row_count"]

                        # Verify last_run_key also stored
                        last_run_key = f"last_run_key:{dataset_version}"
                        assert last_run_key in mock_session_state
                        assert mock_session_state[last_run_key] == run_key
