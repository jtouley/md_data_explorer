"""
Test result lifecycle management (history tracking, O(1) eviction).

Test name follows: test_unit_scenario_expectedBehavior
"""

from collections import deque

import pytest


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session_state."""
    return {}


def test_lifecycle_remember_run_evicts_oldest_when_maxlen_reached(mock_session_state, monkeypatch, ask_questions_page):
    """
    Test that 6th result evicts oldest (O(1) eviction).

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Add 5 results to history
    dataset_version = "test_dataset_v1"

    # Mock st.session_state
    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Add 5 runs
    for i in range(5):
        run_key = f"run_key_{i}"
        # Store result for each run
        result_key = f"analysis_result:{dataset_version}:run_key_{i}"
        mock_session_state[result_key] = {"result": f"data_{i}"}
        ask_questions_page.remember_run(dataset_version, run_key)

    # Verify 5 results stored
    hist_key = f"run_history_{dataset_version}"
    assert len(mock_session_state[hist_key]) == 5

    # Act: Add 6th result
    result_key_5 = f"analysis_result:{dataset_version}:run_key_5"
    mock_session_state[result_key_5] = {"result": "data_5"}
    ask_questions_page.remember_run(dataset_version, "run_key_5")

    # Assert: Oldest result evicted, newest stored
    assert len(mock_session_state[hist_key]) == 5
    assert "run_key_0" not in mock_session_state[hist_key]  # Oldest evicted
    assert "run_key_5" in mock_session_state[hist_key]  # Newest added

    # Verify result for evicted key was deleted
    evicted_result_key = f"analysis_result:{dataset_version}:run_key_0"
    assert evicted_result_key not in mock_session_state


def test_lifecycle_remember_run_lru_behavior_moves_existing_to_end(mock_session_state, monkeypatch, ask_questions_page):
    """
    Test that re-adding existing run_key moves it to end (LRU behavior).

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Add runs
    dataset_version = "test_dataset_v1"

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    ask_questions_page.remember_run(dataset_version, "run_key_1")
    ask_questions_page.remember_run(dataset_version, "run_key_2")
    ask_questions_page.remember_run(dataset_version, "run_key_3")

    hist_key = f"run_history_{dataset_version}"

    # Act: Re-add existing key
    ask_questions_page.remember_run(dataset_version, "run_key_1")

    # Assert: Key moved to end (most recent)
    assert mock_session_state[hist_key][-1] == "run_key_1"
    assert len(mock_session_state[hist_key]) == 3  # No new entry added


def test_lifecycle_clear_all_results_only_clears_dataset_scoped_keys(
    mock_session_state, monkeypatch, ask_questions_page
):
    """
    Test that clear_all_results only clears dataset-scoped keys.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Add results for multiple datasets

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Add results for dataset 1
    dataset1 = "dataset_1"
    ask_questions_page.remember_run(dataset1, "run_key_1")
    ask_questions_page.remember_run(dataset1, "run_key_2")
    mock_session_state[f"analysis_result:{dataset1}:run_key_1"] = {"result": "data1"}
    mock_session_state[f"analysis_result:{dataset1}:run_key_2"] = {"result": "data2"}

    # Add results for dataset 2
    dataset2 = "dataset_2"
    ask_questions_page.remember_run(dataset2, "run_key_1")
    mock_session_state[f"analysis_result:{dataset2}:run_key_1"] = {"result": "data3"}

    # Act: Clear results for dataset 1 only
    ask_questions_page.clear_all_results(dataset1)

    # Assert: Only dataset 1's results cleared
    assert f"run_history_{dataset1}" not in mock_session_state
    assert f"analysis_result:{dataset1}:run_key_1" not in mock_session_state
    assert f"analysis_result:{dataset1}:run_key_2" not in mock_session_state

    # Dataset 2's results still exist
    assert f"run_history_{dataset2}" in mock_session_state
    assert f"analysis_result:{dataset2}:run_key_1" in mock_session_state


def test_lifecycle_remember_run_stores_history_as_list_not_deque(mock_session_state, monkeypatch, ask_questions_page):
    """
    Test that history is stored as list[str] (not deque) to avoid serialization issues.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange
    dataset_version = "test_dataset_v1"

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Act: Add runs
    ask_questions_page.remember_run(dataset_version, "run_key_1")
    ask_questions_page.remember_run(dataset_version, "run_key_2")

    # Assert: History is a list (not deque)
    hist_key = f"run_history_{dataset_version}"
    hist = mock_session_state[hist_key]
    assert isinstance(hist, list)
    assert not isinstance(hist, deque)


def test_lifecycle_remember_run_handles_empty_history(mock_session_state, monkeypatch, ask_questions_page):
    """
    Test that remember_run handles empty history correctly.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Empty session state
    dataset_version = "test_dataset_v1"

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Act: Add first run
    ask_questions_page.remember_run(dataset_version, "run_key_1")

    # Assert: History created and populated
    hist_key = f"run_history_{dataset_version}"
    assert hist_key in mock_session_state
    assert mock_session_state[hist_key] == ["run_key_1"]


def test_lifecycle_cleanup_old_results_removes_orphaned_results(mock_session_state, monkeypatch, ask_questions_page):
    """
    Test that cleanup_old_results removes orphaned results not in history (PR25).

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Add results, some in history, some orphaned
    dataset_version = "test_dataset_v1"

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Add runs to history
    ask_questions_page.remember_run(dataset_version, "run_key_1")
    ask_questions_page.remember_run(dataset_version, "run_key_2")

    # Store results for runs in history
    mock_session_state[f"analysis_result:{dataset_version}:run_key_1"] = {"result": "data1"}
    mock_session_state[f"analysis_result:{dataset_version}:run_key_2"] = {"result": "data2"}

    # Add orphaned result (not in history)
    mock_session_state[f"analysis_result:{dataset_version}:orphaned_key"] = {"result": "orphaned"}

    # Act: Cleanup
    ask_questions_page.cleanup_old_results(dataset_version)

    # Assert: Orphaned result removed, history results kept
    assert f"analysis_result:{dataset_version}:orphaned_key" not in mock_session_state
    assert f"analysis_result:{dataset_version}:run_key_1" in mock_session_state
    assert f"analysis_result:{dataset_version}:run_key_2" in mock_session_state


def test_lifecycle_execution_cache_eviction_prevents_unbounded_growth(
    mock_session_state, monkeypatch, ask_questions_page
):
    """
    Test that execution cache eviction prevents unbounded growth (PR25).

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Add more than MAX_STORED_RESULTS_PER_DATASET execution cache entries
    dataset_version = "test_dataset_v1"
    max_results = 5  # Should match MAX_STORED_RESULTS_PER_DATASET in implementation

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Add max_results + 1 execution cache entries
    for i in range(max_results + 1):
        exec_key = f"exec_result:{dataset_version}:query_hash_{i}"
        mock_session_state[exec_key] = {"result": f"exec_{i}"}

    # Act: Trigger eviction (simulate adding new entry)
    # Import the function directly from the module
    _evict_old_execution_cache = ask_questions_page._evict_old_execution_cache
    _evict_old_execution_cache(dataset_version)

    # Assert: Only max_results entries remain
    exec_keys = [k for k in mock_session_state.keys() if k.startswith(f"exec_result:{dataset_version}:")]
    assert len(exec_keys) <= max_results


def test_lifecycle_remember_run_called_for_cached_results(mock_session_state, monkeypatch, ask_questions_page):
    """
    Test that remember_run() is called even for cached results (PR25 invariant).

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Pre-populate cached result
    dataset_version = "test_dataset_v1"
    run_key = "cached_run_key"

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Pre-store result (simulating cache hit)
    result_key = f"analysis_result:{dataset_version}:{run_key}"
    mock_session_state[result_key] = {"result": "cached_data"}

    # History should be empty initially
    hist_key = f"run_history_{dataset_version}"
    assert hist_key not in mock_session_state or len(mock_session_state.get(hist_key, [])) == 0

    # Act: Call remember_run (simulating what happens in cached path)
    ask_questions_page.remember_run(dataset_version, run_key)

    # Assert: History updated even though result was cached
    assert hist_key in mock_session_state
    assert run_key in mock_session_state[hist_key]
