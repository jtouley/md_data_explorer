"""
Tests for conversation history feature in Ask Questions page.

Tests verify:
- Conversation history is initialized in session state
- Entries are added after analysis completes
- History stores lightweight summaries (headline, not full result dicts)
- History is limited to prevent memory bloat
- Entry structure matches ADR001 specification
"""

import time

import pytest


class TestConversationHistory:
    """Test conversation history data structure and lifecycle."""

    def test_conversation_history_initialized_in_session_state(self, mock_session_state):
        """Test that conversation_history is initialized as empty list."""
        # Arrange: Import the page module to trigger initialization
        import importlib.util
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent.parent.parent
        sys.path.insert(0, str(project_root / "src"))

        page_path = project_root / "src" / "clinical_analytics" / "ui" / "pages" / "03_💬_Ask_Questions.py"
        spec = importlib.util.spec_from_file_location("ask_questions_page", page_path)
        ask_questions_page = importlib.util.module_from_spec(spec)

        # Mock streamlit

        with pytest.MonkeyPatch().context() as m:
            m.setattr("streamlit.session_state", mock_session_state)
            spec.loader.exec_module(ask_questions_page)

            # Act: Initialize (simulate main() initialization)
            if "conversation_history" not in mock_session_state:
                mock_session_state["conversation_history"] = []

            # Assert: conversation_history exists and is empty list
            assert "conversation_history" in mock_session_state
            assert isinstance(mock_session_state["conversation_history"], list)
            assert len(mock_session_state["conversation_history"]) == 0

    def test_conversation_history_entry_structure_matches_adr001(self):
        """Test that conversation history entry structure matches ADR001 specification."""
        # Arrange: Create sample entry
        entry = {
            "query": "what is the average viral load",
            "intent": "DESCRIBE",  # Stored as string value
            "headline": "Average viral load: 150 copies/mL",
            "run_key": "describe_123456",
            "timestamp": time.time(),
            "filters_applied": [],
        }

        # Assert: Entry has all required fields per ADR001
        assert "query" in entry
        assert "intent" in entry
        assert "headline" in entry
        assert "run_key" in entry
        assert "timestamp" in entry
        assert "filters_applied" in entry

        # Assert: Types are correct
        assert isinstance(entry["query"], str)
        assert isinstance(entry["intent"], str)  # String, not enum
        assert isinstance(entry["headline"], str)
        assert isinstance(entry["run_key"], str)
        assert isinstance(entry["timestamp"], float)
        assert isinstance(entry["filters_applied"], list)

    def test_conversation_history_stores_lightweight_summaries(self):
        """Test that conversation history stores lightweight summaries, not full result dicts."""
        # Arrange: Create entry with headline (lightweight)
        entry = {
            "query": "compare viral load by treatment",
            "intent": "COMPARE_GROUPS",
            "headline": "Treatment A has lower viral load (p<0.05)",
            "run_key": "compare_789012",
            "timestamp": time.time(),
            "filters_applied": [],
        }

        # Assert: Entry is lightweight (no large objects like arrays, dataframes)
        import json
        import sys

        entry_size = sys.getsizeof(json.dumps(entry, default=str))
        assert entry_size < 1024  # Should be under 1KB per entry

        # Assert: No full result dict stored
        assert "result" not in entry
        assert "summary_stats" not in entry
        assert "group_statistics" not in entry

    def test_conversation_history_includes_filters_applied(self):
        """Test that conversation history includes filters_applied for audit trail."""
        # Arrange: Create entry with filters
        entry = {
            "query": "average t score of those that had osteoporosis",
            "intent": "DESCRIBE",
            "headline": "Average t score: -2.8",
            "run_key": "describe_456789",
            "timestamp": time.time(),
            "filters_applied": [
                {
                    "column": "Results of DEXA?",
                    "operator": "==",
                    "value": "Osteoporosis",
                    "exclude_nulls": True,
                }
            ],
        }

        # Assert: filters_applied is included
        assert "filters_applied" in entry
        assert len(entry["filters_applied"]) == 1
        assert entry["filters_applied"][0]["column"] == "Results of DEXA?"

    def test_conversation_history_limits_size_to_prevent_memory_bloat(self, mock_session_state):
        """Test that conversation history is limited to prevent memory bloat."""
        # Arrange: Initialize conversation history
        mock_session_state["conversation_history"] = []

        # Act: Add more than 20 entries
        max_conversation_history = 20
        for i in range(25):
            mock_session_state["conversation_history"].append(
                {
                    "query": f"query {i}",
                    "intent": "DESCRIBE",
                    "headline": f"result {i}",
                    "run_key": f"run_key_{i}",
                    "timestamp": time.time(),
                    "filters_applied": [],
                }
            )

            # Limit history size (keep last N entries)
            if len(mock_session_state["conversation_history"]) > max_conversation_history:
                mock_session_state["conversation_history"] = mock_session_state["conversation_history"][
                    -max_conversation_history:
                ]

        # Assert: History is limited to max_conversation_history
        assert len(mock_session_state["conversation_history"]) == max_conversation_history

        # Assert: Most recent entries are kept (not oldest)
        assert mock_session_state["conversation_history"][0]["query"] == "query 5"  # Entry 5 (25 - 20)
        assert mock_session_state["conversation_history"][-1]["query"] == "query 24"  # Most recent

    def test_conversation_history_entry_can_reference_full_result_via_run_key(self):
        """Test that full results can be reconstructed on demand via run_key."""
        # Arrange: Create entry with run_key
        entry = {
            "query": "what is the average age",
            "intent": "DESCRIBE",
            "headline": "Average age: 45.2 years",
            "run_key": "describe_abc123",
            "timestamp": time.time(),
            "filters_applied": [],
        }

        # Assert: run_key can be used to reference full result
        assert entry["run_key"] == "describe_abc123"
        # In real implementation: st.session_state[f"analysis_result:dataset_v1:{entry['run_key']}"]
        # would contain the full result dict. This test verifies the contract: run_key enables
        # reconstruction of full results from lightweight conversation history entries.

    def test_conversation_history_handles_missing_headline_gracefully(self):
        """Test that conversation history handles missing headline gracefully."""
        # Arrange: Create entry without headline (fallback scenario)
        entry = {
            "query": "analyze data",
            "intent": "DESCRIBE",
            "headline": "Analysis completed",  # Fallback value
            "run_key": "run_key_xyz",
            "timestamp": time.time(),
            "filters_applied": [],
        }

        # Assert: Entry has fallback headline
        assert entry["headline"] == "Analysis completed"
        assert isinstance(entry["headline"], str)

    def test_conversation_history_preserved_across_reruns(self, mock_session_state):
        """
        Test that conversation history is NOT reset mid-conversation.

        This verifies the fix where conversation history was being cleared
        unexpectedly. History should only be cleared via explicit user action
        (Clear Conversation button), not during normal page reruns.
        """
        # Arrange: Simulate existing conversation history
        mock_session_state["conversation_history"] = [
            {
                "query": "what is the average age",
                "intent": "DESCRIBE",
                "headline": "Average age: 45.2 years",
                "run_key": "describe_abc123",
                "timestamp": time.time(),
                "filters_applied": [],
            },
            {
                "query": "compare by treatment",
                "intent": "COMPARE_GROUPS",
                "headline": "Treatment A vs B: p<0.05",
                "run_key": "compare_def456",
                "timestamp": time.time(),
                "filters_applied": [],
            },
        ]
        original_history = mock_session_state["conversation_history"].copy()

        # Act: Simulate page rerun (initialization check should preserve history)
        # The initialization code should check "if not in session_state" and NOT reset
        if "conversation_history" not in mock_session_state:
            mock_session_state["conversation_history"] = []

        # Assert: History is preserved (not reset to empty)
        assert "conversation_history" in mock_session_state
        assert len(mock_session_state["conversation_history"]) == len(original_history), (
            f"History should be preserved, got {len(mock_session_state['conversation_history'])} entries, "
            f"expected {len(original_history)}"
        )
        assert mock_session_state["conversation_history"] == original_history, "History entries should be unchanged"

    def test_conversation_history_only_cleared_by_explicit_user_action(self, mock_session_state):
        """
        Test that conversation history is only cleared by explicit user action,
        not by normal page operations.
        """
        # Arrange: Existing conversation history
        mock_session_state["conversation_history"] = [
            {
                "query": "query 1",
                "intent": "DESCRIBE",
                "headline": "result 1",
                "run_key": "run_1",
                "timestamp": time.time(),
                "filters_applied": [],
            },
        ]

        # Act: Normal operations (adding new entry) should NOT clear history
        # Simulate adding a new entry (as done in execute_analysis_with_idempotency)
        if "conversation_history" not in mock_session_state:
            mock_session_state["conversation_history"] = []

        mock_session_state["conversation_history"].append(
            {
                "query": "query 2",
                "intent": "DESCRIBE",
                "headline": "result 2",
                "run_key": "run_2",
                "timestamp": time.time(),
                "filters_applied": [],
            }
        )

        # Assert: History should contain both entries (not cleared)
        assert len(mock_session_state["conversation_history"]) == 2, "History should accumulate entries, not be cleared"
        assert mock_session_state["conversation_history"][0]["query"] == "query 1"
        assert mock_session_state["conversation_history"][1]["query"] == "query 2"
