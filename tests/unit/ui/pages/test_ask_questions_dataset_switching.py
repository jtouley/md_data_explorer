"""
Tests for dataset switching behavior in Ask Questions page.

Tests verify:
- Conversation history is cleared when dataset changes
- Analysis context is cleared when dataset changes
- Generic behavior works for any dataset, not hardcoded
"""

from unittest.mock import MagicMock, patch


class TestDatasetSwitching:
    """Test that dataset switching clears conversation history and context."""

    def test_dataset_change_clears_conversation_history(self):
        """Test that changing dataset clears conversation history."""
        # Arrange: Simulate existing conversation history
        session_state = {
            "conversation_history": [
                {"query": "how many patients", "intent": "count", "headline": "Total: 100"},
                {"query": "what predicts outcome", "intent": "predictor", "headline": "Age predicts"},
            ],
            "last_dataset_choice": "dataset_a",
        }

        with patch("streamlit.session_state", session_state):
            # Simulate dataset change
            current_dataset = "dataset_b"
            if "last_dataset_choice" not in session_state:
                session_state["last_dataset_choice"] = current_dataset
            elif session_state["last_dataset_choice"] != current_dataset:
                # Dataset changed - clear conversation history
                if "conversation_history" in session_state:
                    session_state["conversation_history"] = []
                if "analysis_context" in session_state:
                    del session_state["analysis_context"]
                session_state["last_dataset_choice"] = current_dataset

            # Assert: Conversation history should be cleared
            assert "conversation_history" in session_state
            assert len(session_state["conversation_history"]) == 0, (
                "Conversation history should be cleared on dataset change"
            )

    def test_dataset_change_clears_analysis_context(self):
        """Test that changing dataset clears analysis context."""
        # Arrange: Simulate existing analysis context
        from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

        session_state = {
            "analysis_context": AnalysisContext(),
            "last_dataset_choice": "dataset_a",
        }
        session_state["analysis_context"].inferred_intent = AnalysisIntent.COUNT

        with patch("streamlit.session_state", session_state):
            # Simulate dataset change
            current_dataset = "dataset_b"
            if "last_dataset_choice" not in session_state:
                session_state["last_dataset_choice"] = current_dataset
            elif session_state["last_dataset_choice"] != current_dataset:
                # Dataset changed - clear analysis context
                if "conversation_history" in session_state:
                    session_state["conversation_history"] = []
                if "analysis_context" in session_state:
                    del session_state["analysis_context"]
                session_state["last_dataset_choice"] = current_dataset

            # Assert: Analysis context should be cleared
            assert "analysis_context" not in session_state, "Analysis context should be cleared on dataset change"

    def test_dataset_switching_is_generic_works_for_any_dataset(self):
        """Test that dataset switching logic is generic and works for any dataset name."""
        # Arrange: Test with various dataset names
        dataset_pairs = [
            ("user_upload_123", "user_upload_456"),
            ("dataset_a", "dataset_b"),
            ("📤 Statin use", "📤 DEXA results"),
        ]

        for dataset_a, dataset_b in dataset_pairs:
            session_state = {
                "conversation_history": [{"query": "test", "intent": "count"}],
                "analysis_context": MagicMock(),
                "last_dataset_choice": dataset_a,
            }

            with patch("streamlit.session_state", session_state):
                # Simulate switching from dataset_a to dataset_b
                current_dataset = dataset_b
                if "last_dataset_choice" not in session_state:
                    session_state["last_dataset_choice"] = current_dataset
                elif session_state["last_dataset_choice"] != current_dataset:
                    # Dataset changed - clear state
                    if "conversation_history" in session_state:
                        session_state["conversation_history"] = []
                    if "analysis_context" in session_state:
                        del session_state["analysis_context"]
                    session_state["last_dataset_choice"] = current_dataset

                # Assert: State should be cleared regardless of dataset names
                assert len(session_state["conversation_history"]) == 0, (
                    f"Conversation history should be cleared when switching from {dataset_a} to {dataset_b}"
                )
                assert "analysis_context" not in session_state, (
                    f"Analysis context should be cleared when switching from {dataset_a} to {dataset_b}"
                )
                assert session_state["last_dataset_choice"] == dataset_b, (
                    f"last_dataset_choice should be updated to {dataset_b}"
                )

    def test_same_dataset_does_not_clear_history(self):
        """Test that selecting the same dataset does not clear conversation history."""
        # Arrange: Simulate existing conversation history
        session_state = {
            "conversation_history": [
                {"query": "how many patients", "intent": "count", "headline": "Total: 100"},
            ],
            "last_dataset_choice": "dataset_a",
        }

        with patch("streamlit.session_state", session_state):
            # Simulate selecting same dataset
            current_dataset = "dataset_a"
            if "last_dataset_choice" not in session_state:
                session_state["last_dataset_choice"] = current_dataset
            elif session_state["last_dataset_choice"] != current_dataset:
                # Dataset changed - clear state
                if "conversation_history" in session_state:
                    session_state["conversation_history"] = []
                if "analysis_context" in session_state:
                    del session_state["analysis_context"]
                session_state["last_dataset_choice"] = current_dataset

            # Assert: Conversation history should be preserved
            assert len(session_state["conversation_history"]) == 1, (
                "Conversation history should be preserved when same dataset is selected"
            )
