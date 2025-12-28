"""
Test UI Messages - Phase 7

Tests that message constants are defined and can be imported.
"""

from clinical_analytics.ui import messages


class TestMessageConstants:
    """Test that message constants are defined and accessible."""

    def test_low_confidence_messages_defined(self):
        """Test that low-confidence feedback messages are defined."""
        assert hasattr(messages, "LOW_CONFIDENCE_WARNING")
        assert hasattr(messages, "SEMANTIC_LAYER_NOT_READY")
        assert hasattr(messages, "COLLISION_SUGGESTION_WARNING")
        assert isinstance(messages.LOW_CONFIDENCE_WARNING, str)
        assert isinstance(messages.SEMANTIC_LAYER_NOT_READY, str)
        assert isinstance(messages.COLLISION_SUGGESTION_WARNING, str)

    def test_analysis_execution_messages_defined(self):
        """Test that analysis execution messages are defined."""
        assert hasattr(messages, "CONFIRM_AND_RUN")
        assert hasattr(messages, "START_OVER")
        assert hasattr(messages, "CLEAR_RESULTS")
        assert hasattr(messages, "RESULTS_CLEARED")
        assert isinstance(messages.CONFIRM_AND_RUN, str)
        assert isinstance(messages.START_OVER, str)
        assert isinstance(messages.CLEAR_RESULTS, str)
        assert isinstance(messages.RESULTS_CLEARED, str)

    def test_dataset_messages_defined(self):
        """Test that dataset-related messages are defined."""
        assert hasattr(messages, "NO_DATASETS_AVAILABLE")
        assert isinstance(messages.NO_DATASETS_AVAILABLE, str)

    def test_nl_query_messages_defined(self):
        """Test that natural language query messages are defined."""
        assert hasattr(messages, "NL_QUERY_UNAVAILABLE")
        assert hasattr(messages, "NL_QUERY_ERROR")
        assert isinstance(messages.NL_QUERY_UNAVAILABLE, str)
        assert isinstance(messages.NL_QUERY_ERROR, str)

    def test_analysis_result_messages_defined(self):
        """Test that analysis result messages are defined."""
        assert hasattr(messages, "ANALYSIS_RUNNING")
        assert hasattr(messages, "UNDERSTANDING_QUESTION")
        assert isinstance(messages.ANALYSIS_RUNNING, str)
        assert isinstance(messages.UNDERSTANDING_QUESTION, str)

    def test_messages_are_non_empty(self):
        """Test that all message constants are non-empty strings."""
        message_attrs = [
            "LOW_CONFIDENCE_WARNING",
            "SEMANTIC_LAYER_NOT_READY",
            "COLLISION_SUGGESTION_WARNING",
            "CONFIRM_AND_RUN",
            "START_OVER",
            "CLEAR_RESULTS",
            "RESULTS_CLEARED",
            "NO_DATASETS_AVAILABLE",
            "NL_QUERY_UNAVAILABLE",
            "NL_QUERY_ERROR",
            "ANALYSIS_RUNNING",
            "UNDERSTANDING_QUESTION",
        ]

        for attr in message_attrs:
            value = getattr(messages, attr)
            assert isinstance(value, str), f"{attr} should be a string"
            assert len(value) > 0, f"{attr} should not be empty"
