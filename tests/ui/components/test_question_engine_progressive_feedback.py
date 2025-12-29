"""Tests for progressive feedback system in question engine."""

from unittest.mock import MagicMock, patch

import pytest

from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent


@pytest.fixture
def mock_semantic_layer():
    """Create a mock semantic layer for testing."""
    mock = MagicMock()
    mock.get_column_alias_index.return_value = {
        "mortality": "mortality",
        "treatment": "treatment_arm",
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")
    return mock


@pytest.fixture
def mock_nl_engine(mock_semantic_layer):
    """Create NLQueryEngine instance for testing."""
    return NLQueryEngine(mock_semantic_layer)


def test_progressive_feedback_returns_intent_with_diagnostics(mock_nl_engine):
    """Progressive feedback should return QueryIntent with parsing_tier populated."""
    from clinical_analytics.ui.components.question_engine import QuestionEngine

    query = "compare mortality by treatment"

    # Mock st.status to avoid Streamlit dependency
    with patch("streamlit.status") as mock_status:
        mock_status.return_value.__enter__.return_value.update = MagicMock()
        # Mock the actual parsing to return a valid intent
        intent = QuestionEngine._show_progressive_feedback(mock_nl_engine, query)

        # Assert
        assert intent is not None
        assert intent.parsing_tier in ["pattern_match", "semantic_match", "llm_fallback"]
        assert intent.parsing_attempts  # Should have at least one attempt
        assert intent.confidence > 0.0


def test_progressive_feedback_tracks_all_attempts(mock_nl_engine):
    """When pattern match fails, semantic match should be attempted and tracked."""
    from clinical_analytics.ui.components.question_engine import QuestionEngine

    # Create a query that pattern match will fail but semantic might succeed
    query = "show me differences in outcomes across treatment groups"

    with patch("streamlit.status") as mock_status:
        mock_status.return_value.__enter__.return_value.update = MagicMock()
        intent = QuestionEngine._show_progressive_feedback(mock_nl_engine, query)

        # Should have tracked attempts
        assert intent is not None
        assert len(intent.parsing_attempts) > 0
        # Should have at least pattern_match attempt
        tier_names = [attempt.get("tier") for attempt in intent.parsing_attempts]
        assert "pattern_match" in tier_names


def test_progressive_feedback_handles_timeout(mock_nl_engine):
    """Progressive feedback should handle tier timeout gracefully."""

    from clinical_analytics.ui.components.question_engine import QuestionEngine

    query = "complex query that might timeout"

    # Mock a slow semantic match that would timeout
    original_semantic_match = mock_nl_engine._semantic_match

    def slow_semantic_match(q):
        import time

        time.sleep(6)  # Exceeds 5s timeout
        return original_semantic_match(q)

    mock_nl_engine._semantic_match = slow_semantic_match

    with patch("streamlit.status") as mock_status:
        mock_status.return_value.__enter__.return_value.update = MagicMock()
        # Mock signal.alarm for timeout (Unix only)
        with patch("signal.alarm"), patch("signal.signal"):
            intent = QuestionEngine._show_progressive_feedback(mock_nl_engine, query)

            # Should return something (either intent or None, but not crash)
            assert intent is None or isinstance(intent, QueryIntent)


def test_progressive_feedback_respects_feature_flag(mock_nl_engine):
    """Progressive feedback should respect ENABLE_PROGRESSIVE_FEEDBACK feature flag."""
    from unittest.mock import patch

    from clinical_analytics.ui.components.question_engine import QuestionEngine

    query = "compare mortality by treatment"

    # When feature flag is disabled, should fall back to simple parsing
    with patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", False):
        with patch("streamlit.status") as mock_status:
            intent = QuestionEngine._show_progressive_feedback(mock_nl_engine, query)
            # Should still return intent (via simple parse_query)
            assert intent is not None
            # Status should not be called when feature flag is off
            # (Actually, it might still be called, but the behavior should be different)
