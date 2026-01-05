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

    # Test behavior, not implementation details
    with patch("streamlit.status") as mock_status:
        mock_status.return_value.__enter__.return_value.update = MagicMock()
        intent = QuestionEngine._show_progressive_feedback(mock_nl_engine, query)

        # Assert behavior: correct intent returned with diagnostics
        assert intent is not None
        assert intent.parsing_tier in ["pattern_match", "semantic_match", "llm_fallback"]
        assert intent.parsing_attempts  # Should have at least one attempt
        assert intent.confidence > 0.0
        assert intent.intent_type == "COMPARE_GROUPS"  # Verify correct parsing


def test_progressive_feedback_tracks_all_attempts(mock_nl_engine):
    """When pattern match fails, semantic match should be attempted and tracked."""
    from clinical_analytics.ui.components.question_engine import QuestionEngine

    # Create a query that pattern match will fail but semantic might succeed
    query = "show me differences in outcomes across treatment groups"

    # Mock the LLM tier to prevent Ollama timeout in tests
    mock_llm_intent = QueryIntent(
        intent_type="COMPARE_GROUPS",
        primary_variable="outcomes",
        grouping_variable="treatment",
        confidence=0.7,
        parsing_tier="llm_fallback",
        parsing_attempts=[],
    )

    with (
        patch("streamlit.status") as mock_status,
        patch.object(mock_nl_engine, "_llm_parse", return_value=mock_llm_intent),
    ):
        mock_status.return_value.__enter__.return_value.update = MagicMock()
        intent = QuestionEngine._show_progressive_feedback(mock_nl_engine, query)

        # Test behavior: all attempts tracked, correct tier selected
        assert intent is not None
        assert len(intent.parsing_attempts) > 0
        tier_names = [attempt.get("tier") for attempt in intent.parsing_attempts]
        assert "pattern_match" in tier_names
        # Verify correct tier was selected based on confidence
        assert intent.parsing_tier in tier_names


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
    from clinical_analytics.core.nl_query_engine import QueryIntent
    from clinical_analytics.ui.components.question_engine import QuestionEngine

    query = "compare mortality by treatment"

    # Mock parse_query to return a QueryIntent when feature flag is disabled
    mock_intent = QueryIntent(
        intent_type="COMPARE_GROUPS",
        primary_variable="mortality",
        grouping_variable="treatment",
        confidence=0.8,
        parsing_tier="pattern_match",
        parsing_attempts=[{"tier": "pattern_match", "confidence": 0.8}],
    )

    # When feature flag is disabled, should fall back to simple parsing
    with (
        patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", False),
        patch.object(mock_nl_engine, "parse_query", return_value=mock_intent) as mock_parse,
    ):
        intent = QuestionEngine._show_progressive_feedback(mock_nl_engine, query)
        # Should call parse_query when feature flag is disabled
        mock_parse.assert_called_once_with(query)
        # Should return the intent from parse_query
        assert intent is not None
        assert intent == mock_intent
