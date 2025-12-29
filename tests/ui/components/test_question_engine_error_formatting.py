"""Tests for error message formatting in question engine."""

from clinical_analytics.core.nl_query_engine import QueryIntent
from clinical_analytics.ui.components.question_engine import QuestionEngine


def test_format_diagnostic_error_shows_suggestions():
    """_format_diagnostic_error should show suggestions when available."""
    intent = QueryIntent(
        intent_type="DESCRIBE",
        confidence=0.0,
        failure_reason="All parsing tiers failed",
        suggestions=["Try mentioning specific variable names", "Use phrases like 'compare X by Y'"],
    )

    formatted = QuestionEngine._format_diagnostic_error(intent)

    # Should include suggestions
    assert "Suggestions" in formatted or "💡" in formatted
    assert "variable names" in formatted or "Try mentioning" in formatted


def test_format_diagnostic_error_shows_parsing_attempts():
    """_format_diagnostic_error should show what tiers were tried."""
    intent = QueryIntent(
        intent_type="DESCRIBE",
        confidence=0.0,
        parsing_attempts=[
            {"tier": "pattern_match", "result": "failed", "confidence": 0.0},
            {"tier": "semantic_match", "result": "failed", "confidence": 0.0},
            {"tier": "llm_fallback", "result": "failed", "confidence": 0.0},
        ],
        failure_reason="All parsing tiers failed",
    )

    formatted = QuestionEngine._format_diagnostic_error(intent)

    # Should mention what was tried
    assert "tried" in formatted.lower() or "pattern" in formatted.lower() or "semantic" in formatted.lower()


def test_format_diagnostic_error_shows_failure_reason():
    """_format_diagnostic_error should show failure reason."""
    intent = QueryIntent(
        intent_type="DESCRIBE",
        confidence=0.0,
        failure_reason="Query too ambiguous",
    )

    formatted = QuestionEngine._format_diagnostic_error(intent)

    # Should include failure reason
    assert "ambiguous" in formatted.lower() or "reason" in formatted.lower()
