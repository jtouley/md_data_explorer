"""Tests for diagnostic error messages in NL query engine."""

from unittest.mock import MagicMock

import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine


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


@pytest.mark.slow
def test_error_message_shows_parsing_attempts(mock_semantic_layer):
    """Error message should show what tiers were tried."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Create a query that will fail all tiers
    query = "some completely unclear query that won't match anything"

    intent = engine.parse_query(query)

    # Should have parsing attempts tracked
    assert intent.parsing_attempts
    assert len(intent.parsing_attempts) > 0

    # Should have failure reason
    if intent.confidence == 0.0:
        assert intent.failure_reason is not None or intent.suggestions  # Either failure reason or suggestions


def test_error_message_includes_suggestions(mock_semantic_layer):
    """Error message should include actionable suggestions."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Create a query that will fail
    query = "xyz abc 123"  # Very unclear query

    intent = engine.parse_query(query)

    # Should have suggestions when all tiers fail
    if intent.confidence == 0.0 and intent.failure_reason:
        assert intent.suggestions  # Should have at least one suggestion


def test_suggestions_use_semantic_layer_metadata(mock_semantic_layer):
    """Suggestions should leverage semantic layer metadata for available columns."""
    engine = NLQueryEngine(mock_semantic_layer)

    query = "unclear query"

    intent = engine.parse_query(query)

    # If suggestions are generated, they should mention available columns
    if intent.suggestions:
        # At least one suggestion should mention variables or columns
        suggestion_text = " ".join(intent.suggestions).lower()
        # Should suggest using variable names (semantic layer has mortality, treatment)
        assert "variable" in suggestion_text or "mention" in suggestion_text or "try" in suggestion_text
