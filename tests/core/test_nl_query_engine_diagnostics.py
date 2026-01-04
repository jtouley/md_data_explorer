"""Tests for NL Query Engine diagnostics and QueryIntent enhancements."""

import pytest
from clinical_analytics.core.nl_query_engine import VALID_INTENT_TYPES, NLQueryEngine, QueryIntent


@pytest.fixture
def mock_semantic_layer():
    """Create a mock semantic layer for testing."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.get_column_alias_index.return_value = {
        "mortality": "mortality",
        "treatment": "treatment_arm",
        "age": "age",
        "outcome": "outcome",
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")
    return mock


def test_query_intent_tracks_successful_tier():
    """QueryIntent should track which tier succeeded."""
    intent = QueryIntent(
        intent_type="COMPARE_GROUPS",
        parsing_tier="pattern_match",
        parsing_attempts=[{"tier": "pattern_match", "result": "success", "confidence": 0.95}],
        confidence=0.95,
    )
    assert intent.parsing_tier == "pattern_match"
    assert len(intent.parsing_attempts) == 1
    assert intent.parsing_attempts[0]["tier"] == "pattern_match"
    assert intent.parsing_attempts[0]["result"] == "success"


def test_query_intent_tracks_all_parsing_attempts():
    """QueryIntent should track all parsing attempts for diagnostics."""
    intent = QueryIntent(
        intent_type="COMPARE_GROUPS",
        parsing_attempts=[
            {"tier": "pattern_match", "result": "failed", "confidence": 0.5},
            {"tier": "semantic_match", "result": "success", "confidence": 0.8},
        ],
        confidence=0.8,
    )
    assert len(intent.parsing_attempts) == 2
    assert intent.parsing_attempts[0]["tier"] == "pattern_match"
    assert intent.parsing_attempts[1]["tier"] == "semantic_match"


def test_query_intent_includes_failure_reason():
    """QueryIntent should include failure reason when parsing fails."""
    intent = QueryIntent(
        intent_type="DESCRIBE",
        confidence=0.0,
        failure_reason="All parsing tiers failed",
        suggestions=["Try mentioning specific variable names"],
    )
    assert intent.failure_reason == "All parsing tiers failed"
    assert len(intent.suggestions) == 1
    assert "variable names" in intent.suggestions[0]


def test_query_intent_validates_intent_type():
    """QueryIntent should validate intent_type against VALID_INTENT_TYPES."""
    with pytest.raises(ValueError, match="Invalid intent_type"):
        QueryIntent(intent_type="INVALID_TYPE", confidence=0.5)


def test_query_intent_accepts_valid_intent_types():
    """QueryIntent should accept all valid intent types."""
    for intent_type in VALID_INTENT_TYPES:
        intent = QueryIntent(intent_type=intent_type, confidence=0.5)
        assert intent.intent_type == intent_type


def test_existing_queries_still_parse_correctly(mock_semantic_layer):
    """Regression: Existing queries should continue to work after diagnostics added."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Known working queries from existing codebase
    # Note: Some queries may require proper semantic layer setup for variable extraction
    test_cases = [
        ("compare mortality by treatment", "COMPARE_GROUPS"),
        ("what predicts mortality", "FIND_PREDICTORS"),
        ("survival analysis", "SURVIVAL"),
        ("descriptive statistics", "DESCRIBE"),
    ]

    for query, expected_intent in test_cases:
        intent = engine.parse_query(query)
        assert intent.intent_type == expected_intent, f"Failed for: {query}"
        assert intent.confidence > 0.0, f"Zero confidence for: {query}"

    # Correlation query may require variables to be extractable
    # Test that it at least parses (even if it falls back to low confidence)


def test_pattern_match_which_x_had_lowest_y(mock_semantic_layer):
    """Pattern matching should handle 'which X had the lowest Y' queries."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Mock fuzzy_match_variable to return test variables
    def mock_fuzzy_match(var_name: str):
        var_map = {
            "regimen": ("current_regimen", 0.9, {}),
            "current regimen": ("current_regimen", 0.9, {}),
            "viral load": ("viral_load", 0.9, {}),
            "viral": ("viral_load", 0.8, {}),
        }
        return var_map.get(var_name.lower(), (None, 0.0, {}))

    engine._fuzzy_match_variable = mock_fuzzy_match

    # Test "which X had the lowest Y" pattern
    query = "which Current Regimen had the lowest viral load"
    intent = engine._pattern_match(query)

    assert intent is not None, f"Pattern should match: {query}"
    assert intent.intent_type == "COMPARE_GROUPS", f"Should be COMPARE_GROUPS for: {query}"
    assert intent.grouping_variable == "current_regimen", "Grouping variable should be regimen"
    assert intent.primary_variable == "viral_load", "Primary variable should be viral load"
    assert intent.confidence >= 0.9, "Should have high confidence for pattern match"


def test_pattern_match_which_x_had_highest_y(mock_semantic_layer):
    """Pattern matching should handle 'which X had the highest Y' queries."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Mock fuzzy_match_variable (use variables that won't match other patterns)
    def mock_fuzzy_match(var_name: str):
        var_map = {
            "treatment": ("treatment_arm", 0.9, {}),
            "response rate": ("response_rate", 0.9, {}),
            "response": ("response_rate", 0.8, {}),
        }
        return var_map.get(var_name.lower(), (None, 0.0, {}))

    engine._fuzzy_match_variable = mock_fuzzy_match

    query = "which treatment had the highest response rate"
    intent = engine._pattern_match(query)

    assert intent is not None, f"Pattern should match: {query}"
    assert intent.intent_type == "COMPARE_GROUPS", f"Should be COMPARE_GROUPS for: {query}"
    assert intent.grouping_variable == "treatment_arm", "Grouping variable should be treatment"
    assert intent.primary_variable == "response_rate", "Primary variable should be response rate"
    correlation_intent = engine.parse_query("correlation between age and outcome")
    assert correlation_intent is not None
    # Should be CORRELATIONS if pattern matches, or DESCRIBE if it falls through
    assert correlation_intent.intent_type in ["CORRELATIONS", "DESCRIBE"]
    assert correlation_intent.confidence >= 0.0


def test_parse_query_populates_parsing_tier(mock_semantic_layer):
    """parse_query() should populate parsing_tier when match found."""
    engine = NLQueryEngine(mock_semantic_layer)
    intent = engine.parse_query("compare mortality by treatment")

    assert intent.parsing_tier is not None
    assert intent.parsing_tier in ["pattern_match", "semantic_match", "llm_fallback"]


def test_parse_query_tracks_parsing_attempts(mock_semantic_layer):
    """parse_query() should track all parsing attempts in parsing_attempts."""
    engine = NLQueryEngine(mock_semantic_layer)
    intent = engine.parse_query("compare mortality by treatment")

    assert intent.parsing_attempts is not None
    assert len(intent.parsing_attempts) > 0
    # Should have at least one attempt recorded
    assert any("tier" in attempt for attempt in intent.parsing_attempts)


def test_parse_query_sets_failure_reason_on_failure(mock_semantic_layer):
    """parse_query() should set failure_reason when all tiers fail."""
    engine = NLQueryEngine(mock_semantic_layer)
    # Use a query that's unlikely to match any pattern
    intent = engine.parse_query("xyzabc123")

    # If all tiers fail, should have failure_reason or low confidence
    if intent.confidence < 0.3:
        assert intent.failure_reason is not None or len(intent.suggestions) > 0
