"""Tests for pattern match fallback behavior when below threshold.

When pattern matching returns a result below TIER_1_PATTERN_MATCH_THRESHOLD (0.9),
but semantic match returns something worse (e.g., DESCRIBE with low confidence),
we should preserve the pattern match result.
"""

import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent


@pytest.fixture
def mock_semantic_layer():
    """Create a mock semantic layer for testing."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    # Long column name (typical of Excel-style data with embedded legends)
    regimen_col = (
        "Current Regimen     1: Biktarvy                   2: Symtuza                 "
        "3: Triumeq                 4: Odefsey                  5: Dovato                      "
        "6: Juluca                         7: combination"
    )
    mock.get_column_alias_index.return_value = {
        "viral_load": "Viral Load",
        "viral": "Viral Load",
        "current_regimen": regimen_col,
        "regimen": regimen_col,
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")
    return mock


def test_parse_query_preserves_pattern_match_below_threshold(mock_semantic_layer):
    """parse_query() should preserve pattern match result even if below threshold, if semantic match is worse."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Query that matches pattern but fuzzy matching fails for grouping variable
    query = "which Current Regimen had the lowest viral load"

    # Mock fuzzy matching to fail for "current regimen" but succeed for "viral load"
    # This will cause pattern match to return 0.85 confidence (below 0.9 threshold)
    original_fuzzy = engine._fuzzy_match_variable

    def mock_fuzzy_match(term: str):
        if "regimen" in term.lower():
            return None, 0.0, None  # Fail for regimen
        elif "viral" in term.lower():
            return ("Viral Load", 0.9, None)  # Succeed for viral load
        return original_fuzzy(term)

    engine._fuzzy_match_variable = mock_fuzzy_match

    # Pattern match will return COMPARE_GROUPS with 0.85 confidence (below 0.9 threshold)
    # Semantic match will return DESCRIBE with 0.3 confidence
    # We should preserve the COMPARE_GROUPS result

    intent = engine.parse_query(query)

    # Should be COMPARE_GROUPS, not DESCRIBE
    assert (
        intent.intent_type == "COMPARE_GROUPS"
    ), f"Should preserve COMPARE_GROUPS from pattern match, got {intent.intent_type}"
    assert intent.confidence == 0.85, f"Should preserve pattern match confidence 0.85, got {intent.confidence}"
    assert intent.parsing_tier == "pattern_match", "Should mark as pattern_match tier"


def test_parse_query_uses_semantic_if_better_than_pattern(mock_semantic_layer):
    """parse_query() should use semantic match if it meets threshold, even if pattern match is below threshold."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Mock fuzzy matching to fail so pattern match returns 0.85
    original_fuzzy = engine._fuzzy_match_variable

    def mock_fuzzy_match(term: str):
        if "regimen" in term.lower():
            return None, 0.0, None  # Fail for regimen
        elif "viral" in term.lower():
            return ("Viral Load", 0.9, None)  # Succeed for viral load
        return original_fuzzy(term)

    engine._fuzzy_match_variable = mock_fuzzy_match

    # Mock semantic match to return a good result that meets threshold
    def mock_semantic_match(query: str):
        return QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.8, primary_variable="viral_load")

    engine._semantic_match = mock_semantic_match

    query = "which Current Regimen had the lowest viral load"
    intent = engine.parse_query(query)

    # Should use semantic match if it meets threshold (0.75)
    assert intent.intent_type == "COMPARE_GROUPS"
    assert intent.confidence == 0.8
    assert intent.parsing_tier == "semantic_match"


def test_parse_query_pattern_below_threshold_vs_semantic_describe(mock_semantic_layer):
    """parse_query() should prefer pattern match COMPARE_GROUPS (0.85) over semantic DESCRIBE (0.3)."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Mock fuzzy matching to fail for regimen so pattern match returns 0.85
    original_fuzzy = engine._fuzzy_match_variable

    def mock_fuzzy_match(term: str):
        if "regimen" in term.lower():
            return None, 0.0, None  # Fail for regimen
        elif "viral" in term.lower():
            return ("Viral Load", 0.9, None)  # Succeed for viral load
        return original_fuzzy(term)

    engine._fuzzy_match_variable = mock_fuzzy_match

    query = "which Current Regimen had the lowest viral load"

    # Pattern match: COMPARE_GROUPS, 0.85 (below 0.9 threshold)
    # Semantic match: DESCRIBE, 0.3 (below 0.75 threshold)
    # Should use pattern match

    intent = engine.parse_query(query)

    assert intent.intent_type == "COMPARE_GROUPS", "Should prefer pattern match COMPARE_GROUPS over semantic DESCRIBE"
    assert intent.confidence == 0.85, f"Should use pattern match confidence 0.85, got {intent.confidence}"
    assert intent.parsing_tier == "pattern_match", "Should be marked as pattern_match"


def test_parse_query_logs_partial_pattern_match(mock_semantic_layer):
    """parse_query() should log when using pattern match below threshold."""
    from unittest.mock import patch

    engine = NLQueryEngine(mock_semantic_layer)

    # Mock fuzzy matching to fail for regimen so pattern match returns 0.85
    original_fuzzy = engine._fuzzy_match_variable

    def mock_fuzzy_match(term: str):
        if "regimen" in term.lower():
            return None, 0.0, None  # Fail for regimen
        elif "viral" in term.lower():
            return ("Viral Load", 0.9, None)  # Succeed for viral load
        return original_fuzzy(term)

    engine._fuzzy_match_variable = mock_fuzzy_match

    query = "which Current Regimen had the lowest viral load"

    with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
        _ = engine.parse_query(query)  # Result checked via logs

        # Should log query_parse_partial_pattern_match
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any(
            "query_parse_partial_pattern_match" in str(call) for call in log_calls
        ), "Should log partial pattern match when using pattern result below threshold"
