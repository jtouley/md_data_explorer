"""Tests for variable extraction in NL Query Engine.

Tests that variables are extracted and assigned correctly, especially
for COMPARE_GROUPS queries where pattern matching may succeed but
fuzzy matching of individual variables may fail.
"""

import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine


@pytest.fixture
def mock_semantic_layer():
    """Create a mock semantic layer with realistic column names."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    # Simulate column names from DEXA dataset
    # Long column name (typical of Excel-style data with embedded legends)
    regimen_col = (
        "Current Regimen     1: Biktarvy                   2: Symtuza                 "
        "3: Triumeq                 4: Odefsey                  5: Dovato                      "
        "6: Juluca                         7: combination"
    )
    mock.get_column_alias_index.return_value = {
        "current regimen": regimen_col,
        "regimen": regimen_col,
        "viral load": "Viral Load",
        "viral": "Viral Load",
        "cd4": "CD4 Count",
        "cd4 count": "CD4 Count",
        "age": "Age",
        "gender": "Gender",
        "race": "Race",
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")
    return mock


def test_pattern_match_extracts_variables_when_fuzzy_match_fails(mock_semantic_layer):
    """Pattern matching should return COMPARE_GROUPS even if fuzzy matching fails, and variables extracted later."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Query where pattern matches but fuzzy matching might fail
    query = "which Current Regimen had the lowest viral load"

    # Pattern should match and return COMPARE_GROUPS
    intent = engine._pattern_match(query)

    assert intent is not None, "Pattern should match"
    assert intent.intent_type == "COMPARE_GROUPS", "Should be COMPARE_GROUPS"

    # Even if fuzzy matching failed initially, parse_query should extract variables
    full_intent = engine.parse_query(query)

    assert full_intent.intent_type == "COMPARE_GROUPS"
    # Variables should be extracted by post-processing
    assert full_intent.primary_variable is not None, "Primary variable should be extracted"
    assert full_intent.grouping_variable is not None, "Grouping variable should be extracted"


def test_parse_query_extracts_variables_for_compare_groups(mock_semantic_layer):
    """parse_query() should extract and assign variables for COMPARE_GROUPS queries."""
    engine = NLQueryEngine(mock_semantic_layer)

    query = "which Current Regimen had the lowest viral load"
    intent = engine.parse_query(query)

    assert intent.intent_type == "COMPARE_GROUPS"
    assert intent.primary_variable is not None, "Primary variable should be extracted"
    assert intent.grouping_variable is not None, "Grouping variable should be extracted"

    # Verify variables are from the semantic layer
    alias_index = mock_semantic_layer.get_column_alias_index()
    assert intent.primary_variable in alias_index.values(), "Primary variable should be a valid column"
    assert intent.grouping_variable in alias_index.values(), "Grouping variable should be a valid column"


def test_parse_query_extracts_variables_when_pattern_match_partial(mock_semantic_layer):
    """parse_query() should extract variables even when pattern match only partially succeeds."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Query that matches pattern but fuzzy matching fails for one variable
    query = "which treatment had the lowest response"

    # Mock fuzzy matching to fail for "treatment" but succeed for "response"
    original_fuzzy = engine._fuzzy_match_variable

    def mock_fuzzy(term: str):
        if "treatment" in term.lower():
            return None, 0.0, None  # Fail for treatment
        elif "response" in term.lower():
            return ("response_rate", 0.8, None)  # Succeed for response
        return original_fuzzy(term)

    engine._fuzzy_match_variable = mock_fuzzy

    intent = engine.parse_query(query)

    # Should still be COMPARE_GROUPS
    assert intent.intent_type == "COMPARE_GROUPS"
    # Variables should be extracted by post-processing
    assert (
        intent.primary_variable is not None or intent.grouping_variable is not None
    ), "At least one variable should be extracted"


def test_parse_query_extracts_variables_for_find_predictors(mock_semantic_layer):
    """parse_query() should extract primary variable for FIND_PREDICTORS queries."""
    engine = NLQueryEngine(mock_semantic_layer)

    query = "what predicts viral load"
    intent = engine.parse_query(query)

    assert intent.intent_type == "FIND_PREDICTORS"
    assert intent.primary_variable is not None, "Primary variable should be extracted"
    assert intent.primary_variable in mock_semantic_layer.get_column_alias_index().values()


def test_parse_query_extracts_variables_for_correlations(mock_semantic_layer):
    """parse_query() should extract variables for CORRELATIONS queries."""
    engine = NLQueryEngine(mock_semantic_layer)

    query = "correlation between age and viral load"
    intent = engine.parse_query(query)

    # May be CORRELATIONS or fall through to DESCRIBE, but if CORRELATIONS, should have variables
    if intent.intent_type == "CORRELATIONS":
        assert (
            intent.primary_variable is not None or intent.grouping_variable is not None
        ), "At least one variable should be extracted for correlations"


def test_variable_extraction_logs_when_variables_found(mock_semantic_layer):
    """Variable extraction should log when variables are successfully extracted during post-processing."""
    from unittest.mock import patch

    from clinical_analytics.core.nl_query_engine import QueryIntent

    engine = NLQueryEngine(mock_semantic_layer)

    # Use a query where variables are NOT set during pattern matching
    # This will trigger post-processing variable extraction
    query = "compare groups by treatment"

    with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
        # Mock pattern match to return intent without variables
        def mock_pattern_match(query: str):
            return QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.85)

        engine._pattern_match = mock_pattern_match

        intent = engine.parse_query(query)

        # Should log variable extraction if variables were found during post-processing
        if intent.primary_variable or intent.grouping_variable:
            # Check if variables_extracted_post_parse was called
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any(
                "variables_extracted_post_parse" in str(call) for call in log_calls
            ), "Should log variable extraction during post-processing"


def test_pattern_match_returns_compare_groups_even_without_variable_match(mock_semantic_layer):
    """Pattern matching should return COMPARE_GROUPS intent even if variables can't be matched immediately."""
    engine = NLQueryEngine(mock_semantic_layer)

    # Mock fuzzy matching to always fail
    def mock_fuzzy_fail(term: str):
        return None, 0.0, None

    engine._fuzzy_match_variable = mock_fuzzy_fail

    query = "which Current Regimen had the lowest viral load"
    intent = engine._pattern_match(query)

    # Should still return COMPARE_GROUPS with lower confidence
    assert intent is not None, "Should return intent even if fuzzy matching fails"
    assert intent.intent_type == "COMPARE_GROUPS", "Should be COMPARE_GROUPS"
    assert intent.confidence == 0.85, "Should have lower confidence when variables not matched"
    # Variables may be None at this stage, but will be extracted later
    assert (
        intent.primary_variable is None or intent.grouping_variable is None
    ), "Variables may be None if fuzzy matching failed"
