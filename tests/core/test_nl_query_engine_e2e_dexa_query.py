"""End-to-end test for DEXA dataset query: 'which Current Regimen had the lowest viral load'.

This test verifies the complete flow from query parsing through variable extraction
to execution readiness, with logging verification at each step.
"""

from unittest.mock import MagicMock, patch

import pytest

from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.ui.components.question_engine import AnalysisContext, QuestionEngine


@pytest.fixture
def mock_dexa_semantic_layer():
    """Create a mock semantic layer matching the actual DEXA dataset structure."""
    mock = MagicMock()

    # Long column names (typical of Excel-style data with embedded legends)
    col_dexa_scan = "Had DEXA Scan?            Yes: 1 No: 2"
    col_dexa_results = "Results of DEXA?  1: Normal                  2: Osteopenia                 3: Osteoporosis"
    col_prior_tenofovir = "Prior Tenofovir TDF) use?                          1: Yes    2: No           3: Unknown"
    col_regimen = (
        "Current Regimen     1: Biktarvy                   2: Symtuza                 "
        "3: Triumeq                 4: Odefsey                  5: Dovato                      "
        "6: Juluca                         7: combination"
    )

    # Real column names from DEXA dataset (from logs)
    real_columns = {
        "Race": "Race",
        "Gender": "Gender",
        "Age": "Age",
        col_dexa_scan: col_dexa_scan,
        col_dexa_results: col_dexa_results,
        "DEXA Score          (T score)": "DEXA Score          (T score)",
        "DEXA Score          (Z score)": "DEXA Score          (Z score)",
        "CD4 Count": "CD4 Count",
        "Viral Load": "Viral Load",
        col_prior_tenofovir: col_prior_tenofovir,
        col_regimen: col_regimen,
    }

    # Build alias index with normalized keys
    alias_index = {}
    for canonical_name in real_columns.values():
        # Add multiple aliases for each column
        normalized = (
            canonical_name.lower().replace(" ", "_").replace("?", "").replace(":", "").replace("(", "").replace(")", "")
        )
        alias_index[normalized] = canonical_name

        # Add common aliases
        if "current regimen" in canonical_name.lower():
            alias_index["current_regimen"] = canonical_name
            alias_index["regimen"] = canonical_name
            alias_index["treatment"] = canonical_name
        if "viral load" in canonical_name.lower():
            alias_index["viral_load"] = canonical_name
            alias_index["viral"] = canonical_name
        if "cd4" in canonical_name.lower():
            alias_index["cd4"] = canonical_name
            alias_index["cd4_count"] = canonical_name

    mock.get_column_alias_index.return_value = alias_index
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock.get_available_dimensions.return_value = {
        "Current Regimen": ["Biktarvy", "Symtuza", "Triumeq", "Odefsey", "Dovato", "Juluca", "combination"]
    }
    mock.get_data_quality_warnings.return_value = []

    # Normalize function that matches semantic layer behavior
    def normalize_alias(alias: str) -> str:
        """Normalize alias to match index keys."""
        return alias.lower().replace(" ", "_").replace("?", "").replace(":", "").replace("(", "").replace(")", "")

    mock._normalize_alias = normalize_alias
    return mock


def test_e2e_dexa_query_which_regimen_lowest_viral_load(mock_dexa_semantic_layer):
    """End-to-end test: 'which Current Regimen had the lowest viral load' parses correctly."""

    # Arrange: Set up query and engine
    query = "which Current Regimen had the lowest viral load"
    engine = NLQueryEngine(mock_dexa_semantic_layer)

    # Act: Parse query with logging verification
    with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
        intent = engine.parse_query(query)

        # Assert: Verify logging happened
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("query_parse_start" in str(call) for call in log_calls), "Should log query parse start"

        # Verify pattern matching was attempted and preserved
        assert intent is not None, "Query should parse successfully"
        assert intent.intent_type == "COMPARE_GROUPS", (
            f"Should be COMPARE_GROUPS (pattern match preserved), got {intent.intent_type}"
        )
        # Pattern match may return 0.85 confidence (below 0.9 threshold) if fuzzy matching fails
        # or 0.95 if both variables match. Either is valid.
        assert intent.confidence >= 0.85, f"Should have confidence >= 0.85, got {intent.confidence}"
        assert intent.parsing_tier == "pattern_match", "Should be marked as pattern_match tier"

        # Verify variables were extracted (post-processing)
        assert intent.primary_variable is not None, "Primary variable (viral load) should be extracted"
        assert intent.grouping_variable is not None, "Grouping variable (Current Regimen) should be extracted"

        # Verify variables match actual columns
        alias_index = mock_dexa_semantic_layer.get_column_alias_index()
        assert intent.primary_variable in alias_index.values(), (
            f"Primary variable '{intent.primary_variable}' should be a valid column"
        )
        assert intent.grouping_variable in alias_index.values(), (
            f"Grouping variable '{intent.grouping_variable}' should be a valid column"
        )

        # Verify confidence is sufficient for auto-execution
        assert intent.confidence >= 0.75, f"Confidence {intent.confidence} should be >= 0.75 for auto-execution"

        # Verify logging for variable extraction
        log_info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("variables_extracted_post_parse" in str(call) for call in log_info_calls) or any(
            "query_parse_success" in str(call) for call in log_info_calls
        ), "Should log variable extraction or parse success"


def test_e2e_dexa_query_to_analysis_context(mock_dexa_semantic_layer):
    """End-to-end test: Query should convert to AnalysisContext with all required fields."""

    # Arrange
    query = "which Current Regimen had the lowest viral load"

    # Act: Full flow through QuestionEngine
    with patch("streamlit.text_input", return_value=query):
        with patch("streamlit.markdown"):
            with patch("streamlit.expander"):
                with patch("streamlit.radio", return_value="Yes, that's correct"):
                    with patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", False):
                        with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", False):
                            with patch("streamlit.spinner"):
                                context = QuestionEngine.ask_free_form_question(mock_dexa_semantic_layer)

                                # Assert: Context should be complete
                                assert context is not None, "Should return AnalysisContext"
                                assert context.inferred_intent.value == "compare_groups", (
                                    "Should be COMPARE_GROUPS intent"
                                )
                                assert context.primary_variable is not None, "Primary variable should be set"
                                assert context.grouping_variable is not None, "Grouping variable should be set"
                                assert context.confidence >= 0.75, f"Confidence {context.confidence} should be >= 0.75"
                                assert context.is_complete_for_intent(), (
                                    "Context should be complete for COMPARE_GROUPS analysis"
                                )


def test_e2e_dexa_query_logging_throughout_process(mock_dexa_semantic_layer):
    """End-to-end test: Verify logging happens at all key steps."""

    query = "which Current Regimen had the lowest viral load"
    engine = NLQueryEngine(mock_dexa_semantic_layer)

    # Capture all log calls
    with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
        intent = engine.parse_query(query)

        # Verify logging at each tier
        all_log_calls = []
        all_log_calls.extend(mock_logger.debug.call_args_list)
        all_log_calls.extend(mock_logger.info.call_args_list)
        all_log_calls.extend(mock_logger.warning.call_args_list)

        log_messages = [str(call) for call in all_log_calls]

        # Should log query parse start
        assert any("query_parse_start" in msg for msg in log_messages), "Should log query parse start"

        # Should log pattern matching attempt (debug or info)
        assert any("pattern_match" in msg.lower() or "query_parse_success" in msg for msg in log_messages), (
            "Should log pattern matching or parse success"
        )

        # Should log variable extraction if variables were extracted
        if intent.primary_variable or intent.grouping_variable:
            assert any(
                "variables_extracted_post_parse" in msg or "query_parse_success" in msg for msg in log_messages
            ), "Should log variable extraction when variables are found"


def test_e2e_dexa_query_variable_matching_accuracy(mock_dexa_semantic_layer):
    """End-to-end test: Verify variables are matched correctly to actual column names."""

    query = "which Current Regimen had the lowest viral load"
    engine = NLQueryEngine(mock_dexa_semantic_layer)

    intent = engine.parse_query(query)

    # Verify primary variable matches "Viral Load"
    assert intent.primary_variable == "Viral Load", (
        f"Primary variable should be 'Viral Load', got '{intent.primary_variable}'"
    )

    # Verify grouping variable matches "Current Regimen" column
    expected_regimen_col = (
        "Current Regimen     1: Biktarvy                   2: Symtuza                 "
        "3: Triumeq                 4: Odefsey                  5: Dovato                      "
        "6: Juluca                         7: combination"
    )
    assert intent.grouping_variable == expected_regimen_col, (
        f"Grouping variable should match Current Regimen column, got '{intent.grouping_variable}'"
    )


def test_e2e_dexa_query_execution_readiness(mock_dexa_semantic_layer):
    """End-to-end test: Verify query is ready for execution (confidence and completeness)."""

    query = "which Current Regimen had the lowest viral load"
    engine = NLQueryEngine(mock_dexa_semantic_layer)

    intent = engine.parse_query(query)

    # Verify intent is execution-ready
    assert intent.intent_type == "COMPARE_GROUPS", "Should be COMPARE_GROUPS"
    assert intent.primary_variable is not None, "Primary variable required for execution"
    assert intent.grouping_variable is not None, "Grouping variable required for execution"
    assert intent.confidence >= 0.75, f"Confidence {intent.confidence} should be >= 0.75 for auto-execution"

    # Convert to AnalysisContext and verify completeness
    context = AnalysisContext()
    context.inferred_intent = type(context.inferred_intent)("compare_groups")
    context.primary_variable = intent.primary_variable
    context.grouping_variable = intent.grouping_variable
    context.confidence = intent.confidence

    assert context.is_complete_for_intent(), "Context should be complete for COMPARE_GROUPS execution"

    # Verify confidence threshold for auto-execution
    from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD

    assert context.confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD, (
        f"Confidence {context.confidence} should be >= threshold {AUTO_EXECUTE_CONFIDENCE_THRESHOLD}"
    )
