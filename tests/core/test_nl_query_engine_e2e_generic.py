"""Generic end-to-end tests for NL query engine.

Tests verify the complete flow from query parsing through variable extraction
to execution readiness, with logging verification at each step.

These tests are generic and extensible - they test patterns, not specific datasets.
"""

from unittest.mock import patch

from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.ui.components.question_engine import AnalysisContext, QuestionEngine


class TestE2EQueryParsingAndExecution:
    """Generic E2E tests for query parsing and execution readiness."""

    def test_e2e_query_parses_with_grouping_and_metric(self, mock_semantic_layer):
        """Test that queries with grouping and metric variables parse correctly."""
        # Arrange: Create generic semantic layer with grouping and metric columns
        grouping_col = "Treatment Group"
        metric_col = "Outcome Score"
        mock = mock_semantic_layer(
            columns={
                "treatment": grouping_col,
                "treatment_group": grouping_col,
                "outcome": metric_col,
                "outcome_score": metric_col,
                "score": metric_col,
            }
        )
        mock.get_available_dimensions.return_value = {grouping_col: ["Control", "Intervention"]}
        mock.get_data_quality_warnings.return_value = []

        query = f"which {grouping_col} had the lowest {metric_col}"
        engine = NLQueryEngine(mock)

        # Act: Parse query with logging verification
        with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
            intent = engine.parse_query(query)

            # Assert: Verify logging happened
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("query_parse_start" in str(call) for call in log_calls), "Should log query parse start"

            # Verify pattern matching was attempted and preserved
            assert intent is not None, "Query should parse successfully"
            assert (
                intent.intent_type == "COMPARE_GROUPS"
            ), f"Should be COMPARE_GROUPS (pattern match preserved), got {intent.intent_type}"
            assert intent.confidence >= 0.85, f"Should have confidence >= 0.85, got {intent.confidence}"
            assert intent.parsing_tier == "pattern_match", "Should be marked as pattern_match tier"

            # Verify variables were extracted (post-processing)
            assert intent.primary_variable is not None, "Primary variable should be extracted"
            assert intent.grouping_variable is not None, "Grouping variable should be extracted"

            # Verify variables match actual columns (generic check)
            alias_index = mock.get_column_alias_index()
            assert (
                intent.primary_variable in alias_index.values()
            ), f"Primary variable '{intent.primary_variable}' should be a valid column"
            assert (
                intent.grouping_variable in alias_index.values()
            ), f"Grouping variable '{intent.grouping_variable}' should be a valid column"

            # Verify confidence is sufficient for auto-execution
            assert intent.confidence >= 0.75, f"Confidence {intent.confidence} should be >= 0.75 for auto-execution"

            # Verify logging for variable extraction
            log_info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("variables_extracted_post_parse" in str(call) for call in log_info_calls) or any(
                "query_parse_success" in str(call) for call in log_info_calls
            ), "Should log variable extraction or parse success"

    def test_e2e_query_converts_to_analysis_context(self, mock_semantic_layer):
        """Test that parsed query converts to AnalysisContext with all required fields."""
        # Arrange: Create generic semantic layer
        grouping_col = "Treatment Group"
        metric_col = "Outcome Score"
        mock = mock_semantic_layer(
            columns={
                "treatment": grouping_col,
                "treatment_group": grouping_col,
                "outcome": metric_col,
                "outcome_score": metric_col,
            }
        )
        mock.get_available_dimensions.return_value = {grouping_col: ["Control", "Intervention"]}
        mock.get_data_quality_warnings.return_value = []

        query = f"which {grouping_col} had the lowest {metric_col}"

        # Act: Full flow through QuestionEngine
        with patch("streamlit.text_input", return_value=query):
            with patch("streamlit.markdown"):
                with patch("streamlit.expander"):
                    with patch("streamlit.radio", return_value="Yes, that's correct"):
                        with patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", False):
                            with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", False):
                                with patch("streamlit.spinner"):
                                    context = QuestionEngine.ask_free_form_question(mock)

                                    # Assert: Context should be complete
                                    assert context is not None, "Should return AnalysisContext"
                                    assert (
                                        context.inferred_intent.value == "compare_groups"
                                    ), "Should be COMPARE_GROUPS intent"
                                    assert context.primary_variable is not None, "Primary variable should be set"
                                    assert context.grouping_variable is not None, "Grouping variable should be set"
                                    assert (
                                        context.confidence >= 0.75
                                    ), f"Confidence {context.confidence} should be >= 0.75"
                                    assert (
                                        context.is_complete_for_intent()
                                    ), "Context should be complete for COMPARE_GROUPS analysis"

    def test_e2e_query_logging_throughout_process(self, mock_semantic_layer):
        """Test that logging happens at all key steps during query parsing."""
        # Arrange: Create generic semantic layer
        grouping_col = "Treatment Group"
        metric_col = "Outcome Score"
        mock = mock_semantic_layer(
            columns={
                "treatment": grouping_col,
                "treatment_group": grouping_col,
                "outcome": metric_col,
                "outcome_score": metric_col,
            }
        )
        mock.get_available_dimensions.return_value = {grouping_col: ["Control", "Intervention"]}
        mock.get_data_quality_warnings.return_value = []

        query = f"which {grouping_col} had the lowest {metric_col}"
        engine = NLQueryEngine(mock)

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
            assert any(
                "pattern_match" in msg.lower() or "query_parse_success" in msg for msg in log_messages
            ), "Should log pattern matching or parse success"

            # Should log variable extraction if variables were extracted
            if intent.primary_variable or intent.grouping_variable:
                assert any(
                    "variables_extracted_post_parse" in msg or "query_parse_success" in msg for msg in log_messages
                ), "Should log variable extraction when variables are found"

    def test_e2e_query_execution_readiness(self, mock_semantic_layer):
        """Test that parsed query is ready for execution (confidence and completeness)."""
        # Arrange: Create generic semantic layer
        grouping_col = "Treatment Group"
        metric_col = "Outcome Score"
        mock = mock_semantic_layer(
            columns={
                "treatment": grouping_col,
                "treatment_group": grouping_col,
                "outcome": metric_col,
                "outcome_score": metric_col,
            }
        )
        mock.get_available_dimensions.return_value = {grouping_col: ["Control", "Intervention"]}
        mock.get_data_quality_warnings.return_value = []

        query = f"which {grouping_col} had the lowest {metric_col}"
        engine = NLQueryEngine(mock)

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

        assert (
            context.confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD
        ), f"Confidence {context.confidence} should be >= threshold {AUTO_EXECUTE_CONFIDENCE_THRESHOLD}"
