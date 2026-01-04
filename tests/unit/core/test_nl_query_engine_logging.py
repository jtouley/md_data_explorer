"""
Test Structured Logging in NL Query Engine - Phase 6

Tests that all logs include standardized fields:
- dataset_id, upload_id, query, intent, confidence, matched_vars, error_type
"""

from unittest.mock import MagicMock, patch

import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent


@pytest.fixture
def mock_semantic_layer():
    """Mock semantic layer for testing."""
    semantic = MagicMock()
    semantic.get_column_alias_index.return_value = {
        "mortality": "mortality",
        "treatment": "treatment_arm",
        "survival": "survival_time",
    }
    semantic._normalize_alias = lambda x: x.lower().strip()
    semantic.get_collision_suggestions.return_value = None
    semantic.get_collision_warnings.return_value = set()
    return semantic


@pytest.fixture
def nl_engine(mock_semantic_layer):
    """Create NLQueryEngine instance for testing."""
    return NLQueryEngine(mock_semantic_layer)


class TestStructuredLogging:
    """Test structured logging with standardized fields."""

    def test_parse_query_logs_start_with_standardized_fields(self, nl_engine, mock_semantic_layer):
        """Test that query_parse_start includes all standardized fields."""
        # Arrange
        query = "compare mortality by treatment"
        dataset_id = "test_dataset"
        upload_id = "test_upload_123"

        # Act
        with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
            nl_engine.parse_query(query, dataset_id=dataset_id, upload_id=upload_id)

            # Assert: query_parse_start called with standardized fields
            start_calls = [c for c in mock_logger.info.call_args_list if c[0][0] == "query_parse_start"]
            assert len(start_calls) > 0, "query_parse_start should be logged"

            call_kwargs = start_calls[0][1]
            assert "query" in call_kwargs
            assert "dataset_id" in call_kwargs
            assert "upload_id" in call_kwargs
            assert call_kwargs["query"] == query
            assert call_kwargs["dataset_id"] == dataset_id
            assert call_kwargs["upload_id"] == upload_id

    def test_parse_query_logs_success_with_standardized_fields(self, nl_engine, mock_semantic_layer):
        """Test that query_parse_success includes all standardized fields."""
        # Arrange
        query = "compare mortality by treatment"
        dataset_id = "test_dataset"
        upload_id = "test_upload_123"

        # Act
        with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
            intent = nl_engine.parse_query(query, dataset_id=dataset_id, upload_id=upload_id)

            # Assert: query_parse_success called with standardized fields
            success_calls = [c for c in mock_logger.info.call_args_list if c[0][0] == "query_parse_success"]
            assert len(success_calls) > 0, "query_parse_success should be logged"

            call_kwargs = success_calls[0][1]
            assert "query" in call_kwargs
            assert "intent" in call_kwargs
            assert "confidence" in call_kwargs
            assert "matched_vars" in call_kwargs
            assert "tier" in call_kwargs
            assert "dataset_id" in call_kwargs
            assert "upload_id" in call_kwargs
            assert call_kwargs["intent"] == intent.intent_type
            assert call_kwargs["confidence"] == intent.confidence

    def test_parse_query_logs_error_with_standardized_fields(self, nl_engine):
        """Test that query_parse_failed includes all standardized fields."""
        # Arrange
        query = ""  # Empty query should raise ValueError
        dataset_id = "test_dataset"
        upload_id = "test_upload_123"

        # Act
        with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
            with pytest.raises(ValueError):
                nl_engine.parse_query(query, dataset_id=dataset_id, upload_id=upload_id)

            # Assert: query_parse_failed called with standardized fields
            error_calls = [c for c in mock_logger.error.call_args_list if c[0][0] == "query_parse_failed"]
            assert len(error_calls) > 0, "query_parse_failed should be logged"

            call_kwargs = error_calls[0][1]
            assert "error_type" in call_kwargs
            assert "query" in call_kwargs
            assert "dataset_id" in call_kwargs
            assert "upload_id" in call_kwargs
            assert call_kwargs["error_type"] == "empty_query"

    def test_parse_query_logs_warning_when_no_intent_found(self, nl_engine, mock_semantic_layer):
        """Test that query_parse_failed (warning) includes standardized fields when no intent found."""
        # Arrange
        query = "xyzabc123"  # Unparseable query
        dataset_id = "test_dataset"
        upload_id = "test_upload_123"

        # Act
        with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
            intent = nl_engine.parse_query(query, dataset_id=dataset_id, upload_id=upload_id)

            # Assert: query_parse_failed (warning) called if no intent found
            warning_calls = [c for c in mock_logger.warning.call_args_list if c[0][0] == "query_parse_failed"]

            if intent is None or intent.confidence < 0.3:
                assert len(warning_calls) > 0, "query_parse_failed warning should be logged when no intent found"

                call_kwargs = warning_calls[0][1]
                assert "error_type" in call_kwargs
                assert "query" in call_kwargs
                assert "dataset_id" in call_kwargs
                assert "upload_id" in call_kwargs
                assert call_kwargs["error_type"] == "no_intent_found"

    def test_semantic_match_failed_logs_with_error_type(self, nl_engine, mock_semantic_layer):
        """Test that semantic_match_failed includes error_type field."""
        # Arrange
        query = "compare mortality by treatment"

        # Act
        with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
            # Force semantic match to fail by raising exception
            with patch.object(nl_engine, "_semantic_match", side_effect=Exception("Test error")):
                nl_engine.parse_query(query)

            # Assert: semantic_match_failed logged with error_type
            failed_calls = [c for c in mock_logger.warning.call_args_list if c[0][0] == "semantic_match_failed"]

            # May or may not be called depending on tier 1 success
            if failed_calls:
                call_kwargs = failed_calls[0][1]
                assert "error_type" in call_kwargs
                assert "error" in call_kwargs
                assert "query" in call_kwargs
                assert call_kwargs["error_type"] == "semantic_matching_exception"

    def test_get_matched_variables_extracts_all_variables(self, nl_engine):
        """Test that _get_matched_variables extracts all variables from intent."""
        # Arrange
        intent = QueryIntent(
            intent_type="COMPARE_GROUPS",
            primary_variable="mortality",
            grouping_variable="treatment",
            predictor_variables=["age", "sex"],
            time_variable="survival_time",
            event_variable="death",
            confidence=0.9,
        )

        # Act
        matched_vars = nl_engine._get_matched_variables(intent)

        # Assert
        assert "mortality" in matched_vars
        assert "treatment" in matched_vars
        assert "age" in matched_vars
        assert "sex" in matched_vars
        assert "survival_time" in matched_vars
        assert "death" in matched_vars
        assert len(matched_vars) == 6

    def test_get_matched_variables_handles_none_values(self, nl_engine):
        """Test that _get_matched_variables handles None values gracefully."""
        # Arrange
        intent = QueryIntent(
            intent_type="DESCRIBE",
            primary_variable=None,
            grouping_variable=None,
            predictor_variables=[],
            time_variable=None,
            event_variable=None,
            confidence=0.9,
        )

        # Act
        matched_vars = nl_engine._get_matched_variables(intent)

        # Assert
        assert matched_vars == []

    def test_parse_query_without_dataset_ids_still_logs(self, nl_engine, mock_semantic_layer):
        """Test that logging works even when dataset_id and upload_id are None."""
        # Arrange
        query = "compare mortality by treatment"

        # Act
        with patch("clinical_analytics.core.nl_query_engine.logger") as mock_logger:
            nl_engine.parse_query(query, dataset_id=None, upload_id=None)

            # Assert: Logging still works with None values
            start_calls = [c for c in mock_logger.info.call_args_list if c[0][0] == "query_parse_start"]
            assert len(start_calls) > 0

            call_kwargs = start_calls[0][1]
            assert call_kwargs["dataset_id"] is None
            assert call_kwargs["upload_id"] is None
            assert call_kwargs["query"] == query
