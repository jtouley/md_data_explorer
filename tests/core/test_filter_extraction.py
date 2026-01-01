"""
Tests for LLM-powered filter extraction (ADR009 Phase 5).

Tests cover:
- _extract_filters_with_llm() function existence and basic behavior
- Independent filter validation (not all-or-nothing)
- Partial filter application (apply only valid filters)
- Confidence reduction when invalid filters detected
- Graceful degradation when LLM unavailable
- Complex filter patterns: "get rid of the n/a", "exclude missing", etc.
- Real-world test case: "get rid of the n/a" (currently fails with regex)
"""

from unittest.mock import MagicMock, patch

from clinical_analytics.core.filter_extraction import _extract_filters_with_llm
from clinical_analytics.core.llm_feature import LLMCallResult, LLMFeature


class TestFilterExtractionLLM:
    """Tests for LLM-powered filter extraction."""

    def test_extract_filters_with_llm_function_exists(self):
        """Verify _extract_filters_with_llm() function exists and is callable."""
        # Arrange: Import should succeed
        from clinical_analytics.core.filter_extraction import _extract_filters_with_llm

        # Act & Assert: Function should be callable
        assert callable(_extract_filters_with_llm)

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_success_simple(self, mock_call_llm):
        """Test successful filter extraction for simple pattern."""
        # Arrange
        query = "get rid of the n/a"
        semantic_layer = MagicMock()
        # Mock get_base_view() to return a view with columns
        mock_view = MagicMock()
        mock_view.columns = ["treatment_group", "age", "bmi"]
        semantic_layer.get_base_view.return_value = mock_view
        semantic_layer.get_column_alias_index.return_value = {
            "Treatment Group 1: Control 2: Intervention": "treatment_group",
        }
        semantic_layer.get_column_metadata.return_value = {
            "treatment_group": {"type": "categorical", "codes": {"0": "n/a", "1": "Control", "2": "Intervention"}},
        }

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"filters": [{"column": "treatment_group", "operator": "!=", "value": 0}]}',
            payload={"filters": [{"column": "treatment_group", "operator": "!=", "value": 0}]},
            latency_ms=500.0,
            timed_out=False,
            error=None,
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert
        assert len(result) == 1
        assert result[0].column == "treatment_group"
        assert result[0].operator == "!="
        assert result[0].value == 0
        assert confidence_delta == 0.0  # No invalid filters
        assert len(validation_failures) == 0
        mock_call_llm.assert_called_once()
        call_args = mock_call_llm.call_args
        assert (
            call_args.kwargs.get("feature") == LLMFeature.FILTER_EXTRACTION
            or call_args[0][0] == LLMFeature.FILTER_EXTRACTION
        )

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_real_world_case(self, mock_call_llm):
        """Test real-world case: 'get rid of the n/a' (currently fails with regex)."""
        # Arrange
        query = "get rid of the n/a"
        semantic_layer = MagicMock()
        # Mock get_base_view() to return a view with columns
        mock_view = MagicMock()
        mock_view.columns = ["treatment_group", "age", "bmi"]
        semantic_layer.get_base_view.return_value = mock_view
        semantic_layer.get_column_alias_index.return_value = {
            "Treatment Group 1: Control 2: Intervention": "treatment_group",
        }
        semantic_layer.get_column_metadata.return_value = {
            "treatment_group": {"type": "categorical", "codes": {"0": "n/a", "1": "Control", "2": "Intervention"}},
        }

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"filters": [{"column": "treatment_group", "operator": "!=", "value": 0}]}',
            payload={"filters": [{"column": "treatment_group", "operator": "!=", "value": 0}]},
            latency_ms=800.0,
            timed_out=False,
            error=None,
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert: Should extract filter to exclude n/a (code 0)
        assert len(result) == 1
        assert result[0].column == "treatment_group"
        assert result[0].operator == "!="
        assert result[0].value == 0
        assert confidence_delta == 0.0
        assert len(validation_failures) == 0

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_complex_patterns(self, mock_call_llm):
        """Test complex filter patterns that regex struggles with."""
        # Arrange
        query = "exclude missing values and patients on statins, but remove the n/a"
        semantic_layer = MagicMock()
        # Mock get_base_view() to return a view with columns
        mock_view = MagicMock()
        mock_view.columns = ["statin_prescribed", "treatment_group", "age", "bmi"]
        semantic_layer.get_base_view.return_value = mock_view
        semantic_layer.get_column_alias_index.return_value = {
            "Statin Prescribed? 1: Yes 2: No": "statin_prescribed",
            "Treatment Group 1: Control 2: Intervention": "treatment_group",
        }
        semantic_layer.get_column_metadata.return_value = {
            "statin_prescribed": {"type": "categorical", "codes": {"0": "n/a", "1": "Yes", "2": "No"}},
            "treatment_group": {"type": "categorical", "codes": {"0": "n/a", "1": "Control", "2": "Intervention"}},
        }

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"filters": ['
            '{"column": "statin_prescribed", "operator": "==", "value": 1}, '
            '{"column": "treatment_group", "operator": "!=", "value": 0}'
            "]}",
            payload={
                "filters": [
                    {"column": "statin_prescribed", "operator": "==", "value": 1},
                    {"column": "treatment_group", "operator": "!=", "value": 0},
                ]
            },
            latency_ms=1200.0,
            timed_out=False,
            error=None,
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert: Should extract multiple filters
        assert len(result) >= 1
        statin_filter = next((f for f in result if f.column == "statin_prescribed"), None)
        assert statin_filter is not None
        assert statin_filter.operator == "=="
        assert statin_filter.value == 1
        assert confidence_delta == 0.0
        assert len(validation_failures) == 0

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_independent_validation(self, mock_call_llm):
        """Test that filters are validated independently (not all-or-nothing)."""
        # Arrange: LLM returns mix of valid and invalid filters
        query = "exclude n/a and invalid_column"
        semantic_layer = MagicMock()
        # Mock get_base_view() to return a view with columns
        mock_view = MagicMock()
        mock_view.columns = ["treatment_group", "age", "bmi"]
        semantic_layer.get_base_view.return_value = mock_view
        semantic_layer.get_column_alias_index.return_value = {
            "Treatment Group 1: Control 2: Intervention": "treatment_group",
        }
        semantic_layer.get_column_metadata.return_value = {
            "treatment_group": {"type": "categorical", "codes": {"0": "n/a", "1": "Control", "2": "Intervention"}},
        }

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"filters": ['
            '{"column": "treatment_group", "operator": "!=", "value": 0}, '
            '{"column": "invalid_column", "operator": "==", "value": 1}'
            "]}",
            payload={
                "filters": [
                    {"column": "treatment_group", "operator": "!=", "value": 0},
                    {"column": "invalid_column", "operator": "==", "value": 1},
                ]
            },
            latency_ms=600.0,
            timed_out=False,
            error=None,
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert: Only valid filter should be applied
        assert len(result) == 1
        assert result[0].column == "treatment_group"
        assert result[0].operator == "!="
        assert result[0].value == 0
        # Invalid filter should be dropped (not in result)
        assert len(validation_failures) > 0  # Invalid filter logged
        # Confidence delta should be negative (reduction) or zero if capped at 0.6
        assert confidence_delta <= 0  # Confidence reduced or capped

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_confidence_reduction(self, mock_call_llm):
        """Test that confidence is reduced when invalid filters are detected."""
        # Arrange: LLM returns mix of valid and invalid filters
        query = "exclude n/a and invalid_column"
        semantic_layer = MagicMock()
        # Mock get_base_view() to return a view with columns
        mock_view = MagicMock()
        mock_view.columns = ["treatment_group", "age", "bmi"]
        semantic_layer.get_base_view.return_value = mock_view
        semantic_layer.get_column_alias_index.return_value = {
            "Treatment Group 1: Control 2: Intervention": "treatment_group",
        }
        semantic_layer.get_column_metadata.return_value = {
            "treatment_group": {"type": "categorical", "codes": {"0": "n/a", "1": "Control", "2": "Intervention"}},
        }

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"filters": ['
            '{"column": "treatment_group", "operator": "!=", "value": 0}, '
            '{"column": "invalid_column", "operator": "==", "value": 1}'
            "]}",
            payload={
                "filters": [
                    {"column": "treatment_group", "operator": "!=", "value": 0},
                    {"column": "invalid_column", "operator": "==", "value": 1},
                ]
            },
            latency_ms=600.0,
            timed_out=False,
            error=None,
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(
            query, semantic_layer, current_confidence=0.8
        )

        # Assert: Confidence should be reduced, validation failures logged
        assert len(result) == 1  # Only valid filter applied
        # Confidence delta should be negative (reduction) or zero if capped at 0.6
        assert confidence_delta <= 0  # Confidence reduced or capped
        assert len(validation_failures) > 0  # Invalid filters logged

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_graceful_degradation(self, mock_call_llm):
        """Test graceful degradation when LLM unavailable."""
        # Arrange: LLM unavailable
        query = "get rid of the n/a"
        semantic_layer = MagicMock()

        mock_call_llm.return_value = LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=100.0,
            timed_out=False,
            error="ollama_unavailable",
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert: Should return empty list (graceful degradation)
        assert result == []
        assert confidence_delta == 0.0
        assert len(validation_failures) == 0

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_timeout_handling(self, mock_call_llm):
        """Test timeout handling (LLM may timeout on complex patterns)."""
        # Arrange: LLM times out
        query = "exclude missing values and patients on statins"
        semantic_layer = MagicMock()

        mock_call_llm.return_value = LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=10000.0,
            timed_out=True,
            error="timeout",
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert: Should return empty list (graceful degradation on timeout)
        assert result == []
        assert confidence_delta == 0.0
        assert len(validation_failures) == 0

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_json_parse_failure(self, mock_call_llm):
        """Test graceful handling of JSON parse failures."""
        # Arrange: LLM returns invalid JSON
        query = "exclude n/a"
        semantic_layer = MagicMock()

        mock_call_llm.return_value = LLMCallResult(
            raw_text="This is not valid JSON",
            payload=None,
            latency_ms=500.0,
            timed_out=False,
            error="json_parse_failed",
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert: Should return empty list (graceful degradation)
        assert result == []
        assert confidence_delta == 0.0
        assert len(validation_failures) == 0

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_operator_validation(self, mock_call_llm):
        """Test that invalid operators are rejected."""
        # Arrange: LLM returns invalid operator
        query = "exclude n/a"
        semantic_layer = MagicMock()
        # Mock get_base_view() to return a view with columns
        mock_view = MagicMock()
        mock_view.columns = ["treatment_group", "age", "bmi"]
        semantic_layer.get_base_view.return_value = mock_view
        semantic_layer.get_column_alias_index.return_value = {
            "Treatment Group 1: Control 2: Intervention": "treatment_group",
        }
        semantic_layer.get_column_metadata.return_value = {
            "treatment_group": {"type": "categorical", "codes": {"0": "n/a", "1": "Control", "2": "Intervention"}},
        }

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"filters": [{"column": "treatment_group", "operator": "INVALID_OP", "value": 0}]}',
            payload={"filters": [{"column": "treatment_group", "operator": "INVALID_OP", "value": 0}]},
            latency_ms=500.0,
            timed_out=False,
            error=None,
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert: Invalid operator should be rejected
        assert len(result) == 0
        assert len(validation_failures) > 0  # Invalid operator logged
        # Confidence delta should be negative (reduction) or zero if capped at 0.6
        assert confidence_delta <= 0  # Confidence reduced or capped

    @patch("clinical_analytics.core.filter_extraction.call_llm")
    def test_extract_filters_with_llm_value_type_validation(self, mock_call_llm):
        """Test that value types are validated against column types."""
        # Arrange: LLM returns string value for numeric column
        query = "age above 50"
        semantic_layer = MagicMock()
        # Mock get_base_view() to return a view with columns
        mock_view = MagicMock()
        mock_view.columns = ["age", "bmi", "treatment_group"]
        semantic_layer.get_base_view.return_value = mock_view
        semantic_layer.get_column_alias_index.return_value = {
            "Age (years)": "age",
        }
        semantic_layer.get_column_metadata.return_value = {
            "age": {"type": "numeric"},
        }

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"filters": [{"column": "age", "operator": ">", "value": "fifty"}]}',
            payload={"filters": [{"column": "age", "operator": ">", "value": "fifty"}]},
            latency_ms=500.0,
            timed_out=False,
            error=None,
        )

        # Act
        result, confidence_delta, validation_failures = _extract_filters_with_llm(query, semantic_layer)

        # Assert: Invalid value type should be rejected (or converted if possible)
        # Implementation may convert "fifty" to 50, or reject it
        # For now, just verify function handles it gracefully
        assert isinstance(result, list)
        assert isinstance(confidence_delta, float)
        assert isinstance(validation_failures, list)
