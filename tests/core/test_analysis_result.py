"""
Tests for AnalysisResult - Typed domain model for analysis execution results.

Tests verify immutability, error handling, and factory construction.
"""

from typing import Any

import pytest
from clinical_analytics.core.analysis_result import AnalysisResult


class TestAnalysisResult:
    """Test suite for AnalysisResult dataclass."""

    def test_analysis_result_creation_valid_type_returns_frozen_instance(self):
        """Test that AnalysisResult creates a frozen (immutable) instance."""
        # Arrange
        payload: dict[str, Any] = {"mean": 45.5, "std": 12.3}

        # Act
        result = AnalysisResult(type="descriptive", payload=payload)

        # Assert
        assert result.type == "descriptive"
        assert result.payload == {"mean": 45.5, "std": 12.3}
        assert result.friendly_error_message is None
        assert result.llm_interpretation is None
        assert result.run_key is None

        # Verify frozen - should raise on attribute modification
        with pytest.raises(AttributeError):
            result.type = "comparison"  # type: ignore[misc]

    def test_analysis_result_with_error_has_friendly_message(self):
        """Test that AnalysisResult stores friendly error message for failures."""
        # Arrange
        error_payload: dict[str, Any] = {"error": "Division by zero"}
        friendly_msg = "The calculation could not be completed due to missing values."

        # Act
        result = AnalysisResult(
            type="descriptive",
            payload=error_payload,
            friendly_error_message=friendly_msg,
        )

        # Assert
        assert result.friendly_error_message == friendly_msg
        assert result.payload == {"error": "Division by zero"}
        assert result.llm_interpretation is None

    def test_analysis_result_from_dict_valid_payload_returns_result(self):
        """Test that AnalysisResult.from_dict factory constructs from raw dict."""
        # Arrange
        raw_data: dict[str, Any] = {
            "type": "comparison",
            "payload": {"group_a": 45.5, "group_b": 52.1, "p_value": 0.03},
            "llm_interpretation": "Group B shows significantly higher values.",
            "run_key": "abc123",
        }

        # Act
        result = AnalysisResult.from_dict(raw_data)

        # Assert
        assert result.type == "comparison"
        assert result.payload == {"group_a": 45.5, "group_b": 52.1, "p_value": 0.03}
        assert result.llm_interpretation == "Group B shows significantly higher values."
        assert result.run_key == "abc123"
        assert result.friendly_error_message is None
