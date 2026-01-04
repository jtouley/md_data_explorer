"""
Tests for 'average X' or 'mean X' query pattern detection.

Tests verify:
- 'average X' queries are recognized as DESCRIBE intent
- Variable is extracted correctly (e.g., "BMI", "ldl")
- Pattern works for "average", "mean", "avg"
- Handles "of patients", "of all patients" gracefully
"""

import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine


@pytest.fixture
def mock_semantic_layer_with_bmi_ldl():
    """Create a mock semantic layer with BMI and LDL columns."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.get_column_alias_index.return_value = {
        "bmi": "BMI",
        "body mass index": "BMI",
        "ldl": "LDL mg/dL",
        "ldl cholesterol": "LDL mg/dL",
        "age": "age",
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")

    def mock_fuzzy_match(term: str):
        """Mock fuzzy matching for test variables."""
        term_lower = term.lower().strip()
        var_map = {
            "bmi": ("BMI", 0.9, None),
            "ldl": ("LDL mg/dL", 0.9, None),
            "age": ("age", 0.9, None),
        }
        return var_map.get(term_lower, (None, 0.0, None))

    mock._fuzzy_match_variable = mock_fuzzy_match
    return mock


class TestAveragePatternDetection:
    """Test that 'average X' queries are recognized as DESCRIBE with variable extraction."""

    def test_average_bmi_pattern_detected(self, mock_semantic_layer_with_bmi_ldl):
        """Test that 'average BMI of patients' is detected as DESCRIBE with BMI variable."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_layer_with_bmi_ldl)

        # Act
        query = "average BMI of patients"
        intent = engine._pattern_match(query)

        # Assert
        assert intent is not None, f"Pattern should match: {query}"
        assert intent.intent_type == "DESCRIBE", f"Expected DESCRIBE, got {intent.intent_type}"
        assert intent.primary_variable == "BMI", f"Expected BMI variable, got {intent.primary_variable}"
        assert intent.confidence >= 0.85, f"Expected confidence >= 0.85, got {intent.confidence}"

    def test_average_ldl_pattern_detected(self, mock_semantic_layer_with_bmi_ldl):
        """Test that 'average ldl of all patients' is detected as DESCRIBE with LDL variable."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_layer_with_bmi_ldl)

        # Act
        query = "average ldl of all patients"
        intent = engine._pattern_match(query)

        # Assert
        assert intent is not None, f"Pattern should match: {query}"
        assert intent.intent_type == "DESCRIBE", f"Expected DESCRIBE, got {intent.intent_type}"
        assert intent.primary_variable == "LDL mg/dL", f"Expected LDL variable, got {intent.primary_variable}"
        assert intent.confidence >= 0.85, f"Expected confidence >= 0.85, got {intent.confidence}"

    def test_mean_pattern_works(self, mock_semantic_layer_with_bmi_ldl):
        """Test that 'mean X' pattern works (synonym for average)."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_layer_with_bmi_ldl)

        # Act
        query = "mean BMI"
        intent = engine._pattern_match(query)

        # Assert
        assert intent is not None, f"Pattern should match: {query}"
        assert intent.intent_type == "DESCRIBE"
        assert intent.primary_variable == "BMI"

    def test_avg_pattern_works(self, mock_semantic_layer_with_bmi_ldl):
        """Test that 'avg X' pattern works (abbreviation for average)."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_layer_with_bmi_ldl)

        # Act
        query = "avg age"
        intent = engine._pattern_match(query)

        # Assert
        assert intent is not None, f"Pattern should match: {query}"
        assert intent.intent_type == "DESCRIBE"
        assert intent.primary_variable == "age"

    def test_average_pattern_without_variable_match_still_returns_describe(self, mock_semantic_layer_with_bmi_ldl):
        """Test that 'average X' returns DESCRIBE even if variable doesn't match (variable extracted later)."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_layer_with_bmi_ldl)

        # Act: Query with variable that won't match
        query = "average unknown_variable"
        intent = engine._pattern_match(query)

        # Assert: Should still return DESCRIBE intent (variable will be extracted later)
        assert intent is not None, f"Pattern should match: {query}"
        assert intent.intent_type == "DESCRIBE"
        assert intent.confidence >= 0.85  # Lower confidence since variable didn't match

    def test_average_pattern_handles_various_phrasings(self, mock_semantic_layer_with_bmi_ldl):
        """Test that pattern handles various phrasings like 'of patients', 'in patients', etc."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_layer_with_bmi_ldl)

        test_cases = [
            ("average BMI of patients", "BMI"),
            ("average BMI in patients", "BMI"),
            ("average BMI for patients", "BMI"),
            ("average BMI", "BMI"),  # No trailing phrase
        ]

        for query, expected_var in test_cases:
            # Act
            intent = engine._pattern_match(query)

            # Assert
            assert intent is not None, f"Pattern should match: {query}"
            assert intent.intent_type == "DESCRIBE", f"Expected DESCRIBE for: {query}"
            if expected_var:
                assert intent.primary_variable == expected_var, f"Expected {expected_var} for: {query}"
