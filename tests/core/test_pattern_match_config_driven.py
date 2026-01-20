"""Tests for config-driven _pattern_match() refactoring (Phase 2b).

Verifies that _pattern_match() uses patterns from config/nl_query_patterns.yaml
instead of hardcoded regex patterns, while preserving existing behavior.

Following AAA pattern and test naming convention:
test_unit_scenario_expectedBehavior
"""

from unittest.mock import MagicMock, patch

import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine


@pytest.fixture
def mock_semantic_for_pattern():
    """Create mock semantic layer for pattern match testing."""
    mock = MagicMock()
    mock.get_column_alias_index.return_value = {
        "age": "age",
        "bmi": "BMI",
        "status": "status",
        "treatment": "treatment",
        "outcome": "outcome",
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")
    return mock


class TestPatternMatchConfigDriven:
    """Test suite for config-driven pattern matching."""

    def test_pattern_match_count_how_many_returns_count_intent(self, mock_semantic_for_pattern):
        """Test that 'how many' pattern returns COUNT intent from config."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("how many patients are there?")

        # Assert
        assert result is not None
        assert result.intent_type == "COUNT"
        assert result.confidence == 0.9

    def test_pattern_match_count_number_of_returns_count_intent(self, mock_semantic_for_pattern):
        """Test that 'number of' pattern returns COUNT intent from config."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("what is the number of patients?")

        # Assert
        assert result is not None
        assert result.intent_type == "COUNT"
        assert result.confidence == 0.9

    def test_pattern_match_survival_keyword_returns_survival_intent(self, mock_semantic_for_pattern):
        """Test that survival patterns return SURVIVAL intent from config."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("analyze survival outcomes")

        # Assert
        assert result is not None
        assert result.intent_type == "SURVIVAL"
        assert result.confidence == 0.9

    def test_pattern_match_correlations_keyword_returns_correlations_intent(self, mock_semantic_for_pattern):
        """Test that correlation patterns return CORRELATIONS intent from config."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("what is the relationship between age and bmi")

        # Assert
        assert result is not None
        assert result.intent_type == "CORRELATIONS"

    def test_pattern_match_compare_by_extracts_variables(self, mock_semantic_for_pattern):
        """Test that 'compare X by Y' pattern extracts variables from config groups."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("compare age by treatment")

        # Assert
        assert result is not None
        assert result.intent_type == "COMPARE_GROUPS"
        assert result.primary_variable == "age"
        assert result.grouping_variable == "treatment"
        assert result.confidence == 0.95

    def test_pattern_match_find_predictors_extracts_outcome(self, mock_semantic_for_pattern):
        """Test that 'what predicts X' pattern extracts outcome variable."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("what predicts outcome")

        # Assert
        assert result is not None
        assert result.intent_type == "FIND_PREDICTORS"
        assert result.primary_variable == "outcome"
        assert result.confidence == 0.95

    def test_pattern_match_describe_average_extracts_variable(self, mock_semantic_for_pattern):
        """Test that 'average X' pattern extracts variable for DESCRIBE."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("average age of patients")

        # Assert
        assert result is not None
        assert result.intent_type == "DESCRIBE"
        assert result.primary_variable == "age"
        assert result.confidence == 0.9

    def test_pattern_match_unrecognized_query_returns_none(self, mock_semantic_for_pattern):
        """Test that unrecognized patterns return None."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("xyzzy foobar gibberish")

        # Assert
        assert result is None

    def test_pattern_match_uses_config_patterns(self, mock_semantic_for_pattern):
        """Test that _pattern_match loads patterns from config."""
        import re as regex_module

        # Arrange - patch at the import location in nl_query_engine
        with patch("clinical_analytics.core.nl_query_engine.load_patterns_config") as mock_load_patterns:
            mock_load_patterns.return_value = {
                "COUNT": [
                    {
                        "regex": regex_module.compile(r"how many", regex_module.IGNORECASE),
                        "groups": {},
                        "confidence": 0.9,
                    }
                ]
            }
            engine = NLQueryEngine(mock_semantic_for_pattern)

            # Act
            result = engine._pattern_match("how many patients?")

            # Assert
            mock_load_patterns.assert_called()
            assert result is not None
            assert result.intent_type == "COUNT"

    def test_pattern_match_case_insensitive(self, mock_semantic_for_pattern):
        """Test that pattern matching is case-insensitive."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result_lower = engine._pattern_match("how many patients?")
        result_upper = engine._pattern_match("HOW MANY PATIENTS?")
        result_mixed = engine._pattern_match("How Many Patients?")

        # Assert
        assert result_lower is not None
        assert result_upper is not None
        assert result_mixed is not None
        assert result_lower.intent_type == result_upper.intent_type == result_mixed.intent_type


class TestPatternMatchPreservesBehavior:
    """Tests to ensure config-driven refactor preserves existing behavior."""

    def test_pattern_match_what_was_most_returns_count(self, mock_semantic_for_pattern):
        """Test 'what was the most X' still returns COUNT."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("what was the most common treatment?")

        # Assert
        assert result is not None
        assert result.intent_type == "COUNT"

    def test_pattern_match_fuzzy_matches_variables(self, mock_semantic_for_pattern):
        """Test that variables are fuzzy-matched via semantic layer."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # Act
        result = engine._pattern_match("compare bmi by status")

        # Assert
        assert result is not None
        assert result.primary_variable == "BMI"  # Fuzzy matched to proper column name
        assert result.grouping_variable == "status"

    def test_pattern_match_order_matters_first_match_wins(self, mock_semantic_for_pattern):
        """Test that pattern order matters - first match should win."""
        # Arrange
        engine = NLQueryEngine(mock_semantic_for_pattern)

        # "compare X by Y" should match COMPARE_GROUPS, not fall through to DESCRIBE
        # Act
        result = engine._pattern_match("compare age by treatment")

        # Assert
        assert result is not None
        assert result.intent_type == "COMPARE_GROUPS"
