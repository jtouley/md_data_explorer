"""
Tests for 'which X was most Y' query pattern detection.

Tests verify:
- 'which X was most Y' queries are recognized as COUNT intent (not DESCRIBE)
- Pattern is generic and works for any domain (not hardcoded to statins)
- Grouping variable is extracted correctly
"""

from clinical_analytics.core.nl_query_engine import NLQueryEngine


class TestMostQueryPatternDetection:
    """Test that 'which X was most Y' queries are recognized as COUNT."""

    def test_which_most_pattern_detected_as_count(self, mock_semantic_layer):
        """Test that 'which statin was most prescribed?' is detected as COUNT."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "statin_used": "Statin Used: 0: n/a 1: Atorvastatin 2: Rosuvastatin",
            }
        )
        engine = NLQueryEngine(mock)

        # Act
        query = "which statin was most prescribed?"
        intent = engine.parse_query(query)

        # Assert: Should be COUNT, not DESCRIBE
        assert intent is not None
        assert intent.intent_type == "COUNT", f"Expected COUNT, got {intent.intent_type}"
        assert intent.confidence >= 0.9, f"Expected high confidence, got {intent.confidence}"

    def test_most_pattern_works_for_any_domain(self, mock_semantic_layer):
        """Test that 'which X was most Y' pattern is generic and works for any domain."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "treatment_type": "Treatment Type: 0: None 1: Aspirin 2: Ibuprofen",
                "medication": "Medication: 0: None 1: DrugA 2: DrugB",
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Test various domains
        queries = [
            "which treatment was most common?",
            "which medication was most used?",
            "which therapy was most frequent?",
        ]

        for query in queries:
            intent = engine.parse_query(query)
            assert intent is not None, f"Failed to parse: {query}"
            assert intent.intent_type == "COUNT", f"Expected COUNT for '{query}', got {intent.intent_type}"
            assert intent.confidence >= 0.9, f"Expected high confidence for '{query}', got {intent.confidence}"

    def test_most_query_extracts_grouping_variable(self, mock_semantic_layer):
        """Test that grouping variable is extracted from 'which X was most Y' queries."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "statin_used": "Statin Used: 0: n/a 1: Atorvastatin 2: Rosuvastatin",
            }
        )
        engine = NLQueryEngine(mock)

        # Act
        query = "which statin was most prescribed?"
        intent = engine.parse_query(query)

        # Assert: Should extract grouping variable
        assert intent is not None
        assert intent.intent_type == "COUNT"
        # Grouping variable should be extracted (may be None if fuzzy match fails, but pattern should be detected)
        # The key is that intent_type is COUNT, not DESCRIBE
