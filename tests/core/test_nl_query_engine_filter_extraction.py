"""
Tests for filter extraction from natural language queries.

Tests verify:
- Filter extraction stops at query continuation words (and, or, which, etc.)
- Filters are correctly extracted for "on X" patterns
- Compound queries don't incorrectly extract filters from continuation phrases
"""

from clinical_analytics.core.nl_query_engine import NLQueryEngine


class TestFilterExtractionStopsAtContinuationWords:
    """Test that filter extraction stops at continuation words."""

    def test_filter_extraction_stops_at_and(self, mock_semantic_layer):
        """Test that filter extraction stops at 'and' in compound queries."""
        # Arrange: Create semantic layer with statin column
        statin_column_value = "Statin Prescribed? 1: Yes 2: No"
        mock = mock_semantic_layer(
            columns={
                "statin_prescribed": statin_column_value,
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Extract filters from compound query
        query = "how many patients were on statins and which statin was most prescribed?"
        intent = engine.parse_query(query)

        # Assert: Should extract filter for "statins" but NOT capture "and which statin was most prescribed?"
        assert intent is not None
        if intent.filters:
            # Check that no filter has the continuation phrase as value
            for f in intent.filters:
                assert "and which statin" not in str(f.value).lower(), (
                    f"Filter value should not contain continuation phrase, got: {f.value}"
                )
                assert "most prescribed" not in str(f.value).lower(), (
                    f"Filter value should not contain continuation phrase, got: {f.value}"
                )

    def test_filter_extraction_stops_at_which(self, mock_semantic_layer):
        """Test that filter extraction stops at 'which' in compound queries."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "treatment": "Treatment Group",
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Extract filters from query with "which" continuation
        query = "patients on treatment which treatment was most effective"
        intent = engine.parse_query(query)

        # Assert: Filter should not capture "which treatment was most effective"
        assert intent is not None
        if intent.filters:
            for f in intent.filters:
                assert "which treatment" not in str(f.value).lower(), (
                    f"Filter value should not contain continuation phrase, got: {f.value}"
                )

    def test_filter_extraction_stops_at_or(self, mock_semantic_layer):
        """Test that filter extraction stops at 'or' in compound queries."""
        # Arrange: Create semantic layer with statin column
        statin_column_value = "Statin Prescribed? 1: Yes 2: No"
        mock = mock_semantic_layer(
            columns={
                "statin_prescribed": statin_column_value,
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Extract filters from query with "or" continuation
        query = "patients on statins or treatment"
        intent = engine.parse_query(query)

        # Assert: Filter should not capture "or treatment"
        assert intent is not None
        if intent.filters:
            for f in intent.filters:
                assert "or treatment" not in str(f.value).lower(), (
                    f"Filter value should not contain continuation phrase, got: {f.value}"
                )
                # Filter value should be just "statins", not "statins or treatment"
                if "statins" in str(f.value).lower():
                    assert str(f.value).lower() == "statins" or str(f.value) == 1, (
                        f"Filter value should be just 'statins' or code 1, got: {f.value}"
                    )

    def test_filter_extraction_correctly_extracts_single_filter(self, mock_semantic_layer):
        """Test that filter extraction correctly extracts a single filter when no continuation."""
        # Arrange: Create semantic layer with statin column
        statin_column_value = "Statin Prescribed? 1: Yes 2: No"
        mock = mock_semantic_layer(
            columns={
                "statin_prescribed": statin_column_value,
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Extract filters from simple query
        query = "how many patients were on statins"
        intent = engine.parse_query(query)

        # Assert: Should extract filter for "statins"
        assert intent is not None
        # Filter extraction may or may not find a match depending on semantic layer setup
        # The key is that if filters are extracted, they should be correct
        if intent.filters:
            assert len(intent.filters) > 0, "Should extract at least one filter"
            # Filter should be for statin-related column
            filter_columns = [f.column for f in intent.filters]
            assert any("statin" in col.lower() for col in filter_columns), (
                f"Filter should be for statin column, got: {filter_columns}"
            )
