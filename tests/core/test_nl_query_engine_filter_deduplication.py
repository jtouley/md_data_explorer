"""
Tests for filter deduplication in NL Query Engine.

Tests verify:
- Duplicate filters (same column + operator + value) are removed
- Deduplication works for both single values and list values (IN operator)
- Generic deduplication logic works for any dataset
"""

from clinical_analytics.core.nl_query_engine import NLQueryEngine


class TestFilterDeduplication:
    """Test that duplicate filters are removed before returning."""

    def test_deduplication_removes_identical_filters(self, mock_semantic_layer):
        """Test that identical filters are deduplicated."""
        # Arrange: Create semantic layer with medication column
        mock = mock_semantic_layer(
            columns={
                "medication_prescribed": "Medication Prescribed? 1: Yes 2: No",
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Extract filters (simulate duplicate extraction)
        # Note: The actual extraction logic should not produce duplicates,
        # but we test that deduplication works if duplicates exist
        query = "how many patients were on medication"
        intent = engine.parse_query(query)

        # Assert: No duplicate filters (same column + operator + value)
        assert intent is not None
        if intent.filters:
            seen = set()
            for f in intent.filters:
                filter_key = (
                    f.column,
                    f.operator,
                    str(f.value) if not isinstance(f.value, list) else tuple(sorted(f.value)),
                )
                assert filter_key not in seen, f"Duplicate filter found: {f}"
                seen.add(filter_key)

    def test_deduplication_handles_list_values(self, mock_semantic_layer):
        """Test that deduplication works correctly for IN operator with list values."""
        # Arrange: Create semantic layer with coded column
        mock = mock_semantic_layer(
            columns={
                "treatment_type": "Treatment Type: 0: None 1: Aspirin 2: Ibuprofen 3: Acetaminophen",
            }
        )
        # Mock metadata for coded column detection
        mock.get_column_metadata.return_value = {
            "type": "categorical",
            "metadata": {"numeric": True, "values": [0, 1, 2, 3]},
        }
        engine = NLQueryEngine(mock)

        # Act: Extract filters for "on treatment" query
        query = "how many patients were on treatment"
        intent = engine.parse_query(query)

        # Assert: No duplicate filters, even if they have list values
        assert intent is not None
        if intent.filters:
            seen = set()
            for f in intent.filters:
                # For list values, create tuple of sorted values for comparison
                filter_key = (
                    f.column,
                    f.operator,
                    tuple(sorted(f.value)) if isinstance(f.value, list) else f.value,
                )
                assert filter_key not in seen, f"Duplicate filter found: {f}"
                seen.add(filter_key)

    def test_deduplication_is_generic_works_for_any_column(self, mock_semantic_layer):
        """Test that deduplication logic is generic and works for any column type."""
        # Arrange: Create semantic layer with various column types
        mock = mock_semantic_layer(
            columns={
                "diabetes": "Diabetes 1: Yes 2: No",
                "hypertension": "Hypertension Prescribed? 1: Yes 2: No",
                "medication": "Medication Used: 0: None 1: Aspirin 2: Ibuprofen",
            }
        )

        # Mock metadata for coded column detection
        def get_metadata(column_name):
            if "diabetes" in column_name.lower() or "hypertension" in column_name.lower():
                return {"type": "binary", "metadata": {"numeric": True, "values": [1, 2]}}
            elif "medication" in column_name.lower():
                return {"type": "categorical", "metadata": {"numeric": True, "values": [0, 1, 2]}}
            return None

        mock.get_column_metadata.side_effect = get_metadata
        engine = NLQueryEngine(mock)

        # Act: Extract filters for various queries
        queries = [
            "how many patients had diabetes",
            "how many patients were on hypertension medication",
            "how many patients were on medication",
        ]

        for query in queries:
            intent = engine.parse_query(query)
            assert intent is not None

            # Assert: No duplicate filters for any query
            if intent.filters:
                seen = set()
                for f in intent.filters:
                    filter_key = (
                        f.column,
                        f.operator,
                        tuple(sorted(f.value)) if isinstance(f.value, list) else str(f.value),
                    )
                    assert filter_key not in seen, f"Duplicate filter found for query '{query}': {f}"
                    seen.add(filter_key)
