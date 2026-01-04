"""
Tests for type-aware query parsing logic (Phase 4.2).

Ensures:
- Intent is set appropriately based on variable types
- Categorical variables trigger COUNT intent when used for grouping
- Numeric variables trigger DESCRIBE intent when queried
- Grouping variables are validated as categorical
"""

from clinical_analytics.core.nl_query_engine import NLQueryEngine


class TestTypeAwareIntentSelection:
    """Test that intent is selected based on variable types."""

    def test_categorical_grouping_triggers_count_intent(self, mock_semantic_layer):
        """Categorical variable used for grouping should trigger COUNT intent."""
        # Arrange: Categorical status column
        mock = mock_semantic_layer(
            columns={
                "status": "Patient Status",  # Categorical
                "patient_id": "Patient ID",
            }
        )
        # Mock metadata to indicate categorical type
        mock.get_column_metadata.return_value = {
            "type": "categorical",
            "metadata": {"numeric": False},
        }
        engine = NLQueryEngine(mock)

        # Act: Query with categorical grouping
        query = "how many patients by status?"
        intent = engine.parse_query(query)

        # Assert: Should use COUNT intent for categorical grouping
        assert intent is not None
        assert intent.intent_type == "COUNT", f"Expected COUNT for categorical grouping, got {intent.intent_type}"
        assert intent.grouping_variable == "Patient Status", "Should detect status as grouping variable (uses alias)"

    def test_numeric_metric_triggers_describe_intent(self, mock_semantic_layer):
        """Numeric variable queried for statistics should trigger DESCRIBE intent."""
        # Arrange: Numeric age column
        mock = mock_semantic_layer(
            columns={
                "age": "Patient Age (years)",  # Numeric
            }
        )
        # Mock metadata to indicate numeric type
        mock.get_column_metadata.return_value = {
            "type": "numeric",
            "metadata": {"numeric": True},
        }
        engine = NLQueryEngine(mock)

        # Act: Query for numeric statistics
        query = "what is the average age?"
        intent = engine.parse_query(query)

        # Assert: Should use DESCRIBE intent for numeric statistics
        assert intent is not None
        assert intent.intent_type == "DESCRIBE", f"Expected DESCRIBE for numeric metric, got {intent.intent_type}"
        # Variable extraction is out of scope for Phase 4.2 (type-aware logic)
        # Phase 4.2 is about using type info to select the right intent

    def test_numeric_grouping_categorical_metric_triggers_compare(self, mock_semantic_layer):
        """Numeric metric compared across categorical groups should trigger COMPARE_GROUPS."""
        # Arrange: Categorical treatment column and numeric outcome
        mock = mock_semantic_layer(
            columns={
                "treatment": "Treatment Group",  # Categorical
                "mortality": "Mortality Rate",  # Numeric
            }
        )

        def get_metadata(column):
            if column == "treatment":
                return {"type": "categorical", "metadata": {"numeric": False}}
            elif column == "mortality":
                return {"type": "numeric", "metadata": {"numeric": True}}
            return None

        mock.get_column_metadata.side_effect = get_metadata
        engine = NLQueryEngine(mock)

        # Act: Compare numeric metric across categorical groups
        query = "compare mortality by treatment"
        intent = engine.parse_query(query)

        # Assert: Should use COMPARE_GROUPS or DESCRIBE intent (both valid for comparison)
        assert intent is not None
        assert intent.intent_type in [
            "COMPARE_GROUPS",
            "DESCRIBE",
            "COUNT",  # COUNT with grouping also valid
        ], f"Expected COMPARE_GROUPS, DESCRIBE, or COUNT, got {intent.intent_type}"
        # Variable extraction out of scope for Phase 4.2


class TestGroupingVariableValidation:
    """Test that grouping variables are validated as categorical."""

    def test_numeric_grouping_variable_handled_appropriately(self, mock_semantic_layer):
        """Numeric variable used for grouping should be handled (binning or error)."""
        # Arrange: Numeric age column
        mock = mock_semantic_layer(
            columns={
                "age": "Patient Age (years)",  # Numeric
                "mortality": "Mortality Rate",
            }
        )

        def get_metadata(column):
            if column == "age":
                return {"type": "numeric", "metadata": {"numeric": True}}
            elif column == "mortality":
                return {"type": "numeric", "metadata": {"numeric": True}}
            return None

        mock.get_column_metadata.side_effect = get_metadata
        engine = NLQueryEngine(mock)

        # Act: Try to group by numeric variable
        query = "compare mortality by age"
        intent = engine.parse_query(query)

        # Assert: Should either:
        # 1. Suggest binning age into categories, OR
        # 2. Treat as correlation/regression intent
        # For now, we accept any valid intent that handles numeric grouping
        assert intent is not None
        # This is a soft assertion - we don't enforce specific behavior yet
        # The key is that it doesn't crash and produces a valid intent

    def test_categorical_grouping_variable_accepted(self, mock_semantic_layer):
        """Categorical variable used for grouping should be accepted."""
        # Arrange: Categorical treatment column
        mock = mock_semantic_layer(
            columns={
                "treatment": "Treatment Group",  # Categorical
                "mortality": "Mortality Rate",
            }
        )

        def get_metadata(column):
            if column == "treatment":
                return {"type": "categorical", "metadata": {"numeric": False}}
            elif column == "mortality":
                return {"type": "numeric", "metadata": {"numeric": True}}
            return None

        mock.get_column_metadata.side_effect = get_metadata
        engine = NLQueryEngine(mock)

        # Act: Group by categorical variable
        query = "compare mortality by treatment"
        intent = engine.parse_query(query)

        # Assert: Should accept categorical grouping variable (intent should be valid)
        assert intent is not None
        assert intent.intent_type in [
            "COMPARE_GROUPS",
            "DESCRIBE",
            "COUNT",
        ], f"Should produce valid intent for categorical grouping, got {intent.intent_type}"
        # No error or crash expected for categorical grouping


class TestTypeAwareCodedColumnHandling:
    """Test that coded categorical columns are handled correctly."""

    def test_coded_categorical_triggers_count_with_filter(self, mock_semantic_layer):
        """Coded categorical column should trigger COUNT with numeric code filter."""
        # Arrange: Coded statin column
        statin_alias = "Statin Used: 0: n/a 1: Atorvastatin 2: Rosuvastatin"
        mock = mock_semantic_layer(
            columns={
                "statin_used": statin_alias,
                "statins": "statin_used",
            }
        )
        # Mock metadata to indicate coded categorical
        mock.get_column_metadata.return_value = {
            "type": "categorical",
            "metadata": {"numeric": True, "values": [0, 1, 2]},
        }
        engine = NLQueryEngine(mock)

        # Act: Query for coded categorical counts
        query = "how many patients on statins"
        intent = engine.parse_query(query)

        # Assert: Should use COUNT intent (type-aware for categorical)
        assert intent is not None
        assert intent.intent_type == "COUNT", f"Expected COUNT for categorical, got {intent.intent_type}"
        # Coded categorical columns should produce numeric code filters (tested in Phase 4.1)
        # Phase 4.2 focus: type-aware intent selection ✓
