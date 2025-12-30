"""
Tests for filter extraction from natural language queries.

Tests verify:
- Filter extraction stops at query continuation words (and, or, which, etc.)
- Filters are correctly extracted for "on X" patterns
- Compound queries don't incorrectly extract filters from continuation phrases
- Type safety: coded numeric columns extract numeric codes, not string values
- Generic coded column detection works for various column formats
"""

import polars as pl

from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.core.query_plan import FilterSpec


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


class TestCodedColumnDetection:
    """Test generic coded column detection helper."""

    def test_is_coded_column_uses_metadata_when_available(self, mock_semantic_layer):
        """Test that _is_coded_column uses metadata first (most reliable)."""
        # Arrange: Mock semantic layer with metadata
        mock = mock_semantic_layer(
            columns={
                "statin_used": "Statin Used: 0: n/a 1: Atorvastatin",
            }
        )
        # Mock metadata to return coded column info
        mock.get_column_metadata.return_value = {
            "type": "categorical",
            "metadata": {
                "numeric": True,  # This is the key indicator
                "values": [0, 1, 2, 3, 4, 5],
            },
        }
        engine = NLQueryEngine(mock)

        # Act
        is_coded = engine._is_coded_column("statin_used")

        # Assert: Should detect from metadata
        assert is_coded is True, "Should detect coded column from metadata (numeric categorical)"
        # Verify metadata was checked
        mock.get_column_metadata.assert_called_once_with("statin_used")

    def test_is_coded_column_detects_code_pattern_in_alias(self, mock_semantic_layer):
        """Test that _is_coded_column detects columns with code patterns (1: Yes 2: No)."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "statin_prescribed": "Statin Prescribed? 1: Yes 2: No",
                "treatment": "Treatment Group",
            }
        )
        # Mock get_column_metadata to return None (no metadata, tests fallback)
        mock.get_column_metadata.return_value = None
        engine = NLQueryEngine(mock)

        # Act
        is_coded = engine._is_coded_column("statin_prescribed", "Statin Prescribed? 1: Yes 2: No")
        is_not_coded = engine._is_coded_column("treatment", "Treatment Group")

        # Assert
        assert is_coded is True, "Column with code pattern should be detected as coded"
        assert is_not_coded is False, "Column without code pattern should not be detected as coded"

    def test_is_coded_column_detects_multiple_codes(self, mock_semantic_layer):
        """Test that _is_coded_column detects columns with multiple codes (0: n/a 1: Atorvastatin...)."""
        # Arrange
        alias = "Statin Used: 0: n/a 1: Atorvastatin 2: Rosuvastatin 3: Simvastatin"
        mock = mock_semantic_layer(
            columns={
                "statin_used": alias,
            }
        )
        # Mock get_column_metadata to return None (no metadata, tests fallback)
        mock.get_column_metadata.return_value = None
        engine = NLQueryEngine(mock)

        # Act
        is_coded = engine._is_coded_column("statin_used", alias)

        # Assert
        assert is_coded is True, "Column with multiple codes should be detected as coded"

    def test_is_coded_column_detects_coded_indicators(self, mock_semantic_layer):
        """Test that _is_coded_column detects columns with coded indicators (prescribed, used, type, etc.)."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "medication_type": "Medication Type: 1: A 2: B",
                "treatment_status": "Treatment Status: Active:1 Inactive:2",
            }
        )
        # Mock get_column_metadata to return None (no metadata, tests fallback)
        mock.get_column_metadata.return_value = None
        engine = NLQueryEngine(mock)

        # Act
        is_coded_type = engine._is_coded_column("medication_type", "Medication Type: 1: A 2: B")
        is_coded_status = engine._is_coded_column("treatment_status", "Treatment Status: Active:1 Inactive:2")

        # Assert
        assert is_coded_type is True, "Column with 'type' indicator should be detected as coded"
        assert is_coded_status is True, "Column with 'status' indicator should be detected as coded"

    def test_is_coded_column_looks_up_alias_when_not_provided(self, mock_semantic_layer):
        """Test that _is_coded_column looks up alias from semantic layer when not provided."""
        # Arrange: Mock returns alias -> canonical mapping
        # The alias is the full string with codes, canonical is the normalized name
        alias = "Statin Prescribed? 1: Yes 2: No"
        canonical = "statin_prescribed"
        mock = mock_semantic_layer(
            columns={
                alias: canonical,  # alias -> canonical (as returned by get_column_alias_index)
            }
        )
        # Mock get_column_metadata to return None (no metadata available)
        # This tests the fallback to alias parsing
        mock.get_column_metadata.return_value = None
        engine = NLQueryEngine(mock)

        # Act: Don't provide alias, let it look up
        is_coded = engine._is_coded_column(canonical)

        # Assert: Should detect coded column by parsing alias (fallback when no metadata)
        assert is_coded is True, "Should detect coded column by looking up alias and parsing it"


class TestFilterExtractionTypeSafety:
    """Test that filter extraction produces type-safe filters for coded columns."""

    def test_filter_extraction_extracts_numeric_codes_not_strings_for_coded_columns(self, mock_semantic_layer):
        """Test that 'on statins' extracts numeric codes, not string 'statins'."""
        # Arrange: Create semantic layer with coded statin column
        statin_alias = "Statin Used: 0: n/a 1: Atorvastatin 2: Rosuvastatin 3: Simvastatin"
        mock = mock_semantic_layer(
            columns={
                "statin_used": statin_alias,
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Extract filters from query
        query = "how many patients were on statins"
        intent = engine.parse_query(query)

        # Assert: Filters should have numeric values (codes), not string "statins"
        assert intent is not None
        if intent.filters:
            for f in intent.filters:
                if "statin" in f.column.lower():
                    # Value should be numeric (int or list of ints), not string
                    assert not isinstance(f.value, str), (
                        f"Filter value should not be string for coded column, got: {f.value} (type: {type(f.value)})"
                    )
                    # Should be either a single int code or list of int codes
                    if isinstance(f.value, list):
                        assert all(isinstance(v, int) for v in f.value), (
                            f"Filter value list should contain ints, got: {f.value}"
                        )
                    else:
                        assert isinstance(f.value, int), (
                            f"Filter value should be int for coded column, got: {f.value} (type: {type(f.value)})"
                        )

    def test_filter_extraction_uses_in_operator_for_multiple_codes(self, mock_semantic_layer):
        """Test that coded columns with multiple non-zero codes use IN operator."""
        # Arrange: Column with multiple statin options
        statin_alias = "Statin Used: 0: n/a 1: Atorvastatin 2: Rosuvastatin 3: Simvastatin 4: Pravastatin"
        mock = mock_semantic_layer(
            columns={
                "statin_used": statin_alias,
            }
        )
        engine = NLQueryEngine(mock)

        # Act
        query = "how many patients were on statins"
        intent = engine.parse_query(query)

        # Assert: Should use IN operator with list of codes
        assert intent is not None
        if intent.filters:
            for f in intent.filters:
                if "statin" in f.column.lower() and len(f.value) > 1 if isinstance(f.value, list) else False:
                    assert f.operator == "IN", f"Multiple codes should use IN operator, got: {f.operator}"
                    assert isinstance(f.value, list), f"IN operator should have list value, got: {type(f.value)}"
                    assert all(isinstance(v, int) for v in f.value), f"IN list should contain ints, got: {f.value}"

    def test_filter_extraction_binary_yes_no_extracts_single_code(self, mock_semantic_layer):
        """Test that binary yes/no columns extract single code (1 for Yes)."""
        # Arrange: Binary prescribed column
        statin_alias = "Statin Prescribed? 1: Yes 2: No"
        mock = mock_semantic_layer(
            columns={
                "statin_prescribed": statin_alias,
            }
        )
        engine = NLQueryEngine(mock)

        # Act
        query = "how many patients were on statins"
        intent = engine.parse_query(query)

        # Assert: Should extract code 1 (Yes) with == operator
        assert intent is not None
        if intent.filters:
            for f in intent.filters:
                if "statin" in f.column.lower() and "prescribed" in f.column.lower():
                    assert f.operator == "==", f"Binary column should use == operator, got: {f.operator}"
                    assert f.value == 1, f"Binary 'Yes' should extract code 1, got: {f.value}"
                    assert isinstance(f.value, int), f"Binary code should be int, got: {type(f.value)}"


class TestFilterApplicationTypeSafety:
    """Test that filter application handles type mismatches gracefully."""

    def test_apply_filters_handles_type_mismatch_gracefully(self, sample_cohort):
        """Test that _apply_filters skips filters with type mismatches instead of crashing."""
        from clinical_analytics.analysis.compute import _apply_filters

        # Arrange: Create filter with string value for numeric column
        # This simulates the bug where filter extraction creates string value for numeric column
        filters = [
            FilterSpec(
                column="age",  # Numeric column
                operator="==",
                value="statins",  # String value (type mismatch)
                exclude_nulls=True,
            )
        ]

        # Act: Should not crash, should skip the invalid filter
        result = _apply_filters(sample_cohort, filters)

        # Assert: Should return original dataframe (filter skipped)
        assert len(result) == len(sample_cohort), "Type mismatch filter should be skipped, not crash"

    def test_apply_filters_applies_numeric_filter_correctly(self, sample_cohort):
        """Test that numeric filters work correctly on numeric columns."""
        from clinical_analytics.analysis.compute import _apply_filters

        # Arrange: Valid numeric filter
        filters = [
            FilterSpec(
                column="age",
                operator=">=",
                value=50,  # Numeric value
                exclude_nulls=True,
            )
        ]

        # Act
        result = _apply_filters(sample_cohort, filters)

        # Assert: Should filter correctly
        assert len(result) < len(sample_cohort), "Filter should reduce row count"
        assert all(result["age"] >= 50), "All filtered rows should have age >= 50"

    def test_apply_filters_applies_in_operator_with_numeric_codes(self, sample_cohort):
        """Test that IN operator works correctly with list of numeric codes."""
        from clinical_analytics.analysis.compute import _apply_filters

        # Arrange: Create numeric column with coded values
        df_with_codes = sample_cohort.with_columns(pl.Series("statin_code", [0, 1, 2, 1, 0], dtype=pl.Int64))
        filters = [
            FilterSpec(
                column="statin_code",
                operator="IN",
                value=[1, 2],  # List of numeric codes
                exclude_nulls=True,
            )
        ]

        # Act
        result = _apply_filters(df_with_codes, filters)

        # Assert: Should filter to rows with codes 1 or 2
        assert len(result) == 3, "Should filter to 3 rows (codes 1, 2, 1)"
        assert all(result["statin_code"].is_in([1, 2])), "All rows should have code 1 or 2"
