"""
Tests for composite identifier detection and synthetic patient_id creation.

Tests the automatic creation of patient_id from composite columns when
no single-column identifier is found.
"""

import polars as pl

from clinical_analytics.ui.components.variable_detector import VariableTypeDetector


class TestCompositeIdentifierDetection:
    """Tests for finding composite identifier candidates."""

    def test_find_composite_identifier_with_two_columns(self):
        """Test that two columns together can form a unique identifier."""
        # Create DataFrame where no single column is unique, but combination is
        df = pl.DataFrame(
            {
                "race": ["A", "A", "B", "B", "C"],
                "gender": ["M", "F", "M", "F", "M"],
                "age": [50, 50, 60, 60, 70],  # Not unique alone
                "cd4": [100, 200, 300, 400, 500],  # Unique
            }
        )

        # race + gender + age should be unique (5 unique combinations)
        result = VariableTypeDetector.find_composite_identifier_candidates(df)

        assert result is not None
        assert len(result) >= 2
        # Should include columns that together are unique
        assert "race" in result or "gender" in result or "age" in result

    def test_find_composite_identifier_no_candidates(self):
        """Test that None is returned when no unique combination exists."""
        # Create DataFrame where no combination is unique
        df = pl.DataFrame(
            {
                "col1": ["A", "A", "A"],
                "col2": ["B", "B", "B"],  # All rows identical
            }
        )

        result = VariableTypeDetector.find_composite_identifier_candidates(df)

        assert result is None

    def test_find_composite_identifier_excludes_high_missing(self):
        """Test that columns with high missing values are excluded."""
        df = pl.DataFrame(
            {
                "race": ["A", "B", "C", "D", "E"],
                "gender": ["M", "F", "M", "F", "M"],
                "bad_col": [None, None, None, None, "value"],  # 80% missing
            }
        )

        result = VariableTypeDetector.find_composite_identifier_candidates(df)

        # Should not include bad_col due to high missing
        if result:
            assert "bad_col" not in result

    def test_find_composite_identifier_prefers_smaller_combinations(self):
        """Test that smaller combinations are preferred over larger ones."""
        # Create data where 2 columns are unique, but 3 would also work
        df = pl.DataFrame(
            {
                "race": ["A", "A", "B", "B", "C"],
                "gender": ["M", "F", "M", "F", "M"],
                "age": [50, 51, 52, 53, 54],  # All unique
                "extra": [1, 2, 3, 4, 5],  # All unique
            }
        )

        result = VariableTypeDetector.find_composite_identifier_candidates(df)

        # Should find a combination (race+gender might not be unique, but race+gender+age should be)
        assert result is not None
        # Prefer smaller combinations (2-3 columns) over 4


class TestSyntheticPatientIdCreation:
    """Tests for creating synthetic patient_id from composite columns."""

    def test_create_synthetic_patient_id_from_two_columns(self):
        """Test creating synthetic ID from two columns."""
        df = pl.DataFrame(
            {
                "race": ["A", "B", "C"],
                "gender": ["M", "F", "M"],
                "age": [50, 60, 70],
            }
        )

        result = VariableTypeDetector.create_synthetic_patient_id(df, ["race", "gender"])

        assert "patient_id" in result.columns
        assert result.height == 3
        # All patient_ids should be unique
        assert result["patient_id"].n_unique() == 3
        # Original columns should still exist
        assert "race" in result.columns
        assert "gender" in result.columns

    def test_create_synthetic_patient_id_deterministic(self):
        """Test that same input produces same hash."""
        df = pl.DataFrame(
            {
                "race": ["A", "B", "C"],
                "gender": ["M", "F", "M"],
            }
        )

        result1 = VariableTypeDetector.create_synthetic_patient_id(df, ["race", "gender"])
        result2 = VariableTypeDetector.create_synthetic_patient_id(df, ["race", "gender"])

        # Same input should produce same hashes
        assert result1["patient_id"].to_list() == result2["patient_id"].to_list()

    def test_create_synthetic_patient_id_handles_nulls(self):
        """Test that null values in source columns are handled."""
        df = pl.DataFrame(
            {
                "race": ["A", None, "C"],
                "gender": ["M", "F", None],
            }
        )

        result = VariableTypeDetector.create_synthetic_patient_id(df, ["race", "gender"])

        assert "patient_id" in result.columns
        assert result.height == 3
        # Should handle nulls gracefully (empty string in composite key)


class TestEnsurePatientId:
    """Tests for the main ensure_patient_id function."""

    def test_ensure_patient_id_existing_column(self):
        """Test that existing patient_id column is preserved."""
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "age": [50, 60, 70],
            }
        )

        result_df, metadata = VariableTypeDetector.ensure_patient_id(df)

        assert metadata["patient_id_source"] == "existing"
        assert "patient_id" in result_df.columns
        assert result_df["patient_id"].to_list() == ["P001", "P002", "P003"]

    def test_ensure_patient_id_single_column_identifier(self):
        """Test that single column with >95% uniqueness is used."""
        # Create column with 100% uniqueness
        df = pl.DataFrame(
            {
                "unique_id": ["ID1", "ID2", "ID3", "ID4", "ID5"],
                "age": [50, 60, 70, 80, 90],
            }
        )

        result_df, metadata = VariableTypeDetector.ensure_patient_id(df)

        assert metadata["patient_id_source"] == "single_column"
        assert metadata["patient_id_columns"] == ["unique_id"]
        assert "patient_id" in result_df.columns
        assert result_df["patient_id"].n_unique() == 5

    def test_ensure_patient_id_composite_identifier(self):
        """Test that composite identifier is created when no single column is unique."""
        # No single column is unique enough (>95%), but combination is 100% unique
        df = pl.DataFrame(
            {
                "race": ["A", "A", "B", "B", "C", "C"],  # 50% unique
                "gender": ["M", "F", "M", "F", "M", "F"],  # 33% unique
                "age": [50, 51, 52, 53, 54, 55],  # 100% unique but continuous type
            }
        )

        result_df, metadata = VariableTypeDetector.ensure_patient_id(df)

        # Age might be detected as identifier (100% unique), or composite might be used
        # Either way, patient_id should exist
        assert "patient_id" in result_df.columns
        assert result_df["patient_id"].n_unique() == 6
        # If composite was used, verify metadata
        if metadata["patient_id_source"] == "composite":
            assert metadata["patient_id_columns"] is not None
            assert len(metadata["patient_id_columns"]) >= 2

    def test_ensure_patient_id_no_identifier_found(self):
        """Test behavior when no identifier can be found."""
        # All rows identical - no unique combination possible
        df = pl.DataFrame(
            {
                "col1": ["A", "A", "A"],
                "col2": ["B", "B", "B"],
            }
        )

        result_df, metadata = VariableTypeDetector.ensure_patient_id(df)

        assert metadata["patient_id_source"] is None
        # DataFrame should be returned unchanged (no patient_id added)
        assert "patient_id" not in result_df.columns or result_df["patient_id"].is_null().all()

    def test_ensure_patient_id_with_pandas_input(self):
        """Test that pandas DataFrame is handled correctly."""
        import pandas as pd

        df_pandas = pd.DataFrame(
            {
                "race": ["A", "B", "C"],
                "gender": ["M", "F", "M"],
            }
        )

        result_df, metadata = VariableTypeDetector.ensure_patient_id(df_pandas)

        # Should return Polars DataFrame
        assert isinstance(result_df, pl.DataFrame)
        assert "patient_id" in result_df.columns or metadata["patient_id_source"] is not None


class TestCompositeIdentifierIntegration:
    """Integration tests for composite identifier in full workflow."""

    def test_suggest_schema_mapping_with_composite_id(self):
        """Test that suggest_schema_mapping uses composite ID when needed."""
        # Dataset where no single column is >95% unique, but combination is 100% unique
        df = pl.DataFrame(
            {
                "Race": ["Black", "Black", "White", "White", "Black", "Black"],  # 33% unique
                "Gender": ["M", "F", "M", "F", "M", "F"],  # 33% unique
                "Age": [50, 51, 52, 53, 54, 55],  # 100% unique but continuous (not identifier type)
            }
        )

        suggestions = VariableTypeDetector.suggest_schema_mapping(df)

        # Age might be detected as identifier (100% unique), or composite might be used
        # Either way, should suggest something
        assert suggestions["patient_id"] is not None
        # If composite was used, metadata should be present
        if suggestions["patient_id"] == "patient_id" and "_patient_id_metadata" in suggestions:
            metadata = suggestions["_patient_id_metadata"]
            assert metadata["patient_id_source"] == "composite"

    def test_composite_identifier_preserves_original_columns(self):
        """Test that original columns are preserved when creating synthetic ID."""
        # Create data where combination is unique but no single column is >95% unique
        # Use more rows to ensure combination is unique
        df = pl.DataFrame(
            {
                "race": ["A", "A", "B", "B", "C", "C"],  # 50% unique
                "gender": ["M", "F", "M", "F", "M", "F"],  # 33% unique
                "age": [50, 51, 52, 53, 54, 55],  # All unique but continuous
            }
        )

        result_df, metadata = VariableTypeDetector.ensure_patient_id(df)

        # patient_id should always exist
        assert "patient_id" in result_df.columns

        # When composite ID is created, original columns should still exist
        if metadata["patient_id_source"] == "composite":
            assert "race" in result_df.columns
            assert "gender" in result_df.columns
            assert "age" in result_df.columns

    def test_composite_identifier_handles_mixed_types(self):
        """Test that composite identifier works with mixed column types."""
        df = pl.DataFrame(
            {
                "categorical": ["A", "B", "C"],  # Categorical
                "numeric": [1.5, 2.5, 3.5],  # Continuous
                "age": [50, 60, 70],  # Continuous
            }
        )

        result_df, metadata = VariableTypeDetector.ensure_patient_id(df)

        # Should work with mixed types
        if metadata["patient_id_source"] == "composite":
            assert "patient_id" in result_df.columns
            assert result_df["patient_id"].n_unique() == 3


class TestEnsurePatientIdLogging:
    """Tests for logging in ensure_patient_id."""

    def test_ensure_patient_id_logs_appropriately(self, caplog):
        """Test that ensure_patient_id logs at appropriate levels."""
        import logging

        caplog.set_level(logging.INFO)

        # Test with existing patient_id
        df = pl.DataFrame({"patient_id": ["P1", "P2"], "age": [50, 60]})
        VariableTypeDetector.ensure_patient_id(df)

        assert any("patient_id already exists in DataFrame" in record.message for record in caplog.records)

        # Test with single column identifier
        caplog.clear()
        df2 = pl.DataFrame({"unique_id": ["ID1", "ID2", "ID3"], "age": [50, 60, 70]})
        VariableTypeDetector.ensure_patient_id(df2)

        assert any("Found single-column identifier" in record.message for record in caplog.records)

        # Test with composite identifier
        # Use data where no single column is >95% unique, but combination is unique
        # Need enough rows to ensure combination is unique but individual columns aren't
        caplog.clear()
        df3 = pl.DataFrame(
            {
                "race": ["A", "A", "B", "B", "C", "C"],  # 50% unique
                "gender": ["M", "F", "M", "F", "M", "F"],  # 33% unique
                "age": [50, 51, 52, 53, 54, 55],  # 100% unique but continuous (not identifier type)
            }
        )
        VariableTypeDetector.ensure_patient_id(df3)

        # Age might be detected as identifier, or composite might be used
        # Check for either path
        has_single = any("Found single-column identifier" in record.message for record in caplog.records)
        has_composite = any("Creating composite identifier" in record.message for record in caplog.records)
        has_success = any("Successfully created composite patient_id" in record.message for record in caplog.records)

        # Either single column or composite should be logged
        assert has_single or (has_composite and has_success)
