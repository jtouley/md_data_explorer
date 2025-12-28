"""
Tests for DataQualityValidator

The validator uses a non-blocking philosophy:
- Structural issues (empty data) are errors
- Quality issues (missing data, duplicates) are warnings/observations
- All quality observations are stored as metadata for semantic layer
"""

import polars as pl


class TestDataQualityValidatorComplete:
    """Tests for the complete validation flow."""

    def test_validate_complete_empty_dataframe(self):
        """Test that empty DataFrame returns structural error."""
        from clinical_analytics.ui.components.data_validator import DataQualityValidator

        df = pl.DataFrame()

        result = DataQualityValidator.validate_complete(df)

        assert result["is_valid"] is False
        assert len(result["schema_errors"]) > 0
        assert any("empty" in err.lower() or "no rows" in err.lower() for err in result["schema_errors"])

    def test_validate_complete_no_columns(self):
        """Test that DataFrame with no columns returns structural error."""
        from clinical_analytics.ui.components.data_validator import DataQualityValidator

        # Create empty schema DataFrame
        df = pl.DataFrame().lazy().collect()

        result = DataQualityValidator.validate_complete(df)

        assert result["is_valid"] is False

    def test_validate_complete_valid_data_always_passes(self):
        """Test that valid data passes (quality issues are warnings, not errors)."""
        from clinical_analytics.ui.components.data_validator import DataQualityValidator

        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "age": [45, 62, 38],
                "outcome": [0, 1, 0],
            }
        )

        result = DataQualityValidator.validate_complete(df, id_column="patient_id")

        # Should always be valid (no structural errors)
        assert result["is_valid"] is True
        assert len(result["schema_errors"]) == 0

    def test_validate_complete_high_missing_is_warning_not_error(self):
        """Test that high missing data is a warning, not a blocking error."""
        from clinical_analytics.ui.components.data_validator import DataQualityValidator

        # Create DataFrame with 80%+ missing in one column
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004", "P005"],
                "age": [45, None, None, None, None],  # 80% missing
                "outcome": [0, 1, 0, 1, 0],
            }
        )

        result = DataQualityValidator.validate_complete(df, id_column="patient_id")

        # Should still be valid (missing data is warning, not error)
        assert result["is_valid"] is True
        assert len(result["schema_errors"]) == 0

        # Should have quality warnings
        assert len(result["quality_warnings"]) > 0
        high_missing_warnings = [w for w in result["quality_warnings"] if "missing" in w.get("type", "")]
        assert len(high_missing_warnings) > 0

    def test_validate_complete_duplicates_are_warnings(self):
        """Test that duplicate IDs are warnings (may be valid for multi-row data)."""
        from clinical_analytics.ui.components.data_validator import DataQualityValidator

        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P001", "P002"],  # Duplicate P001
                "visit": [1, 2, 1],
                "outcome": [0, 1, 0],
            }
        )

        result = DataQualityValidator.validate_complete(df, id_column="patient_id")

        # Should still be valid (duplicates are warnings for multi-row data)
        assert result["is_valid"] is True
        assert len(result["schema_errors"]) == 0

    def test_validate_complete_stores_quality_metadata(self):
        """Test that quality observations are stored as metadata."""
        from clinical_analytics.ui.components.data_validator import DataQualityValidator

        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "age": [45, None, 38],  # Some missing
                "outcome": [0, 1, 0],
            }
        )

        result = DataQualityValidator.validate_complete(df, id_column="patient_id", outcome_column="outcome")

        # Should have quality_metadata for semantic layer
        assert "quality_metadata" in result
        assert "columns_with_high_missing" in result["quality_metadata"]
        assert "overall_missing_pct" in result["quality_metadata"]

    def test_validate_complete_preview_mode_no_id_column(self):
        """Test that validation works in preview mode (no id_column mapping yet)."""
        from clinical_analytics.ui.components.data_validator import DataQualityValidator

        df = pl.DataFrame(
            {
                "secret_name": ["P001", "P002", "P003"],
                "age": [45, 62, 38],
            }
        )

        # No id_column = preview mode
        result = DataQualityValidator.validate_complete(df)

        # Should be valid (only quality observations)
        assert result["is_valid"] is True
        assert len(result["schema_errors"]) == 0

    def test_validate_complete_summary_statistics(self):
        """Test that summary statistics are returned."""
        from clinical_analytics.ui.components.data_validator import DataQualityValidator

        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "age": [45, 62, 38],
            }
        )

        result = DataQualityValidator.validate_complete(df)

        assert "summary" in result
        assert result["summary"]["total_rows"] == 3
        assert result["summary"]["total_columns"] == 2


class TestVariableTypeDetector:
    """Tests for the variable type detector."""

    def test_detect_identifier_by_uniqueness(self):
        """Test that high-uniqueness columns are detected as identifiers."""
        from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

        df = pl.DataFrame(
            {
                "secret_name": ["C_1001", "C_1002", "C_1003", "C_1004", "C_1005"],
                "age": [45, 62, 38, 55, 42],
            }
        )

        variable_info = VariableTypeDetector.detect_all_variables(df)

        # secret_name should be detected as identifier (100% unique, no nulls)
        assert variable_info["secret_name"]["type"] == "identifier"

    def test_detect_binary_outcome(self):
        """Test that binary columns are detected."""
        from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "outcome": [0, 1, 0],
            }
        )

        variable_info = VariableTypeDetector.detect_all_variables(df)

        assert variable_info["outcome"]["type"] == "binary"

    def test_detect_continuous(self):
        """Test that high-cardinality numeric with nulls is detected as continuous."""
        from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

        # Add more rows and some nulls so uniqueness < 95%
        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "age": [45.5 + i * 0.1 if i % 10 != 0 else None for i in range(100)],
            }
        )

        variable_info = VariableTypeDetector.detect_all_variables(df)

        # Age should be continuous (high cardinality numeric, has nulls so not identifier)
        assert variable_info["age"]["type"] == "continuous"

    def test_suggest_schema_mapping_data_driven(self):
        """Test that schema mapping is suggested based on data, not column names."""
        from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

        # Column names don't match patterns, but data characteristics should drive suggestions
        df = pl.DataFrame(
            {
                "secret_name": ["C_1001", "C_1002", "C_1003", "C_1004", "C_1005"],
                "hospitalized": [0, 1, 0, 1, 0],  # Binary
                "age": [45, 62, 38, 55, 42],
            }
        )

        suggestions = VariableTypeDetector.suggest_schema_mapping(df)

        # Should suggest secret_name as ID (highest uniqueness)
        assert suggestions["patient_id"] == "secret_name"

        # Should suggest hospitalized as outcome (binary)
        assert suggestions["outcome"] == "hospitalized"
