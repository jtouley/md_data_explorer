"""
Tests for UnifiedCohort schema validation.

Validates:
- validate_unified_cohort_schema() correctly validates required columns
- Missing columns are detected
- Type validation works correctly
- Strict mode raises exceptions
"""

from datetime import datetime

# PANDAS EXCEPTION: Tests validate pd.DataFrame schema (current get_cohort return type)
# TODO: Update to Polars when get_cohort() returns pl.DataFrame
import pandas as pd
import pytest

from clinical_analytics.core.schema import (
    SchemaValidationError,
    UnifiedCohort,
    validate_unified_cohort_schema,
)


class TestValidateUnifiedCohortSchema:
    """Tests for schema validation function."""

    def test_schema_validation_valid_cohort_returns_true(self) -> None:
        """Test that a valid cohort DataFrame passes validation."""
        # Arrange
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P002", "P003"],
                UnifiedCohort.TIME_ZERO: [datetime.now()] * 3,
                UnifiedCohort.OUTCOME: [0, 1, 0],
                UnifiedCohort.OUTCOME_LABEL: ["mortality"] * 3,
            }
        )

        # Act
        is_valid, errors = validate_unified_cohort_schema(df)

        # Assert
        assert is_valid is True
        assert errors == []

    def test_schema_validation_missing_columns_returns_error(self) -> None:
        """Test that missing required columns are detected."""
        # Arrange
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P002"],
                # Missing: TIME_ZERO, OUTCOME, OUTCOME_LABEL
            }
        )

        # Act
        is_valid, errors = validate_unified_cohort_schema(df)

        # Assert
        assert is_valid is False
        assert len(errors) >= 1
        assert any("Missing required columns" in e for e in errors)

    def test_schema_validation_null_patient_id_returns_error(self) -> None:
        """Test that NULL patient IDs are detected."""
        # Arrange
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", None, "P003"],
                UnifiedCohort.TIME_ZERO: [datetime.now()] * 3,
                UnifiedCohort.OUTCOME: [0, 1, 0],
                UnifiedCohort.OUTCOME_LABEL: ["mortality"] * 3,
            }
        )

        # Act
        is_valid, errors = validate_unified_cohort_schema(df)

        # Assert
        assert is_valid is False
        assert any("NULL" in e for e in errors)

    def test_schema_validation_non_numeric_outcome_returns_error(self) -> None:
        """Test that non-numeric outcome is detected."""
        # Arrange
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P002"],
                UnifiedCohort.TIME_ZERO: [datetime.now()] * 2,
                UnifiedCohort.OUTCOME: ["yes", "no"],  # Should be numeric
                UnifiedCohort.OUTCOME_LABEL: ["mortality"] * 2,
            }
        )

        # Act
        is_valid, errors = validate_unified_cohort_schema(df)

        # Assert
        assert is_valid is False
        assert any("numeric" in e.lower() for e in errors)

    def test_schema_validation_invalid_binary_outcome_returns_error(self) -> None:
        """Test that binary outcomes with invalid values are detected."""
        # Arrange
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P002"],
                UnifiedCohort.TIME_ZERO: [datetime.now()] * 2,
                UnifiedCohort.OUTCOME: [0, 2],  # 2 is invalid for binary
                UnifiedCohort.OUTCOME_LABEL: ["mortality"] * 2,
            }
        )

        # Act
        is_valid, errors = validate_unified_cohort_schema(df)

        # Assert
        assert is_valid is False
        assert any("binary" in e.lower() for e in errors)

    def test_schema_validation_strict_mode_raises_exception(self) -> None:
        """Test that strict mode raises SchemaValidationError."""
        # Arrange
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001"],
                # Missing required columns
            }
        )

        # Act & Assert
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_unified_cohort_schema(df, strict=True)

        assert len(exc_info.value.errors) > 0

    def test_schema_validation_extra_columns_still_valid(self) -> None:
        """Test that extra columns don't cause validation failure."""
        # Arrange
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P002"],
                UnifiedCohort.TIME_ZERO: [datetime.now()] * 2,
                UnifiedCohort.OUTCOME: [0, 1],
                UnifiedCohort.OUTCOME_LABEL: ["mortality"] * 2,
                "extra_col": [1, 2],
                "another_col": ["a", "b"],
            }
        )

        # Act
        is_valid, errors = validate_unified_cohort_schema(df)

        # Assert
        assert is_valid is True
        assert errors == []
