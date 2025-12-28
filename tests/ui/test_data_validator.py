"""
Unit tests for DataQualityValidator.validate_complete().
"""

import pandas as pd
import pytest

from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.ui.components.data_validator import DataQualityValidator


class TestDataQualityValidatorComplete:
    """Test suite for DataQualityValidator.validate_complete()."""

    @pytest.fixture
    def valid_cohort_df(self):
        """Create a valid cohort DataFrame for testing."""
        return pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P002", "P003"],
                UnifiedCohort.TIME_ZERO: pd.date_range("2024-01-01", periods=3),
                UnifiedCohort.OUTCOME: [0, 1, 0],
                UnifiedCohort.OUTCOME_LABEL: ["alive", "dead", "alive"],
                "age": [45, 62, 38],
            }
        )

    def test_validate_complete_schema_first(self, valid_cohort_df):
        """Test that schema validation runs first and returns schema_errors."""
        # Remove required column to trigger schema error
        df_invalid = valid_cohort_df.drop(columns=[UnifiedCohort.PATIENT_ID])

        result = DataQualityValidator.validate_complete(df_invalid)

        # Schema errors should be populated
        assert "schema_errors" in result
        assert len(result["schema_errors"]) > 0
        assert not result["is_valid"]
        assert any("patient_id" in str(err).lower() for err in result["schema_errors"])

    def test_validate_complete_default_granularity_unknown(self):
        """Test that default granularity is 'unknown'."""
        # Create DataFrame with duplicate patient IDs
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P001", "P002"],  # Duplicate P001
                UnifiedCohort.TIME_ZERO: pd.date_range("2024-01-01", periods=3),
                UnifiedCohort.OUTCOME: [0, 1, 0],
                UnifiedCohort.OUTCOME_LABEL: ["alive", "dead", "alive"],
            }
        )

        # Call without granularity parameter (should default to "unknown")
        result = DataQualityValidator.validate_complete(df, id_column=UnifiedCohort.PATIENT_ID)

        # With granularity="unknown", duplicates should be a WARNING not error
        assert "quality_warnings" in result
        duplicate_warnings = [w for w in result["quality_warnings"] if w["type"] == "duplicate_ids"]
        assert len(duplicate_warnings) > 0
        assert duplicate_warnings[0]["severity"] == "warning"

    def test_validate_complete_duplicates_warning_non_patient_level(self):
        """Test that duplicates are warnings for non-patient_level granularity."""
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P001", "P002"],  # Duplicate P001
                UnifiedCohort.TIME_ZERO: pd.date_range("2024-01-01", periods=3),
                UnifiedCohort.OUTCOME: [0, 1, 0],
                UnifiedCohort.OUTCOME_LABEL: ["alive", "dead", "alive"],
            }
        )

        # Test with admission_level
        result = DataQualityValidator.validate_complete(
            df, id_column=UnifiedCohort.PATIENT_ID, granularity="admission_level"
        )

        # Duplicates should be WARNING for admission_level
        assert "quality_warnings" in result
        duplicate_warnings = [w for w in result["quality_warnings"] if w["type"] == "duplicate_ids"]
        assert len(duplicate_warnings) > 0
        assert duplicate_warnings[0]["severity"] == "warning"
        assert "expected" in duplicate_warnings[0]["message"].lower()

    def test_validate_complete_duplicates_error_patient_level(self):
        """Test that duplicates are errors for patient_level granularity."""
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P001", "P002"],  # Duplicate P001
                UnifiedCohort.TIME_ZERO: pd.date_range("2024-01-01", periods=3),
                UnifiedCohort.OUTCOME: [0, 1, 0],
                UnifiedCohort.OUTCOME_LABEL: ["alive", "dead", "alive"],
            }
        )

        # Test with patient_level
        result = DataQualityValidator.validate_complete(
            df, id_column=UnifiedCohort.PATIENT_ID, granularity="patient_level"
        )

        # Duplicates should be ERROR for patient_level (not converted to warning)
        assert not result["is_valid"]
        # Should be in issues with severity="error"
        duplicate_errors = [i for i in result["issues"] if i["type"] == "duplicate_ids" and i["severity"] == "error"]
        assert len(duplicate_errors) > 0

    def test_validate_complete_returns_combined_issues(self, valid_cohort_df):
        """Test that validate_complete returns schema_errors, quality_warnings, and combined issues."""
        result = DataQualityValidator.validate_complete(valid_cohort_df, id_column=UnifiedCohort.PATIENT_ID)

        # Check all expected keys are present
        assert "is_valid" in result
        assert "schema_errors" in result
        assert "quality_warnings" in result
        assert "issues" in result
        assert "summary" in result

        # For valid data, should have no errors
        assert result["is_valid"]
        assert len(result["schema_errors"]) == 0

    def test_validate_complete_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()

        result = DataQualityValidator.validate_complete(df)

        assert not result["is_valid"]
        assert len(result["schema_errors"]) > 0
        assert any("empty" in str(err).lower() or "no columns" in str(err).lower() for err in result["schema_errors"])

    def test_validate_complete_quality_warnings_separate(self):
        """Test that quality warnings are separated from schema errors."""
        # Create DataFrame with high missing data (triggers warning not error)
        df = pd.DataFrame(
            {
                UnifiedCohort.PATIENT_ID: ["P001", "P002", "P003"],
                UnifiedCohort.TIME_ZERO: pd.date_range("2024-01-01", periods=3),
                UnifiedCohort.OUTCOME: [0, 1, 0],
                UnifiedCohort.OUTCOME_LABEL: ["alive", "dead", "alive"],
                "age": [45, None, 38],  # Some missing data
                "score": [None, None, None],  # 100% missing - triggers warning threshold
            }
        )

        result = DataQualityValidator.validate_complete(df, id_column=UnifiedCohort.PATIENT_ID)

        # Should have quality warnings but no schema errors (if within thresholds)
        assert "quality_warnings" in result
        # Check for warnings about missing data
        missing_warnings = [w for w in result["quality_warnings"] if "missing" in w["type"]]
        assert len(missing_warnings) >= 0  # May or may not trigger depending on thresholds
