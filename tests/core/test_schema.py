"""
Tests for schema module.
"""

from clinical_analytics.core.schema import UnifiedCohort


class TestUnifiedCohort:
    """Test suite for UnifiedCohort schema."""

    def test_required_columns(self):
        """Test that required columns are defined."""
        assert UnifiedCohort.PATIENT_ID in UnifiedCohort.REQUIRED_COLUMNS
        assert UnifiedCohort.TIME_ZERO in UnifiedCohort.REQUIRED_COLUMNS
        assert UnifiedCohort.OUTCOME in UnifiedCohort.REQUIRED_COLUMNS
        assert UnifiedCohort.OUTCOME_LABEL in UnifiedCohort.REQUIRED_COLUMNS

    def test_column_names(self):
        """Test column name constants."""
        assert UnifiedCohort.PATIENT_ID == "patient_id"
        assert UnifiedCohort.TIME_ZERO == "time_zero"
        assert UnifiedCohort.OUTCOME == "outcome"
        assert UnifiedCohort.OUTCOME_LABEL == "outcome_label"
        assert UnifiedCohort.FEATURES_JSON == "features_json"

    def test_required_columns_list(self):
        """Test that REQUIRED_COLUMNS is a list with all required columns."""
        assert isinstance(UnifiedCohort.REQUIRED_COLUMNS, list)
        assert len(UnifiedCohort.REQUIRED_COLUMNS) == 4
