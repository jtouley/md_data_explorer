"""
Tests for COVID-MS dataset implementation.
"""

import pytest
import pandas as pd
from clinical_analytics.datasets.covid_ms.definition import CovidMSDataset
from clinical_analytics.core.schema import UnifiedCohort


class TestCovidMSDataset:
    """Test suite for CovidMSDataset."""

    @pytest.fixture
    def dataset(self, sample_covid_ms_path):
        """Create dataset instance."""
        if sample_covid_ms_path is None:
            pytest.skip("COVID-MS data not available")

        return CovidMSDataset()

    def test_initialization(self):
        """Test dataset initialization."""
        dataset = CovidMSDataset()

        assert dataset.name == 'covid_ms'
        assert dataset.config is not None
        assert dataset.mapper is not None

    def test_validation(self, dataset):
        """Test dataset validation."""
        is_valid = dataset.validate()

        assert isinstance(is_valid, bool)
        # Will be True if data exists, False otherwise

    def test_load_data(self, dataset):
        """Test data loading."""
        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        dataset.load()

        assert dataset._data is not None
        assert len(dataset._data) > 0

    def test_get_cohort_schema_compliance(self, dataset):
        """Test that get_cohort returns UnifiedCohort-compliant DataFrame."""
        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        cohort = dataset.get_cohort()

        # Check it's a Pandas DataFrame
        assert isinstance(cohort, pd.DataFrame)

        # Check required columns present
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort.columns, f"Missing required column: {col}"

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(cohort[UnifiedCohort.TIME_ZERO])
        assert pd.api.types.is_numeric_dtype(cohort[UnifiedCohort.OUTCOME])

    def test_get_cohort_with_filters(self, dataset):
        """Test get_cohort with custom filters."""
        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        # Test with confirmed_only filter
        cohort = dataset.get_cohort(confirmed_only=True)

        assert isinstance(cohort, pd.DataFrame)
        assert len(cohort) > 0

    def test_config_driven_defaults(self, dataset):
        """Test that dataset uses config-driven defaults."""
        assert hasattr(dataset, 'mapper')
        assert dataset.mapper is not None

        # Check mapper has config-loaded values
        predictors = dataset.mapper.get_default_predictors()
        assert isinstance(predictors, list)
        assert len(predictors) > 0

        outcome = dataset.mapper.get_default_outcome()
        assert isinstance(outcome, str)

    def test_get_cohort_no_data_loaded(self):
        """Test get_cohort auto-loads data if not loaded."""
        dataset = CovidMSDataset()

        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        # Don't call load() explicitly
        cohort = dataset.get_cohort()

        # Should still work - auto-loads
        assert isinstance(cohort, pd.DataFrame)
        assert len(cohort) > 0

    def test_idempotency(self, dataset):
        """Test that same filters produce same result (idempotency)."""
        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        # Get cohort with same filters multiple times
        cohort1 = dataset.get_cohort(confirmed_only=True)
        cohort2 = dataset.get_cohort(confirmed_only=True)

        # Results should be identical
        assert len(cohort1) == len(cohort2)
        assert set(cohort1.columns) == set(cohort2.columns)
        
        # Check that required columns are present
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort1.columns
            assert col in cohort2.columns

    def test_config_driven_time_zero(self, dataset):
        """Test that time_zero comes from config."""
        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        time_zero_value = dataset.mapper.get_time_zero_value()
        assert time_zero_value is not None
        assert isinstance(time_zero_value, str)

        cohort = dataset.get_cohort()
        # Time zero should be set from config
        assert UnifiedCohort.TIME_ZERO in cohort.columns

    def test_config_driven_outcome_label(self, dataset):
        """Test that outcome_label comes from config."""
        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        outcome_col = dataset.mapper.get_default_outcome()
        outcome_label = dataset.mapper.get_default_outcome_label(outcome_col)
        
        assert outcome_label is not None
        assert isinstance(outcome_label, str)

        cohort = dataset.get_cohort()
        # Outcome label should be set from config
        assert UnifiedCohort.OUTCOME_LABEL in cohort.columns

    def test_get_cohort_with_granularity_patient_level(self, dataset):
        """Test that get_cohort(granularity="patient_level") works (M8 integration test)."""
        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        cohort = dataset.get_cohort(granularity="patient_level")

        assert isinstance(cohort, pd.DataFrame)
        # Should have required columns
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort.columns

    def test_get_cohort_rejects_non_patient_level_granularity(self, dataset):
        """Test that non-patient_level granularity raises ValueError at dataset level (M8)."""
        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        # Single-table datasets only support patient_level
        with pytest.raises(ValueError, match="granularity"):
            dataset.get_cohort(granularity="admission_level")

        with pytest.raises(ValueError, match="granularity"):
            dataset.get_cohort(granularity="event_level")