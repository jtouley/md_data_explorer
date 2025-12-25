"""
Integration tests for UI components.

These tests verify that the UI can interact with datasets correctly
without requiring manual browser testing.
"""

import pytest
import pandas as pd
from clinical_analytics.datasets.covid_ms.definition import CovidMSDataset
from clinical_analytics.datasets.sepsis.definition import SepsisDataset
from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.core.schema import UnifiedCohort


class TestUIDatasetIntegration:
    """Test dataset interactions as they would occur in the UI."""

    def test_dataset_registry_discovery(self):
        """Test that registry can discover all datasets."""
        datasets = DatasetRegistry.list_datasets()

        assert isinstance(datasets, list)
        assert 'covid_ms' in datasets
        assert 'sepsis' in datasets

    def test_dataset_info_retrieval(self):
        """Test that dataset info can be retrieved for UI display."""
        dataset_info = DatasetRegistry.get_all_dataset_info()

        assert isinstance(dataset_info, dict)
        assert 'covid_ms' in dataset_info

        covid_info = dataset_info['covid_ms']
        assert 'config' in covid_info
        assert 'class' in covid_info

    def test_dataset_factory_creation(self):
        """Test that datasets can be created via factory."""
        # This is the pattern used in the UI
        dataset = DatasetRegistry.get_dataset('covid_ms')

        assert dataset is not None
        assert isinstance(dataset, CovidMSDataset)

    def test_covid_ms_cohort_retrieval_with_filters(self):
        """Test COVID-MS dataset cohort retrieval with different filter types."""
        dataset = DatasetRegistry.get_dataset('covid_ms')

        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        # Test with no filters (should use defaults)
        cohort = dataset.get_cohort()
        assert isinstance(cohort, pd.DataFrame)
        assert len(cohort) > 0

        # Verify schema compliance
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort.columns, f"Missing required column: {col}"

        # Test with boolean filter (this was causing the type comparison error)
        cohort_filtered = dataset.get_cohort(confirmed_only=True)
        assert isinstance(cohort_filtered, pd.DataFrame)
        assert len(cohort_filtered) > 0

    def test_sepsis_cohort_retrieval(self):
        """Test Sepsis dataset cohort retrieval."""
        dataset = DatasetRegistry.get_dataset('sepsis')

        if not dataset.validate():
            pytest.skip("Sepsis data not available")

        # Test cohort retrieval
        cohort = dataset.get_cohort()
        assert isinstance(cohort, pd.DataFrame)
        assert len(cohort) > 0

        # Verify schema compliance
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort.columns

    def test_ui_workflow_end_to_end(self):
        """Test complete UI workflow: select dataset -> get cohort -> verify schema."""
        # Simulate UI workflow
        available_datasets = DatasetRegistry.list_datasets()

        for dataset_name in available_datasets:
            if dataset_name == 'mimic3':
                # Skip MIMIC3 as it requires database connection
                continue

            # Create dataset instance
            dataset = DatasetRegistry.get_dataset(dataset_name)

            if not dataset.validate():
                continue

            # Get cohort (as UI would)
            cohort = dataset.get_cohort()

            # Verify results
            assert isinstance(cohort, pd.DataFrame)
            assert len(cohort) > 0

            # Check schema compliance
            for col in UnifiedCohort.REQUIRED_COLUMNS:
                assert col in cohort.columns, \
                    f"Dataset {dataset_name} missing required column: {col}"

    def test_boolean_filter_type_safety(self):
        """Regression test: Ensure boolean filters work with different column types."""
        dataset = DatasetRegistry.get_dataset('covid_ms')

        if not dataset.validate():
            pytest.skip("COVID-MS data not available")

        # This should not raise ComputeError about type comparison
        try:
            cohort = dataset.get_cohort(confirmed_only=True)
            assert isinstance(cohort, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"Boolean filter failed with error: {e}")


class TestUIErrorHandling:
    """Test error handling scenarios that might occur in UI."""

    def test_invalid_dataset_name(self):
        """Test that invalid dataset name raises appropriate error."""
        with pytest.raises(KeyError):
            DatasetRegistry.get_dataset('nonexistent_dataset')

    def test_dataset_without_validation(self):
        """Test that dataset without valid data can still be created."""
        # MIMIC3 without database connection
        dataset = DatasetRegistry.get_dataset('mimic3')

        # Should be created but validation fails
        assert dataset is not None
        assert not dataset.validate()
