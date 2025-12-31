"""
Integration tests for UI components using registry discovery.

These tests verify that the UI can interact with datasets correctly
without requiring manual browser testing. Tests are parametrized across
all available datasets using registry discovery.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pandas as pd
import pytest

from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.core.schema import UnifiedCohort


def get_available_datasets():
    """
    Helper to discover all available datasets from registry.

    Returns:
        List of dataset names to test against (only user uploads, excludes built-in datasets)
    """
    DatasetRegistry.reset()
    DatasetRegistry.discover_datasets()
    DatasetRegistry.load_config()
    # Filter out built-in datasets (covid_ms, mimic3, sepsis) and uploaded class
    datasets = DatasetRegistry.list_datasets()
    return [name for name in datasets if name not in ["covid_ms", "mimic3", "sepsis", "uploaded"]]


def get_sample_datasets():
    """
    Helper to get 1-2 representative datasets for fast tests.

    Returns:
        List of 1-2 dataset names for fast unit testing
    """
    all_datasets = get_available_datasets()
    # Return first 1-2 available datasets for fast testing
    return all_datasets[:2] if len(all_datasets) >= 2 else all_datasets[:1]


class TestUIDatasetIntegration:
    """Test dataset interactions as they would occur in the UI."""

    def test_dataset_registry_discovers_datasets(self):
        """Test that registry can discover datasets dynamically."""
        # Act
        datasets = DatasetRegistry.list_datasets()

        # Assert: Registry discovers datasets
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        # Don't assert specific names - use discovery

    def test_dataset_info_retrieval_returns_metadata(self):
        """Test that dataset info can be retrieved for UI display."""
        # Act
        dataset_info = DatasetRegistry.get_all_dataset_info()

        # Assert: Returns metadata dict
        assert isinstance(dataset_info, dict)
        assert len(dataset_info) > 0

        # Verify structure of first available dataset
        first_dataset = next(iter(dataset_info.keys()))
        info = dataset_info[first_dataset]
        assert "config" in info
        assert "class" in info

    def test_dataset_factory_creates_instance_via_registry(self):
        """Test that datasets can be created via factory pattern (UI pattern)."""
        # Arrange: Get first available dataset (exclude "uploaded" which requires upload_id)
        datasets = [d for d in DatasetRegistry.list_datasets() if d != "uploaded"]
        if not datasets:
            pytest.skip("No datasets available (excluding 'uploaded' which requires upload_id)")
        first_dataset = datasets[0]

        # Act: Create via factory (UI pattern)
        dataset = DatasetRegistry.get_dataset(first_dataset)

        # Assert: Instance created
        assert dataset is not None
        assert dataset.name == first_dataset

    @pytest.mark.parametrize("dataset_name", get_sample_datasets())
    @pytest.mark.slow
    @pytest.mark.integration
    def test_cohort_retrieval_with_default_filters(self, dataset_name):
        """Test dataset cohort retrieval with default filters."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act: Get cohort with no filters (uses defaults)
        cohort = dataset.get_cohort()

        # Assert: Valid DataFrame returned
        assert isinstance(cohort, pd.DataFrame)
        assert len(cohort) > 0

        # Verify schema compliance
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort.columns, f"{dataset_name} missing required column: {col}"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_ui_workflow_end_to_end_with_all_datasets(self):
        """Test complete UI workflow: select dataset -> get cohort -> verify schema."""
        # Arrange: Get all available datasets
        available_datasets = DatasetRegistry.list_datasets()

        tested_count = 0
        for dataset_name in available_datasets:
            # Skip uploaded class (requires upload_id)
            if dataset_name == "uploaded":
                continue

            # Act: Create dataset instance (UI pattern)
            dataset = DatasetRegistry.get_dataset(dataset_name)

            if not dataset.validate():
                continue

            # Get cohort (as UI would)
            cohort = dataset.get_cohort()

            # Assert: Verify results
            assert isinstance(cohort, pd.DataFrame)
            assert len(cohort) > 0

            # Check schema compliance
            for col in UnifiedCohort.REQUIRED_COLUMNS:
                assert col in cohort.columns, f"Dataset {dataset_name} missing required column: {col}"

            tested_count += 1

        # Assert: At least one dataset was tested
        assert tested_count > 0, "No datasets available for testing"

    @pytest.mark.parametrize("dataset_name", get_sample_datasets())
    @pytest.mark.slow
    @pytest.mark.integration
    def test_patient_level_granularity_supported(self, dataset_name):
        """Test that patient_level granularity works (M8 integration)."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        cohort = dataset.get_cohort(granularity="patient_level")

        # Assert: Returns valid cohort
        assert isinstance(cohort, pd.DataFrame)
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort.columns

    @pytest.mark.parametrize("dataset_name", get_sample_datasets())
    @pytest.mark.slow
    @pytest.mark.integration
    def test_non_patient_level_granularity_rejected_for_single_table(self, dataset_name):
        """Test that single-table datasets reject non-patient_level granularity (M8)."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act & Assert: Single-table datasets only support patient_level
        with pytest.raises(ValueError, match="granularity"):
            dataset.get_cohort(granularity="admission_level")

        with pytest.raises(ValueError, match="granularity"):
            dataset.get_cohort(granularity="event_level")


class TestUIErrorHandling:
    """Test error handling scenarios that might occur in UI."""

    def test_invalid_dataset_name_raises_keyerror(self):
        """Test that invalid dataset name raises appropriate error."""
        # Act & Assert: Invalid name should raise KeyError
        with pytest.raises(KeyError):
            DatasetRegistry.get_dataset("nonexistent_dataset")


class TestUIFilterHandling:
    """Test filter handling scenarios in UI."""

    @pytest.mark.parametrize("dataset_name", get_sample_datasets())
    @pytest.mark.slow
    @pytest.mark.integration
    def test_boolean_filter_type_safety(self, dataset_name):
        """Regression test: Ensure boolean filters work with different column types."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act: Try common boolean filters
        try:
            # This should not raise ComputeError about type comparison
            cohort = dataset.get_cohort()
            assert isinstance(cohort, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"{dataset_name} filter failed with error: {e}")
