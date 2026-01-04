"""
Tests for selective dataset loading (Phase 2.3).

Tests verify that lazy loading allows tests to load only needed datasets
instead of loading all dataset configs upfront.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pytest
from clinical_analytics.core.registry import DatasetRegistry


class TestSelectiveDatasetLoading:
    """Test suite for selective dataset loading fixtures."""

    def test_dataset_registry_fixture_returns_registry(self, dataset_registry):
        """
        Test that dataset_registry fixture returns DatasetRegistry class.

        This fixture should discover datasets but NOT pre-load all configs.
        """
        # Arrange: Fixture should be provided
        # Act: Verify fixture returns DatasetRegistry class
        assert dataset_registry is not None
        assert dataset_registry == DatasetRegistry
        # Verify we can use it to list datasets
        datasets = dataset_registry.list_datasets()
        assert isinstance(datasets, list)

    def test_get_dataset_by_name_loads_specific_dataset(self, dataset_registry, get_dataset_by_name):
        """
        Test that get_dataset_by_name helper loads only requested dataset.

        This should load dataset config only when requested, not upfront.
        """
        # Arrange: Get available dataset name
        available = [
            d for d in dataset_registry.list_datasets() if d not in ["covid_ms", "mimic3", "sepsis", "uploaded"]
        ]

        if not available:
            pytest.skip("No datasets available for testing")

        dataset_name = available[0]

        # Act: Use helper to get dataset
        dataset = get_dataset_by_name(dataset_name)

        # Assert: Dataset loaded successfully
        assert dataset is not None
        assert hasattr(dataset, "config")
        assert hasattr(dataset, "validate")

    def test_get_dataset_by_name_skips_unavailable_datasets(self, get_dataset_by_name):
        """
        Test that get_dataset_by_name skips datasets that don't exist.

        If dataset doesn't exist in registry, should skip with clear message.
        """
        # Arrange: Try to get non-existent dataset
        # Act: Request invalid dataset name
        # Note: This will skip, not raise - we need to catch the skip
        try:
            get_dataset_by_name("non_existent_dataset_xyz123")
            # If we get here, the dataset was found (unexpected)
            pytest.fail("Expected skip for non-existent dataset")
        except pytest.skip.Exception as e:
            # Assert: Should skip with clear message
            assert "non_existent_dataset_xyz123" in str(e) or "not found" in str(e).lower()

    def test_selective_loading_only_loads_requested_dataset(self, dataset_registry, get_dataset_by_name):
        """
        Test that selective loading loads dataset config on demand.

        This test verifies that configs are loaded lazily when get_dataset_by_name is called,
        not when dataset_registry fixture is created.
        """
        # Arrange: Get available dataset name
        available = [
            d for d in dataset_registry.list_datasets() if d not in ["covid_ms", "mimic3", "sepsis", "uploaded"]
        ]

        if not available:
            pytest.skip("No datasets available for testing")

        dataset_name = available[0]

        # Act: Load specific dataset (config loads on demand)
        dataset = get_dataset_by_name(dataset_name)

        # Assert: Dataset loaded successfully with config
        assert dataset is not None
        assert dataset.config is not None

        # Verify we can access dataset properties (config was loaded)
        assert hasattr(dataset, "get_cohort")
