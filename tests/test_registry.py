"""
Tests for the dataset registry module.
"""

import pytest

from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.registry import DatasetRegistry


class TestDatasetRegistry:
    """Test suite for DatasetRegistry."""

    def test_discover_datasets(self):
        """Test that registry discovers datasets."""
        # Reset registry to ensure clean state
        DatasetRegistry.reset()

        datasets = DatasetRegistry.discover_datasets()

        assert isinstance(datasets, dict)
        assert len(datasets) > 0
        assert "covid_ms" in datasets or "sepsis" in datasets

    def test_list_datasets(self):
        """Test listing available datasets."""
        DatasetRegistry.reset()

        dataset_list = DatasetRegistry.list_datasets()

        assert isinstance(dataset_list, list)
        assert len(dataset_list) > 0

    def test_get_dataset_info(self):
        """Test getting dataset information."""
        DatasetRegistry.reset()
        DatasetRegistry.discover_datasets()
        DatasetRegistry.load_config()

        datasets = DatasetRegistry.list_datasets()
        if datasets:
            info = DatasetRegistry.get_dataset_info(datasets[0])

            assert isinstance(info, dict)
            assert "name" in info
            assert "available" in info
            assert "config" in info

    def test_get_dataset_factory(self):
        """Test factory method for creating datasets."""
        DatasetRegistry.reset()

        dataset = DatasetRegistry.get_dataset("covid_ms")

        assert isinstance(dataset, ClinicalDataset)
        assert dataset.name == "covid_ms"

    def test_get_all_dataset_info(self):
        """Test getting info for all datasets."""
        DatasetRegistry.reset()

        all_info = DatasetRegistry.get_all_dataset_info()

        assert isinstance(all_info, dict)
        assert len(all_info) > 0

        for name, info in all_info.items():
            assert "name" in info
            assert "config" in info

    def test_reset(self):
        """Test registry reset functionality."""
        DatasetRegistry.discover_datasets()
        DatasetRegistry.load_config()

        assert len(DatasetRegistry._datasets) > 0
        assert DatasetRegistry._config_loaded

        DatasetRegistry.reset()

        assert len(DatasetRegistry._datasets) == 0
        assert not DatasetRegistry._config_loaded

    def test_get_nonexistent_dataset(self):
        """Test that getting nonexistent dataset raises KeyError."""
        DatasetRegistry.reset()
        DatasetRegistry.discover_datasets()

        with pytest.raises(KeyError):
            DatasetRegistry.get_dataset("nonexistent_dataset")
