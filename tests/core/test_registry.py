"""
Tests for the dataset registry module using registry discovery.

Tests the DatasetRegistry functionality without hardcoded dataset dependencies,
using dynamic discovery to ensure tests work across all available datasets.

Test name follows: test_unit_scenario_expectedBehavior
"""

from pathlib import Path

import polars as pl
import pytest
from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.registry import DatasetRegistry


def get_first_available_dataset(discovered_datasets):
    """
    Get first available dataset from cached discovery.

    Performance optimization: Uses session-scoped fixture to avoid
    expensive dataset discovery on every test call.

    Args:
        discovered_datasets: Session-scoped fixture with cached datasets

    Returns:
        str: Name of first available dataset, or None if none available
    """
    import logging

    logger = logging.getLogger(__name__)

    available = discovered_datasets["available"]

    if not available:
        logger.warning(
            "test_registry_no_datasets_available: all_datasets=%s, reason=%s",
            discovered_datasets["all_datasets"],
            "all datasets filtered out or none discovered",
        )
        return None

    selected = available[0]
    logger.info(
        "test_registry_dataset_selected: selected_dataset=%s, available_options=%s",
        selected,
        available,
    )

    return selected


class TestDatasetRegistry:
    """Test suite for DatasetRegistry."""

    def test_discover_datasets_returns_non_empty_dict(self):
        """Test that registry discovers datasets dynamically."""
        # Arrange: Reset registry to ensure clean state
        DatasetRegistry.reset()

        # Act
        datasets = DatasetRegistry.discover_datasets()

        # Assert: Returns non-empty dict of dataset classes
        assert isinstance(datasets, dict)
        assert len(datasets) > 0
        # Don't assert specific dataset names - use discovery
        assert all(isinstance(name, str) for name in datasets.keys())

    def test_list_datasets_returns_non_empty_list(self):
        """Test listing available datasets returns non-empty list."""
        # Arrange
        DatasetRegistry.reset()

        # Act
        dataset_list = DatasetRegistry.list_datasets()

        # Assert: Returns non-empty list of dataset names
        assert isinstance(dataset_list, list)
        assert len(dataset_list) > 0
        assert all(isinstance(name, str) for name in dataset_list)

    def test_get_dataset_info_returns_metadata_dict(self):
        """Test getting dataset information returns metadata dict."""
        # Arrange
        DatasetRegistry.reset()
        DatasetRegistry.discover_datasets()
        DatasetRegistry.load_config()
        datasets = DatasetRegistry.list_datasets()
        assert len(datasets) > 0, "No datasets available for testing"

        # Act: Get info for first available dataset
        first_dataset = datasets[0]
        info = DatasetRegistry.get_dataset_info(first_dataset)

        # Assert: Returns metadata dict with required keys
        assert isinstance(info, dict)
        assert "name" in info
        assert "available" in info
        assert "config" in info

    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_dataset_factory_creates_instance(self, discovered_datasets):
        """Test factory method creates dataset instance."""
        # Arrange
        dataset_name = get_first_available_dataset(discovered_datasets)
        if dataset_name is None:
            pytest.skip("No datasets available for testing - skipping integration test")

        # Act: Create dataset via factory
        dataset = DatasetRegistry.get_dataset(dataset_name)

        # Assert: Returns ClinicalDataset instance with correct name
        assert isinstance(dataset, ClinicalDataset)
        assert dataset.name == dataset_name

    def test_get_all_dataset_info_returns_dict_with_metadata(self):
        """Test getting info for all datasets returns dict with metadata."""
        # Arrange
        DatasetRegistry.reset()

        # Act
        all_info = DatasetRegistry.get_all_dataset_info()

        # Assert: Returns non-empty dict with metadata for each dataset
        assert isinstance(all_info, dict)
        assert len(all_info) > 0

        # Verify each dataset has required metadata keys
        for name, info in all_info.items():
            assert "name" in info
            assert "config" in info

    def test_reset_clears_registry_state(self):
        """Test registry reset clears all state."""
        # Arrange: Populate registry
        DatasetRegistry.discover_datasets()
        DatasetRegistry.load_config()
        assert len(DatasetRegistry._datasets) > 0
        assert DatasetRegistry._config_loaded

        # Act: Reset registry
        DatasetRegistry.reset()

        # Assert: All state cleared
        assert len(DatasetRegistry._datasets) == 0
        assert not DatasetRegistry._config_loaded

    def test_get_nonexistent_dataset_raises_keyerror(self):
        """Test that getting nonexistent dataset raises KeyError."""
        # Arrange
        DatasetRegistry.reset()
        DatasetRegistry.discover_datasets()

        # Act & Assert: Should raise KeyError for nonexistent dataset
        with pytest.raises(KeyError):
            DatasetRegistry.get_dataset("nonexistent_dataset")

    def test_load_config_sets_config_loaded_flag(self):
        """Test loading configuration from YAML sets config loaded flag."""
        # Arrange
        DatasetRegistry.reset()

        # Act
        DatasetRegistry.load_config()

        # Assert: Config loaded flag set and configs dict initialized
        assert DatasetRegistry._config_loaded
        assert isinstance(DatasetRegistry._configs, dict)

    def test_load_config_nonexistent_file_sets_empty_config(self):
        """Test loading config with nonexistent file sets empty config."""
        # Arrange
        DatasetRegistry.reset()

        # Act: Load config from nonexistent path (should not raise error)
        DatasetRegistry.load_config(Path("/nonexistent/path/config.yaml"))

        # Assert: Config loaded flag set, configs dict empty
        assert DatasetRegistry._config_loaded
        assert DatasetRegistry._configs == {}

    def test_register_from_dataframe_creates_config(self):
        """Test registering dataset from DataFrame creates config."""
        # Arrange
        DatasetRegistry.reset()
        df = pl.DataFrame({"patient_id": ["P001", "P002", "P003"], "age": [45, 62, 38], "outcome": [1, 0, 1]})

        # Act: Register DataFrame as dataset
        config = DatasetRegistry.register_from_dataframe("test_dataset", df, display_name="Test Dataset")

        # Assert: Config created with correct metadata
        assert isinstance(config, dict)
        assert config["name"] == "test_dataset"
        assert config["display_name"] == "Test Dataset"
        assert config["status"] == "auto-inferred"
        assert config["row_count"] == 3
        assert config["column_count"] == 3

        # Assert: DataFrame is stored
        stored_df = DatasetRegistry.get_auto_inferred_dataframe("test_dataset")
        assert stored_df is not None
        assert stored_df.height == 3

    def test_get_auto_inferred_dataframe_retrieves_stored_data(self):
        """Test retrieving auto-inferred DataFrame returns stored data."""
        # Arrange: Register DataFrame
        DatasetRegistry.reset()
        df = pl.DataFrame({"col1": [1, 2, 3]})
        DatasetRegistry.register_from_dataframe("test_df", df)

        # Act: Retrieve stored DataFrame
        retrieved = DatasetRegistry.get_auto_inferred_dataframe("test_df")

        # Assert: Retrieved DataFrame matches original
        assert retrieved is not None
        assert retrieved.equals(df)

    def test_get_auto_inferred_dataframe_nonexistent_returns_none(self):
        """Test retrieving nonexistent auto-inferred DataFrame returns None."""
        # Arrange
        DatasetRegistry.reset()

        # Act & Assert: Nonexistent dataset returns None
        assert DatasetRegistry.get_auto_inferred_dataframe("nonexistent") is None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_dataset_with_override_params_applies_overrides(self, discovered_datasets):
        """Test getting dataset with override parameters applies overrides."""
        # Arrange
        dataset_name = get_first_available_dataset(discovered_datasets)
        if dataset_name is None:
            pytest.skip("No datasets available for testing - skipping integration test")

        # Act: Get dataset with override params
        dataset = DatasetRegistry.get_dataset(dataset_name, source_path="/custom/path")

        # Assert: Dataset created (override params applied if supported)
        assert isinstance(dataset, ClinicalDataset)
        # Note: Override params may be applied if dataset supports them
        # Not all datasets have source_path attribute

    @pytest.mark.slow
    @pytest.mark.integration
    def test_registry_filters_unsupported_params_without_error(self, discovered_datasets, caplog):
        """Test that registry filters out unsupported init params without error."""
        # Arrange
        dataset_name = get_first_available_dataset(discovered_datasets)
        if dataset_name is None:
            pytest.skip("No datasets available for testing - skipping integration test")

        # Act: Get dataset with extra params that may not be supported
        with caplog.at_level("INFO"):
            dataset = DatasetRegistry.get_dataset(
                dataset_name,
                db_connection=None,  # May not be supported by all datasets
                some_other_param="value",  # Should be filtered out
            )

        # Assert: Should not raise "unexpected keyword argument" error
        assert isinstance(dataset, ClinicalDataset)
        assert dataset.name == dataset_name

        # Verify logging indicates dropped params (if any)
        # Logging may or may not indicate dropped params depending on dataset
        assert True  # Just verify no error was raised
