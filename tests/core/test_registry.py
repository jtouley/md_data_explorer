"""
Tests for the dataset registry module.
"""

import pytest
import polars as pl
from pathlib import Path
from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.core.dataset import ClinicalDataset


class TestDatasetRegistry:
    """Test suite for DatasetRegistry."""

    def test_discover_datasets(self):
        """Test that registry discovers datasets."""
        # Reset registry to ensure clean state
        DatasetRegistry.reset()

        datasets = DatasetRegistry.discover_datasets()

        assert isinstance(datasets, dict)
        assert len(datasets) > 0
        assert 'covid_ms' in datasets or 'sepsis' in datasets

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
            assert 'name' in info
            assert 'available' in info
            assert 'config' in info

    def test_get_dataset_factory(self):
        """Test factory method for creating datasets."""
        DatasetRegistry.reset()

        dataset = DatasetRegistry.get_dataset('covid_ms')

        assert isinstance(dataset, ClinicalDataset)
        assert dataset.name == 'covid_ms'

    def test_get_all_dataset_info(self):
        """Test getting info for all datasets."""
        DatasetRegistry.reset()

        all_info = DatasetRegistry.get_all_dataset_info()

        assert isinstance(all_info, dict)
        assert len(all_info) > 0

        for name, info in all_info.items():
            assert 'name' in info
            assert 'config' in info

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
            DatasetRegistry.get_dataset('nonexistent_dataset')

    def test_load_config(self):
        """Test loading configuration from YAML."""
        DatasetRegistry.reset()
        DatasetRegistry.load_config()

        assert DatasetRegistry._config_loaded
        assert isinstance(DatasetRegistry._configs, dict)

    def test_load_config_nonexistent_file(self):
        """Test loading config with nonexistent file path."""
        DatasetRegistry.reset()
        
        # Should not raise error, just set empty config
        DatasetRegistry.load_config(Path('/nonexistent/path/config.yaml'))
        
        assert DatasetRegistry._config_loaded
        assert DatasetRegistry._configs == {}

    def test_register_from_dataframe(self):
        """Test registering dataset from DataFrame."""
        DatasetRegistry.reset()

        # Create test DataFrame
        df = pl.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'age': [45, 62, 38],
            'outcome': [1, 0, 1]
        })

        config = DatasetRegistry.register_from_dataframe(
            'test_dataset',
            df,
            display_name='Test Dataset'
        )

        assert isinstance(config, dict)
        assert config['name'] == 'test_dataset'
        assert config['display_name'] == 'Test Dataset'
        assert config['status'] == 'auto-inferred'
        assert config['row_count'] == 3
        assert config['column_count'] == 3

        # Check DataFrame is stored
        stored_df = DatasetRegistry.get_auto_inferred_dataframe('test_dataset')
        assert stored_df is not None
        assert stored_df.height == 3

    def test_get_auto_inferred_dataframe(self):
        """Test retrieving auto-inferred DataFrame."""
        DatasetRegistry.reset()

        df = pl.DataFrame({'col1': [1, 2, 3]})
        DatasetRegistry.register_from_dataframe('test_df', df)

        retrieved = DatasetRegistry.get_auto_inferred_dataframe('test_df')
        assert retrieved is not None
        assert retrieved.equals(df)

        # Test nonexistent dataset
        assert DatasetRegistry.get_auto_inferred_dataframe('nonexistent') is None

    def test_get_dataset_with_override_params(self):
        """Test getting dataset with override parameters."""
        DatasetRegistry.reset()

        # Get dataset with override params
        dataset = DatasetRegistry.get_dataset(
            'covid_ms',
            source_path='/custom/path'
        )

        assert isinstance(dataset, ClinicalDataset)
        # Override params should be applied
        if dataset.source_path:
            assert str(dataset.source_path) == '/custom/path'

