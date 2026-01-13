"""Test that registry discovers only uploaded dataset after built-in removal."""

from clinical_analytics.core.registry import DatasetRegistry


class TestDatasetRegistryUploadedOnly:
    """Test suite for uploaded-only dataset registry."""

    def test_dataset_registry_uploaded_only_discovers_uploaded(self):
        """Test that registry discovers only the uploaded dataset class."""
        # Arrange
        DatasetRegistry.reset()

        # Act
        DatasetRegistry.discover_datasets()
        datasets = DatasetRegistry.list_datasets()

        # Assert
        assert len(datasets) == 1, f"Expected 1 dataset, found {len(datasets)}: {datasets}"
        assert "uploaded" in datasets, "Expected 'uploaded' dataset to be discovered"
        assert "covid_ms" not in datasets, "Built-in 'covid_ms' should not be discovered"
        assert "mimic3" not in datasets, "Built-in 'mimic3' should not be discovered"
        assert "sepsis" not in datasets, "Built-in 'sepsis' should not be discovered"

    def test_dataset_registry_uploaded_only_no_import_errors(self):
        """Test that dataset package imports without errors after removal."""
        import importlib.util

        # Act
        datasets_spec = importlib.util.find_spec("clinical_analytics.datasets")
        uploaded_spec = importlib.util.find_spec("clinical_analytics.datasets.uploaded.definition")

        # Assert
        assert datasets_spec is not None, "clinical_analytics.datasets should be importable"
        assert uploaded_spec is not None, "UploadedDataset definition should be importable"
