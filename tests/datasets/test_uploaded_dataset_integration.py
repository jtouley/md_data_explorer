"""
Integration tests for UploadedDataset semantic layer end-to-end flow.

Phase 3: Integration tests for NL query flow with single-table uploads.
"""

import json

import pandas as pd

from clinical_analytics.datasets.uploaded.definition import UploadedDataset
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


class TestUploadedDatasetSemanticLayerIntegration:
    """Integration tests for semantic layer with uploaded datasets."""

    def test_single_table_upload_semantic_layer_enables_nl_queries(self, tmp_path):
        """Test that single-table uploads can use semantic layer for NL queries."""
        # Arrange: Create complete single-table upload with variable_mapping
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_nl_queries"

        # Create test data
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "mortality": [0, 1, 0],
                "age": [50, 60, 70],
                "sex": ["M", "F", "M"],
            }
        )

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Create metadata with variable_mapping
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "original_filename": "test_dataset.csv",
            "dataset_name": "test_dataset",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": "mortality",
                "time_variables": {},
                "predictors": ["age", "sex"],
            },
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)

        # Act: Call get_semantic_layer() (simulating UI call)
        semantic_layer = dataset.get_semantic_layer()

        # Assert: Semantic layer available, no "Semantic layer not ready" error
        assert semantic_layer is not None
        assert dataset.semantic is not None
        # Verify semantic layer is functional
        assert hasattr(semantic_layer, "config")
        assert "column_mapping" in semantic_layer.config
        assert "outcomes" in semantic_layer.config

    def test_multi_table_upload_semantic_layer_still_works_regression(self, tmp_path):
        """Test that multi-table uploads still work (regression test)."""
        # Arrange: Create complete multi-table upload with inferred_schema
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_multi_table_regression"

        # Create test data
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "mortality": [0, 1],
                "age": [50, 60],
            }
        )

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Create tables directory and save individual table
        tables_dir = storage.raw_dir / f"{upload_id}_tables"
        tables_dir.mkdir(exist_ok=True)
        patients_table = pd.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "gender": ["M", "F"],
            }
        )
        patients_table.to_csv(tables_dir / "patients.csv", index=False)

        # Create metadata with inferred_schema
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "original_filename": "test_dataset.zip",
            "dataset_name": "test_dataset",
            "tables": ["patients"],
            "inferred_schema": {
                "column_mapping": {"patient_id": "patient_id"},
                "outcomes": {"mortality": {"source_column": "mortality", "type": "binary"}},
                "time_zero": {},
                "predictors": ["age"],
                "categorical_columns": [],
            },
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)

        # Act: Call get_semantic_layer()
        semantic_layer = dataset.get_semantic_layer()

        # Assert: Semantic layer available, all tables registered
        assert semantic_layer is not None
        assert dataset.semantic is not None
        # Verify multi-table path was used
        assert "mortality" in semantic_layer.config["outcomes"]
        # Verify tables directory was accessed (multi-table upload)
        assert tables_dir.exists()

    def test_semantic_layer_config_structure_matches_expected_format(self, tmp_path):
        """Test that semantic layer config structure matches SemanticLayer expectations."""
        # Arrange: Create single-table upload
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_config_structure"

        # Create test data
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "mortality": [0, 1],
                "age": [50, 60],
                "sex": ["M", "F"],
            }
        )

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Create metadata
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "original_filename": "test_dataset.csv",
            "dataset_name": "test_dataset",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": "mortality",
                "time_variables": {},
                "predictors": ["age", "sex"],
            },
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)

        # Act: Get semantic layer config
        semantic_layer = dataset.get_semantic_layer()
        config = semantic_layer.config

        # Assert: Config structure matches SemanticLayer expectations
        # Required top-level keys
        assert "name" in config
        assert "display_name" in config
        assert "status" in config
        assert "init_params" in config
        assert "column_mapping" in config
        assert "outcomes" in config
        assert "time_zero" in config
        assert "analysis" in config

        # Analysis sub-structure
        assert "default_outcome" in config["analysis"]
        assert "default_predictors" in config["analysis"]
        assert "categorical_variables" in config["analysis"]

        # Verify types
        assert isinstance(config["column_mapping"], dict)
        assert isinstance(config["outcomes"], dict)
        assert isinstance(config["time_zero"], dict)
        assert isinstance(config["analysis"], dict)
        assert isinstance(config["analysis"]["default_predictors"], list)
        assert isinstance(config["analysis"]["categorical_variables"], list)

    def test_single_table_semantic_layer_init_params_has_absolute_path(self, tmp_path):
        """Test that init_params has absolute path (integration-level verification)."""
        # Arrange: Create single-table upload with CSV file
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_absolute_path"

        # Create test data
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "age": [50, 60],
            }
        )

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)
        expected_absolute_path = str(csv_path.resolve())

        # Create metadata
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "original_filename": "test_dataset.csv",
            "dataset_name": "test_dataset",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": None,
                "time_variables": {},
                "predictors": ["age"],
            },
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)

        # Act: Get semantic layer and check config
        semantic_layer = dataset.get_semantic_layer()
        config = semantic_layer.config

        # Assert: config["init_params"]["source_path"] is absolute path
        assert "init_params" in config
        assert "source_path" in config["init_params"]
        source_path = config["init_params"]["source_path"]
        assert source_path == expected_absolute_path
        # Verify it's an absolute path (starts with / on Unix, or drive letter on Windows)
        assert source_path.startswith("/") or (len(source_path) > 1 and source_path[1] == ":")
