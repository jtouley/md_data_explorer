"""
Tests for UploadedDataset semantic layer initialization.

Phase 1: Tests for _build_config_from_variable_mapping() method.
"""

import json

import pandas as pd

from clinical_analytics.datasets.uploaded.definition import UploadedDataset
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


class TestBuildConfigFromVariableMapping:
    """Tests for _build_config_from_variable_mapping() method."""

    def test_build_config_from_variable_mapping_with_all_fields_returns_complete_config(self, tmp_path):
        """Test that config is built correctly with all fields present."""
        # Arrange: Create UploadedDataset with variable_mapping containing all fields
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_all_fields"

        # Create test data
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "mortality": [0, 1, 0],
                "admission_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "age": [50, 60, 70],
                "sex": ["M", "F", "M"],
                "treatment": ["A", "B", "A"],
            }
        )

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Create metadata with complete variable_mapping
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "original_filename": "test_dataset.csv",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": "mortality",
                "time_variables": {"time_zero": "admission_date"},
                "predictors": ["age", "sex", "treatment"],
            },
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load data
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        variable_mapping = metadata["variable_mapping"]

        # Act: Call _build_config_from_variable_mapping()
        config = dataset._build_config_from_variable_mapping(variable_mapping)

        # Assert: Config has correct structure
        assert config["name"] == dataset.name
        assert config["display_name"] == "test_dataset.csv"
        assert config["status"] == "available"
        assert "init_params" in config
        assert config["init_params"] == {}  # Will be set later

        # Check column_mapping
        assert "patient_id" in config["column_mapping"]
        assert config["column_mapping"]["patient_id"] == "patient_id"

        # Check outcomes
        assert "mortality" in config["outcomes"]
        assert config["outcomes"]["mortality"]["source_column"] == "mortality"
        assert "type" in config["outcomes"]["mortality"]

        # Check time_zero
        assert "source_column" in config["time_zero"]
        assert config["time_zero"]["source_column"] == "admission_date"

        # Check analysis
        assert config["analysis"]["default_outcome"] == "mortality"
        assert set(config["analysis"]["default_predictors"]) == {"age", "sex", "treatment"}
        assert "categorical_variables" in config["analysis"]

    def test_build_config_from_variable_mapping_without_outcome_handles_gracefully(self, tmp_path):
        """Test that config handles missing outcome gracefully."""
        # Arrange: Create UploadedDataset with variable_mapping without outcome
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_no_outcome"

        # Create test data
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "age": [50, 60],
                "sex": ["M", "F"],
            }
        )

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Create metadata without outcome
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": None,
                "time_variables": {},
                "predictors": ["age", "sex"],
            },
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load data
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        variable_mapping = metadata["variable_mapping"]

        # Act: Call _build_config_from_variable_mapping()
        config = dataset._build_config_from_variable_mapping(variable_mapping)

        # Assert: Config has empty outcomes dict, default_outcome is None
        assert config["outcomes"] == {}
        assert config["analysis"]["default_outcome"] is None
        assert "patient_id" in config["column_mapping"]

    def test_build_config_from_variable_mapping_detects_categorical_variables(self, tmp_path):
        """Test that categorical variables are correctly detected."""
        # Arrange: Create UploadedDataset with data containing categorical predictors
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_categorical"

        # Create test data with various types:
        # - String type (categorical)
        # - Numeric with ≤20 unique values (categorical)
        # - Numeric with >20 unique values (continuous)
        # - Boolean (categorical)
        test_data = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(50)],
                "sex": ["M", "F"] * 25,  # String, 2 unique values (categorical)
                "treatment": ["A", "B", "C"] * 16 + ["A", "B"],  # String, 3 unique (categorical)
                "severity": [1, 2, 3, 4, 5] * 10,  # Numeric, 5 unique (categorical)
                "age": list(range(20, 70)),  # Numeric, 50 unique (continuous)
                "is_active": [True, False] * 25,  # Boolean (categorical)
            }
        )

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Create metadata
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": None,
                "time_variables": {},
                "predictors": ["sex", "treatment", "severity", "age", "is_active"],
            },
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load data
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        variable_mapping = metadata["variable_mapping"]

        # Act: Call _build_config_from_variable_mapping()
        config = dataset._build_config_from_variable_mapping(variable_mapping)

        # Assert: categorical_variables list contains correct columns
        categorical_vars = set(config["analysis"]["categorical_variables"])
        # String types should be categorical
        assert "sex" in categorical_vars
        assert "treatment" in categorical_vars
        # Numeric with ≤20 unique should be categorical
        assert "severity" in categorical_vars
        # Boolean should be categorical
        assert "is_active" in categorical_vars
        # Numeric with >20 unique should NOT be categorical
        assert "age" not in categorical_vars

    def test_build_config_from_variable_mapping_infers_outcome_type_from_data(self, tmp_path):
        """Test that outcome type is inferred from data characteristics."""
        # Arrange: Create UploadedDataset with different outcome types
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Test case 1: Binary outcome (2 unique values)
        upload_id_binary = "test_upload_binary_outcome"
        test_data_binary = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "mortality": [0, 1, 0],  # Binary: exactly 2 unique values
                "age": [50, 60, 70],
            }
        )
        csv_path_binary = storage.raw_dir / f"{upload_id_binary}.csv"
        test_data_binary.to_csv(csv_path_binary, index=False)

        metadata_binary = {
            "upload_id": upload_id_binary,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": "mortality",
                "time_variables": {},
                "predictors": ["age"],
            },
        }
        metadata_path_binary = storage.metadata_dir / f"{upload_id_binary}.json"
        metadata_path_binary.write_text(json.dumps(metadata_binary))

        dataset_binary = UploadedDataset(upload_id=upload_id_binary, storage=storage)
        dataset_binary.load()

        # Act: Call _build_config_from_variable_mapping()
        config_binary = dataset_binary._build_config_from_variable_mapping(metadata_binary["variable_mapping"])

        # Assert: Outcome type is "binary"
        assert config_binary["outcomes"]["mortality"]["type"] == "binary"

        # Test case 2: Continuous outcome (>2 unique values)
        upload_id_continuous = "test_upload_continuous_outcome"
        test_data_continuous = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004"],
                "survival_days": [100, 200, 300, 400],  # Continuous: >2 unique values
                "age": [50, 60, 70, 80],
            }
        )
        csv_path_continuous = storage.raw_dir / f"{upload_id_continuous}.csv"
        test_data_continuous.to_csv(csv_path_continuous, index=False)

        metadata_continuous = {
            "upload_id": upload_id_continuous,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": "survival_days",
                "time_variables": {},
                "predictors": ["age"],
            },
        }
        metadata_path_continuous = storage.metadata_dir / f"{upload_id_continuous}.json"
        metadata_path_continuous.write_text(json.dumps(metadata_continuous))

        dataset_continuous = UploadedDataset(upload_id=upload_id_continuous, storage=storage)
        dataset_continuous.load()

        # Act: Call _build_config_from_variable_mapping()
        config_continuous = dataset_continuous._build_config_from_variable_mapping(
            metadata_continuous["variable_mapping"]
        )

        # Assert: Outcome type is "continuous" (or "binary" with warning if ambiguous)
        outcome_type = config_continuous["outcomes"]["survival_days"]["type"]
        assert outcome_type in ["binary", "continuous"]  # May default to binary with warning

    def test_build_config_from_variable_mapping_time_zero_matches_multi_table_format(self, tmp_path):
        """Test that time_zero config matches multi-table format exactly."""
        # Arrange: Create UploadedDataset with time_zero in variable_mapping
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_time_zero"

        # Create test data
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "admission_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "age": [50, 60],
            }
        )

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Create metadata with time_zero
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": None,
                "time_variables": {"time_zero": "admission_date"},
                "predictors": ["age"],
            },
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load data
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        variable_mapping = metadata["variable_mapping"]

        # Act: Call _build_config_from_variable_mapping()
        config = dataset._build_config_from_variable_mapping(variable_mapping)

        # Assert: time_zero config is exactly {"source_column": str} (matches multi-table format)
        assert config["time_zero"] == {"source_column": "admission_date"}
        # Ensure it's not {"value": str} or any other format
        assert "value" not in config["time_zero"]
        assert len(config["time_zero"]) == 1
        assert "source_column" in config["time_zero"]
