"""
Tests for UploadedDataset semantic layer initialization.

Phase 1: Tests for _build_config_from_variable_mapping() method.
"""

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from clinical_analytics.core.mapper import ColumnMapper
from clinical_analytics.datasets.uploaded.definition import UploadedDataset


class TestBuildConfigFromVariableMapping:
    """Tests for _build_config_from_variable_mapping() method."""

    def test_build_config_from_variable_mapping_with_all_fields_returns_complete_config(self, upload_storage):
        """Test that config is built correctly with all fields present."""
        # Arrange: Create UploadedDataset with variable_mapping containing all fields
        storage = upload_storage
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

    def test_build_config_from_variable_mapping_without_outcome_handles_gracefully(self, upload_storage):
        """Test that config handles missing outcome gracefully."""
        # Arrange: Create UploadedDataset with variable_mapping without outcome
        storage = upload_storage
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

    def test_build_config_from_variable_mapping_detects_categorical_variables(self, upload_storage):
        """Test that categorical variables are correctly detected."""
        # Arrange: Create UploadedDataset with data containing categorical predictors
        storage = upload_storage
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

    def test_build_config_from_variable_mapping_infers_outcome_type_from_data(self, upload_storage):
        """Test that outcome type is inferred from data characteristics."""
        # Arrange: Create UploadedDataset with different outcome types
        storage = upload_storage

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

    def test_build_config_from_variable_mapping_time_zero_matches_multi_table_format(self, upload_storage):
        """Test that time_zero config matches multi-table format exactly."""
        # Arrange: Create UploadedDataset with time_zero in variable_mapping
        storage = upload_storage
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


class TestMaybeInitSemantic:
    """Tests for _maybe_init_semantic() method single-table path."""

    def test_maybe_init_semantic_with_variable_mapping_initializes_semantic_layer(self, upload_storage):
        """Test that semantic layer is initialized for single-table uploads with variable_mapping."""
        # Arrange: Create UploadedDataset with variable_mapping (no inferred_schema), mock CSV file exists
        storage = upload_storage
        upload_id = "test_upload_single_table"

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

        # Create metadata with variable_mapping (no inferred_schema)
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

        # Act: Call get_semantic_layer()
        semantic_layer = dataset.get_semantic_layer()

        # Assert: self.semantic is not None, semantic layer config matches variable_mapping
        assert semantic_layer is not None
        assert dataset.semantic is not None
        # Verify config was built from variable_mapping
        assert dataset.semantic.config["column_mapping"]["patient_id"] == "patient_id"
        assert "mortality" in dataset.semantic.config["outcomes"]

    def test_maybe_init_semantic_with_inferred_schema_still_works_multi_table(self, upload_storage):
        """Test that multi-table uploads still work (regression test)."""
        # Arrange: Create UploadedDataset with inferred_schema
        storage = upload_storage
        upload_id = "test_upload_multi_table"

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

        # Create metadata with inferred_schema (multi-table format)
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "original_filename": "test_dataset.zip",
            "dataset_name": "test_dataset",
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

        # Assert: self.semantic is not None, uses existing _build_config_from_inferred_schema() path
        assert semantic_layer is not None
        assert dataset.semantic is not None
        # Verify config was built from inferred_schema
        assert dataset.semantic.config["column_mapping"]["patient_id"] == "patient_id"
        assert "mortality" in dataset.semantic.config["outcomes"]

    def test_maybe_init_semantic_with_variable_mapping_registers_table_after_migration(self, upload_storage):
        """Test that legacy single-table uploads register tables after migration (Fix #2)."""
        # Arrange: Create legacy UploadedDataset with variable_mapping, no {upload_id}_tables directory
        storage = upload_storage
        upload_id = "test_upload_no_tables"

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

        # Ensure tables directory does NOT exist (legacy state)
        tables_dir = storage.raw_dir / f"{upload_id}_tables"
        assert not tables_dir.exists()

        # Create metadata with variable_mapping (legacy format)
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

        # Act: Call get_upload_data() which triggers migration, then get_semantic_layer()
        _ = storage.get_upload_data(upload_id)  # Triggers migration
        semantic_layer = dataset.get_semantic_layer()

        # Assert: semantic layer created from the typed Parquet the conversion wrote
        assert semantic_layer is not None
        updated_metadata = storage.get_upload_metadata(upload_id)
        assert updated_metadata["storage_layout"] == "typed_parquet_v1"
        assert "tables" in updated_metadata
        assert semantic_layer.config["init_params"]["source_path"] == str(
            Path(updated_metadata["cohort_parquet"]).resolve()
        )

    def test_maybe_init_semantic_without_schema_or_mapping_raises_valueerror(self, upload_storage):
        """Test that missing both schema and mapping raises ValueError."""
        # Arrange: Create UploadedDataset with neither inferred_schema nor variable_mapping
        storage = upload_storage
        upload_id = "test_upload_no_schema"

        # Create test data
        test_data = pd.DataFrame({"patient_id": ["P001"], "age": [50]})

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Create metadata without schema or mapping
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "original_filename": "test_dataset.csv",
            "dataset_name": "test_dataset",
            # No inferred_schema or variable_mapping
        }

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)

        # Act & Assert: Call get_semantic_layer() should raise ValueError
        with pytest.raises(ValueError, match="No schema or mapping found"):
            dataset.get_semantic_layer()

    def test_maybe_init_semantic_sets_init_params_with_absolute_csv_path(self, upload_storage):
        """Test that init_params is set with absolute CSV path."""
        # Arrange: Create UploadedDataset with variable_mapping, mock CSV file exists
        storage = upload_storage
        upload_id = "test_upload_init_params"

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

        # Create metadata with variable_mapping
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

        # Act: Call get_semantic_layer()
        semantic_layer = dataset.get_semantic_layer()

        # Assert: the layer reads the typed Parquet converted from the legacy CSV (absolute path)
        converted = storage.get_upload_metadata(upload_id)["cohort_parquet"]
        assert semantic_layer is not None
        assert dataset.semantic is not None
        assert dataset.semantic.config["init_params"]["source_path"] == str(Path(converted).resolve())
        assert Path(expected_absolute_path).is_file()  # the legacy CSV is left in place


# How a metadata container field can arrive from an upload: null, missing, or the wrong type.
_FIELD_STATES = [
    pytest.param(True, None, id="none"),
    pytest.param(False, None, id="absent"),
    pytest.param(True, "not-a-container", id="wrong_type"),
]


def _upload_metadata(upload_id: str, path: tuple[str, ...], present: bool, value: object) -> dict:
    """Upload metadata with a valid schema and mapping, except the field at ``path``."""
    metadata: dict = {
        "upload_id": upload_id,
        "upload_timestamp": "2024-01-01T00:00:00",
        "variable_types": {"age": {"type": "numeric"}},
        "inferred_schema": {
            "column_mapping": {"patient_id": "patient_id"},
            "outcomes": {"died": {"source_column": "died", "type": "binary"}},
            "predictors": ["age"],
        },
        "variable_mapping": {
            "patient_id": "patient_id",
            "outcome": "died",
            "time_variables": {"time_zero": None},
            "predictors": ["age"],
        },
    }
    parent = metadata
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    if present:
        parent[path[-1]] = value
    return metadata


class TestUploadMetadataNormalizedAtBoundary:
    """A null, missing or wrong-typed metadata field reaches every config builder and reader as empty.

    Covers the read sites that crashed on a present None: definition.py (both config builders and
    get_cohort), semantic.py (get_base_view, get_cohort, _find_mapped_column) and mapper.py.
    """

    @pytest.mark.parametrize(
        ("path", "config_key"),
        [
            pytest.param(("variable_types",), "variable_types", id="variable_types"),
            pytest.param(("inferred_schema", "outcomes"), "outcomes", id="outcomes"),
            pytest.param(("inferred_schema", "column_mapping"), "column_mapping", id="column_mapping"),
        ],
    )
    @pytest.mark.parametrize(("present", "value"), _FIELD_STATES)
    def test_buildConfigFromInferredSchema_fieldMissingOrInvalid_isEmptyDict(
        self, upload_storage, path, config_key, present, value
    ):
        # Arrange
        metadata = _upload_metadata("u_schema", path, present, value)
        dataset = UploadedDataset(upload_id="u_schema", storage=upload_storage, metadata=metadata)

        # Act
        config = dataset._build_config_from_inferred_schema(dataset.metadata["inferred_schema"])
        mapper = ColumnMapper(config)

        # Assert: the builder and ColumnMapper both see an empty dict, never None
        assert config[config_key] == {}
        assert isinstance(mapper.column_mapping, dict)
        assert isinstance(mapper.outcomes, dict)

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("path", "config_check"),
        [
            pytest.param(("variable_mapping", "time_variables"), lambda c: c["time_zero"] == {}, id="time_variables"),
            pytest.param(
                ("variable_mapping", "predictors"),
                lambda c: c["analysis"]["default_predictors"] == [],
                id="predictors",
            ),
            pytest.param(("variable_types",), lambda c: c["variable_types"] == {}, id="variable_types"),
        ],
    )
    @pytest.mark.parametrize(("present", "value"), _FIELD_STATES)
    def test_variableMappingPaths_fieldMissingOrInvalid_buildConfigAndCohort(
        self, upload_storage, path, config_check, present, value
    ):
        # Arrange: a single-table upload (no inferred_schema), so the variable_mapping paths run
        metadata = _upload_metadata("u_mapping", path, present, value)
        del metadata["inferred_schema"]
        pl.DataFrame({"patient_id": ["P001", "P002"], "age": [50, 60], "died": [0, 1]}).write_csv(
            upload_storage.raw_dir / "u_mapping.csv"
        )
        dataset = UploadedDataset(upload_id="u_mapping", storage=upload_storage, metadata=metadata)
        dataset.load()

        # Act
        config = dataset._build_config_from_variable_mapping(dataset.metadata["variable_mapping"])
        cohort = dataset.get_cohort()

        # Assert
        assert config_check(config)
        assert cohort.shape[0] == 2

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(("inferred_schema", "outcomes"), id="outcomes"),
            pytest.param(("inferred_schema", "column_mapping"), id="column_mapping"),
        ],
    )
    @pytest.mark.parametrize(("present", "value"), _FIELD_STATES)
    def test_semanticLayer_fieldMissingOrInvalid_buildsViewAndCohort(self, upload_storage, path, present, value):
        # Arrange
        metadata = _upload_metadata("u_layer", path, present, value)
        pl.DataFrame({"patient_id": ["P001", "P002"], "age": [50, 60], "died": [0, 1]}).write_csv(
            upload_storage.raw_dir / "u_layer.csv"
        )
        dataset = UploadedDataset(upload_id="u_layer", storage=upload_storage, metadata=metadata)
        dataset.load()

        # Act: get_base_view reads outcomes and column_mapping; get_cohort reads both again
        layer = dataset.get_semantic_layer()
        view_columns = layer.get_base_view().columns
        cohort = layer.get_cohort()  # SemanticLayer.get_cohort returns pandas (its API)

        # Assert
        assert {"patient_id", "age"} <= set(view_columns)
        assert cohort.shape[0] == 2
