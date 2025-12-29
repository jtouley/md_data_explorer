"""
Tests for unified semantic layer registration (Phase 3 - ADR007).

Tests that both single-table and multi-table uploads register tables identically.
"""


import polars as pl
import pytest

from clinical_analytics.datasets.uploaded.definition import UploadedDataset
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


class TestUnifiedSemanticLayerRegistration:
    """Test suite for unified semantic layer registration."""

    def test_single_table_upload_registers_individual_table(self, tmp_path):
        """Test that single-table uploads register individual table in DuckDB."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create single-table upload with inferred_schema
        df = pl.DataFrame({
            "patient_id": ["P001", "P002"],
            "age": [25, 30],
            "outcome": [0, 1],
        })

        upload_id = "test_single_upload"

        # Save CSV
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        df.write_csv(csv_path)

        # Save individual table (Phase 2 persistence)
        tables_dir = storage.raw_dir / f"{upload_id}_tables"
        tables_dir.mkdir(exist_ok=True)
        df.write_csv(tables_dir / "patient_outcomes.csv")

        # Create metadata with inferred_schema
        metadata = {
            "upload_id": upload_id,
            "dataset_name": "test_single",
            "upload_timestamp": "2024-01-01T00:00:00",
            "tables": ["patient_outcomes"],
            "inferred_schema": {
                "column_mapping": {"patient_id": "patient_id"},
                "outcomes": {
                    "outcome": {
                        "source_column": "outcome",
                        "type": "binary",
                        "confidence": 0.9,
                    }
                },
                "granularities": ["patient_level"],
            },
        }

        import json
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Act
        dataset = UploadedDataset(upload_id, storage)
        semantic = dataset.get_semantic_layer()

        # Assert
        assert semantic is not None
        # Verify table is registered in DuckDB (check via query)
        result = semantic.con.con.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in result]
        assert any("patient_outcomes" in name for name in table_names)

    def test_multi_table_upload_registers_all_tables(self, tmp_path):
        """Test that multi-table uploads register all individual tables."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create multi-table upload
        patients_df = pl.DataFrame({
            "patient_id": ["P001", "P002"],
            "name": ["Alice", "Bob"],
        })

        admissions_df = pl.DataFrame({
            "admission_id": ["A001", "A002"],
            "patient_id": ["P001", "P002"],
        })

        upload_id = "test_multi_upload"

        # Save unified cohort CSV
        cohort_df = pl.DataFrame({
            "patient_id": ["P001", "P002"],
            "outcome": [0, 1],
        })
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        cohort_df.write_csv(csv_path)

        # Save individual tables
        tables_dir = storage.raw_dir / f"{upload_id}_tables"
        tables_dir.mkdir(exist_ok=True)
        patients_df.write_csv(tables_dir / "patients.csv")
        admissions_df.write_csv(tables_dir / "admissions.csv")

        # Create metadata
        metadata = {
            "upload_id": upload_id,
            "dataset_name": "test_multi",
            "upload_timestamp": "2024-01-01T00:00:00",
            "tables": ["patients", "admissions"],
            "inferred_schema": {
                "column_mapping": {"patient_id": "patient_id"},
                "outcomes": {},
                "granularities": ["patient_level", "admission_level"],
            },
        }

        import json
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Act
        dataset = UploadedDataset(upload_id, storage)
        semantic = dataset.get_semantic_layer()

        # Assert
        assert semantic is not None
        result = semantic.con.con.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in result]
        assert any("patients" in name for name in table_names)
        assert any("admissions" in name for name in table_names)

    def test_semantic_layer_registration_is_idempotent(self, tmp_path):
        """Test that re-initializing semantic layer doesn't clobber data (IF NOT EXISTS)."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pl.DataFrame({
            "patient_id": ["P001", "P002"],
            "outcome": [0, 1],
        })

        upload_id = "test_idempotent"

        # Save files
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        df.write_csv(csv_path)

        tables_dir = storage.raw_dir / f"{upload_id}_tables"
        tables_dir.mkdir(exist_ok=True)
        df.write_csv(tables_dir / "data.csv")

        metadata = {
            "upload_id": upload_id,
            "dataset_name": "test_idempotent",
            "upload_timestamp": "2024-01-01T00:00:00",
            "tables": ["data"],
            "inferred_schema": {
                "column_mapping": {"patient_id": "patient_id"},
                "outcomes": {},
                "granularities": ["patient_level"],
            },
        }

        import json
        (storage.metadata_dir / f"{upload_id}.json").write_text(json.dumps(metadata))

        # Act - Call get_semantic_layer() twice (public API)
        dataset1 = UploadedDataset(upload_id, storage)
        semantic1 = dataset1.get_semantic_layer()

        # Get row count from first initialization
        # Find the actual table name (includes hashes for uniqueness)
        tables1 = semantic1.con.con.execute("SHOW TABLES").fetchall()
        data_tables1 = [t[0] for t in tables1 if "data" in t[0]]
        assert len(data_tables1) == 1, f"Expected 1 data table, found: {data_tables1}"
        table_name = data_tables1[0]
        first_count = semantic1.con.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        # Second initialization - creates new dataset instance
        dataset2 = UploadedDataset(upload_id, storage)
        semantic2 = dataset2.get_semantic_layer()

        # Get row count after second initialization (same table should exist)
        tables2 = semantic2.con.con.execute("SHOW TABLES").fetchall()
        data_tables2 = [t[0] for t in tables2 if "data" in t[0]]
        assert len(data_tables2) == 1, f"Expected 1 data table after re-init, found: {data_tables2}"
        second_count = semantic2.con.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        # Assert - Data not lost (IF NOT EXISTS prevented clobbering)
        assert first_count == second_count == 2


class TestGranularityValidation:
    """Test suite for runtime granularity validation."""

    def test_get_cohort_validates_requested_granularity_exists(self, tmp_path):
        """Test that get_cohort() validates requested granularity is supported."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pl.DataFrame({
            "patient_id": ["P001", "P002"],
            "age": [25, 30],
        })

        upload_id = "test_validation"

        # Save CSV
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        df.write_csv(csv_path)

        # Metadata with only patient_level granularity
        metadata = {
            "upload_id": upload_id,
            "dataset_name": "test_validation",
            "upload_timestamp": "2024-01-01T00:00:00",
            "inferred_schema": {
                "column_mapping": {"patient_id": "patient_id"},
                "outcomes": {},
                "granularities": ["patient_level"],  # Only patient_level
            },
        }

        import json
        (storage.metadata_dir / f"{upload_id}.json").write_text(json.dumps(metadata))

        # Act & Assert
        dataset = UploadedDataset(upload_id, storage)
        dataset.load()

        # patient_level should work
        cohort = dataset.get_cohort(granularity="patient_level")
        assert cohort is not None

        # admission_level should fail with clear error
        with pytest.raises(ValueError, match="does not support admission_level granularity"):
            dataset.get_cohort(granularity="admission_level")

    def test_get_cohort_supports_all_inferred_granularities(self, tmp_path):
        """Test that dataset supports all granularities inferred from columns."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pl.DataFrame({
            "patient_id": ["P001", "P002"],
            "admission_id": ["A001", "A002"],
            "event_timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
        })

        upload_id = "test_all_granularities"
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        df.write_csv(csv_path)

        metadata = {
            "upload_id": upload_id,
            "dataset_name": "test_all_gran",
            "upload_timestamp": "2024-01-01T00:00:00",
            "inferred_schema": {
                "column_mapping": {"patient_id": "patient_id"},
                "outcomes": {},
                "granularities": ["patient_level", "admission_level", "event_level"],
            },
        }

        import json
        (storage.metadata_dir / f"{upload_id}.json").write_text(json.dumps(metadata))

        # Act & Assert
        dataset = UploadedDataset(upload_id, storage)
        dataset.load()

        # All granularities should work (no ValueError)
        cohort_patient = dataset.get_cohort(granularity="patient_level")
        assert cohort_patient is not None

        cohort_admission = dataset.get_cohort(granularity="admission_level")
        assert cohort_admission is not None

        cohort_event = dataset.get_cohort(granularity="event_level")
        assert cohort_event is not None


class TestNoConditionalLogic:
    """Test suite verifying no upload-type conditionals in semantic layer."""

    def test_both_upload_types_use_same_registration_code_path(self, tmp_path):
        """Test that single-table and multi-table use identical registration logic."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create two uploads with inferred_schema (no variable_mapping)
        for name, table_name in [("single", "data"), ("multi", "patients")]:
            df = pl.DataFrame({"patient_id": ["P001"], "age": [25]})
            upload_id = f"test_{name}"

            csv_path = storage.raw_dir / f"{upload_id}.csv"
            df.write_csv(csv_path)

            tables_dir = storage.raw_dir / f"{upload_id}_tables"
            tables_dir.mkdir(exist_ok=True)
            df.write_csv(tables_dir / f"{table_name}.csv")

            metadata = {
                "upload_id": upload_id,
                "dataset_name": f"test_{name}",
                "upload_timestamp": "2024-01-01T00:00:00",
                "tables": [table_name],
                "inferred_schema": {
                    "column_mapping": {"patient_id": "patient_id"},
                    "outcomes": {},
                    "granularities": ["patient_level"],
                },
            }

            import json
            (storage.metadata_dir / f"{upload_id}.json").write_text(json.dumps(metadata))

        # Act
        single_dataset = UploadedDataset("test_single", storage)
        multi_dataset = UploadedDataset("test_multi", storage)

        single_semantic = single_dataset.get_semantic_layer()
        multi_semantic = multi_dataset.get_semantic_layer()

        # Assert - Both have semantic layers (no conditionals blocked either)
        assert single_semantic is not None
        assert multi_semantic is not None

        # Both have tables registered
        single_tables = single_semantic.con.con.execute("SHOW TABLES").fetchall()
        multi_tables = multi_semantic.con.con.execute("SHOW TABLES").fetchall()

        assert len(single_tables) > 0
        assert len(multi_tables) > 0
