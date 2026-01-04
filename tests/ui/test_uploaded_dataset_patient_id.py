"""
Tests for UploadedDataset patient_id handling and error recovery.

Integration tests for how UploadedDataset handles patient_id:
- Error recovery when patient_id is missing from loaded data
- Regeneration logic in get_cohort()
- Error messages and logging

Note: Unit tests for VariableTypeDetector.ensure_patient_id() are in
test_composite_identifier.py. This file focuses on UploadedDataset integration.
"""

import json

import pandas as pd
import pytest
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.datasets.uploaded.definition import UploadedDataset
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


class TestUploadedDatasetPatientId:
    """Tests for patient_id handling in UploadedDataset."""

    def test_get_cohort_with_missing_patient_id_regenerates(self, tmp_path):
        """Test that get_cohort regenerates patient_id if missing from loaded data."""
        # Create storage
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create test data without patient_id
        test_data = pd.DataFrame(
            {
                "race": ["A", "B", "C"],
                "gender": ["M", "F", "M"],
                "age": [50, 60, 70],
            }
        )

        # Create upload metadata with patient_id mapping but data doesn't have it
        upload_id = "test_upload_123"
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "patient_id",  # Mapped to patient_id but column missing
                "outcome": None,
                "predictors": ["race", "gender", "age"],
            },
            "synthetic_id_metadata": {
                "patient_id": {
                    "patient_id_source": "composite",
                    "patient_id_columns": ["race", "gender"],
                }
            },
        }

        # Save data directly (bypassing normal save which would create patient_id)
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        # get_cohort should regenerate patient_id
        cohort = dataset.get_cohort()

        # Should have patient_id
        assert UnifiedCohort.PATIENT_ID in cohort.columns
        assert len(cohort[UnifiedCohort.PATIENT_ID]) == 3
        # All patient_ids should be unique
        assert cohort[UnifiedCohort.PATIENT_ID].nunique() == 3

    def test_get_cohort_with_missing_patient_id_raises_error_if_cannot_regenerate(self, tmp_path):
        """Test that get_cohort raises error if patient_id cannot be regenerated."""
        # Create storage
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create test data with all identical rows (no unique combination possible)
        test_data = pd.DataFrame(
            {
                "col1": ["A", "A", "A"],
                "col2": ["B", "B", "B"],
            }
        )

        # Create upload metadata
        upload_id = "test_upload_456"
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": None,
                "predictors": ["col1", "col2"],
            },
        }

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        # load() should raise error during migration (convert_schema validates)
        with pytest.raises(ValueError, match="Patient ID column"):
            dataset.load()

    def test_get_cohort_with_existing_patient_id_uses_it(self, tmp_path):
        """Test that get_cohort uses existing patient_id if present."""
        # Create storage
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create test data with patient_id already present
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "race": ["A", "B", "C"],
                "age": [50, 60, 70],
            }
        )

        # Create upload metadata
        upload_id = "test_upload_789"
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "patient_id",
                "outcome": None,
                "predictors": ["race", "age"],
            },
        }

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        # get_cohort should use existing patient_id
        cohort = dataset.get_cohort()

        assert UnifiedCohort.PATIENT_ID in cohort.columns
        assert cohort[UnifiedCohort.PATIENT_ID].tolist() == ["P001", "P002", "P003"]

    def test_get_cohort_with_wrong_patient_id_column_name_raises_error(self, tmp_path):
        """Test that get_cohort raises error if mapped column doesn't exist."""
        # Create storage
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create test data
        test_data = pd.DataFrame(
            {
                "race": ["A", "B", "C"],
                "age": [50, 60, 70],
            }
        )

        # Create upload metadata with wrong column name
        upload_id = "test_upload_wrong"
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "nonexistent_column",  # Column doesn't exist
                "outcome": None,
                "predictors": ["race", "age"],
            },
        }

        # Save data
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        # load() should raise error during migration (convert_schema validates)
        with pytest.raises(ValueError, match="Patient ID column"):
            dataset.load()

    def test_get_cohort_with_renamed_patient_id_column_uses_patient_id(self, tmp_path):
        """
        Test that get_cohort uses 'patient_id' if mapped column name not found but 'patient_id' exists.

        This handles the case where ensure_patient_id() renamed the column to 'patient_id'
        during ingestion, but metadata still references the original column name (e.g., 'secret_name').
        """
        # Create storage
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create test data with 'patient_id' (renamed during ingestion)
        # but metadata says it should be 'secret_name'
        test_data = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],  # Column was renamed to patient_id
                "race": ["A", "B", "C"],
                "age": [50, 60, 70],
            }
        )

        # Create upload metadata with original column name in mapping
        upload_id = "test_upload_renamed"
        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": {
                "patient_id": "secret_name",  # Metadata says 'secret_name' but CSV has 'patient_id'
                "outcome": None,
                "predictors": ["race", "age"],
            },
            "synthetic_id_metadata": {
                "patient_id": {
                    "patient_id_source": "single_column",
                    "patient_id_columns": ["secret_name"],  # Original column name
                }
            },
        }

        # Save data with 'patient_id' column (as it would be after ingestion)
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.to_csv(csv_path, index=False)

        # Save metadata
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Create dataset and load
        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        # get_cohort should use 'patient_id' even though metadata says 'secret_name'
        cohort = dataset.get_cohort()

        # Should have patient_id and use the values from the CSV
        assert UnifiedCohort.PATIENT_ID in cohort.columns
        assert cohort[UnifiedCohort.PATIENT_ID].tolist() == ["P001", "P002", "P003"]
