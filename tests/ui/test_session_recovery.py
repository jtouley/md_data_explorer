"""
Tests for session recovery functionality.

Tests follow AAA pattern and verify:
- Dataset restoration on app startup
- Handling missing metadata
- Dataset selection UI integration
"""

import polars as pl
import pytest


@pytest.fixture
def sample_datasets(tmp_path):
    """Create sample datasets for testing recovery."""
    from clinical_analytics.storage.datastore import DataStore
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    storage = UserDatasetStorage(upload_dir=tmp_path / "uploads")
    db_path = tmp_path / "analytics.duckdb"
    datastore = DataStore(db_path)

    # Create two sample uploads
    datasets = []
    for i in range(2):
        upload_id = f"test_upload_00{i + 1}"
        dataset_version = f"v{i + 1}"
        df = pl.DataFrame(
            {
                "patient_id": list(range(1 + i * 10, 11 + i * 10)),
                "age": [25 + j for j in range(10)],
            }
        )

        # Save to DuckDB
        datastore.save_table(
            table_name="patients",
            data=df,
            upload_id=upload_id,
            dataset_version=dataset_version,
        )

        # Save metadata
        metadata = {
            "upload_id": upload_id,
            "dataset_version": dataset_version,
            "dataset_name": f"Test Dataset {i + 1}",
            "tables": ["patients"],
            "created_at": f"2025-12-30T10:{i}0:00Z",
        }
        storage.metadata_dir.mkdir(parents=True, exist_ok=True)
        import json

        with open(storage.metadata_dir / f"{upload_id}.json", "w") as f:
            json.dump(metadata, f)

        datasets.append({"upload_id": upload_id, "metadata": metadata})

    datastore.close()
    return {"storage": storage, "db_path": db_path, "datasets": datasets}


class TestRestoreDatasets:
    """Test restore_datasets function."""

    def test_restore_datasets_detects_existing_uploads(self, sample_datasets):
        """restore_datasets should detect and return all uploads with metadata."""
        # Arrange
        from clinical_analytics.ui.app_utils import restore_datasets

        storage = sample_datasets["storage"]
        db_path = sample_datasets["db_path"]

        # Act
        restored = restore_datasets(storage, db_path)

        # Assert: Returns all upload metadata
        assert len(restored) == 2
        upload_ids = [d["upload_id"] for d in restored]
        assert "test_upload_001" in upload_ids
        assert "test_upload_002" in upload_ids

        # Verify metadata fields
        for dataset in restored:
            assert "upload_id" in dataset
            assert "dataset_name" in dataset
            assert "created_at" in dataset

    def test_restore_datasets_handles_missing_metadata(self, tmp_path):
        """restore_datasets should skip datasets with missing metadata gracefully."""
        # Arrange: DuckDB table exists but metadata JSON missing
        from clinical_analytics.storage.datastore import DataStore
        from clinical_analytics.ui.app_utils import restore_datasets
        from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

        storage = UserDatasetStorage(upload_dir=tmp_path / "uploads")
        db_path = tmp_path / "analytics.duckdb"
        datastore = DataStore(db_path)

        # Create table in DuckDB without metadata
        df = pl.DataFrame({"patient_id": [1, 2], "age": [25, 30]})
        datastore.save_table(
            table_name="patients",
            data=df,
            upload_id="orphan_upload",
            dataset_version="v1",
        )
        datastore.close()

        # Act: Restore (should skip orphan upload)
        restored = restore_datasets(storage, db_path)

        # Assert: Returns empty list (no metadata found)
        assert len(restored) == 0

    def test_restore_datasets_returns_empty_list_when_no_db(self, tmp_path):
        """restore_datasets should return empty list when DuckDB doesn't exist."""
        # Arrange: No DuckDB file
        from clinical_analytics.ui.app_utils import restore_datasets
        from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

        storage = UserDatasetStorage(upload_dir=tmp_path / "uploads")
        db_path = tmp_path / "nonexistent.duckdb"

        # Act
        restored = restore_datasets(storage, db_path)

        # Assert: Returns empty list
        assert restored == []

    def test_restore_datasets_sorts_by_created_at_descending(self, sample_datasets):
        """restore_datasets should return datasets sorted by creation time (newest first)."""
        # Arrange
        from clinical_analytics.ui.app_utils import restore_datasets

        storage = sample_datasets["storage"]
        db_path = sample_datasets["db_path"]

        # Act
        restored = restore_datasets(storage, db_path)

        # Assert: Sorted by created_at descending (newest first)
        assert len(restored) == 2
        # Upload 002 created at 10:10, Upload 001 at 10:00
        assert restored[0]["upload_id"] == "test_upload_002"  # Newer first
        assert restored[1]["upload_id"] == "test_upload_001"


class TestSessionRecoveryIntegration:
    """Test session recovery integration."""

    def test_session_recovery_loads_dataset_on_startup(self, sample_datasets):
        """App initialization should restore available datasets."""
        # Arrange
        from clinical_analytics.ui.app_utils import restore_datasets

        storage = sample_datasets["storage"]
        db_path = sample_datasets["db_path"]

        # Act: Simulate app startup
        restored = restore_datasets(storage, db_path)

        # Assert: Datasets available
        assert len(restored) > 0
        first_dataset = restored[0]

        # Verify we can access the metadata
        metadata = storage.get_upload_metadata(first_dataset["upload_id"])
        assert metadata is not None
        assert metadata["dataset_name"] == first_dataset["dataset_name"]
