"""
Tests for user dataset storage.
"""

import json

import pandas as pd
import polars as pl

from clinical_analytics.ui.storage.user_datasets import (
    UploadSecurityValidator,
    UserDatasetStorage,
    save_table_list,
)


class TestUploadSecurityValidator:
    """Test suite for UploadSecurityValidator."""

    def test_validate_file_type_csv(self):
        """Test validating CSV file type."""
        is_valid, error = UploadSecurityValidator.validate_file_type("test.csv")
        assert is_valid is True
        assert error == ""

    def test_validate_file_type_xlsx(self):
        """Test validating XLSX file type."""
        is_valid, error = UploadSecurityValidator.validate_file_type("test.xlsx")
        assert is_valid is True

    def test_validate_file_type_invalid(self):
        """Test validating invalid file type."""
        is_valid, error = UploadSecurityValidator.validate_file_type("test.exe")
        assert is_valid is False
        assert "not allowed" in error

    def test_validate_file_type_no_extension(self):
        """Test validating file with no extension."""
        is_valid, error = UploadSecurityValidator.validate_file_type("test")
        assert is_valid is False
        assert "no extension" in error

    def test_validate_file_size_valid(self):
        """Test validating file size within limits."""
        file_bytes = b"x" * (10 * 1024)  # 10KB
        is_valid, error = UploadSecurityValidator.validate_file_size(file_bytes)
        assert is_valid is True
        assert error == ""

    def test_validate_file_size_too_small(self):
        """Test validating file that's too small."""
        file_bytes = b"x" * 100  # Less than 1KB
        is_valid, error = UploadSecurityValidator.validate_file_size(file_bytes)
        assert is_valid is False
        assert "too small" in error

    def test_validate_file_size_too_large(self):
        """Test validating file that's too large."""
        file_bytes = b"x" * (101 * 1024 * 1024)  # 101MB
        is_valid, error = UploadSecurityValidator.validate_file_size(file_bytes)
        assert is_valid is False
        assert "too large" in error

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        safe = UploadSecurityValidator.sanitize_filename("test_file.csv")
        assert safe == "test_file.csv"

    def test_sanitize_filename_path_traversal(self):
        """Test sanitizing filename with path traversal."""
        safe = UploadSecurityValidator.sanitize_filename("../../../etc/passwd")
        assert ".." not in safe
        assert "/" not in safe

    def test_sanitize_filename_special_chars(self):
        """Test sanitizing filename with special characters."""
        safe = UploadSecurityValidator.sanitize_filename("test@file#name$.csv")
        assert "@" not in safe
        assert "#" not in safe
        assert "$" not in safe


class TestUserDatasetStorage:
    """Test suite for UserDatasetStorage."""

    def test_storage_initialization(self, tmp_path):
        """Test storage initialization."""
        storage = UserDatasetStorage(upload_dir=tmp_path)
        assert storage.upload_dir == tmp_path
        assert storage.raw_dir == tmp_path / "raw"
        assert storage.metadata_dir == tmp_path / "metadata"

    def test_save_upload(self, tmp_path):
        """Test saving an upload."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create larger DataFrame to meet 1KB minimum (need ~150 rows to be safe)
        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                "outcome": [i % 2 for i in range(150)],
            }
        )

        # Convert DataFrame to CSV bytes
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="test_dataset.csv",
            metadata={"dataset_name": "test_dataset", "description": "Test dataset"},
        )

        if not success:
            print(f"Upload failed: {message}")
        assert success is True, f"Upload failed: {message}"
        assert upload_id is not None
        # Check that CSV file exists with upload_id as filename
        assert (tmp_path / "raw" / f"{upload_id}.csv").exists()

    def test_get_upload_data(self, tmp_path):
        """Test loading an upload."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create larger DataFrame to meet 1KB minimum
        df = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [20 + i for i in range(150)]})
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes, original_filename="test.csv", metadata={"dataset_name": "test"}
        )
        loaded_df = storage.get_upload_data(upload_id, lazy=False)

        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 150
        assert "patient_id" in loaded_df.columns

    def test_get_upload_data_large_id_values(self, tmp_path):
        """
        Test loading CSV with large patient_id values that exceed i64 range.

        Composite identifier hashes (e.g., 18325393944197996489) exceed max i64
        (9223372036854775807) and must be read as strings (Utf8) to prevent
        integer overflow errors.
        """
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create DataFrame with large ID values that exceed i64 range
        # These simulate composite identifier hashes
        # Repeat values to meet 1KB minimum file size requirement
        large_ids = [
            "18325393944197996489",  # Exceeds i64 max (9223372036854775807)
            "18446744073709551615",  # Max u64 (would overflow i64)
            "99999999999999999999",  # Very large number
            "12345678901234567890",  # Large but within i64 range (should still work)
        ] * 50  # Repeat 50 times to meet 1KB minimum
        df = pd.DataFrame(
            {
                "patient_id": large_ids,
                "age": [20 + (i % 4) for i in range(200)],
                "outcome": [i % 2 for i in range(200)],
                "diagnosis": [f"Diagnosis_{i}" for i in range(200)],  # Add more data for size
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Save upload
        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes, original_filename="test_large_ids.csv", metadata={"dataset_name": "test_large_ids"}
        )
        assert success is True, f"Upload failed: {message}"

        # Test lazy loading (LazyFrame)
        loaded_lf = storage.get_upload_data(upload_id, lazy=True)
        assert isinstance(loaded_lf, pl.LazyFrame)

        # Collect and verify patient_id is string type
        loaded_df_lazy = loaded_lf.collect()
        assert loaded_df_lazy.schema["patient_id"] == pl.Utf8, "patient_id should be Utf8 (string) type"
        assert len(loaded_df_lazy) == 200
        # Verify all unique large IDs are present (accounting for repetition)
        unique_loaded_ids = set(loaded_df_lazy["patient_id"].to_list())
        unique_expected_ids = set(large_ids)
        assert unique_loaded_ids == unique_expected_ids, "All large ID values should be preserved"

        # Test eager loading (pandas DataFrame)
        loaded_df_eager = storage.get_upload_data(upload_id, lazy=False)
        assert isinstance(loaded_df_eager, pd.DataFrame)
        assert len(loaded_df_eager) == 200
        assert loaded_df_eager["patient_id"].dtype == "object", "patient_id should be string type in pandas"
        # Verify all unique large IDs are present
        unique_loaded_ids = set(loaded_df_eager["patient_id"].tolist())
        unique_expected_ids = set(large_ids)
        assert unique_loaded_ids == unique_expected_ids, "All large ID values should be preserved"

    def test_get_upload_data_synthetic_id_metadata(self, tmp_path):
        """
        Test that synthetic ID metadata is used to identify ID columns for schema override.

        When patient_id is created synthetically (composite hash), metadata should
        indicate this so we can force it to Utf8 during CSV reading.
        """
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create DataFrame with composite identifier columns (no patient_id initially)
        # Add enough rows to meet 1KB minimum file size requirement
        df = pd.DataFrame(
            {
                "age": [20 + i for i in range(150)],
                "cd4_count": [500 + i * 10 for i in range(150)],
                "outcome": [i % 2 for i in range(150)],
                "diagnosis": [f"Diagnosis_{i}" for i in range(150)],  # Add more data for size
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Save upload (this will create synthetic patient_id via ensure_patient_id)
        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="test_synthetic_id.csv",
            metadata={"dataset_name": "test_synthetic_id"},
        )
        assert success is True, f"Upload failed: {message}"

        # Verify metadata contains synthetic_id_metadata
        metadata = storage.get_upload_metadata(upload_id)
        assert "synthetic_id_metadata" in metadata
        assert "patient_id" in metadata["synthetic_id_metadata"]

        # Load with lazy=True - should handle large hash values correctly
        loaded_lf = storage.get_upload_data(upload_id, lazy=True)
        assert isinstance(loaded_lf, pl.LazyFrame)

        # Collect and verify patient_id exists and is string type
        loaded_df = loaded_lf.collect()
        assert "patient_id" in loaded_df.columns
        assert loaded_df.schema["patient_id"] == pl.Utf8, "Synthetic patient_id should be Utf8 (string) type"
        assert len(loaded_df) == 150
        # Verify patient_id values are strings (not integers)
        patient_ids = loaded_df["patient_id"].to_list()
        assert all(isinstance(pid, str) for pid in patient_ids), "All patient_id values should be strings"

    def test_list_uploads(self, tmp_path):
        """Test listing all uploads."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create larger DataFrame to meet 1KB minimum - need multiple columns with more data
        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "name": [f"Patient_{i}" for i in range(100)],
                "age": [20 + i for i in range(100)],
                "diagnosis": [f"Diagnosis_{i}" for i in range(100)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        success1, msg1, id1 = storage.save_upload(
            file_bytes=csv_bytes, original_filename="dataset1.csv", metadata={"dataset_name": "dataset1"}
        )
        assert success1 is True, f"First upload failed: {msg1}"

        success2, msg2, id2 = storage.save_upload(
            file_bytes=csv_bytes, original_filename="dataset2.csv", metadata={"dataset_name": "dataset2"}
        )
        assert success2 is True, f"Second upload failed: {msg2}"

        uploads = storage.list_uploads()

        assert len(uploads) == 2
        assert all("upload_id" in u for u in uploads)
        assert all("dataset_name" in u for u in uploads)

    def test_get_upload_metadata(self, tmp_path):
        """Test getting upload metadata."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create larger DataFrame to meet 1KB minimum
        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "name": [f"Patient_{i}" for i in range(100)],
                "age": [20 + i for i in range(100)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        metadata = {"dataset_name": "test", "description": "Test", "source": "Manual upload"}
        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes, original_filename="test.csv", metadata=metadata
        )
        assert success is True, f"Upload failed: {message}"

        retrieved_metadata = storage.get_upload_metadata(upload_id)

        assert retrieved_metadata is not None, "Metadata should not be None"
        assert retrieved_metadata["description"] == "Test"
        assert retrieved_metadata["source"] == "Manual upload"

    def test_delete_upload(self, tmp_path):
        """Test deleting an upload."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create larger DataFrame to meet 1KB minimum
        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "name": [f"Patient_{i}" for i in range(100)],
                "age": [20 + i for i in range(100)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes, original_filename="test.csv", metadata={"dataset_name": "test"}
        )
        assert success is True, f"Upload failed: {message}"

        success, message = storage.delete_upload(upload_id)

        assert success is True, f"Delete failed: {message}"
        # Upload data should no longer exist
        assert storage.get_upload_data(upload_id) is None

    def test_overwrite_preserves_version_history(self, tmp_path):
        """Overwrite should preserve version history and add new version."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df1 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes1 = df1.to_csv(index=False).encode("utf-8")

        # First upload
        success1, msg1, id1 = storage.save_upload(
            file_bytes=csv_bytes1,
            original_filename="dataset.csv",
            metadata={"dataset_name": "my_dataset"},
        )
        assert success1 is True

        # Get first version
        metadata1 = storage.get_upload_metadata(id1)
        version1 = metadata1["dataset_version"]

        # Second upload with overwrite=True and different content
        df2 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [30 + i for i in range(150)],  # Different data
            }
        )
        csv_bytes2 = df2.to_csv(index=False).encode("utf-8")

        success2, msg2, id2 = storage.save_upload(
            file_bytes=csv_bytes2,
            original_filename="dataset.csv",
            metadata={"dataset_name": "my_dataset"},
            overwrite=True,
        )
        assert success2 is True, f"Overwrite should succeed: {msg2}"

        # Load metadata and verify version history
        metadata2 = storage.get_upload_metadata(id2)
        version2 = metadata2["dataset_version"]

        assert "version_history" in metadata2
        assert len(metadata2["version_history"]) == 2, "Should have 2 versions"

        # Check that old version is preserved but not active
        v1_entry = [v for v in metadata2["version_history"] if v["version"] == version1][0]
        assert v1_entry["is_active"] is False, "Old version should not be active"

        # Check that new version is active
        v2_entry = [v for v in metadata2["version_history"] if v["version"] == version2][0]
        assert v2_entry["is_active"] is True, "New version should be active"

    def test_overwrite_without_flag_rejected(self, tmp_path):
        """Uploading same name without overwrite=True should be rejected."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # First upload
        success1, msg1, id1 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset.csv",
            metadata={"dataset_name": "my_dataset"},
        )
        assert success1 is True

        # Second upload without overwrite=True should be rejected
        success2, msg2, id2 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset.csv",
            metadata={"dataset_name": "my_dataset"},
            # No overwrite parameter (defaults to False)
        )
        assert success2 is False, "Should reject duplicate name without overwrite=True"
        assert "already exists" in msg2.lower()

    def test_duplicate_dataset_name_rejected(self, tmp_path):
        """Test that uploading a dataset with the same name is rejected."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create larger DataFrame to meet 1KB minimum
        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "name": [f"Patient_{i}" for i in range(100)],
                "age": [20 + i for i in range(100)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # First upload should succeed
        success1, msg1, id1 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset.csv",
            metadata={"dataset_name": "my_dataset"},
        )
        assert success1 is True, f"First upload failed: {msg1}"
        assert id1 is not None

        # Second upload with same dataset_name should fail
        success2, msg2, id2 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset2.csv",
            metadata={"dataset_name": "my_dataset"},  # Same name!
        )
        assert success2 is False
        assert "already exists" in msg2
        assert id2 is None

        # Verify only one upload exists
        uploads = storage.list_uploads()
        assert len(uploads) == 1

    def test_different_dataset_names_allowed(self, tmp_path):
        """Test that datasets with different names can be uploaded."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "name": [f"Patient_{i}" for i in range(100)],
                "age": [20 + i for i in range(100)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # First upload
        success1, msg1, id1 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset1.csv",
            metadata={"dataset_name": "dataset_one"},
        )
        assert success1 is True

        # Second upload with different name should succeed
        success2, msg2, id2 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset2.csv",
            metadata={"dataset_name": "dataset_two"},  # Different name
        )
        assert success2 is True

        # Both uploads should exist
        uploads = storage.list_uploads()
        assert len(uploads) == 2


class TestSaveTableList:
    """Test suite for save_table_list() function (Fix #1)."""

    def test_save_table_list_single_table_creates_tables_directory(self, tmp_path):
        """Test that save_table_list() creates {upload_id}_tables/ directory for single-table."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_123"

        df = pl.DataFrame({"patient_id": ["P001", "P002"], "age": [25, 30], "outcome": [0, 1]})
        tables = [{"name": "patient_outcomes", "data": df}]
        metadata = {
            "dataset_name": "test",
            "original_filename": "patient_outcomes.csv",
        }

        # Act
        success, message = save_table_list(storage, tables, upload_id, metadata)

        # Assert
        assert success is True, f"save_table_list failed: {message}"
        assert (tmp_path / "raw" / f"{upload_id}_tables" / "patient_outcomes.csv").exists()
        assert (tmp_path / "raw" / f"{upload_id}.csv").exists()  # Unified cohort

    def test_save_table_list_saves_metadata_with_tables_list(self, tmp_path):
        """Test that save_table_list() saves metadata with tables list."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_456"

        df = pl.DataFrame({"patient_id": ["P001"], "age": [25]})
        tables = [{"name": "data", "data": df}]
        metadata = {"dataset_name": "test"}

        # Act
        success, message = save_table_list(storage, tables, upload_id, metadata)

        # Assert
        assert success is True
        metadata_path = tmp_path / "metadata" / f"{upload_id}.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            saved_metadata = json.load(f)

        assert "tables" in saved_metadata
        assert saved_metadata["tables"] == ["data"]
        assert "inferred_schema" in saved_metadata

    def test_save_table_list_converts_variable_mapping_to_inferred_schema(self, tmp_path):
        """Test that save_table_list() converts variable_mapping to inferred_schema."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_upload_789"

        df = pl.DataFrame(
            {
                "Patient ID": ["P001", "P002"],
                "Outcome": [0, 1],  # Binary outcome
                "Age": [25, 30],
            }
        )
        tables = [{"name": "data", "data": df}]
        metadata = {
            "dataset_name": "test",
            "variable_mapping": {
                "patient_id": "Patient ID",
                "outcome": "Outcome",
                "predictors": ["Age"],
            },
        }

        # Act
        success, message = save_table_list(storage, tables, upload_id, metadata)

        # Assert
        assert success is True
        metadata_path = tmp_path / "metadata" / f"{upload_id}.json"
        with open(metadata_path) as f:
            saved_metadata = json.load(f)

        assert "inferred_schema" in saved_metadata
        inferred = saved_metadata["inferred_schema"]
        assert "column_mapping" in inferred
        assert "Outcome" in inferred.get("outcomes", {})
        assert inferred["outcomes"]["Outcome"]["type"] == "binary"  # Should infer binary

    def test_save_table_list_multi_table_builds_unified_cohort(self, tmp_path):
        """Test that save_table_list() builds unified cohort for multi-table uploads."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_multi_upload"

        patients_df = pl.DataFrame({"patient_id": ["P001", "P002"], "name": ["Alice", "Bob"]})
        admissions_df = pl.DataFrame(
            {"admission_id": ["A001", "A002"], "patient_id": ["P001", "P002"], "date": ["2024-01-01", "2024-01-02"]}
        )

        tables = [
            {"name": "patients", "data": patients_df},
            {"name": "admissions", "data": admissions_df},
        ]
        metadata = {"dataset_name": "test_multi"}

        # Act
        success, message = save_table_list(storage, tables, upload_id, metadata)

        # Assert
        assert success is True
        # Both tables should be saved
        assert (tmp_path / "raw" / f"{upload_id}_tables" / "patients.csv").exists()
        assert (tmp_path / "raw" / f"{upload_id}_tables" / "admissions.csv").exists()
        # Unified cohort should exist
        assert (tmp_path / "raw" / f"{upload_id}.csv").exists()

        # Metadata should have both tables
        metadata_path = tmp_path / "metadata" / f"{upload_id}.json"
        with open(metadata_path) as f:
            saved_metadata = json.load(f)

        assert saved_metadata["tables"] == ["patients", "admissions"]


class TestFileLocking:
    """Test suite for file locking helper (Phase 0)."""

    def test_file_lock_exclusive_access(self, tmp_path):
        """File lock should provide exclusive access to metadata file."""
        import threading
        import time

        from clinical_analytics.ui.storage.user_datasets import file_lock

        # Arrange: Create a metadata file
        metadata_file = tmp_path / "test_metadata.json"
        metadata_file.write_text('{"test": "data"}')

        # Track if second thread acquired lock
        lock_acquired = {"value": False}

        def try_acquire_lock():
            # Try to acquire lock while main thread holds it
            time.sleep(0.1)  # Give main thread time to acquire lock
            try:
                with file_lock(metadata_file, timeout=0.2):
                    lock_acquired["value"] = True
            except Exception:
                # Expected: should timeout
                pass

        # Act: Main thread acquires lock, second thread tries
        thread = threading.Thread(target=try_acquire_lock)
        thread.start()

        with file_lock(metadata_file):
            # Hold lock for a bit
            time.sleep(0.5)

        thread.join()

        # Assert: Second thread should not have acquired lock
        assert lock_acquired["value"] is False, "Second thread should not acquire lock while first holds it"

    def test_file_lock_platform_support(self, tmp_path):
        """File lock should work on both Unix (fcntl) and Windows (msvcrt)."""
        from clinical_analytics.ui.storage.user_datasets import file_lock

        # Arrange: Create metadata file
        metadata_file = tmp_path / "test_metadata.json"
        metadata_file.write_text('{"test": "data"}')

        # Act: Create lock helper (should auto-detect platform)
        with file_lock(metadata_file):
            pass

        # Assert: No exception raised (platform detection works)
        assert True

    def test_file_lock_used_in_all_metadata_writes(self, tmp_path):
        """Verify file_lock() is used in all metadata write operations."""
        import inspect

        from clinical_analytics.ui.storage.user_datasets import (
            UserDatasetStorage,
            save_table_list,
        )

        # Arrange: Get all functions that write metadata
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Functions that write metadata (must use file_lock)
        metadata_write_functions = [
            ("save_table_list", save_table_list),
            ("rollback_to_version", storage.rollback_to_version),
            ("update_metadata", storage.update_metadata),
        ]

        # Act: Check source code for file_lock usage
        missing_locks = []
        for func_name, func in metadata_write_functions:
            source = inspect.getsource(func)
            if "with file_lock" not in source and "file_lock(" not in source:
                missing_locks.append(func_name)

        # Also check _migrate_legacy_upload (internal function)
        from clinical_analytics.ui.storage import user_datasets

        migrate_source = inspect.getsource(user_datasets._migrate_legacy_upload)
        if "with file_lock" not in migrate_source and "file_lock(" not in migrate_source:
            missing_locks.append("_migrate_legacy_upload")

        # Assert: All metadata writes use file locking
        assert len(missing_locks) == 0, f"Metadata write functions missing file_lock: {missing_locks}"


class TestCrossDatasetDeduplication:
    """Test suite for cross-dataset content deduplication (Phase 1)."""

    def test_same_content_different_name_warns_with_link(self, tmp_path):
        """Same file content with different dataset_name should warn with link to existing."""
        # Arrange: Upload same file twice with different names
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                "outcome": [i % 2 for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # First upload should succeed
        success1, msg1, id1 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset1.csv",
            metadata={"dataset_name": "first_dataset"},
        )
        assert success1 is True, f"First upload failed: {msg1}"

        # Second upload with same content but different name should warn (not block)
        success2, msg2, id2 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset2.csv",
            metadata={"dataset_name": "second_dataset"},
        )
        # Upload should succeed (warn, not block)
        assert success2 is True, "Upload should succeed with warning"
        # Message should contain warning about duplicate content
        assert "duplicate" in msg2.lower() or "warning" in msg2.lower(), f"Should warn about duplicate: {msg2}"
        # Should provide link to existing dataset
        assert "first_dataset" in msg2, f"Should mention existing dataset name: {msg2}"

    def test_same_content_same_name_overwrite_allowed(self, tmp_path):
        """Same content + same name + overwrite=True should proceed without warning."""
        # Arrange: Upload, then upload again with overwrite=True
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # First upload
        success1, msg1, id1 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset.csv",
            metadata={"dataset_name": "my_dataset"},
        )
        assert success1 is True

        # Second upload with same content and same name with overwrite=True
        # (overwrite parameter to be added in Phase 4, for now just test same behavior)
        # This will be rejected by existing duplicate name check
        success2, msg2, id2 = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="dataset.csv",
            metadata={"dataset_name": "my_dataset"},  # Same name - will be rejected
        )
        # Currently this is rejected, but in Phase 4 with overwrite=True it will succeed
        assert success2 is False, "Same name currently rejected (until Phase 4 overwrite)"

    def test_different_content_different_name_allowed(self, tmp_path):
        """Different content + different name should be allowed without warning."""
        # Arrange: Two different files with different names
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df1 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes1 = df1.to_csv(index=False).encode("utf-8")

        df2 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [30 + i for i in range(150)],  # Different data
            }
        )
        csv_bytes2 = df2.to_csv(index=False).encode("utf-8")

        # Both uploads should succeed
        success1, msg1, id1 = storage.save_upload(
            file_bytes=csv_bytes1,
            original_filename="dataset1.csv",
            metadata={"dataset_name": "dataset_one"},
        )
        assert success1 is True

        success2, msg2, id2 = storage.save_upload(
            file_bytes=csv_bytes2,
            original_filename="dataset2.csv",
            metadata={"dataset_name": "dataset_two"},
        )
        assert success2 is True
        # Should not warn about duplicates
        assert "duplicate" not in msg2.lower(), f"Should not warn: {msg2}"


class TestEventStructure:
    """Test suite for event structure (Phase 5)."""

    def test_event_id_uses_uuid4_hex_format(self, tmp_path):
        """Event IDs should use UUID4 hex format (no dashes)."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Act: Upload dataset
        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="test.csv",
            metadata={"dataset_name": "test"},
        )
        assert success is True

        # Assert: Check event_id format in version_history
        metadata = storage.get_upload_metadata(upload_id)
        assert "version_history" in metadata
        assert len(metadata["version_history"]) > 0

        version_entry = metadata["version_history"][0]
        event_id = version_entry.get("event_id")

        # Assert: event_id is hex format (32 hex chars, no dashes)
        assert event_id is not None
        assert len(event_id) == 32  # UUID4 hex is 32 chars
        assert all(c in "0123456789abcdef" for c in event_id)
        assert "-" not in event_id

    def test_timestamp_uses_iso8601_utc_with_z_suffix(self, tmp_path):
        """Timestamps should use ISO 8601 UTC format with Z suffix."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Act: Upload dataset
        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="test.csv",
            metadata={"dataset_name": "test"},
        )
        assert success is True

        # Assert: Check timestamp format in version_history
        metadata = storage.get_upload_metadata(upload_id)
        assert "version_history" in metadata
        assert len(metadata["version_history"]) > 0

        version_entry = metadata["version_history"][0]
        created_at = version_entry.get("created_at")

        # Assert: Timestamp is ISO 8601 UTC with Z suffix
        assert created_at is not None
        assert created_at.endswith("Z") or created_at.endswith("+00:00")
        # Parse to verify it's valid ISO 8601
        from datetime import datetime

        # Should parse without error
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        assert dt is not None


class TestVersionHistoryOrdering:
    """Test suite for version_history ordering (Phase 2)."""

    def test_version_history_sorted_by_created_at_ascending(self, tmp_path):
        """version_history should be sorted by created_at ascending (oldest first)."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Arrange: Upload initial dataset
        df1 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes1 = df1.to_csv(index=False).encode("utf-8")

        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes1,
            original_filename="test.csv",
            metadata={"dataset_name": "test"},
        )
        assert success is True

        # Arrange: Overwrite multiple times
        for i in range(2, 4):
            df = pd.DataFrame(
                {
                    "patient_id": [f"P{j:03d}" for j in range(150)],
                    "age": [20 + j + i for j in range(150)],  # Different data
                }
            )
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            success, message, _ = storage.save_upload(
                file_bytes=csv_bytes,
                original_filename="test.csv",
                metadata={"dataset_name": "test"},
                overwrite=True,
            )
            assert success is True

        # Act: Get metadata
        metadata = storage.get_upload_metadata(upload_id)
        version_history = metadata["version_history"]

        # Assert: version_history is sorted by created_at ascending
        assert len(version_history) >= 3, "Should have at least 3 versions"

        created_dates = [v["created_at"] for v in version_history]
        sorted_dates = sorted(created_dates)

        assert created_dates == sorted_dates, "version_history should be sorted by created_at ascending"


class TestLegacyDatasetMigration:
    """Test suite for legacy dataset migration during overwrite (Phase 4.2)."""

    def test_overwrite_legacy_dataset_migrates_to_version_history(self, tmp_path):
        """Overwriting legacy dataset (no version_history) should create synthetic version entry."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Arrange: Create legacy dataset metadata (without version_history)
        legacy_upload_id = "legacy_upload_123"
        legacy_metadata = {
            "upload_id": legacy_upload_id,
            "dataset_name": "legacy_dataset",
            "dataset_version": "legacy_version_abc",
            "created_at": "2024-01-01T00:00:00Z",
            "upload_timestamp": "2024-01-01T00:00:00Z",  # Add upload_timestamp for compatibility
            "original_filename": "legacy.csv",
            "row_count": 100,
            "column_count": 5,
            "tables": ["table_0"],  # Add tables list
            # No version_history - this is a legacy dataset
        }

        # Save legacy metadata
        metadata_path = storage.metadata_dir / f"{legacy_upload_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(legacy_metadata, f)

        # Arrange: Prepare overwrite
        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Act: Overwrite legacy dataset
        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="new_data.csv",
            metadata={"dataset_name": "legacy_dataset"},
            overwrite=True,
        )

        # Assert: Should succeed
        assert success is True, f"Upload failed: {message}"
        assert upload_id == legacy_upload_id  # Should preserve upload_id

        # Assert: version_history should have 2 entries (legacy + new)
        metadata = storage.get_upload_metadata(upload_id)
        assert "version_history" in metadata
        version_history = metadata["version_history"]
        assert len(version_history) == 2, "Should have 2 versions (legacy + new)"

        # Assert: Legacy version should be first (oldest) and inactive
        legacy_version = version_history[0]
        assert legacy_version["version"] == "legacy_version_abc"
        assert legacy_version["is_active"] is False
        assert legacy_version.get("source_filename") == "legacy.csv"

        # Assert: New version should be second and active
        new_version = version_history[1]
        assert new_version["is_active"] is True
        assert new_version["version"] != "legacy_version_abc"


class TestVersionHistoryMetadata:
    """Test suite for version history metadata structure (Phase 2)."""

    def test_metadata_includes_version_history(self, tmp_path):
        """Metadata should include version_history array."""
        # Arrange: Upload dataset
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Act: Upload
        success, msg, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="test.csv",
            metadata={"dataset_name": "test_dataset"},
        )
        assert success is True

        # Load metadata
        metadata = storage.get_upload_metadata(upload_id)

        # Assert: version_history exists and is a list
        assert "version_history" in metadata, "Metadata should include version_history"
        assert isinstance(metadata["version_history"], list), "version_history should be a list"
        assert len(metadata["version_history"]) == 1, "Should have one version entry for initial upload"

    def test_version_entry_has_canonical_tables_structure(self, tmp_path):
        """Version entry should use tables map, not parquet_paths/duckdb_tables lists."""
        # Arrange: Upload dataset
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Act: Upload
        success, msg, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="patient_data.csv",
            metadata={"dataset_name": "test"},
        )
        assert success is True

        # Load metadata
        metadata = storage.get_upload_metadata(upload_id)

        # Assert: Version entry has canonical tables structure
        assert "version_history" in metadata
        version_entry = metadata["version_history"][0]

        assert "tables" in version_entry, "Version entry should have 'tables' map"
        assert isinstance(version_entry["tables"], dict), "tables should be a dict"

        # Check table entry structure
        table_keys = list(version_entry["tables"].keys())
        assert len(table_keys) > 0, "Should have at least one table"

        first_table = version_entry["tables"][table_keys[0]]
        assert "parquet_path" in first_table, "Table should have parquet_path"
        assert "duckdb_table" in first_table, "Table should have duckdb_table"
        assert "row_count" in first_table, "Table should have row_count"
        assert "column_count" in first_table, "Table should have column_count"
        assert "schema_fingerprint" in first_table, "Table should have schema_fingerprint"

    def test_stable_internal_table_identifier_preserved(self, tmp_path):
        """Version entries should preserve stable internal identifier (table_0) for single-table uploads."""
        # Arrange: Upload single-table dataset
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Act: Upload with specific filename
        success, msg, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="MyPatientData.csv",
            metadata={"dataset_name": "test"},
        )
        assert success is True

        # Load metadata
        metadata = storage.get_upload_metadata(upload_id)

        # Assert: Internal identifier is stable (table_0 or original stem), not a generated name
        version_entry = metadata["version_history"][0]
        table_keys = list(version_entry["tables"].keys())

        # For single-table uploads, should use original filename stem as key
        # This is the stable internal identifier
        assert len(table_keys) == 1
        # The key should be the filename stem (MyPatientData), not a generated name
        assert table_keys[0] == "MyPatientData", f"Should preserve filename stem as table key, got {table_keys[0]}"


class TestSchemaDriftDetection:
    """Test suite for schema drift detection and policy (Phase 3)."""

    def test_detect_schema_drift_policy_defined(self, tmp_path):
        """Schema drift detection should have defined policy constants."""
        from clinical_analytics.ui.storage.user_datasets import SchemaDriftPolicy

        # Assert: Policy enum exists with expected values
        assert hasattr(SchemaDriftPolicy, "REJECT")
        assert hasattr(SchemaDriftPolicy, "WARN")
        assert hasattr(SchemaDriftPolicy, "ALLOW")

    def test_detect_schema_drift_same_schema(self, tmp_path):
        """Same schema should not trigger drift detection."""
        from clinical_analytics.ui.storage.user_datasets import detect_schema_drift

        # Arrange: Two identical schemas
        schema1 = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}
        schema2 = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}

        # Act: Check for drift
        has_drift, drift_details = detect_schema_drift(schema1, schema2)

        # Assert: No drift
        assert has_drift is False
        assert drift_details == {}

    def test_detect_schema_drift_new_column(self, tmp_path):
        """New column should trigger drift detection."""
        from clinical_analytics.ui.storage.user_datasets import detect_schema_drift

        # Arrange: Schema with new column
        schema1 = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}
        schema2 = {"columns": [("patient_id", "Utf8"), ("age", "Int64"), ("outcome", "Int64")]}

        # Act: Check for drift
        has_drift, drift_details = detect_schema_drift(schema1, schema2)

        # Assert: Drift detected
        assert has_drift is True
        assert "added_columns" in drift_details
        assert "outcome" in drift_details["added_columns"]

    def test_detect_schema_drift_removed_column(self, tmp_path):
        """Removed column should trigger drift detection."""
        from clinical_analytics.ui.storage.user_datasets import detect_schema_drift

        # Arrange: Schema with removed column
        schema1 = {"columns": [("patient_id", "Utf8"), ("age", "Int64"), ("outcome", "Int64")]}
        schema2 = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}

        # Act: Check for drift
        has_drift, drift_details = detect_schema_drift(schema1, schema2)

        # Assert: Drift detected
        assert has_drift is True
        assert "removed_columns" in drift_details
        assert "outcome" in drift_details["removed_columns"]

    def test_detect_schema_drift_type_change(self, tmp_path):
        """Type change should trigger drift detection."""
        from clinical_analytics.ui.storage.user_datasets import detect_schema_drift

        # Arrange: Schema with type change
        schema1 = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}
        schema2 = {"columns": [("patient_id", "Utf8"), ("age", "Float64")]}

        # Act: Check for drift
        has_drift, drift_details = detect_schema_drift(schema1, schema2)

        # Assert: Drift detected
        assert has_drift is True
        assert "type_changes" in drift_details
        assert "age" in drift_details["type_changes"]


class TestSchemaFingerprint:
    """Test suite for schema fingerprint computation (Phase 3)."""

    def test_compute_schema_fingerprint_uses_utf8_encoding(self, tmp_path):
        """Schema fingerprint should use UTF-8 encoding."""
        from clinical_analytics.ui.storage.user_datasets import compute_schema_fingerprint

        # Arrange: DataFrame with UTF-8 column names
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "âge": [25, 30],  # UTF-8 character
            }
        )

        # Act: Compute fingerprint
        fingerprint = compute_schema_fingerprint(df)

        # Assert: Should not raise encoding error, returns hex string
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64  # SHA256 hex digest length
        assert all(c in "0123456789abcdef" for c in fingerprint)

    def test_compute_schema_fingerprint_sorts_by_column_name(self, tmp_path):
        """Schema fingerprint should sort columns alphabetically by name."""
        from clinical_analytics.ui.storage.user_datasets import compute_schema_fingerprint

        # Arrange: DataFrame with columns in non-alphabetical order
        df1 = pl.DataFrame({"zebra": [1], "alpha": [2], "beta": [3]})
        df2 = pl.DataFrame({"alpha": [2], "beta": [3], "zebra": [1]})

        # Act: Compute fingerprints
        fp1 = compute_schema_fingerprint(df1)
        fp2 = compute_schema_fingerprint(df2)

        # Assert: Same fingerprint regardless of column order
        assert fp1 == fp2

    def test_compute_schema_fingerprint_sorts_by_column_type(self, tmp_path):
        """Schema fingerprint should sort by column type when names are same."""
        from clinical_analytics.ui.storage.user_datasets import compute_schema_fingerprint

        # Arrange: Two DataFrames with same column names but different types
        # Note: Polars doesn't allow same column name with different types in one DataFrame
        # So we test that type is included in fingerprint
        df1 = pl.DataFrame({"col": [1, 2, 3]})  # Int64
        df2 = pl.DataFrame({"col": [1.0, 2.0, 3.0]})  # Float64

        # Act: Compute fingerprints
        fp1 = compute_schema_fingerprint(df1)
        fp2 = compute_schema_fingerprint(df2)

        # Assert: Different fingerprints due to different types
        assert fp1 != fp2

    def test_compute_schema_fingerprint_deterministic(self, tmp_path):
        """Schema fingerprint should be deterministic (same schema = same fingerprint)."""
        from clinical_analytics.ui.storage.user_datasets import compute_schema_fingerprint

        # Arrange: Same schema, different data
        df1 = pl.DataFrame({"patient_id": ["P001"], "age": [25]})
        df2 = pl.DataFrame({"patient_id": ["P999"], "age": [99]})

        # Act: Compute fingerprints
        fp1 = compute_schema_fingerprint(df1)
        fp2 = compute_schema_fingerprint(df2)

        # Assert: Same fingerprint (schema only, not data)
        assert fp1 == fp2


class TestSchemaDriftClassification:
    """Test suite for schema drift classification (Phase 3)."""

    def test_classify_schema_drift_none(self, tmp_path):
        """No drift should be classified as 'none'."""
        from clinical_analytics.ui.storage.user_datasets import classify_schema_drift

        # Arrange: Identical schemas
        old_schema = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}
        new_schema = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}

        # Act: Classify drift
        result = classify_schema_drift(old_schema, new_schema)

        # Assert: No drift
        assert result["drift_type"] == "none"
        assert result.get("added_columns") == []
        assert result.get("removed_columns") == []
        assert result.get("type_changes") == {}

    def test_classify_schema_drift_additive(self, tmp_path):
        """Additive changes (new columns only) should be classified as 'additive'."""
        from clinical_analytics.ui.storage.user_datasets import classify_schema_drift

        # Arrange: Schema with new column added
        old_schema = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}
        new_schema = {"columns": [("patient_id", "Utf8"), ("age", "Int64"), ("outcome", "Int64")]}

        # Act: Classify drift
        result = classify_schema_drift(old_schema, new_schema)

        # Assert: Additive drift
        assert result["drift_type"] == "additive"
        assert "outcome" in result.get("added_columns", [])
        assert result.get("removed_columns") == []
        assert result.get("type_changes") == {}

    def test_classify_schema_drift_breaking_removed_column(self, tmp_path):
        """Removed columns should be classified as 'breaking'."""
        from clinical_analytics.ui.storage.user_datasets import classify_schema_drift

        # Arrange: Schema with column removed
        old_schema = {"columns": [("patient_id", "Utf8"), ("age", "Int64"), ("outcome", "Int64")]}
        new_schema = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}

        # Act: Classify drift
        result = classify_schema_drift(old_schema, new_schema)

        # Assert: Breaking drift
        assert result["drift_type"] == "breaking"
        assert "outcome" in result.get("removed_columns", [])
        assert result.get("added_columns") == []

    def test_classify_schema_drift_breaking_type_change(self, tmp_path):
        """Type changes should be classified as 'breaking'."""
        from clinical_analytics.ui.storage.user_datasets import classify_schema_drift

        # Arrange: Schema with type change
        old_schema = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}
        new_schema = {"columns": [("patient_id", "Utf8"), ("age", "Float64")]}

        # Act: Classify drift
        result = classify_schema_drift(old_schema, new_schema)

        # Assert: Breaking drift
        assert result["drift_type"] == "breaking"
        assert "age" in result.get("type_changes", {})
        assert result.get("added_columns") == []
        assert result.get("removed_columns") == []

    def test_classify_schema_drift_breaking_mixed(self, tmp_path):
        """Mixed additive and breaking changes should be classified as 'breaking'."""
        from clinical_analytics.ui.storage.user_datasets import classify_schema_drift

        # Arrange: Schema with both added and removed columns
        old_schema = {"columns": [("patient_id", "Utf8"), ("age", "Int64")]}
        new_schema = {"columns": [("patient_id", "Utf8"), ("outcome", "Int64")]}

        # Act: Classify drift
        result = classify_schema_drift(old_schema, new_schema)

        # Assert: Breaking drift (removal takes precedence)
        assert result["drift_type"] == "breaking"
        assert "age" in result.get("removed_columns", [])
        assert "outcome" in result.get("added_columns", [])


class TestSchemaDriftPolicy:
    """Test suite for schema drift policy enforcement (Phase 3)."""

    def test_apply_schema_drift_policy_allows_additive(self, tmp_path):
        """Additive changes should be allowed without override."""
        from clinical_analytics.ui.storage.user_datasets import apply_schema_drift_policy

        # Arrange: Additive drift
        drift_result = {
            "drift_type": "additive",
            "added_columns": ["outcome"],
            "removed_columns": [],
            "type_changes": {},
        }

        # Act: Apply policy
        allowed, message, warnings = apply_schema_drift_policy(drift_result, override=False)

        # Assert: Allowed
        assert allowed is True
        assert "allowed" in message.lower() or "additive" in message.lower()
        assert len(warnings) == 0

    def test_apply_schema_drift_policy_blocks_breaking_removal(self, tmp_path):
        """Breaking changes (removed columns) should be blocked without override."""
        from clinical_analytics.ui.storage.user_datasets import apply_schema_drift_policy

        # Arrange: Breaking drift (removed column)
        drift_result = {
            "drift_type": "breaking",
            "added_columns": [],
            "removed_columns": ["outcome"],
            "type_changes": {},
        }

        # Act: Apply policy without override
        allowed, message, warnings = apply_schema_drift_policy(drift_result, override=False)

        # Assert: Blocked
        assert allowed is False
        assert "blocked" in message.lower() or "removed" in message.lower() or "breaking" in message.lower()
        assert len(warnings) > 0

    def test_apply_schema_drift_policy_blocks_breaking_type_change(self, tmp_path):
        """Breaking changes (type changes) should be blocked without override."""
        from clinical_analytics.ui.storage.user_datasets import apply_schema_drift_policy

        # Arrange: Breaking drift (type change)
        drift_result = {
            "drift_type": "breaking",
            "added_columns": [],
            "removed_columns": [],
            "type_changes": {"age": ("Int64", "Float64")},
        }

        # Act: Apply policy without override
        allowed, message, warnings = apply_schema_drift_policy(drift_result, override=False)

        # Assert: Blocked
        assert allowed is False
        assert "blocked" in message.lower() or "type" in message.lower() or "breaking" in message.lower()
        assert len(warnings) > 0

    def test_apply_schema_drift_policy_allows_breaking_with_override(self, tmp_path):
        """Breaking changes should be allowed with override=True."""
        from clinical_analytics.ui.storage.user_datasets import apply_schema_drift_policy

        # Arrange: Breaking drift
        drift_result = {
            "drift_type": "breaking",
            "added_columns": [],
            "removed_columns": ["outcome"],
            "type_changes": {},
        }

        # Act: Apply policy with override
        allowed, message, warnings = apply_schema_drift_policy(drift_result, override=True)

        # Assert: Allowed with override
        assert allowed is True
        assert len(warnings) > 0  # Should warn about breaking changes even with override

    def test_apply_schema_drift_policy_no_drift(self, tmp_path):
        """No drift should always be allowed."""
        from clinical_analytics.ui.storage.user_datasets import apply_schema_drift_policy

        # Arrange: No drift
        drift_result = {
            "drift_type": "none",
            "added_columns": [],
            "removed_columns": [],
            "type_changes": {},
        }

        # Act: Apply policy
        allowed, message, warnings = apply_schema_drift_policy(drift_result, override=False)

        # Assert: Allowed
        assert allowed is True
        assert len(warnings) == 0


class TestSchemaDriftPolicyEnforcement:
    """Test suite for schema drift policy enforcement in overwrite flow (Phase 3 integration)."""

    def test_overwrite_with_breaking_schema_changes_blocked(self, tmp_path):
        """Overwrite with breaking schema changes (removed column) should be blocked."""
        # Arrange: Upload initial dataset
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df1 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                "outcome": [i % 2 for i in range(150)],
            }
        )
        csv_bytes1 = df1.to_csv(index=False).encode("utf-8")

        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes1,
            original_filename="test_dataset.csv",
            metadata={"dataset_name": "test_dataset"},
        )
        assert success is True

        # Arrange: Prepare overwrite with removed column (breaking change)
        df2 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                # outcome column removed - breaking change
            }
        )
        csv_bytes2 = df2.to_csv(index=False).encode("utf-8")

        # Act: Try to overwrite without override
        success, message, _ = storage.save_upload(
            file_bytes=csv_bytes2,
            original_filename="test_dataset.csv",
            metadata={"dataset_name": "test_dataset"},
            overwrite=True,
        )

        # Assert: Should be blocked
        assert success is False
        assert "blocked" in message.lower() or "breaking" in message.lower() or "removed" in message.lower()

    def test_overwrite_with_additive_schema_changes_allowed(self, tmp_path):
        """Overwrite with additive schema changes (new column) should be allowed."""
        # Arrange: Upload initial dataset
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df1 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
            }
        )
        csv_bytes1 = df1.to_csv(index=False).encode("utf-8")

        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes1,
            original_filename="test_dataset.csv",
            metadata={"dataset_name": "test_dataset"},
        )
        assert success is True

        # Arrange: Prepare overwrite with new column (additive change)
        df2 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                "outcome": [i % 2 for i in range(150)],  # New column - additive change
            }
        )
        csv_bytes2 = df2.to_csv(index=False).encode("utf-8")

        # Act: Overwrite with additive changes
        success, message, _ = storage.save_upload(
            file_bytes=csv_bytes2,
            original_filename="test_dataset.csv",
            metadata={"dataset_name": "test_dataset"},
            overwrite=True,
        )

        # Assert: Should be allowed
        assert success is True

    def test_overwrite_with_breaking_changes_allowed_with_override(self, tmp_path):
        """Overwrite with breaking schema changes should be allowed with override=True."""
        # Arrange: Upload initial dataset
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df1 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                "outcome": [i % 2 for i in range(150)],
            }
        )
        csv_bytes1 = df1.to_csv(index=False).encode("utf-8")

        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes1,
            original_filename="test_dataset.csv",
            metadata={"dataset_name": "test_dataset"},
        )
        assert success is True

        # Arrange: Prepare overwrite with removed column (breaking change)
        df2 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                # outcome column removed - breaking change
            }
        )
        csv_bytes2 = df2.to_csv(index=False).encode("utf-8")

        # Act: Overwrite with override=True
        success, message, _ = storage.save_upload(
            file_bytes=csv_bytes2,
            original_filename="test_dataset.csv",
            metadata={"dataset_name": "test_dataset", "schema_drift_override": True},
            overwrite=True,
        )

        # Assert: Should be allowed with override
        assert success is True


class TestMetadataInvariants:
    """Test suite for metadata invariants (Phase 4.1)."""

    def test_assert_invariants_valid_metadata(self, tmp_path):
        """Valid metadata should pass invariant assertions."""
        from clinical_analytics.ui.storage.user_datasets import assert_metadata_invariants

        # Arrange: Valid metadata
        metadata = {
            "upload_id": "test_upload_123",
            "dataset_name": "test_dataset",
            "dataset_version": "abc123",
            "version_history": [
                {
                    "version": "abc123",
                    "created_at": "2024-01-01T00:00:00",
                    "is_active": True,
                    "tables": {
                        "test": {
                            "parquet_path": "path/to/test.parquet",
                            "duckdb_table": "test",
                            "row_count": 100,
                            "column_count": 5,
                            "schema_fingerprint": "fingerprint123",
                        }
                    },
                }
            ],
        }

        # Act & Assert: Should not raise
        assert_metadata_invariants(metadata)

    def test_assert_invariants_missing_version_history(self, tmp_path):
        """Missing version_history should raise ValueError."""
        from clinical_analytics.ui.storage.user_datasets import assert_metadata_invariants

        # Arrange: Metadata without version_history
        metadata = {
            "upload_id": "test_upload_123",
            "dataset_name": "test_dataset",
        }

        # Act & Assert: Should raise ValueError
        try:
            assert_metadata_invariants(metadata)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "version_history" in str(e).lower()

    def test_assert_invariants_no_active_version(self, tmp_path):
        """No active version should raise ValueError."""
        from clinical_analytics.ui.storage.user_datasets import assert_metadata_invariants

        # Arrange: version_history with no active version
        metadata = {
            "upload_id": "test_upload_123",
            "dataset_name": "test_dataset",
            "version_history": [
                {
                    "version": "abc123",
                    "created_at": "2024-01-01T00:00:00",
                    "is_active": False,  # Not active
                    "tables": {},
                }
            ],
        }

        # Act & Assert: Should raise ValueError
        try:
            assert_metadata_invariants(metadata)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "active version" in str(e).lower()


class TestRollbackMechanism:
    """Test suite for rollback mechanism (Phase 6)."""

    def test_rollback_to_previous_version(self, tmp_path):
        """Rollback should switch active version to specified version."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Upload v1
        df1 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [20 + i for i in range(150)]})
        csv1 = df1.to_csv(index=False).encode("utf-8")
        success1, _, id1 = storage.save_upload(csv1, "data.csv", {"dataset_name": "test"})
        assert success1
        meta1 = storage.get_upload_metadata(id1)
        v1 = meta1["dataset_version"]

        # Upload v2 (overwrite)
        df2 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [30 + i for i in range(150)]})
        csv2 = df2.to_csv(index=False).encode("utf-8")
        success2, _, id2 = storage.save_upload(csv2, "data.csv", {"dataset_name": "test"}, overwrite=True)
        assert success2
        assert id2 == id1  # Same upload_id

        # Verify v2 is active
        meta_before = storage.get_upload_metadata(id1)
        v2_entry = [v for v in meta_before["version_history"] if v["version"] != v1][0]
        assert v2_entry["is_active"] is True

        # Rollback to v1
        success, message = storage.rollback_to_version(id1, v1)
        assert success is True, f"Rollback failed: {message}"

        # Verify v1 is now active
        meta_after = storage.get_upload_metadata(id1)
        v1_entry_after = [v for v in meta_after["version_history"] if v["version"] == v1][0]
        v2_entry_after = [v for v in meta_after["version_history"] if v["version"] != v1][0]

        assert v1_entry_after["is_active"] is True, "v1 should be active after rollback"
        assert v2_entry_after["is_active"] is False, "v2 should be inactive after rollback"

    def test_rollback_creates_event_entry(self, tmp_path):
        """Rollback should create rollback event in version history."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Setup: Create v1 and v2
        df1 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [20 + i for i in range(150)]})
        csv1 = df1.to_csv(index=False).encode("utf-8")
        success1, _, id1 = storage.save_upload(csv1, "data.csv", {"dataset_name": "test"})
        meta1 = storage.get_upload_metadata(id1)
        v1 = meta1["dataset_version"]

        df2 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [30 + i for i in range(150)]})
        csv2 = df2.to_csv(index=False).encode("utf-8")
        storage.save_upload(csv2, "data.csv", {"dataset_name": "test"}, overwrite=True)

        # Rollback
        storage.rollback_to_version(id1, v1)

        # Verify rollback event
        meta_after = storage.get_upload_metadata(id1)
        rollback_events = [v for v in meta_after["version_history"] if v.get("event_type") == "rollback"]
        assert len(rollback_events) >= 1, "Should have at least one rollback event"


class TestActiveVersionResolution:
    """Test suite for active version resolution (Phase 7)."""

    def test_get_active_version_returns_active_entry(self, tmp_path):
        """get_active_version should return the currently active version entry."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Upload v1
        df1 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [20 + i for i in range(150)]})
        csv1 = df1.to_csv(index=False).encode("utf-8")
        success1, _, id1 = storage.save_upload(csv1, "data.csv", {"dataset_name": "test"})
        assert success1

        # Get active version
        active_version = storage.get_active_version(id1)
        assert active_version is not None, "Should return active version entry"
        assert active_version.get("is_active") is True, "Returned version should be active"
        assert "version" in active_version, "Should include version hash"
        assert "created_at" in active_version, "Should include timestamp"
        assert active_version.get("event_type") == "upload", "First version should be upload event"

    def test_get_active_version_after_overwrite(self, tmp_path):
        """get_active_version should return v2 after overwrite."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Upload v1
        df1 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [20 + i for i in range(150)]})
        csv1 = df1.to_csv(index=False).encode("utf-8")
        success1, _, id1 = storage.save_upload(csv1, "data.csv", {"dataset_name": "test"})
        assert success1
        meta1 = storage.get_upload_metadata(id1)
        v1_hash = meta1["dataset_version"]

        # Upload v2 (overwrite)
        df2 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [30 + i for i in range(150)]})
        csv2 = df2.to_csv(index=False).encode("utf-8")
        success2, _, id2 = storage.save_upload(csv2, "data.csv", {"dataset_name": "test"}, overwrite=True)
        assert success2
        assert id2 == id1

        # Get active version
        active_version = storage.get_active_version(id1)
        assert active_version is not None
        assert active_version.get("is_active") is True
        assert active_version.get("version") != v1_hash, "Active version should be v2, not v1"
        assert active_version.get("event_type") == "overwrite", "Active version should be overwrite event"

    def test_get_active_version_after_rollback(self, tmp_path):
        """get_active_version should return rolled-back version."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Upload v1, then v2, then rollback to v1
        df1 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [20 + i for i in range(150)]})
        csv1 = df1.to_csv(index=False).encode("utf-8")
        success1, _, id1 = storage.save_upload(csv1, "data.csv", {"dataset_name": "test"})
        assert success1
        meta1 = storage.get_upload_metadata(id1)
        v1_hash = meta1["dataset_version"]

        df2 = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [30 + i for i in range(150)]})
        csv2 = df2.to_csv(index=False).encode("utf-8")
        success2, _, id2 = storage.save_upload(csv2, "data.csv", {"dataset_name": "test"}, overwrite=True)
        assert success2

        # Rollback to v1
        success_rb, _ = storage.rollback_to_version(id1, v1_hash)
        assert success_rb

        # Get active version
        active_version = storage.get_active_version(id1)
        assert active_version is not None
        assert active_version.get("is_active") is True
        assert active_version.get("version") == v1_hash, "Active version should be v1 after rollback"
        assert active_version.get("event_type") == "upload", "Active version should be original upload"

    def test_get_active_version_nonexistent_dataset(self, tmp_path):
        """get_active_version should return None for nonexistent dataset."""
        storage = UserDatasetStorage(upload_dir=tmp_path)
        active_version = storage.get_active_version("nonexistent_id")
        assert active_version is None, "Should return None for nonexistent dataset"


class TestQueryVersionIntegration:
    """Test suite for query execution with versioned datasets (Phase 8)."""

    def test_get_cohort_works_after_rollback(self, tmp_path):
        """Queries should work correctly after rolling back to previous version."""
        from clinical_analytics.datasets.uploaded.definition import UploadedDataset

        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Upload v1 with age column
        df1 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                "outcome": [i % 2 for i in range(150)],
            }
        )
        csv1 = df1.to_csv(index=False).encode("utf-8")
        success1, _, id1 = storage.save_upload(
            csv1,
            "data.csv",
            {
                "dataset_name": "test",
                "variable_mapping": {"patient_id": "patient_id", "outcome": "outcome", "predictors": ["age"]},
            },
        )
        assert success1
        meta1 = storage.get_upload_metadata(id1)
        v1_hash = meta1["dataset_version"]

        # Query v1 works
        dataset1 = UploadedDataset(upload_id=id1, storage=storage)
        cohort1 = dataset1.get_cohort()
        assert "patient_id" in cohort1.columns
        assert len(cohort1) == 150

        # Upload v2 (overwrite) - different data but same schema
        df2 = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [30 + i for i in range(150)],
                "outcome": [(i + 1) % 2 for i in range(150)],
            }
        )
        csv2 = df2.to_csv(index=False).encode("utf-8")
        success2, _, id2 = storage.save_upload(
            csv2,
            "data.csv",
            {
                "dataset_name": "test",
                "variable_mapping": {"patient_id": "patient_id", "outcome": "outcome", "predictors": ["age"]},
            },
            overwrite=True,
        )
        assert success2
        assert id2 == id1

        # Query v2 works
        dataset2 = UploadedDataset(upload_id=id1, storage=storage)
        cohort2 = dataset2.get_cohort()
        assert len(cohort2) == 150

        # Rollback to v1
        success_rb, _ = storage.rollback_to_version(id1, v1_hash)
        assert success_rb

        # Query after rollback still works
        dataset3 = UploadedDataset(upload_id=id1, storage=storage)
        cohort3 = dataset3.get_cohort()
        assert "patient_id" in cohort3.columns
        assert len(cohort3) == 150


class TestSaveUploadIntegration:
    """Test suite for save_upload() integration with save_table_list() (Fix #1)."""

    def test_save_upload_creates_tables_directory(self, tmp_path):
        """Test that save_upload() creates {upload_id}_tables/ directory (unified persistence)."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                "outcome": [i % 2 for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Act
        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="test_dataset.csv",
            metadata={"dataset_name": "test_dataset"},
        )

        # Assert
        assert success is True, f"Upload failed: {message}"
        assert upload_id is not None

        # Verify unified persistence structure
        assert (tmp_path / "raw" / f"{upload_id}.csv").exists()  # Unified cohort
        tables_dir = tmp_path / "raw" / f"{upload_id}_tables"
        assert tables_dir.exists(), "Tables directory should exist"
        assert tables_dir.is_dir()

        # Verify table file exists (should use filename stem)
        table_files = list(tables_dir.glob("*.csv"))
        assert len(table_files) == 1, f"Expected 1 table file, found {len(table_files)}"

        # Verify metadata has tables list
        metadata = storage.get_upload_metadata(upload_id)
        assert metadata is not None
        assert "tables" in metadata
        assert len(metadata["tables"]) == 1

    def test_save_upload_metadata_contains_inferred_schema(self, tmp_path):
        """Test that save_upload() metadata contains inferred_schema."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "age": [20 + i for i in range(150)],
                "outcome": [i % 2 for i in range(150)],
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Act
        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes,
            original_filename="test.csv",
            metadata={"dataset_name": "test"},
        )

        # Assert
        assert success is True
        metadata = storage.get_upload_metadata(upload_id)
        assert "inferred_schema" in metadata
