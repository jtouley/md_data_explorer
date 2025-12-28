"""
Tests for user dataset storage.
"""

import pandas as pd

from clinical_analytics.ui.storage.user_datasets import UploadSecurityValidator, UserDatasetStorage


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
        # Check that CSV file exists with friendly name
        assert (tmp_path / "raw" / "test_dataset.csv").exists()

    def test_get_upload_data(self, tmp_path):
        """Test loading an upload."""
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create larger DataFrame to meet 1KB minimum
        df = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(150)], "age": [20 + i for i in range(150)]})
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        success, message, upload_id = storage.save_upload(
            file_bytes=csv_bytes, original_filename="test.csv", metadata={"dataset_name": "test"}
        )
        loaded_df = storage.get_upload_data(upload_id)

        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 150
        assert "patient_id" in loaded_df.columns

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
