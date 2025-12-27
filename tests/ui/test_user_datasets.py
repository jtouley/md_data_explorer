"""
Tests for user dataset storage.
"""

import pandas as pd
import pytest

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
        storage = UserDatasetStorage(base_dir=tmp_path)
        assert storage.base_dir == tmp_path

    def test_save_dataset(self, tmp_path):
        """Test saving a dataset."""
        storage = UserDatasetStorage(base_dir=tmp_path)

        df = pd.DataFrame(
            {"patient_id": ["P001", "P002", "P003"], "age": [45, 62, 38], "outcome": [1, 0, 1]}
        )

        dataset_id = storage.save_dataset(
            df, dataset_name="test_dataset", metadata={"description": "Test dataset"}
        )

        assert dataset_id is not None
        assert (tmp_path / "raw" / f"{dataset_id}.csv").exists()

    def test_load_dataset(self, tmp_path):
        """Test loading a dataset."""
        storage = UserDatasetStorage(base_dir=tmp_path)

        df = pd.DataFrame({"patient_id": ["P001", "P002"], "age": [45, 62]})

        dataset_id = storage.save_dataset(df, dataset_name="test")
        loaded_df = storage.load_dataset(dataset_id)

        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 2
        assert "patient_id" in loaded_df.columns

    def test_list_datasets(self, tmp_path):
        """Test listing all datasets."""
        storage = UserDatasetStorage(base_dir=tmp_path)

        df = pd.DataFrame({"col": [1, 2, 3]})
        storage.save_dataset(df, dataset_name="dataset1")
        storage.save_dataset(df, dataset_name="dataset2")

        datasets = storage.list_datasets()

        assert len(datasets) == 2
        assert all("id" in d for d in datasets)
        assert all("name" in d for d in datasets)

    def test_get_dataset_metadata(self, tmp_path):
        """Test getting dataset metadata."""
        storage = UserDatasetStorage(base_dir=tmp_path)

        df = pd.DataFrame({"col": [1, 2, 3]})
        metadata = {"description": "Test", "source": "Manual upload"}
        dataset_id = storage.save_dataset(df, dataset_name="test", metadata=metadata)

        retrieved_metadata = storage.get_dataset_metadata(dataset_id)

        assert retrieved_metadata["description"] == "Test"
        assert retrieved_metadata["source"] == "Manual upload"

    def test_delete_dataset(self, tmp_path):
        """Test deleting a dataset."""
        storage = UserDatasetStorage(base_dir=tmp_path)

        df = pd.DataFrame({"col": [1, 2, 3]})
        dataset_id = storage.save_dataset(df, dataset_name="test")

        storage.delete_dataset(dataset_id)

        # Dataset should no longer exist
        with pytest.raises(FileNotFoundError):
            storage.load_dataset(dataset_id)
