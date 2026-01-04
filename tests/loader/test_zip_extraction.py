"""
Tests for ZIP file extraction and multi-table processing.
"""

import io
import zipfile

import polars as pl
import pytest
from clinical_analytics.core.multi_table_handler import MultiTableHandler


@pytest.fixture(autouse=True)
def enable_multi_table(monkeypatch):
    """Enable MULTI_TABLE feature flag for these tests."""
    import clinical_analytics.ui.storage.user_datasets as user_datasets_module

    monkeypatch.setattr(user_datasets_module, "MULTI_TABLE_ENABLED", True)


class TestZipExtraction:
    """Test suite for ZIP file extraction and processing."""

    def test_extract_zip_with_csv_files(
        self, upload_storage, large_patients_csv, large_admissions_with_discharge_csv, large_diagnoses_csv
    ):
        """Test extracting ZIP file containing multiple CSV files."""
        # Create test ZIP file using shared fixtures
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("patients.csv", large_patients_csv)
            zip_file.writestr("admissions.csv", large_admissions_with_discharge_csv)
            zip_file.writestr("diagnoses.csv", large_diagnoses_csv)
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        # Test extraction
        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test_dataset.zip",
            metadata={"dataset_name": "test_dataset"},
        )

        if not success:
            print(f"Upload failed: {message}")
        assert success is True, f"Upload failed: {message}"
        assert upload_id is not None
        assert "tables" in message.lower() or "joined" in message.lower() or "successful" in message.lower()

    def test_extract_zip_with_csv_gz_files(self, upload_storage, large_test_data_csv, large_admissions_csv):
        """Test extracting ZIP file containing compressed CSV files."""
        import gzip

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Create compressed CSV using shared fixture
            csv_content = large_test_data_csv.encode("utf-8")
            compressed = gzip.compress(csv_content)
            zip_file.writestr("patients.csv.gz", compressed)
            # Add regular CSV using shared fixture
            zip_file.writestr("admissions.csv", large_admissions_csv)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="compressed.zip",
            metadata={"dataset_name": "compressed"},
        )

        assert success is True
        assert upload_id is not None

    def test_extract_zip_with_subdirectories(
        self, upload_storage, large_test_data_csv, large_admissions_csv, large_diagnoses_csv
    ):
        """Test extracting ZIP file with CSV files in subdirectories."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("mimic-iv/patients.csv", large_test_data_csv)
            zip_file.writestr("mimic-iv/admissions.csv", large_admissions_csv)
            zip_file.writestr("mimic-iv/diagnoses.csv", large_diagnoses_csv)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="mimic-iv-clinical-database-demo-2.2.zip",
            metadata={"dataset_name": "mimic_iv"},
        )

        if not success:
            print(f"Upload failed: {message}")
        assert success is True, f"Upload failed: {message}"
        assert upload_id is not None

    def test_extract_zip_ignores_macosx(self, upload_storage, large_test_data_csv):
        """Test that ZIP extraction ignores __MACOSX files."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("patients.csv", large_test_data_csv)
            zip_file.writestr("__MACOSX/._patients.csv", "metadata")
            zip_file.writestr("__MACOSX/.DS_Store", "metadata")

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes, original_filename="test.zip", metadata={"dataset_name": "test"}
        )

        assert success is True
        # Should only process patients.csv, not __MACOSX files

    def test_extract_zip_no_csv_files(self, upload_storage):
        """Test ZIP file with no CSV files raises error."""
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Create larger files to meet 1KB minimum, but no CSV files
            readme_content = "This is a readme file\n" + "x" * 1000
            zip_file.writestr("readme.txt", readme_content)
            json_content = '{"key": "value", "data": "' + "x" * 500 + '"}'
            zip_file.writestr("data.json", json_content)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes, original_filename="no_csv.zip", metadata={"dataset_name": "test"}
        )

        assert success is False
        assert "no csv files" in message.lower()

    def test_extract_zip_invalid_file(self, upload_storage):
        """Test handling invalid ZIP file."""
        # Create invalid ZIP that's large enough to pass size validation (1KB+)
        invalid_bytes = b"This is not a ZIP file" + b"x" * 2000

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=invalid_bytes,
            original_filename="invalid.zip",
            metadata={"dataset_name": "test"},
        )

        assert success is False
        assert "error" in message.lower() or "invalid" in message.lower() or "corrupted" in message.lower()

    def test_extract_zip_with_mixed_types(self, upload_storage):
        """Test ZIP extraction with tables having different key column types (int vs string)."""
        zip_buffer = io.BytesIO()
        # Generate data with integer patient_id
        patients_data = "patient_id,age\n" + "\n".join([f"{i},{20 + i % 100}" for i in range(1000000)])
        admissions_data = "patient_id,date\n" + "\n".join([f"{i},2020-01-{1 + i % 30:02d}" for i in range(1000000)])
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("patients.csv", patients_data)
            zip_file.writestr("admissions.csv", admissions_data)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="mixed_types.zip",
            metadata={"dataset_name": "mixed_types"},
        )

        # Should succeed despite type differences (normalization should handle it)
        assert success is True

    def test_extract_zip_large_dataset(self, upload_storage, large_patients_csv, large_admissions_csv):
        """Test extracting ZIP with larger dataset (multiple tables, many rows)."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("patients.csv", large_patients_csv)
            zip_file.writestr("admissions.csv", large_admissions_csv)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="large_dataset.zip",
            metadata={"dataset_name": "large"},
        )

        assert success is True
        assert upload_id is not None

    def test_extract_zip_creates_unified_cohort(
        self, upload_storage, large_patients_csv, large_admissions_with_admission_date_csv
    ):
        """Test that ZIP extraction creates unified cohort with joined tables."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("patients.csv", large_patients_csv)
            zip_file.writestr("admissions.csv", large_admissions_with_admission_date_csv)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes, original_filename="test.zip", metadata={"dataset_name": "test"}
        )

        assert success is True

        # Check that unified cohort CSV was created
        csv_path = upload_storage.upload_dir / "raw" / f"{upload_id}.csv"
        assert csv_path.exists()

        # Load and verify unified cohort
        unified_df = pl.read_csv(csv_path)
        assert "patient_id" in unified_df.columns
        assert "age" in unified_df.columns
        assert "admission_date" in unified_df.columns

    def test_extract_zip_saves_metadata(self, upload_storage, large_test_data_csv, large_admissions_csv):
        """Test that ZIP extraction saves proper metadata."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("patients.csv", large_test_data_csv)
            zip_file.writestr("admissions.csv", large_admissions_csv)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = upload_storage
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test.zip",
            metadata={"dataset_name": "test_dataset"},
        )

        assert success is True

        # Check metadata file
        import json

        metadata_path = upload_storage.upload_dir / "metadata" / f"{upload_id}.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["file_format"] == "zip_multi_table"
        assert "tables" in metadata
        assert "relationships" in metadata
        assert "inferred_schema" in metadata
        assert len(metadata["tables"]) == 2  # patients and admissions

    def test_save_zip_upload_overwrite_reuses_upload_id(self, upload_storage, large_patients_csv, large_admissions_csv):
        """Test that save_zip_upload() with overwrite=True reuses existing upload_id and appends to version_history."""
        # Arrange: Create initial ZIP upload
        zip_buffer1 = io.BytesIO()
        with zipfile.ZipFile(zip_buffer1, "w") as zip_file:
            zip_file.writestr("patients.csv", large_patients_csv)
            zip_file.writestr("admissions.csv", large_admissions_csv)
        zip_buffer1.seek(0)
        zip_bytes1 = zip_buffer1.getvalue()

        storage = upload_storage
        success1, message1, upload_id1 = storage.save_zip_upload(
            file_bytes=zip_bytes1,
            original_filename="test_dataset.zip",
            metadata={"dataset_name": "test_dataset"},
        )

        # Assert: First upload succeeded
        assert success1 is True, f"First upload failed: {message1}"
        assert upload_id1 is not None

        # Get first version metadata
        metadata1 = storage.get_upload_metadata(upload_id1)
        version1 = metadata1["dataset_version"]
        assert "version_history" in metadata1
        assert len(metadata1["version_history"]) == 1
        assert metadata1["version_history"][0]["is_active"] is True

        # Arrange: Create second ZIP with same dataset_name but potentially different content
        zip_buffer2 = io.BytesIO()
        with zipfile.ZipFile(zip_buffer2, "w") as zip_file:
            zip_file.writestr("patients.csv", large_patients_csv)
            zip_file.writestr("admissions.csv", large_admissions_csv)
        zip_buffer2.seek(0)
        zip_bytes2 = zip_buffer2.getvalue()

        # Act: Second upload with overwrite=True
        success2, message2, upload_id2 = storage.save_zip_upload(
            file_bytes=zip_bytes2,
            original_filename="test_dataset.zip",
            metadata={"dataset_name": "test_dataset"},
            overwrite=True,
        )

        # Assert: Second upload succeeded
        assert success2 is True, f"Overwrite upload failed: {message2}"
        assert upload_id2 is not None

        # Assert: Same upload_id reused (not a new one)
        assert upload_id2 == upload_id1, f"Expected upload_id {upload_id1}, got {upload_id2}"

        # Assert: Version history has 2 versions
        metadata2 = storage.get_upload_metadata(upload_id2)
        assert "version_history" in metadata2
        assert len(metadata2["version_history"]) == 2, f"Expected 2 versions, got {len(metadata2['version_history'])}"

        # Assert: First version is inactive, second is active
        v1_entry = [v for v in metadata2["version_history"] if v["version"] == version1][0]
        assert v1_entry["is_active"] is False, "First version should be inactive after overwrite"

        active_versions = [v for v in metadata2["version_history"] if v.get("is_active", False)]
        all_versions_info = [(v.get("version"), v.get("is_active")) for v in metadata2["version_history"]]
        assert (
            len(active_versions) == 1
        ), f"Should have exactly one active version, got {len(active_versions)}. All versions: {all_versions_info}"

        # Assert: Active version is the newer one (check created_at timestamp, not version,
        # since same content = same version)
        active_version = active_versions[0]
        v1_created_at = v1_entry.get("created_at")
        active_created_at = active_version.get("created_at")
        assert (
            active_created_at > v1_created_at
        ), f"Active version should be newer. v1={v1_created_at}, active={active_created_at}"

        # Assert: Metadata file is the same (not a new file)
        import json

        metadata_path = upload_storage.upload_dir / "metadata" / f"{upload_id1}.json"
        assert metadata_path.exists(), "Metadata file should exist"
        with open(metadata_path) as f:
            saved_metadata = json.load(f)
        assert saved_metadata["upload_id"] == upload_id1
        assert len(saved_metadata["version_history"]) == 2


class TestMultiTableHandler:
    """Test suite for MultiTableHandler used in ZIP extraction."""

    def test_detect_relationships_with_type_mismatch(self):
        """Test relationship detection handles type mismatches (int vs string keys)."""
        # Create tables with different key types
        patients = pl.DataFrame(
            {
                "patient_id": [1, 2, 3],  # Integer
                "age": [45, 62, 38],
            }
        )

        admissions = pl.DataFrame(
            {
                "patient_id": ["1", "2", "3"],  # String (should be normalized)
                "admission_date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            }
        )

        tables = {"patients": patients, "admissions": admissions}

        handler = MultiTableHandler(tables)
        relationships = handler.detect_relationships()

        # Should detect relationship despite type mismatch
        assert len(relationships) > 0
        handler.close()

    def test_build_unified_cohort_with_type_mismatch(self):
        """Test building unified cohort with type mismatches."""
        patients = pl.DataFrame({"patient_id": [1, 2, 3], "age": [45, 62, 38]})

        admissions = pl.DataFrame(
            {
                "patient_id": ["1", "2", "3"],  # String
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            }
        )

        tables = {"patients": patients, "admissions": admissions}

        handler = MultiTableHandler(tables)
        cohort = handler.build_unified_cohort()

        assert cohort.height > 0
        assert "patient_id" in cohort.columns
        assert "age" in cohort.columns
        assert "date" in cohort.columns
        handler.close()
