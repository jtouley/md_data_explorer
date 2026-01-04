"""
Tests for external PDF JSON serialization bug.

Tests ensure that:
1. external_pdf_bytes is removed from metadata before JSON serialization
2. Metadata can be saved to JSON without serialization errors
"""

import json

import polars as pl
import pytest
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage, save_table_list

# ============================================================================
# Helper Fixtures (Module-level - shared across test classes)
# ============================================================================


@pytest.fixture
def upload_storage(tmp_path):
    """Create UserDatasetStorage with temp directory."""
    return UserDatasetStorage(upload_dir=tmp_path / "uploads")


@pytest.fixture
def sample_test_tables():
    """Standard test tables fixture."""
    return [
        {
            "name": "test_table",
            "data": pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
        }
    ]


@pytest.fixture
def make_metadata_with_external_pdf():
    """Factory fixture for creating metadata with external PDF."""

    def _make(
        dataset_name: str = "test_dataset",
        pdf_content: bytes = b"fake pdf content",
        pdf_filename: str = "test_dictionary.pdf",
    ) -> dict:
        return {
            "dataset_name": dataset_name,
            "external_pdf_bytes": pdf_content,
            "external_pdf_filename": pdf_filename,
        }

    return _make


class TestExternalPdfJsonSerialization:
    """Test suite for external PDF JSON serialization."""

    def test_save_table_list_removes_external_pdf_bytes_before_json_save(
        self,
        upload_storage,
        sample_test_tables,
        make_metadata_with_external_pdf,
    ):
        """Test that external_pdf_bytes is removed from metadata before JSON save."""
        # Arrange
        upload_id = "test_upload_123"
        metadata = make_metadata_with_external_pdf()

        # Act: Save table list (this should process external_pdf_bytes and remove it)
        success, message = save_table_list(upload_storage, sample_test_tables, upload_id, metadata)

        # Assert: Save succeeded
        assert success, f"save_table_list should succeed, got: {message}"

        # Assert: external_pdf_bytes is removed from metadata
        assert (
            "external_pdf_bytes" not in metadata
        ), "external_pdf_bytes should be removed from metadata after processing"

        # Assert: external_pdf_filename is removed from metadata
        assert (
            "external_pdf_filename" not in metadata
        ), "external_pdf_filename should be removed from metadata after processing"

        # Assert: Metadata can be serialized to JSON (no bytes objects)
        try:
            json.dumps(metadata)
            # If we get here, serialization succeeded
            assert True, "Metadata should be JSON serializable"
        except TypeError as e:
            pytest.fail(f"Metadata contains non-serializable objects: {e}")

    def test_save_table_list_metadata_json_serializable_after_save(
        self,
        upload_storage,
        sample_test_tables,
        make_metadata_with_external_pdf,
    ):
        """Test that metadata retrieved after save is JSON serializable."""
        # Arrange
        upload_id = "test_upload_456"
        metadata = make_metadata_with_external_pdf()

        # Act: Save table list
        success, message = save_table_list(upload_storage, sample_test_tables, upload_id, metadata)
        assert success, f"save_table_list should succeed, got: {message}"

        # Act: Retrieve metadata (this reads from JSON file)
        saved_metadata = upload_storage.get_upload_metadata(upload_id)

        # Assert: Saved metadata exists
        assert saved_metadata is not None, "Saved metadata should exist"

        # Assert: Saved metadata is JSON serializable
        try:
            json.dumps(saved_metadata)
            # If we get here, serialization succeeded
            assert True, "Saved metadata should be JSON serializable"
        except TypeError as e:
            pytest.fail(f"Saved metadata contains non-serializable objects: {e}")

        # Assert: external_pdf_bytes is not in saved metadata
        assert "external_pdf_bytes" not in saved_metadata, "external_pdf_bytes should not be in saved metadata"

        # Assert: external_pdf_filename is not in saved metadata
        assert "external_pdf_filename" not in saved_metadata, "external_pdf_filename should not be in saved metadata"
