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


class TestExternalPdfJsonSerialization:
    """Test suite for external PDF JSON serialization."""

    def test_save_table_list_removes_external_pdf_bytes_before_json_save(self, tmp_path):
        """Test that external_pdf_bytes is removed from metadata before JSON save."""
        # Arrange: Create storage with temp directory
        storage = UserDatasetStorage(tmp_path / "uploads")

        # Create test data
        tables = [
            {
                "name": "test_table",
                "data": pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
            }
        ]

        upload_id = "test_upload_123"

        # Metadata with external_pdf_bytes (simulating what UI passes)
        metadata = {
            "dataset_name": "test_dataset",
            "external_pdf_bytes": b"fake pdf content",
            "external_pdf_filename": "test_dictionary.pdf",
        }

        # Act: Save table list (this should process external_pdf_bytes and remove it)
        success, message = save_table_list(storage, tables, upload_id, metadata)

        # Assert: Save succeeded
        assert success, f"save_table_list should succeed, got: {message}"

        # Assert: external_pdf_bytes is removed from metadata
        assert "external_pdf_bytes" not in metadata, (
            "external_pdf_bytes should be removed from metadata after processing"
        )

        # Assert: external_pdf_filename is removed from metadata
        assert "external_pdf_filename" not in metadata, (
            "external_pdf_filename should be removed from metadata after processing"
        )

        # Assert: Metadata can be serialized to JSON (no bytes objects)
        try:
            json.dumps(metadata)
            # If we get here, serialization succeeded
            assert True, "Metadata should be JSON serializable"
        except TypeError as e:
            pytest.fail(f"Metadata contains non-serializable objects: {e}")

    def test_save_table_list_metadata_json_serializable_after_save(self, tmp_path):
        """Test that metadata retrieved after save is JSON serializable."""
        # Arrange: Create storage with temp directory
        storage = UserDatasetStorage(tmp_path / "uploads")

        # Create test data
        tables = [
            {
                "name": "test_table",
                "data": pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
            }
        ]

        upload_id = "test_upload_456"

        # Metadata with external_pdf_bytes
        metadata = {
            "dataset_name": "test_dataset",
            "external_pdf_bytes": b"fake pdf content",
            "external_pdf_filename": "test_dictionary.pdf",
        }

        # Act: Save table list
        success, message = save_table_list(storage, tables, upload_id, metadata)
        assert success, f"save_table_list should succeed, got: {message}"

        # Act: Retrieve metadata (this reads from JSON file)
        saved_metadata = storage.get_upload_metadata(upload_id)

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
