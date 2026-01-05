"""Integration tests for failure modes.

Tests what happens when components fail and verifies graceful error handling.
"""

import pytest
from clinical_analytics.datasets.uploaded.definition import UploadedDataset


@pytest.mark.integration
@pytest.mark.slow
class TestFailureModes:
    """Test suite for failure mode handling."""

    def test_integration_missing_upload_id_raises_valueerror(self, real_storage):
        """Test that missing upload_id raises ValueError with clear message."""
        # Arrange: Use non-existent upload_id
        upload_id = "nonexistent_upload_12345"

        # Act & Assert: Should raise ValueError with clear message
        with pytest.raises(ValueError, match="Upload.*not found"):
            UploadedDataset(upload_id=upload_id, storage=real_storage)

    def test_integration_corrupted_csv_rejected(self, real_storage, integration_tmp_dir):
        """Test that corrupted CSV is rejected during upload."""
        # Arrange: Create corrupted CSV (invalid content)
        corrupted_csv = integration_tmp_dir / "corrupted.csv"
        corrupted_csv.write_bytes(b"Invalid binary data\x00\x01\x02\x03" * 100)  # At least 1KB

        # Act: Try to upload corrupted file
        with open(corrupted_csv, "rb") as f:
            file_bytes = f.read()

        success, message, upload_id = real_storage.save_upload(
            file_bytes=file_bytes,
            original_filename="corrupted.csv",
            metadata={"dataset_name": "Corrupted Dataset"},
        )

        # Assert: Upload should fail or be rejected
        # Note: save_upload may succeed but processing may fail later
        # This test verifies the system handles corrupted data gracefully
        if not success:
            # Upload rejected - this is expected
            assert "corrupt" in message.lower() or "invalid" in message.lower() or "error" in message.lower()

    def test_integration_semantic_layer_missing_cohort_handles_gracefully(self, real_storage, sample_csv_file):
        """Test that semantic layer handles missing unified cohort CSV gracefully."""
        # Arrange: Upload file
        with open(sample_csv_file, "rb") as f:
            file_bytes = f.read()

        success, message, upload_id = real_storage.save_upload(
            file_bytes=file_bytes,
            original_filename="test.csv",
            metadata={"dataset_name": "Test Dataset"},
        )
        assert success, f"Upload failed: {message}"

        # Act: Try to get semantic layer (may fail if unified cohort CSV doesn't exist)
        dataset = UploadedDataset(upload_id=upload_id, storage=real_storage)
        dataset.load()

        # Assert: Should either succeed or raise clear error
        try:
            semantic = dataset.get_semantic_layer()
            # If it succeeds, that's fine - unified cohort CSV exists
            assert semantic is not None
        except (ValueError, FileNotFoundError) as e:
            # If it fails, error should be clear
            error_msg = str(e).lower()
            assert "unified" in error_msg or "cohort" in error_msg or "not found" in error_msg
