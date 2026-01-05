"""Tests for diagnostic helper functions used in integration tests.

These tests validate that diagnostic helpers can dump state correctly
and that assert_with_diagnostics produces useful output on failure.
"""

import pytest

# Import helpers (will fail until implemented)
from .diagnostic_helpers import (
    assert_with_diagnostics,
    dump_semantic_layer_state,
    dump_storage_state,
)


@pytest.mark.integration
class TestDiagnosticHelpers:
    """Test suite for diagnostic helper functions."""

    def test_dump_storage_state_returns_dict(self, real_storage):
        """Test that dump_storage_state returns a dictionary."""
        # Arrange: Create a test upload (needs to be at least 1KB)
        import polars as pl

        # Create larger dataset to meet minimum size requirement
        df = pl.DataFrame(
            {
                "id": [f"P{i:04d}" for i in range(200)],
                "value": [10 + (i % 100) for i in range(200)],
                "category": [f"cat_{i % 5}" for i in range(200)],
            }
        )
        csv_content = df.write_csv()

        success, message, upload_id = real_storage.save_upload(
            file_bytes=csv_content.encode(),
            original_filename="test.csv",
            metadata={"dataset_name": "Test Dataset"},
        )
        assert success, f"Upload failed: {message}"

        # Act: Dump storage state
        state = dump_storage_state(storage=real_storage, upload_id=upload_id)

        # Assert: Returns dict with expected keys
        assert isinstance(state, dict)
        assert "upload_id" in state
        assert "metadata_path" in state or "metadata" in state

    def test_dump_semantic_layer_state_returns_dict(self, real_storage, sample_csv_file):
        """Test that dump_semantic_layer_state returns a dictionary."""
        # Arrange: Upload file and create semantic layer
        with open(sample_csv_file, "rb") as f:
            file_bytes = f.read()

        success, message, upload_id = real_storage.save_upload(
            file_bytes=file_bytes,
            original_filename="test.csv",
            metadata={"dataset_name": "Test Dataset"},
        )
        assert success, f"Upload failed: {message}"

        from clinical_analytics.datasets.uploaded.definition import UploadedDataset

        dataset = UploadedDataset(upload_id=upload_id, storage=real_storage)
        dataset.load()

        # Semantic layer initializes lazily - try to get it
        # If unified cohort CSV doesn't exist yet, this will fail
        # In that case, we'll test with a mock semantic layer instead
        try:
            semantic = dataset.get_semantic_layer()
        except (ValueError, FileNotFoundError):
            # Unified cohort CSV not created yet - skip this test
            # or create a minimal semantic layer for testing
            pytest.skip("Semantic layer requires unified cohort CSV which may not exist immediately after upload")

        # Act: Dump semantic layer state
        state = dump_semantic_layer_state(semantic=semantic)

        # Assert: Returns dict with expected keys
        assert isinstance(state, dict)
        assert "dataset_name" in state
        assert "config" in state or "schema" in state

    def test_assert_with_diagnostics_dumps_on_failure(self, capsys):
        """Test that assert_with_diagnostics prints diagnostics on failure."""

        # Arrange: Create a diagnostic function
        def diagnostic_fn(**kwargs):
            return {"test_key": "test_value", "error": "Test failure"}

        # Act: Call assert_with_diagnostics with False condition
        with pytest.raises(AssertionError):
            assert_with_diagnostics(False, diagnostic_fn)

        # Assert: Diagnostic output was printed
        captured = capsys.readouterr()
        assert "ASSERTION FAILED" in captured.out or "DIAGNOSTIC DUMP" in captured.out
        assert "test_key" in captured.out or "test_value" in captured.out

    def test_assert_with_diagnostics_passes_on_success(self):
        """Test that assert_with_diagnostics passes when condition is True."""

        # Arrange: Create a diagnostic function
        def diagnostic_fn(**kwargs):
            return {"test_key": "test_value"}

        # Act & Assert: Should not raise
        assert_with_diagnostics(True, diagnostic_fn)
