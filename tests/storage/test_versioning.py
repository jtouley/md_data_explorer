"""
Tests for dataset versioning.

Tests follow AAA pattern (Arrange, Act, Assert) and MVP scope:
- Basic content hashing (MVP)
- Canonicalization (order-independent)
- Version stability (same data = same hash)
- Deferred: Re-upload detection, perfect deduplication (Phase 5+)
"""

import polars as pl
import pytest

from clinical_analytics.storage.versioning import compute_dataset_version


@pytest.fixture
def patient_age_table():
    """Patient table. Column order is explicit because one test swaps it."""

    def _make(patient_ids, ages, column_order=("patient_id", "age")):
        columns = {"patient_id": patient_ids, "age": ages}
        return pl.DataFrame({name: columns[name] for name in column_order})

    return _make


@pytest.fixture
def upload_storage(tmp_path):
    """Storage rooted in a temporary upload directory."""
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    return UserDatasetStorage(upload_dir=tmp_path / "uploads")


class TestComputeDatasetVersion:
    """Test dataset version computation (MVP scope only)."""

    def test_compute_dataset_version_empty_list_raises(self):
        """An empty table list has no content hash."""
        with pytest.raises(ValueError, match="empty table list"):
            compute_dataset_version([])

    def test_compute_dataset_version_rejects_lazyframe(self):
        """Versioning requires a materialized frame."""
        lazy = pl.DataFrame({"patient_id": [1]}).lazy()
        with pytest.raises(TypeError, match="LazyFrame"):
            compute_dataset_version([lazy])

    def test_compute_dataset_version_identical_tables_same_version(self, patient_age_table):
        """Identical DataFrames should produce the same version hash."""
        # Arrange: Two identical DataFrames
        from clinical_analytics.storage.versioning import compute_dataset_version

        df1 = patient_age_table([1, 2, 3], [25, 30, 35])
        df2 = patient_age_table([1, 2, 3], [25, 30, 35])

        # Act: Compute versions
        version1 = compute_dataset_version([df1])
        version2 = compute_dataset_version([df2])

        # Assert: Versions match
        assert version1 == version2
        assert len(version1) == 16  # 16-char hex hash

    def test_compute_dataset_version_different_tables_different_version(self, patient_age_table):
        """Different DataFrames should produce different version hashes."""
        # Arrange: Two different DataFrames
        from clinical_analytics.storage.versioning import compute_dataset_version

        df1 = patient_age_table([1, 2, 3], [25, 30, 35])
        df2 = patient_age_table([1, 2, 3], [25, 30, 40])  # Different age

        # Act: Compute versions
        version1 = compute_dataset_version([df1])
        version2 = compute_dataset_version([df2])

        # Assert: Versions differ
        assert version1 != version2

    def test_compute_dataset_version_canonicalization_row_order_independent(self, patient_age_table):
        """Same data with different row order should produce the same version."""
        # Arrange: Same data, different row order
        from clinical_analytics.storage.versioning import compute_dataset_version

        df1 = patient_age_table([1, 2, 3], [25, 30, 35])
        df2 = patient_age_table([3, 1, 2], [35, 25, 30])  # Different row order

        # Act: Compute versions
        version1 = compute_dataset_version([df1])
        version2 = compute_dataset_version([df2])

        # Assert: Versions match (canonicalization works)
        assert version1 == version2

    def test_compute_dataset_version_canonicalization_column_order_independent(self, patient_age_table):
        """Same data with different column order should produce the same version."""
        # Arrange: Same data, different column order
        from clinical_analytics.storage.versioning import compute_dataset_version

        df1 = patient_age_table([1, 2, 3], [25, 30, 35])
        df2 = patient_age_table([1, 2, 3], [25, 30, 35], ("age", "patient_id"))  # Different column order

        # Act: Compute versions
        version1 = compute_dataset_version([df1])
        version2 = compute_dataset_version([df2])

        # Assert: Versions match (canonicalization works)
        assert version1 == version2

    def test_compute_dataset_version_multi_table(self, patient_age_table):
        """Multi-table uploads should produce stable versions."""
        # Arrange: Multiple tables
        from clinical_analytics.storage.versioning import compute_dataset_version

        df1 = patient_age_table([1, 2, 3], [25, 30, 35])
        df2 = pl.DataFrame({"visit_id": [1, 2], "patient_id": [1, 2], "date": ["2020-01-01", "2020-01-02"]})

        # Act: Compute version
        version = compute_dataset_version([df1, df2])

        # Assert: Version is stable
        assert len(version) == 16
        assert version == compute_dataset_version([df1, df2])

    def test_compute_dataset_version_handles_null_values(self, patient_age_table):
        """Datasets with null values should produce stable versions."""
        # Arrange: DataFrame with nulls
        from clinical_analytics.storage.versioning import compute_dataset_version

        df1 = patient_age_table([1, 2, 3], [25, None, 35])
        df2 = patient_age_table([1, 2, 3], [25, None, 35])

        # Act: Compute versions
        version1 = compute_dataset_version([df1])
        version2 = compute_dataset_version([df2])

        # Assert: Versions match
        assert version1 == version2

    def test_compute_dataset_version_different_schemas_different_version(self, patient_age_table):
        """Tables with different schemas should produce different versions."""
        # Arrange: Different schemas (different columns)
        from clinical_analytics.storage.versioning import compute_dataset_version

        df1 = patient_age_table([1, 2, 3], [25, 30, 35])
        df2 = pl.DataFrame({"patient_id": [1, 2, 3], "weight": [70, 80, 90]})  # Different column

        # Act: Compute versions
        version1 = compute_dataset_version([df1])
        version2 = compute_dataset_version([df2])

        # Assert: Versions differ
        assert version1 != version2


class TestSaveTableListStoresVersion:
    """Test that save_table_list stores dataset_version in metadata."""

    def test_save_table_list_stores_dataset_version(self, upload_storage, patient_age_table):
        """save_table_list should compute and store dataset_version in metadata."""
        # Arrange: Storage, tables, metadata

        from clinical_analytics.ui.storage.user_datasets import save_table_list

        storage = upload_storage
        upload_id = "test_upload_123"
        tables = [
            {"name": "patients", "data": patient_age_table([1, 2, 3], [25, 30, 35])},
        ]
        metadata = {
            "upload_id": upload_id,
            "created_at": "2025-12-30T10:00:00Z",
            "source_files": ["test.csv"],
        }

        # Act: Save tables
        success, message = save_table_list(storage, tables, upload_id, metadata)

        # Assert: Metadata contains dataset_version
        assert success, f"save_table_list failed: {message}"
        saved_metadata = storage.get_upload_metadata(upload_id)
        assert saved_metadata is not None
        assert "dataset_version" in saved_metadata
        assert len(saved_metadata["dataset_version"]) == 16  # 16-char hex hash

    def test_save_table_list_stores_table_fingerprints(self, upload_storage, patient_age_table):
        """save_table_list should store basic table fingerprints in provenance."""
        # Arrange: Storage, tables, metadata
        from clinical_analytics.ui.storage.user_datasets import save_table_list

        storage = upload_storage
        upload_id = "test_upload_456"
        tables = [
            {"name": "patients", "data": patient_age_table([1, 2, 3], [25, 30, 35])},
            {"name": "visits", "data": pl.DataFrame({"visit_id": [1, 2], "patient_id": [1, 2]})},
        ]
        metadata = {
            "upload_id": upload_id,
            "created_at": "2025-12-30T10:00:00Z",
            "source_files": ["test.zip"],
        }

        # Act: Save tables
        success, message = save_table_list(storage, tables, upload_id, metadata)

        # Assert: Metadata contains provenance with table fingerprints
        assert success, f"save_table_list failed: {message}"
        saved_metadata = storage.get_upload_metadata(upload_id)
        assert saved_metadata is not None
        assert "provenance" in saved_metadata
        assert "tables" in saved_metadata["provenance"]
        assert len(saved_metadata["provenance"]["tables"]) == 2

        # Check each table has required fields
        for table_info in saved_metadata["provenance"]["tables"]:
            assert "name" in table_info
            assert "row_count" in table_info
            assert "fingerprint" in table_info
