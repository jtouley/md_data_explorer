"""
Tests for DataStore class (persistent DuckDB storage).

Tests follow AAA pattern (Arrange, Act, Assert) and MVP scope:
- Basic persistence (save/load tables)
- Lazy frame returns
- Persistence across connections
- Table listing
- Deferred: Table deduplication, storage optimization (Phase 5+)
"""

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def datastore(tmp_path):
    """Create DataStore with temporary database."""
    from clinical_analytics.storage.datastore import DataStore

    db_path = tmp_path / "test.duckdb"
    return DataStore(db_path)


@pytest.fixture
def sample_table():
    """Sample patient table for testing."""
    return pl.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, 45],
            "diagnosis": ["A", "B", "A", "C", "B"],
        }
    )


class TestDataStoreSaveLoad:
    """Test DataStore save and load operations."""

    def test_datastore_save_table_persists_data(self, datastore, sample_table):
        """Saved tables should persist in DuckDB."""
        # Arrange: DataStore, sample table
        upload_id = "test_upload_123"
        table_name = "patients"
        dataset_version = "abc123def456"

        # Act: Save table
        datastore.save_table(
            table_name=table_name,
            data=sample_table,
            upload_id=upload_id,
            dataset_version=dataset_version,
        )

        # Assert: Table exists in DuckDB
        tables = datastore.list_tables()
        assert len(tables) > 0
        # Table name format: {upload_id}_{table_name}_{dataset_version}
        expected_table_name = f"{upload_id}_{table_name}_{dataset_version}"
        assert expected_table_name in tables

    def test_datastore_load_table_returns_lazy_frame(self, datastore, sample_table):
        """Loading tables should return Polars lazy frames."""
        # Arrange: Save table first
        upload_id = "test_upload_456"
        table_name = "patients"
        dataset_version = "xyz789abc123"

        datastore.save_table(
            table_name=table_name,
            data=sample_table,
            upload_id=upload_id,
            dataset_version=dataset_version,
        )

        # Act: Load table
        lazy_df = datastore.load_table(
            upload_id=upload_id,
            table_name=table_name,
            dataset_version=dataset_version,
        )

        # Assert: Returns LazyFrame
        assert isinstance(lazy_df, pl.LazyFrame)

        # Assert: Data matches (collect to verify)
        loaded_data = lazy_df.collect()
        assert loaded_data.height == sample_table.height
        assert loaded_data.columns == sample_table.columns
        assert loaded_data.to_dicts() == sample_table.to_dicts()

    def test_datastore_save_multiple_tables(self, datastore):
        """DataStore should handle multiple tables from different uploads."""
        # Arrange: Multiple tables
        tables = [
            {
                "upload_id": "upload_001",
                "table_name": "patients",
                "version": "v1",
                "data": pl.DataFrame({"patient_id": [1, 2], "age": [25, 30]}),
            },
            {
                "upload_id": "upload_001",
                "table_name": "visits",
                "version": "v1",
                "data": pl.DataFrame({"visit_id": [1, 2], "patient_id": [1, 2]}),
            },
            {
                "upload_id": "upload_002",
                "table_name": "patients",
                "version": "v2",
                "data": pl.DataFrame({"patient_id": [3, 4], "age": [35, 40]}),
            },
        ]

        # Act: Save all tables
        for table in tables:
            datastore.save_table(
                table_name=table["table_name"],
                data=table["data"],
                upload_id=table["upload_id"],
                dataset_version=table["version"],
            )

        # Assert: All tables exist
        all_tables = datastore.list_tables()
        assert len(all_tables) >= 3


class TestDataStorePersistence:
    """Test persistence across connection restarts."""

    def test_datastore_table_survives_restart(self, tmp_path, sample_table):
        """Tables should persist across DataStore reconnections."""
        # Arrange: Save table, close connection
        from clinical_analytics.storage.datastore import DataStore

        db_path = tmp_path / "persistent.duckdb"

        # First connection: save table
        datastore1 = DataStore(db_path)
        upload_id = "persistent_upload"
        table_name = "patients"
        dataset_version = "v1"

        datastore1.save_table(
            table_name=table_name,
            data=sample_table,
            upload_id=upload_id,
            dataset_version=dataset_version,
        )
        datastore1.close()

        # Act: Create new DataStore, load table
        datastore2 = DataStore(db_path)
        lazy_df = datastore2.load_table(
            upload_id=upload_id,
            table_name=table_name,
            dataset_version=dataset_version,
        )

        # Assert: Data persists across connections
        loaded_data = lazy_df.collect()
        assert loaded_data.height == sample_table.height
        assert loaded_data.to_dicts() == sample_table.to_dicts()

        datastore2.close()


class TestDataStoreListDatasets:
    """Test listing datasets in DuckDB."""

    def test_datastore_list_datasets_returns_all_uploads(self, datastore):
        """list_datasets should return all unique upload_ids."""
        # Arrange: Save multiple tables from different uploads
        tables = [
            ("upload_001", "patients", "v1", pl.DataFrame({"patient_id": [1, 2]})),
            ("upload_001", "visits", "v1", pl.DataFrame({"visit_id": [1, 2]})),
            ("upload_002", "patients", "v2", pl.DataFrame({"patient_id": [3, 4]})),
        ]

        for upload_id, table_name, version, data in tables:
            datastore.save_table(
                table_name=table_name,
                data=data,
                upload_id=upload_id,
                dataset_version=version,
            )

        # Act: List datasets
        datasets = datastore.list_datasets()

        # Assert: Returns all upload_ids
        assert len(datasets) >= 2
        upload_ids = [d["upload_id"] for d in datasets]
        assert "upload_001" in upload_ids
        assert "upload_002" in upload_ids

    def test_datastore_list_tables_returns_table_names(self, datastore, sample_table):
        """list_tables should return all table names in DuckDB."""
        # Arrange: Save table
        datastore.save_table(
            table_name="patients",
            data=sample_table,
            upload_id="test_upload",
            dataset_version="v1",
        )

        # Act: List tables
        tables = datastore.list_tables()

        # Assert: Returns table name
        assert len(tables) > 0
        assert "test_upload_patients_v1" in tables


class TestDataStoreParquetExport:
    """Test Parquet export functionality (Phase 3)."""

    def test_export_to_parquet_creates_file(self, datastore, sample_table, tmp_path):
        """Exporting to Parquet should create a valid Parquet file."""
        # Arrange: Save table to DuckDB
        upload_id = "parquet_test_001"
        table_name = "patients"
        dataset_version = "v1"

        datastore.save_table(
            table_name=table_name,
            data=sample_table,
            upload_id=upload_id,
            dataset_version=dataset_version,
        )

        # Act: Export to Parquet
        parquet_dir = tmp_path / "parquet"
        parquet_path = datastore.export_to_parquet(
            upload_id=upload_id,
            table_name=table_name,
            dataset_version=dataset_version,
            parquet_dir=parquet_dir,
        )

        # Assert: Parquet file exists and is valid
        assert parquet_path.exists()
        assert parquet_path.suffix == ".parquet"

        # Verify can be scanned by Polars
        scanned = pl.scan_parquet(parquet_path)
        loaded = scanned.collect()
        assert loaded.height == sample_table.height
        assert loaded.columns == sample_table.columns

    def test_parquet_compression_smaller_than_csv(self, datastore, tmp_path):
        """Parquet files should be ≥40% smaller than CSV."""
        # Arrange: Create larger dataset for compression test
        large_df = pl.DataFrame(
            {
                "patient_id": list(range(1000)),
                "age": [25 + (i % 50) for i in range(1000)],
                "diagnosis": ["Diagnosis_" + str(i % 10) for i in range(1000)],
                "value": [100.5 + (i % 100) for i in range(1000)],
            }
        )

        upload_id = "compression_test"
        table_name = "large_table"
        dataset_version = "v1"

        datastore.save_table(
            table_name=table_name,
            data=large_df,
            upload_id=upload_id,
            dataset_version=dataset_version,
        )

        # Save CSV for comparison
        csv_path = tmp_path / "test.csv"
        large_df.write_csv(csv_path)
        csv_size = csv_path.stat().st_size

        # Act: Export to Parquet
        parquet_dir = tmp_path / "parquet"
        parquet_path = datastore.export_to_parquet(
            upload_id=upload_id,
            table_name=table_name,
            dataset_version=dataset_version,
            parquet_dir=parquet_dir,
        )

        # Assert: Parquet is ≥40% smaller than CSV
        parquet_size = parquet_path.stat().st_size
        compression_ratio = (csv_size - parquet_size) / csv_size

        assert compression_ratio >= 0.40, (
            f"Parquet compression ratio {compression_ratio:.1%} is less than 40%. "
            f"CSV: {csv_size:,} bytes, Parquet: {parquet_size:,} bytes"
        )

    def test_load_from_parquet_returns_lazy_frame(self, datastore, sample_table, tmp_path):
        """Loading from Parquet should return LazyFrame."""
        # Arrange: Save and export to Parquet
        upload_id = "lazy_test"
        table_name = "patients"
        dataset_version = "v1"

        datastore.save_table(
            table_name=table_name,
            data=sample_table,
            upload_id=upload_id,
            dataset_version=dataset_version,
        )

        parquet_dir = tmp_path / "parquet"
        parquet_path = datastore.export_to_parquet(
            upload_id=upload_id,
            table_name=table_name,
            dataset_version=dataset_version,
            parquet_dir=parquet_dir,
        )

        # Act: Load from Parquet
        lazy_df = datastore.load_from_parquet(parquet_path)

        # Assert: Returns LazyFrame
        assert isinstance(lazy_df, pl.LazyFrame)

        # Verify data matches
        loaded = lazy_df.collect()
        assert loaded.height == sample_table.height
        assert loaded.to_dicts() == sample_table.to_dicts()
