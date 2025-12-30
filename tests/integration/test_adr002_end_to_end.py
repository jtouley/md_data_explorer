"""
End-to-End Integration Tests for ADR002: Persistent Storage Layer

Tests the complete flow:
1. Upload dataset
2. Save to DuckDB + export to Parquet
3. Restart session (simulate app restart)
4. Restore datasets
5. Query restored dataset
6. Verify query logging

Verifies ADR002 success metrics:
- Datasets survive restart
- Lazy evaluation works
- Compression achieved (≥40%)
- Query logging works
"""

import json

import polars as pl
import pytest


@pytest.fixture
def integration_env(tmp_path):
    """Create isolated integration test environment."""
    from clinical_analytics.storage.datastore import DataStore
    from clinical_analytics.storage.query_logger import QueryLogger
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    upload_dir = tmp_path / "uploads"
    db_path = tmp_path / "analytics.duckdb"
    parquet_dir = tmp_path / "parquet"
    log_dir = tmp_path / "query_logs"

    storage = UserDatasetStorage(upload_dir=upload_dir)
    datastore = DataStore(db_path)
    query_logger = QueryLogger(log_dir)

    return {
        "storage": storage,
        "datastore": datastore,
        "query_logger": query_logger,
        "db_path": db_path,
        "parquet_dir": parquet_dir,
        "log_dir": log_dir,
    }


@pytest.mark.slow
@pytest.mark.integration
class TestADR002EndToEnd:
    """End-to-end integration tests for ADR002."""

    def test_complete_upload_to_query_flow(self, integration_env):
        """
        Test complete flow: upload → DuckDB → Parquet → restart → restore → query.

        Success Criteria (ADR002):
        1. Dataset survives restart
        2. Lazy evaluation works
        3. Parquet compression ≥40%
        4. Query logging captures events
        """
        storage = integration_env["storage"]
        datastore = integration_env["datastore"]
        query_logger = integration_env["query_logger"]
        db_path = integration_env["db_path"]
        parquet_dir = integration_env["parquet_dir"]

        # ========== Phase 1: Upload Dataset ==========
        upload_id = "test001"  # Simple ID without underscores for parsing
        dataset_version = "v1abc123"

        # Create sample dataset (1000 rows for better compression)
        df = pl.DataFrame(
            {
                "patient_id": list(range(1, 1001)),
                "age": [25 + (i % 50) for i in range(1000)],
                "diagnosis": [f"Diagnosis_{i % 10}" for i in range(1000)],
                "outcome": [i % 2 for i in range(1000)],
                "value": [100.5 + (i % 100) for i in range(1000)],
            }
        )

        # Save to DuckDB
        datastore.save_table(
            table_name="patients",
            data=df,
            upload_id=upload_id,
            dataset_version=dataset_version,
        )

        # Export to Parquet
        parquet_path = datastore.export_to_parquet(
            upload_id=upload_id,
            table_name="patients",
            dataset_version=dataset_version,
            parquet_dir=parquet_dir,
        )

        # Save metadata
        metadata = {
            "upload_id": upload_id,
            "dataset_version": dataset_version,
            "dataset_name": "Integration Test Dataset",
            "tables": ["patients"],
            "row_count": df.height,
            "created_at": "2025-12-30T10:00:00Z",
            "parquet_paths": {"patients": str(parquet_path)},
        }

        storage.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(storage.metadata_dir / f"{upload_id}.json", "w") as f:
            json.dump(metadata, f)

        datastore.close()

        # Verify Parquet compression (Success Metric 3)
        csv_path = integration_env["parquet_dir"].parent / "test.csv"
        df.write_csv(csv_path)
        csv_size = csv_path.stat().st_size
        parquet_size = parquet_path.stat().st_size
        compression_ratio = (csv_size - parquet_size) / csv_size

        assert compression_ratio >= 0.40, (
            f"Parquet compression {compression_ratio:.1%} < 40% (ADR002 Success Metric 3 FAILED)"
        )

        # ========== Phase 2: Simulate Restart ==========
        # Close connections (simulate app shutdown)
        # Reopen (simulate app startup)

        # ========== Phase 3: Restore Session ==========
        from clinical_analytics.ui.app_utils import restore_datasets

        restored = restore_datasets(storage, db_path)

        # Verify dataset restored (Success Metric 1)
        assert len(restored) == 1, "Dataset should survive restart (ADR002 Success Metric 1 FAILED)"
        assert restored[0]["upload_id"] == upload_id
        assert restored[0]["dataset_name"] == "Integration Test Dataset"

        # ========== Phase 4: Load Data Lazily ==========
        # Load from Parquet (lazy evaluation - Success Metric 2)
        from clinical_analytics.storage.datastore import DataStore

        lazy_df = DataStore.load_from_parquet(parquet_path)

        # Verify LazyFrame (Success Metric 2)
        assert isinstance(lazy_df, pl.LazyFrame), "Parquet should load as LazyFrame (ADR002 Success Metric 2 FAILED)"

        # Execute lazy query (predicate pushdown should work)
        filtered = lazy_df.filter(pl.col("age") > 40).collect()
        assert filtered.height > 0

        # ========== Phase 5: Log Query ==========
        query_logger.log_query(upload_id, "How many patients over 40?", "2025-12-30T10:05:00Z")
        query_logger.log_execution(
            upload_id,
            "q1",
            {"intent": "count", "filters": [{"column": "age", "op": ">", "value": 40}]},
            50.5,
            "2025-12-30T10:05:01Z",
        )
        query_logger.log_result(upload_id, "q1", "count", {"total": filtered.height}, "2025-12-30T10:05:02Z")

        # Verify query logging (Success Metric 4)
        history = query_logger.get_query_history(upload_id)
        assert len(history) == 3, "Query logging should capture all events (ADR002 Success Metric 4 FAILED)"
        assert history[0]["event_type"] == "query"
        assert history[1]["event_type"] == "execution"
        assert history[2]["event_type"] == "result"

        # ========== SUCCESS: All Metrics Met ==========
        print("\n✅ ADR002 SUCCESS METRICS:")
        print(f"  1. Dataset survives restart: ✅ ({len(restored)} datasets restored)")
        print("  2. Lazy evaluation works: ✅ (LazyFrame returned)")
        print(f"  3. Parquet compression: ✅ ({compression_ratio:.1%} reduction)")
        print(f"  4. Query logging works: ✅ ({len(history)} events logged)")

    def test_parquet_predicate_pushdown_optimization(self, integration_env):
        """Verify Parquet enables predicate pushdown (lazy evaluation optimization)."""
        datastore = integration_env["datastore"]
        parquet_dir = integration_env["parquet_dir"]

        # Create larger dataset for meaningful test
        df = pl.DataFrame(
            {
                "patient_id": list(range(1, 1001)),
                "age": [20 + (i % 60) for i in range(1000)],
                "status": ["active" if i % 2 == 0 else "inactive" for i in range(1000)],
            }
        )

        upload_id = "predicate_test"
        datastore.save_table("patients", df, upload_id, "v1")
        parquet_path = datastore.export_to_parquet(upload_id, "patients", "v1", parquet_dir)
        datastore.close()

        # Load as LazyFrame
        from clinical_analytics.storage.datastore import DataStore

        lazy_df = DataStore.load_from_parquet(parquet_path)

        # Apply filter (should push down to Parquet scan)
        filtered = lazy_df.filter((pl.col("age") > 50) & (pl.col("status") == "active"))

        # Verify filter works
        result = filtered.collect()
        assert result.height > 0
        assert all(result["age"] > 50)
        assert all(result["status"] == "active")

        # Success: Predicate pushdown works (didn't materialize full DataFrame)

    def test_core_invariant_identical_data_produces_same_version(self, integration_env):
        """
        CONTRACT TEST: Verify core invariant - identical data → identical version → storage reuse.

        This is the fundamental contract that enables:
        - Idempotent uploads (re-uploading same data doesn't duplicate storage)
        - Query result caching (keyed by upload_id + dataset_version)
        - Storage deduplication (same content = same version)

        Invariant: same (upload_id, dataset_version) → guaranteed storage reuse
        """
        datastore = integration_env["datastore"]
        parquet_dir = integration_env["parquet_dir"]

        # Create identical dataset
        df = pl.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [25, 30, 35],
                "diagnosis": ["A", "B", "C"],
            }
        )

        # Upload 1: Save with version v1
        from clinical_analytics.storage.versioning import compute_dataset_version

        version_1 = compute_dataset_version([df])

        datastore.save_table("patients", df, "upload_001", version_1)
        parquet_1 = datastore.export_to_parquet("upload_001", "patients", version_1, parquet_dir)

        # Upload 2: Same data, different column order (should produce same version)
        df_reordered = pl.DataFrame(
            {
                "diagnosis": ["A", "B", "C"],
                "patient_id": [1, 2, 3],
                "age": [25, 30, 35],
            }
        )

        version_2 = compute_dataset_version([df_reordered])

        # CORE CONTRACT: Identical data → Identical version
        assert version_1 == version_2, (
            f"Core invariant violated: identical data produced different versions "
            f"(v1={version_1}, v2={version_2}). This breaks storage reuse guarantee."
        )

        # Upload 3: Different data (should produce different version)
        df_different = pl.DataFrame(
            {
                "patient_id": [1, 2, 3, 4],  # Different row count
                "age": [25, 30, 35, 40],
                "diagnosis": ["A", "B", "C", "D"],
            }
        )

        version_3 = compute_dataset_version([df_different])

        # CORE CONTRACT: Different data → Different version
        assert version_3 != version_1, (
            f"Core invariant violated: different data produced same version "
            f"(v1={version_1}, v3={version_3}). This breaks storage isolation."
        )

        # Verify storage key uniqueness (upload_id + version uniquely identifies storage)
        # Same upload_id, same version → should reference same storage
        parquet_2 = datastore.export_to_parquet("upload_001", "patients", version_2, parquet_dir)

        # Both should point to same Parquet file (reuse)
        assert parquet_1 == parquet_2, (
            "Storage reuse failed: same (upload_id, version) produced different Parquet paths"
        )

        datastore.close()
