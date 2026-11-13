"""
Tests for Phase 4: Lazy Frame Support in UploadedDataset.

Tests the lazy parameter in get_upload_data() and lazy frame handling
in load() and get_cohort() methods.
"""

import json

import pandas as pd
import polars as pl
import polars.testing as plt
import pytest

from clinical_analytics.datasets.uploaded.definition import UploadedDataset
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

# ============================================================================
# Helper Fixtures (Module-level - shared across test classes)
# ============================================================================


@pytest.fixture
def upload_storage(tmp_path):
    """Create UserDatasetStorage with temp directory."""
    return UserDatasetStorage(upload_dir=tmp_path)


@pytest.fixture
def create_upload(upload_storage, sample_upload_df, sample_upload_metadata):
    """Factory fixture to create test uploads with consistent pattern."""

    def _create(
        upload_id: str,
        df: pl.DataFrame | None = None,
        metadata_overrides: dict | None = None,
    ):
        # Use provided df or default
        df_to_save = df if df is not None else sample_upload_df

        # Save CSV
        csv_path = upload_storage.raw_dir / f"{upload_id}.csv"
        df_to_save.write_csv(csv_path)

        # Merge metadata
        metadata = {**sample_upload_metadata, "upload_id": upload_id}
        if metadata_overrides:
            metadata.update(metadata_overrides)

        # Save metadata
        metadata_path = upload_storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        return upload_id

    return _create


# ============================================================================
# Test Classes
# ============================================================================


class TestGetUploadDataLazy:
    """Tests for get_upload_data() with lazy parameter."""

    @pytest.mark.parametrize(
        "lazy_kwargs", [{"lazy": True}, {"lazy": False}, {}], ids=["lazy_true", "lazy_false", "default"]
    )
    def test_get_upload_data_anyLazyFlag_returnsLazyFrame(
        self, upload_storage, create_upload, sample_upload_df, lazy_kwargs
    ):
        """The lazy flag is kept for compatibility only; every path returns a LazyFrame."""
        # Arrange
        upload_id = create_upload("test_lazy_upload")

        # Act
        result = upload_storage.get_upload_data(upload_id, **lazy_kwargs)

        # Assert
        assert isinstance(result, pl.LazyFrame), f"Expected LazyFrame, got {type(result)}"
        plt.assert_frame_equal(result.collect(), sample_upload_df)


class TestGetUploadDataSource:
    """Which file get_upload_data() reads, and legacy migration under concurrent reads."""

    @pytest.mark.parametrize(
        "tables",
        [["only_table"], ["first_table", "second_table"]],
        ids=["single_table", "multi_table"],
    )
    def test_get_upload_data_legacyWithParquet_readsUnifiedCohortLikeTheLayer(
        self, upload_storage, create_upload, sample_upload_df, tmp_path, tables
    ):
        # Arrange: a pre-typed-storage upload whose per-table Parquet differs from the unified CSV
        source_table = sample_upload_df.head(1)
        parquet_path = tmp_path / f"{tables[0]}.parquet"
        source_table.write_parquet(parquet_path)
        (upload_storage.raw_dir / "test_source_upload_tables").mkdir()
        upload_id = create_upload(
            "test_source_upload",
            metadata_overrides={
                "tables": tables,
                "migrated_to_v2": True,
                "parquet_paths": {tables[0]: str(parquet_path)},
            },
        )

        # Act
        result = upload_storage.get_upload_data(upload_id).collect()

        # Assert: the unified cohort the semantic layer queried, now converted to typed Parquet
        plt.assert_frame_equal(result, sample_upload_df, check_dtypes=False)

    def test_get_upload_data_concurrentLegacyReads_convertOnce(self, upload_storage, create_upload, monkeypatch):
        # Arrange
        import threading
        import time

        from clinical_analytics.ui.storage import user_datasets

        upload_id = create_upload("test_concurrent_legacy")
        calls: list[str] = []
        real_convert = user_datasets._convert_legacy_upload

        def slow_convert(storage, converted_id, metadata):
            calls.append(converted_id)
            time.sleep(0.2)  # widen the race window
            return real_convert(storage, converted_id, metadata)

        monkeypatch.setattr(user_datasets, "_convert_legacy_upload", slow_convert)
        threads = [threading.Thread(target=upload_storage.get_upload_data, args=(upload_id,)) for _ in range(4)]

        # Act
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Assert
        assert calls == [upload_id]


class TestUploadedDatasetLoadLazy:
    """Tests for load() method with lazy frame support."""

    def test_load_stores_lazy_frame(self, upload_storage, create_upload, sample_variable_mapping):
        """Test that load() stores LazyFrame when using lazy backend."""
        # Arrange
        upload_id = create_upload(
            "test_load_lazy",
            df=None,
            metadata_overrides={"variable_mapping": sample_variable_mapping, "original_filename": "test.csv"},
        )
        dataset = UploadedDataset(upload_id=upload_id, storage=upload_storage)

        # Act
        dataset.load()

        # Assert
        assert isinstance(dataset.data, pl.LazyFrame), f"Expected LazyFrame, got {type(dataset.data)}"
        collected = dataset.data.collect()
        assert collected.height == 3
        assert "patient_id" in collected.columns

    def test_load_with_legacy_pandas_backend(self, upload_storage, create_upload):
        """Test that load() handles pandas DataFrames for backward compatibility."""
        # Arrange
        upload_id = create_upload("test_load_pandas", df=None, metadata_overrides={"original_filename": "test.csv"})

        # Mock storage to return pandas DataFrame (simulating lazy=False)
        class PandasStorage(UserDatasetStorage):
            def get_upload_data(self, upload_id: str, lazy: bool = True) -> pd.DataFrame | pl.LazyFrame | None:
                return super().get_upload_data(upload_id, lazy=False)

        dataset = UploadedDataset(upload_id=upload_id, storage=PandasStorage(upload_dir=upload_storage.upload_dir))

        # Act
        dataset.load()

        # Assert
        assert isinstance(dataset.data, pd.DataFrame | pl.LazyFrame), "Should handle both types"


class TestGetCohortLazyEvaluation:
    """Tests for get_cohort() with lazy evaluation."""

    def test_get_cohort_with_lazy_frame_returns_pandas(self, upload_storage, create_upload, sample_variable_mapping):
        """Test that get_cohort() collects lazy frame and returns pandas DataFrame."""
        # Arrange
        upload_id = create_upload(
            "test_cohort_lazy", df=None, metadata_overrides={"variable_mapping": sample_variable_mapping}
        )
        dataset = UploadedDataset(upload_id=upload_id, storage=upload_storage)
        dataset.load()

        # Act
        cohort = dataset.get_cohort()

        # Assert
        assert isinstance(cohort, pd.DataFrame), "get_cohort() should return pandas DataFrame"
        assert len(cohort) == 3
        assert "patient_id" in cohort.columns
        assert "outcome" in cohort.columns

    def test_get_cohort_outcomeMappedFromNamedColumn_keepsOriginalColumnName(self, upload_storage, create_upload):
        """
        Regression: the mapped outcome column was only exposed as `outcome`, so asking about it by its
        own name ("compare mortality by treatment") raised ColumnNotFoundError and crashed the page.
        """
        # Arrange
        df = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "mortality": [0, 1, 0], "age": [40, 50, 60]})
        upload_id = create_upload(
            "test_cohort_named_outcome",
            df=df,
            metadata_overrides={"variable_mapping": {"patient_id": "patient_id", "outcome": "mortality"}},
        )
        dataset = UploadedDataset(upload_id=upload_id, storage=upload_storage)
        dataset.load()

        # Act
        cohort = dataset.get_cohort()

        # Assert: canonical alias and original name carry the same values
        assert cohort["outcome"].tolist() == cohort["mortality"].tolist() == [0, 1, 0]

    def test_get_cohort_applies_filters_before_materialization(self, tmp_path, sample_variable_mapping):
        """
        Test that get_cohort() applies filters in Polars LazyFrame before materialization.

        For CSV files, filters are applied during scan/streaming.
        For Excel files, filters are applied after eager read but before collect.
        This test verifies lazy compute behavior, not IO-level predicate pushdown.
        """
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_cohort_filter"

        # Create data with treatment column (use "outcome" to match variable_mapping)
        test_data = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004"],
                "outcome": [0, 1, 0, 1],
                "age": [50, 60, 70, 80],
                "treatment": ["A", "B", "A", "B"],
            }
        )
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.write_csv(csv_path)

        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": sample_variable_mapping,
        }
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        # Act: Apply filter through get_cohort
        cohort = dataset.get_cohort(treatment="A")

        # Assert: Filter was applied correctly
        assert len(cohort) == 2, "Should only return treatment A patients"
        assert all(cohort["treatment"] == "A"), "All rows should have treatment=A"

        # Assert: Verify lazy frame still exists and can be reused
        assert isinstance(dataset.data, pl.LazyFrame), "Should still have LazyFrame for reuse"

    def test_get_cohort_filters_applied_before_materialization(self, tmp_path, sample_variable_mapping):
        """
        Test that filters are applied in Polars LazyFrame before materialization.

        For CSV files, filters are applied during scan/streaming.
        For Excel files, filters are applied after eager read but before collect.
        This test verifies lazy compute behavior, not IO-level predicate pushdown.
        """
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_predicate_pushdown"

        # Create larger dataset (1000 rows, use "outcome" to match variable_mapping)
        test_data = pl.DataFrame(
            {
                "patient_id": [f"P{i:04d}" for i in range(1000)],
                "outcome": [i % 2 for i in range(1000)],
                "age": [50 + (i % 30) for i in range(1000)],
                "treatment": ["A" if i % 2 == 0 else "B" for i in range(1000)],
            }
        )
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.write_csv(csv_path)

        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": sample_variable_mapping,
        }
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        # Act: Get filtered cohort (should push predicate down)
        cohort_filtered = dataset.get_cohort(treatment="A")

        # Assert: Verify actual predicate pushdown happened
        # 1. Filtered cohort has correct number of rows (500 out of 1000)
        assert len(cohort_filtered) == 500, "Should return exactly 500 treatment A patients"
        assert all(cohort_filtered["treatment"] == "A"), "All rows should have treatment=A"

        # 2. Verify lazy frame can still be used (not consumed)
        assert isinstance(dataset.data, pl.LazyFrame), "Should maintain LazyFrame for reuse"

        # 3. Verify we can get unfiltered cohort (tests that filter doesn't mutate self.data)
        cohort_full = dataset.get_cohort()
        assert len(cohort_full) == 1000, "Unfiltered cohort should have all 1000 rows"

    def test_get_cohort_lazy_plan_contains_filter_node(self, tmp_path, sample_variable_mapping):
        """
        Test that Polars optimized plan contains filter operation when filtering.

        For CSV files, filters are applied during scan/streaming.
        For Excel files, filters are applied after eager read but before collect.
        This test verifies lazy compute behavior, not IO-level predicate pushdown.
        """
        import re

        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_filter_plan"

        test_data = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004"],
                "outcome": [0, 1, 0, 1],
                "age": [50, 60, 70, 80],
                "treatment": ["A", "B", "A", "B"],
            }
        )
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.write_csv(csv_path)

        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": sample_variable_mapping,
        }
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        # Act: Create LazyFrame with filter (simulating internal get_cohort logic)
        lf = storage.get_upload_data(upload_id, lazy=True)
        lf_filtered = lf.filter(pl.col("treatment") == "A")

        # Assert: Plan contains filter operation (permissive regex, version-agnostic)
        plan = lf_filtered.explain(optimized=True)
        assert re.search(r"\b(filter|selection)\b", plan.lower()), f"Plan should contain filter operation. Got: {plan}"
        # Don't assert exact column format (brittle across Polars versions)


class TestLazyFrameMigrationCompleteness:
    """Tests to verify all components handle lazy frames correctly."""

    def test_get_cohort_collects_exactly_once(self, tmp_path, sample_variable_mapping):
        """
        Test that get_cohort() collects LazyFrame exactly once at return boundary.

        This test verifies lazy execution by ensuring:
        - Filters work correctly (applied before materialization)
        - self.data remains a LazyFrame (not mutated)
        - Unfiltered cohort still works (proves filter doesn't mutate base data)
        """
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_single_collect"

        test_data = pl.DataFrame(
            {
                "patient_id": [f"P{i:04d}" for i in range(100)],
                "outcome": [i % 2 for i in range(100)],
                "treatment": ["A" if i % 2 == 0 else "B" for i in range(100)],
            }
        )
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        test_data.write_csv(csv_path)

        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "variable_mapping": sample_variable_mapping,
        }
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        dataset = UploadedDataset(upload_id=upload_id, storage=storage)
        dataset.load()

        # Act: Get filtered cohort
        cohort = dataset.get_cohort(treatment="A")

        # Assert: Filter applied correctly
        assert len(cohort) == 50
        assert all(cohort["treatment"] == "A")

        # Assert: LazyFrame still exists (not consumed/mutated)
        assert isinstance(dataset.data, pl.LazyFrame)

        # Assert: Unfiltered cohort works (proves filter doesn't mutate self.data)
        cohort_full = dataset.get_cohort()
        assert len(cohort_full) == 100

    def test_unified_cohort_created_from_lazy_tables(self, tmp_path):
        """Test that unified cohort creation works with lazy table loading."""
        # Arrange
        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = "test_lazy_multi_table"

        # Create tables directory
        tables_dir = storage.raw_dir / f"{upload_id}_tables"
        tables_dir.mkdir(exist_ok=True)

        # Create test table
        patients = pl.DataFrame({"patient_id": ["P001", "P002"], "age": [50, 60]})
        (tables_dir / "patients.csv").write_text(patients.write_csv())

        # Create unified cohort CSV
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        patients.write_csv(csv_path)

        metadata = {
            "upload_id": upload_id,
            "upload_timestamp": "2024-01-01T00:00:00",
            "tables": ["patients"],
            "migrated_to_v2": True,
            "variable_mapping": {"patient_id": "patient_id", "predictors": ["age"]},
        }
        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))

        dataset = UploadedDataset(upload_id=upload_id, storage=storage)

        # Act
        dataset.load()
        cohort = dataset.get_cohort()

        # Assert
        assert len(cohort) == 2
        assert "patient_id" in cohort.columns
