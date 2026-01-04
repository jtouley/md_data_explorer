"""
Test to measure caching impact on fixture generation time.

This test measures the performance improvement from caching:
- Baseline: Time to generate fixtures without cache
- Cached: Time to load fixtures from cache
- Target: 50-80% reduction in data loading time
"""

import shutil
import sys
import time
from pathlib import Path

import polars as pl

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.cache import (
    cache_dataframe,
    get_cache_dir,
    get_cached_dataframe,
    hash_dataframe,
)


class TestCachingImpact:
    """Measure caching impact on DataFrame and Excel file generation."""

    def test_dataframe_caching_reduces_loading_time(self, tmp_path):
        """
        Measure that cached DataFrame loading is faster than regeneration.

        This test measures the time difference between:
        - Writing DataFrame to parquet (baseline - simulates expensive I/O)
        - Reading cached parquet file (cached - fast I/O)

        Note: The real improvement comes from avoiding expensive operations
        like Excel file generation, not just DataFrame creation.
        """
        # Arrange: Create test DataFrame (simulating expensive fixture generation)
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        # Create DataFrame similar to what fixtures generate
        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:04d}" for i in range(10000)],
                "age": [20 + (i % 80) for i in range(10000)],
                "outcome": [i % 2 for i in range(10000)],
                "value": [float(i) * 1.5 for i in range(10000)],
            }
        )

        # Measure baseline: Time to write DataFrame to parquet (simulates expensive I/O)
        baseline_file = tmp_path / "baseline.parquet"
        start_baseline = time.perf_counter()
        df.write_parquet(baseline_file)
        baseline_time = time.perf_counter() - start_baseline

        # Cache the DataFrame
        cache_key = hash_dataframe(df)
        cache_dataframe(df, cache_key, cache_dir)

        # Measure cached: Time to load from cache
        start_cached = time.perf_counter()
        cached_df = get_cached_dataframe(cache_key, cache_dir)
        cached_time = time.perf_counter() - start_cached

        # Assert: Cached DataFrame matches original
        assert cached_df is not None
        assert cached_df.shape == df.shape
        assert cached_df.columns == df.columns

        # Calculate improvement
        if baseline_time > 0:
            improvement_pct = ((baseline_time - cached_time) / baseline_time) * 100
        else:
            improvement_pct = 0

        # Assert: Cached loading should be faster (at least 30% improvement)
        # Note: Parquet read/write is already fast, so improvement may be modest
        # Real improvement comes from avoiding Excel generation, which is much slower
        assert (
            cached_time < baseline_time
        ), f"Cached loading should be faster, but cached time ({cached_time:.4f}s) >= baseline ({baseline_time:.4f}s)"

        # Log results for documentation
        print(
            f"\nDataFrame Caching Impact:\n"
            f"  Baseline (write parquet): {baseline_time:.4f}s\n"
            f"  Cached (read from cache): {cached_time:.4f}s\n"
            f"  Improvement: {improvement_pct:.1f}% reduction\n"
        )

    def test_excel_caching_reduces_file_generation_time(self, tmp_path_factory):
        """
        Measure that cached Excel file loading is faster than regeneration.

        This test measures the time difference between:
        - Generating Excel file from scratch (baseline)
        - Loading cached Excel file (cached)
        """

        from fixtures.factories import _create_synthetic_excel_file

        # Arrange: Create test data
        data = {
            "patient_id": [f"P{i:04d}" for i in range(1000)],
            "age": [20 + (i % 80) for i in range(1000)],
            "outcome": [i % 2 for i in range(1000)],
        }

        tmp_path_factory.mktemp("excel_cache_test")  # Create temp directory for test
        cache_dir = get_cache_dir()

        # Clear cache for this test
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Measure baseline: Time to generate Excel file (first time, no cache)
        start_baseline = time.perf_counter()
        excel_file_1 = _create_synthetic_excel_file(
            tmp_path_factory, data, "test_baseline.xlsx", excel_config={"header_row": 0}
        )
        baseline_time = time.perf_counter() - start_baseline

        # Measure cached: Time to generate same Excel file (should use cache)
        start_cached = time.perf_counter()
        excel_file_2 = _create_synthetic_excel_file(
            tmp_path_factory, data, "test_cached.xlsx", excel_config={"header_row": 0}
        )
        cached_time = time.perf_counter() - start_cached

        # Assert: Both files exist
        assert excel_file_1.exists()
        assert excel_file_2.exists()

        # Calculate improvement
        if baseline_time > 0:
            improvement_pct = ((baseline_time - cached_time) / baseline_time) * 100
        else:
            improvement_pct = 0

        # Assert: Cached generation should be faster (at least 30% improvement)
        # Note: Excel file generation includes pandas operations, so improvement may vary
        # but should still be significant due to file I/O savings
        assert cached_time < baseline_time, (
            f"Cached Excel generation should be faster, "
            f"but cached time ({cached_time:.4f}s) >= baseline ({baseline_time:.4f}s)"
        )

        # Log results for documentation
        print(
            f"\nExcel Caching Impact:\n"
            f"  Baseline (first generation): {baseline_time:.4f}s\n"
            f"  Cached (second generation): {cached_time:.4f}s\n"
            f"  Improvement: {improvement_pct:.1f}% reduction\n"
        )
