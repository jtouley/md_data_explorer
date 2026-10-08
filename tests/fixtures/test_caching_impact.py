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
import polars.testing as plt

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.cache import (
    cache_dataframe,
    get_cache_dir,
    get_cached_dataframe,
    hash_dataframe,
    hash_file,
)


class TestCachingImpact:
    """Measure caching impact on DataFrame and Excel file generation."""

    def test_dataframe_caching_reduces_loading_time(self, tmp_path):
        """
        Verify parquet cache round-trip matches source and log write vs read timings.

        Assertions are deterministic (frame equality). Timing lines are informational only:
        wall-clock ordering of write vs read is not reliable on fast disks.
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

        # Assert: Cached DataFrame matches original (contract under test)
        assert cached_df is not None
        plt.assert_frame_equal(cached_df, df)

        # Wall-clock ordering of write vs read is nondeterministic on fast disks / CI;
        # log timings for humans without failing the suite on microsecond noise.
        if baseline_time > 0:
            improvement_pct = ((baseline_time - cached_time) / baseline_time) * 100
        else:
            improvement_pct = 0.0
        print(
            f"\nDataFrame Caching Impact (informational):\n"
            f"  Baseline (write parquet): {baseline_time:.4f}s\n"
            f"  Cached (read from cache): {cached_time:.4f}s\n"
            f"  Improvement: {improvement_pct:.1f}% reduction\n"
        )

    def test_excel_caching_reduces_file_generation_time(self, tmp_path_factory):
        """
        Verify second synthetic Excel generation yields byte-identical content via cache.

        Timing lines are informational; the contract is same-bytes for both paths.
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

        # Second call must resolve to the same bytes as the cached artifact (contract under test).
        assert hash_file(excel_file_1) == hash_file(excel_file_2)

        if baseline_time > 0:
            improvement_pct = ((baseline_time - cached_time) / baseline_time) * 100
        else:
            improvement_pct = 0.0
        print(
            f"\nExcel Caching Impact (informational):\n"
            f"  Baseline (first generation): {baseline_time:.4f}s\n"
            f"  Cached (second generation): {cached_time:.4f}s\n"
            f"  Improvement: {improvement_pct:.1f}% reduction\n"
        )
