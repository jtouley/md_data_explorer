"""
Test to measure caching impact on fixture generation time.

This test measures the performance improvement from caching:
- Baseline: Time to generate fixtures without cache
- Cached: Time to load fixtures from cache
- Target: 50-80% reduction in data loading time
"""

import sys
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.cache import (
    cache_dataframe,
    get_cached_dataframe,
    hash_dataframe,
)


class TestCachingImpact:
    """Verify the fixture cache round-trips data and short-circuits regeneration.

    Wall-clock comparisons were removed: they measured polars/openpyxl I/O speed, not this code,
    and flipped under parallel load.
    """

    def test_dataframe_cache_roundtrip_returnsIdenticalFrame(self, tmp_path):
        # Arrange
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:04d}" for i in range(1000)],
                "age": [20 + (i % 80) for i in range(1000)],
                "outcome": [i % 2 for i in range(1000)],
                "value": [float(i) * 1.5 for i in range(1000)],
            }
        )
        cache_key = hash_dataframe(df)

        # Act
        cache_dataframe(df, cache_key, cache_dir)
        cached_df = get_cached_dataframe(cache_key, cache_dir)

        # Assert
        assert cached_df is not None
        assert_frame_equal(cached_df, df)

    def test_excel_cache_second_call_skipsRegeneration(self, tmp_path_factory, tmp_path, monkeypatch):
        # Arrange: isolated cache dir (the shared default dir must not be wiped under xdist)
        import fixtures.factories as factories

        monkeypatch.setenv("TEST_CACHE_DIR", str(tmp_path / "excel_cache"))
        (tmp_path / "excel_cache").mkdir()
        data = {"patient_id": [f"P{i:04d}" for i in range(50)], "outcome": [i % 2 for i in range(50)]}
        writer = factories._write_excel_with_layout
        calls: list[Path] = []

        def counting_writer(path, *args, **kwargs):
            calls.append(path)
            return writer(path, *args, **kwargs)

        monkeypatch.setattr(factories, "_write_excel_with_layout", counting_writer)

        # Act
        first = factories._create_synthetic_excel_file(tmp_path_factory, data, "a.xlsx", {"header_row": 0})
        second = factories._create_synthetic_excel_file(tmp_path_factory, data, "b.xlsx", {"header_row": 0})

        # Assert
        assert first.exists() and second.exists()
        assert len(calls) == 1

    def test_excel_cache_differentLayout_doesNotCollide(self, tmp_path_factory, tmp_path, monkeypatch):
        # Arrange
        import fixtures.factories as factories

        monkeypatch.setenv("TEST_CACHE_DIR", str(tmp_path / "excel_cache"))
        (tmp_path / "excel_cache").mkdir()
        data = {"patient_id": [f"P{i:04d}" for i in range(50)], "outcome": [i % 2 for i in range(50)]}
        metadata_rows = [{"row_index": 0, "cells": ["note", ""]}]

        # Act
        plain = factories._create_synthetic_excel_file(tmp_path_factory, data, "plain.xlsx", {"header_row": 0})
        with_meta = factories._create_synthetic_excel_file(
            tmp_path_factory, data, "meta.xlsx", {"header_row": 1, "metadata_rows": metadata_rows}
        )

        # Assert: a layout change must produce a distinct cached file
        assert plain.read_bytes() != with_meta.read_bytes()
