"""
Tests for cached fixture behavior.

Tests verify that expensive fixtures use caching to avoid regeneration.
These tests are generic and work with any data structure.
"""

import pytest


@pytest.mark.slow
class TestCachedExcelFixtures:
    """Test that Excel fixtures use caching with any data structure."""

    @pytest.mark.parametrize(
        "test_data,description",
        [
            # Simple numeric data
            ({"id": [1, 2, 3], "value": [10, 20, 30]}, "simple_numeric"),
            # Mixed data types
            (
                {
                    "category": ["A", "B", "C"],
                    "count": [100, 200, 300],
                    "active": [True, False, True],
                },
                "mixed_types",
            ),
            # String data with duplicates (realistic clinical data pattern)
            (
                {
                    "group": ["Group1"] * 30 + ["Group2"] * 20,
                    "status": ["Active", "Inactive"] * 25,
                    "score": list(range(50)),
                },
                "string_with_duplicates",
            ),
            # Large dataset (realistic size)
            (
                {
                    "id": list(range(1000)),
                    "value": [i * 0.1 for i in range(1000)],
                    "label": [f"item_{i}" for i in range(1000)],
                },
                "large_dataset",
            ),
        ],
    )
    def test_excel_file_caching_creates_cache_entry(self, tmp_path, test_data, description):
        """Test that Excel file caching creates cache entry for any data structure."""
        # Arrange: Set up cache directory
        cache_dir = tmp_path / ".test_cache"
        cache_dir.mkdir()

        import pandas as pd

        from tests.fixtures.cache import (
            cache_excel_file,
            get_cached_excel_file,
            hash_file,
        )

        # Create Excel file with test data
        df = pd.DataFrame(test_data)
        test_file = tmp_path / f"test_{description}.xlsx"
        df.to_excel(test_file, index=False, engine="openpyxl")

        # Generate cache key from file content
        cache_key = hash_file(test_file)

        # Act: Cache the file
        cached_path = cache_excel_file(test_file, cache_key, cache_dir)

        # Assert: Cache entry exists and can be retrieved
        assert cached_path.exists(), "Cached file should exist"
        retrieved = get_cached_excel_file(cache_key, cache_dir)
        assert retrieved is not None, "Should retrieve cached file"
        assert retrieved == cached_path, "Retrieved path should match cached path"

    @pytest.mark.parametrize(
        "source_data,description",
        [
            # Simple numeric data
            ({"id": [1, 2, 3], "value": [10, 20, 30]}, "simple_numeric"),
            # Mixed data types
            (
                {
                    "category": ["A", "B", "C"],
                    "count": [100, 200, 300],
                    "active": [True, False, True],
                },
                "mixed_types",
            ),
            # String data with duplicates
            (
                {
                    "group": ["Group1"] * 30 + ["Group2"] * 20,
                    "status": ["Active", "Inactive"] * 25,
                },
                "string_with_duplicates",
            ),
        ],
    )
    def test_cached_excel_file_retrieves_from_cache(self, tmp_path, source_data, description):
        """Test that cached Excel file can be retrieved from cache for any data structure."""
        # Arrange: Pre-populate cache
        cache_dir = tmp_path / ".test_cache"
        cache_dir.mkdir()

        import pandas as pd

        from tests.fixtures.cache import (
            cache_excel_file,
            get_cached_excel_file,
            hash_file,
        )

        # Create source file with test data
        source_file = tmp_path / f"source_{description}.xlsx"
        pd.DataFrame(source_data).to_excel(source_file, index=False, engine="openpyxl")

        # Cache it
        cache_key = hash_file(source_file)
        cache_excel_file(source_file, cache_key, cache_dir)

        # Act: Retrieve from cache
        cached_file = get_cached_excel_file(cache_key, cache_dir)

        # Assert: Retrieved file matches source
        assert cached_file is not None, "Should retrieve from cache"
        assert cached_file.exists(), "Cached file should exist"

        # Verify content matches
        cached_df = pd.read_excel(cached_file, engine="openpyxl")
        source_df = pd.read_excel(source_file, engine="openpyxl")
        pd.testing.assert_frame_equal(cached_df, source_df)
