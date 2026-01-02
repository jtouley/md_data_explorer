"""
Tests for test data caching infrastructure.

Tests verify:
- Content-based hashing generates consistent hashes for same data
- DataFrame caching stores and retrieves parquet files correctly
- Excel file caching stores and retrieves Excel files correctly
- Cache invalidation works when data changes
"""

import polars as pl


class TestContentBasedHashing:
    """Test content-based hashing for cache keys."""

    def test_hash_dataframe_generates_consistent_hash(self):
        """Test that same DataFrame generates same hash."""
        # Arrange: Create DataFrame
        df1 = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        # Act: Generate hashes
        from tests.fixtures.cache import hash_dataframe

        hash1 = hash_dataframe(df1)
        hash2 = hash_dataframe(df2)

        # Assert: Same data = same hash
        assert hash1 == hash2, "Same DataFrame should generate same hash"

    def test_hash_dataframe_detects_different_data(self):
        """Test that different DataFrames generate different hashes."""
        # Arrange: Create different DataFrames
        df1 = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 40]})  # Different value

        # Act: Generate hashes
        from tests.fixtures.cache import hash_dataframe

        hash1 = hash_dataframe(df1)
        hash2 = hash_dataframe(df2)

        # Assert: Different data = different hash
        assert hash1 != hash2, "Different DataFrames should generate different hashes"

    def test_hash_file_generates_consistent_hash(self, tmp_path):
        """Test that same file content generates same hash."""
        # Arrange: Create file with content
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Act: Generate hash
        from tests.fixtures.cache import hash_file

        hash1 = hash_file(test_file)
        hash2 = hash_file(test_file)

        # Assert: Same file = same hash
        assert hash1 == hash2, "Same file should generate same hash"

    def test_hash_file_detects_different_content(self, tmp_path):
        """Test that different file content generates different hashes."""
        # Arrange: Create files with different content
        file1 = tmp_path / "test1.txt"
        file1.write_text("content 1")
        file2 = tmp_path / "test2.txt"
        file2.write_text("content 2")

        # Act: Generate hashes
        from tests.fixtures.cache import hash_file

        hash1 = hash_file(file1)
        hash2 = hash_file(file2)

        # Assert: Different content = different hash
        assert hash1 != hash2, "Different file content should generate different hashes"


class TestDataFrameCaching:
    """Test DataFrame caching with parquet storage."""

    def test_cache_dataframe_stores_parquet_file(self, tmp_path):
        """Test that cache_dataframe stores DataFrame as parquet."""
        # Arrange: Create DataFrame and cache directory
        df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        cache_dir = tmp_path / ".test_cache"
        cache_dir.mkdir()

        # Act: Cache DataFrame
        from tests.fixtures.cache import cache_dataframe

        cache_path = cache_dataframe(df, "test_key", cache_dir)

        # Assert: Parquet file exists
        assert cache_path.exists(), "Cache file should exist"
        assert cache_path.suffix == ".parquet", "Cache file should be parquet"

    def test_get_cached_dataframe_retrieves_parquet_file(self, tmp_path):
        """Test that get_cached_dataframe retrieves cached DataFrame."""
        # Arrange: Create and cache DataFrame
        original_df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        cache_dir = tmp_path / ".test_cache"
        cache_dir.mkdir()

        from tests.fixtures.cache import cache_dataframe, get_cached_dataframe

        cache_key = "test_key"
        cache_dataframe(original_df, cache_key, cache_dir)

        # Act: Retrieve cached DataFrame
        cached_df = get_cached_dataframe(cache_key, cache_dir)

        # Assert: Retrieved DataFrame matches original
        assert cached_df is not None, "Should retrieve cached DataFrame"
        pl.testing.assert_frame_equal(cached_df, original_df)

    def test_get_cached_dataframe_returns_none_when_not_cached(self, tmp_path):
        """Test that get_cached_dataframe returns None when cache miss."""
        # Arrange: Empty cache directory
        cache_dir = tmp_path / ".test_cache"
        cache_dir.mkdir()

        # Act: Try to retrieve non-existent cache
        from tests.fixtures.cache import get_cached_dataframe

        cached_df = get_cached_dataframe("non_existent_key", cache_dir)

        # Assert: Should return None
        assert cached_df is None, "Should return None when cache miss"


class TestExcelFileCaching:
    """Test Excel file caching."""

    def test_cache_excel_file_stores_file(self, tmp_path):
        """Test that cache_excel_file stores Excel file."""
        # Arrange: Create Excel file
        import pandas as pd

        source_file = tmp_path / "source.xlsx"
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df.to_excel(source_file, index=False, engine="openpyxl")

        cache_dir = tmp_path / ".test_cache"
        cache_dir.mkdir()

        # Act: Cache Excel file
        from tests.fixtures.cache import cache_excel_file

        cache_path = cache_excel_file(source_file, "test_key", cache_dir)

        # Assert: Cached file exists
        assert cache_path.exists(), "Cached file should exist"
        assert cache_path.suffix == ".xlsx", "Cached file should be Excel"

    def test_get_cached_excel_file_retrieves_file(self, tmp_path):
        """Test that get_cached_excel_file retrieves cached Excel file."""
        # Arrange: Create and cache Excel file
        import pandas as pd

        source_file = tmp_path / "source.xlsx"
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df.to_excel(source_file, index=False, engine="openpyxl")

        cache_dir = tmp_path / ".test_cache"
        cache_dir.mkdir()

        from tests.fixtures.cache import cache_excel_file, get_cached_excel_file

        cache_key = "test_key"
        cache_excel_file(source_file, cache_key, cache_dir)

        # Act: Retrieve cached file
        cached_file = get_cached_excel_file(cache_key, cache_dir)

        # Assert: Retrieved file exists and matches
        assert cached_file is not None, "Should retrieve cached file"
        assert cached_file.exists(), "Cached file should exist"

        # Verify content matches
        cached_df = pd.read_excel(cached_file, engine="openpyxl")
        pd.testing.assert_frame_equal(cached_df, df)

    def test_get_cached_excel_file_returns_none_when_not_cached(self, tmp_path):
        """Test that get_cached_excel_file returns None when cache miss."""
        # Arrange: Empty cache directory
        cache_dir = tmp_path / ".test_cache"
        cache_dir.mkdir()

        # Act: Try to retrieve non-existent cache
        from tests.fixtures.cache import get_cached_excel_file

        cached_file = get_cached_excel_file("non_existent_key", cache_dir)

        # Assert: Should return None
        assert cached_file is None, "Should return None when cache miss"
