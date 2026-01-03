"""
Tests for ResultCache - Pure Python result caching with LRU eviction.

Tests verify UI-agnostic result storage, retrieval, and LRU eviction per dataset.
"""

from datetime import datetime

from clinical_analytics.core.result_cache import CachedResult, ResultCache


class TestResultCache:
    """Test suite for ResultCache."""

    def test_result_cache_initializes_empty(self):
        """Test that ResultCache initializes with empty cache."""
        # Arrange
        cache = ResultCache(max_size=50)

        # Act
        result = cache.get("run_key_1", "dataset_v1")

        # Assert
        assert result is None

    def test_result_cache_put_stores_result(self):
        """Test that put stores a result in the cache."""
        # Arrange
        cache = ResultCache(max_size=50)
        cached_result = CachedResult(
            run_key="run_key_1",
            query="What is the average age?",
            result={"mean": 45.5},
            timestamp=datetime.now(),
            dataset_version="dataset_v1",
        )

        # Act
        cache.put(cached_result)
        retrieved = cache.get("run_key_1", "dataset_v1")

        # Assert
        assert retrieved is not None
        assert retrieved.run_key == "run_key_1"
        assert retrieved.query == "What is the average age?"
        assert retrieved.result == {"mean": 45.5}

    def test_result_cache_get_returns_none_for_missing_key(self):
        """Test that get returns None for non-existent run key."""
        # Arrange
        cache = ResultCache(max_size=50)

        # Act
        result = cache.get("nonexistent_key", "dataset_v1")

        # Assert
        assert result is None

    def test_result_cache_get_returns_none_for_different_dataset(self):
        """Test that get returns None for same run_key but different dataset."""
        # Arrange
        cache = ResultCache(max_size=50)
        cached_result = CachedResult(
            run_key="run_key_1",
            query="test query",
            result={"data": 1},
            timestamp=datetime.now(),
            dataset_version="dataset_v1",
        )
        cache.put(cached_result)

        # Act
        result = cache.get("run_key_1", "dataset_v2")  # Different dataset

        # Assert
        assert result is None

    def test_result_cache_lru_eviction_removes_oldest(self):
        """Test that LRU eviction removes oldest result when max_size reached."""
        # Arrange
        cache = ResultCache(max_size=3)  # Small max_size for testing
        dataset_version = "dataset_v1"

        # Act: Add 4 results (one more than max_size)
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key2",
                query="query2",
                result={"data": 2},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key3",
                query="query3",
                result={"data": 3},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key4",
                query="query4",
                result={"data": 4},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )

        # Assert: Oldest (key1) should be evicted
        assert cache.get("key1", dataset_version) is None
        assert cache.get("key2", dataset_version) is not None
        assert cache.get("key3", dataset_version) is not None
        assert cache.get("key4", dataset_version) is not None

    def test_result_cache_lru_preserves_recently_accessed(self):
        """Test that accessing a result moves it to end of LRU (most recent)."""
        # Arrange
        cache = ResultCache(max_size=3)
        dataset_version = "dataset_v1"

        # Act: Add 3 results, then access first one
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key2",
                query="query2",
                result={"data": 2},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key3",
                query="query3",
                result={"data": 3},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        # Access key1 (should move it to end)
        cache.get("key1", dataset_version)
        # Add key4 (should evict key2, not key1)
        cache.put(
            CachedResult(
                run_key="key4",
                query="query4",
                result={"data": 4},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )

        # Assert: key1 should still be there (was accessed), key2 should be evicted
        assert cache.get("key1", dataset_version) is not None
        assert cache.get("key2", dataset_version) is None
        assert cache.get("key3", dataset_version) is not None
        assert cache.get("key4", dataset_version) is not None

    def test_result_cache_evict_oldest_removes_oldest(self):
        """Test that evict_oldest removes oldest result for dataset."""
        # Arrange
        cache = ResultCache(max_size=50)
        dataset_version = "dataset_v1"
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key2",
                query="query2",
                result={"data": 2},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )

        # Act
        cache.evict_oldest(dataset_version)

        # Assert: key1 (oldest) should be gone, key2 should remain
        assert cache.get("key1", dataset_version) is None
        assert cache.get("key2", dataset_version) is not None

    def test_result_cache_clear_removes_all_for_dataset(self):
        """Test that clear removes all results for specified dataset."""
        # Arrange
        cache = ResultCache(max_size=50)
        dataset_version = "dataset_v1"
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key2",
                query="query2",
                result={"data": 2},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )

        # Act
        cache.clear(dataset_version)

        # Assert
        assert cache.get("key1", dataset_version) is None
        assert cache.get("key2", dataset_version) is None

    def test_result_cache_clear_all_removes_all_datasets(self):
        """Test that clear(None) removes all results for all datasets."""
        # Arrange
        cache = ResultCache(max_size=50)
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version="dataset_v1",
            )
        )
        cache.put(
            CachedResult(
                run_key="key2",
                query="query2",
                result={"data": 2},
                timestamp=datetime.now(),
                dataset_version="dataset_v2",
            )
        )

        # Act
        cache.clear(None)

        # Assert
        assert cache.get("key1", "dataset_v1") is None
        assert cache.get("key2", "dataset_v2") is None

    def test_result_cache_get_history_returns_run_keys_in_lru_order(self):
        """Test that get_history returns run keys in LRU order (oldest to newest)."""
        # Arrange
        cache = ResultCache(max_size=50)
        dataset_version = "dataset_v1"
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key2",
                query="query2",
                result={"data": 2},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key3",
                query="query3",
                result={"data": 3},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )

        # Act
        history = cache.get_history(dataset_version)

        # Assert: Should be in order added (oldest to newest)
        assert history == ["key1", "key2", "key3"]

    def test_result_cache_get_history_updates_order_on_access(self):
        """Test that accessing a result updates its position in history."""
        # Arrange
        cache = ResultCache(max_size=50)
        dataset_version = "dataset_v1"
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        cache.put(
            CachedResult(
                run_key="key2",
                query="query2",
                result={"data": 2},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )
        # Access key1 (should move it to end)
        cache.get("key1", dataset_version)

        # Act
        history = cache.get_history(dataset_version)

        # Assert: key1 should be at end (most recently accessed)
        assert history == ["key2", "key1"]

    def test_result_cache_get_history_returns_empty_for_empty_dataset(self):
        """Test that get_history returns empty list for dataset with no results."""
        # Arrange
        cache = ResultCache(max_size=50)

        # Act
        history = cache.get_history("dataset_v1")

        # Assert
        assert history == []

    def test_result_cache_serialize_returns_dict(self):
        """Test that serialize returns serializable dict."""
        # Arrange
        cache = ResultCache(max_size=50)
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version="dataset_v1",
            )
        )

        # Act
        serialized = cache.serialize()

        # Assert
        assert isinstance(serialized, dict)
        assert "results" in serialized
        assert "histories" in serialized

    def test_result_cache_deserialize_restores_state(self):
        """Test that deserialize restores ResultCache state."""
        # Arrange
        cache = ResultCache(max_size=50)
        cache.put(
            CachedResult(
                run_key="key1",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version="dataset_v1",
            )
        )
        serialized = cache.serialize()

        # Act
        restored = ResultCache.deserialize(serialized)

        # Assert
        assert restored.get("key1", "dataset_v1") is not None
        assert restored.get("key1", "dataset_v1").query == "query1"

    def test_result_cache_per_dataset_isolation(self):
        """Test that results are isolated per dataset version."""
        # Arrange
        cache = ResultCache(max_size=50)
        cache.put(
            CachedResult(
                run_key="same_key",
                query="query1",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version="dataset_v1",
            )
        )
        cache.put(
            CachedResult(
                run_key="same_key",
                query="query2",
                result={"data": 2},
                timestamp=datetime.now(),
                dataset_version="dataset_v2",
            )
        )

        # Act
        result1 = cache.get("same_key", "dataset_v1")
        result2 = cache.get("same_key", "dataset_v2")

        # Assert: Same run_key but different datasets should return different results
        assert result1 is not None
        assert result2 is not None
        assert result1.query == "query1"
        assert result2.query == "query2"
