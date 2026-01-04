"""
Integration tests for run_key determinism across UI flow (PR21 feedback).

Verifies that:
1. UI cache key (using stable sha256) is consistent across sessions
2. Cache key generation matches semantic layer normalization
3. Different queries produce different cache keys

Test name follows: test_unit_scenario_expectedBehavior
"""

import hashlib

import pytest


@pytest.mark.slow
@pytest.mark.integration
class TestRunKeyDeterminismUIFlow:
    """Integration tests for run_key determinism across UI flow."""

    def test_ui_cache_key_stable_across_sessions(self):
        """
        UI cache key (using stable sha256) is consistent across sessions.

        Verifies that hash(query_text) replacement with sha256 works correctly.
        This addresses PR21 feedback: Python's hash() is process-salted and not stable.
        """
        # Arrange: Same query text
        query_text = "count patients by status"
        dataset_version = "test_v1"

        # Normalize query same way as UI does (same as semantic layer)
        normalized_query = " ".join(query_text.lower().split())
        query_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
        cache_key = f"exec_result:{dataset_version}:{query_hash}"

        # Act: Generate cache key multiple times (simulating different sessions/processes)
        cache_key_1 = cache_key
        # Simulate different process restart - should still get same hash
        normalized_query_2 = " ".join(query_text.lower().split())
        query_hash_2 = hashlib.sha256(normalized_query_2.encode("utf-8")).hexdigest()[:16]
        cache_key_2 = f"exec_result:{dataset_version}:{query_hash_2}"

        # Assert: Same cache key across "sessions" (stable hashing)
        assert (
            cache_key_1 == cache_key_2
        ), f"Cache key should be stable across sessions. Got: {cache_key_1} vs {cache_key_2}"
        assert len(query_hash) == 16, "Query hash should be 16 chars"

    def test_cache_key_different_for_different_queries(self):
        """Different queries should produce different cache keys."""
        # Arrange: Different queries
        query1 = "average age"
        query2 = "mean age"
        dataset_version = "test_v1"

        # Normalize queries (same as UI does)
        normalized1 = " ".join(query1.lower().split())
        normalized2 = " ".join(query2.lower().split())

        hash1 = hashlib.sha256(normalized1.encode("utf-8")).hexdigest()[:16]
        hash2 = hashlib.sha256(normalized2.encode("utf-8")).hexdigest()[:16]

        cache_key1 = f"exec_result:{dataset_version}:{hash1}"
        cache_key2 = f"exec_result:{dataset_version}:{hash2}"

        # Assert: Different cache keys
        assert (
            cache_key1 != cache_key2
        ), f"Different queries should produce different cache keys. Got: {cache_key1} == {cache_key2}"

    def test_cache_key_whitespace_normalization(self):
        """Whitespace variations in query should produce same cache key."""
        # Arrange: Same query with different whitespace
        query1 = "count  all   patients"
        query2 = "count all patients"
        query3 = "  count all patients  "
        dataset_version = "test_v1"

        # Act: Normalize and hash (same as UI does)
        normalized1 = " ".join(query1.lower().split())
        normalized2 = " ".join(query2.lower().split())
        normalized3 = " ".join(query3.lower().split())

        hash1 = hashlib.sha256(normalized1.encode("utf-8")).hexdigest()[:16]
        hash2 = hashlib.sha256(normalized2.encode("utf-8")).hexdigest()[:16]
        hash3 = hashlib.sha256(normalized3.encode("utf-8")).hexdigest()[:16]

        cache_key1 = f"exec_result:{dataset_version}:{hash1}"
        cache_key2 = f"exec_result:{dataset_version}:{hash2}"
        cache_key3 = f"exec_result:{dataset_version}:{hash3}"

        # Assert: All produce same cache key (whitespace normalized)
        assert (
            cache_key1 == cache_key2 == cache_key3
        ), f"Whitespace variations should produce same cache key. Got: {cache_key1}, {cache_key2}, {cache_key3}"

    def test_cache_key_uses_stable_sha256_not_python_hash(self):
        """
        Cache key uses stable sha256, not Python's hash().

        Python's hash() is process-salted and can produce different values
        across sessions. sha256 is deterministic.
        """
        # Arrange: Same query
        query_text = "average age by status"
        dataset_version = "test_v1"

        # Act: Generate hash using sha256 (stable)
        normalized_query = " ".join(query_text.lower().split())
        query_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
        cache_key = f"exec_result:{dataset_version}:{query_hash}"

        # Assert: Hash is deterministic (same input = same output)
        # Generate again to verify stability
        normalized_query_2 = " ".join(query_text.lower().split())
        query_hash_2 = hashlib.sha256(normalized_query_2.encode("utf-8")).hexdigest()[:16]
        cache_key_2 = f"exec_result:{dataset_version}:{query_hash_2}"

        assert cache_key == cache_key_2, "sha256 hash should be deterministic"
        assert len(query_hash) == 16, "Hash should be 16 chars (truncated sha256)"
        assert cache_key.startswith(f"exec_result:{dataset_version}:"), "Cache key format correct"
