"""
ResultCache - Pure Python result caching with LRU eviction.

Extracted from Streamlit UI to enable UI-agnostic execution.
Manages result storage and LRU eviction per dataset version.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class CachedResult:
    """Represents a cached analysis result."""

    run_key: str
    query: str
    result: dict[str, Any]  # Serializable analysis result
    timestamp: datetime
    dataset_version: str


class ResultCache:
    """
    Manages result caching with LRU eviction per dataset version.

    Pure Python class with zero Streamlit dependencies.
    Extracted from Ask_Questions.py (remember_run, cleanup_old_results).
    """

    def __init__(self, max_size: int = 50) -> None:
        """
        Initialize result cache.

        Args:
            max_size: Maximum number of results to store per dataset (default: 50)
        """
        self._max_size = max_size
        # Store results: {dataset_version: {run_key: CachedResult}}
        self._results: dict[str, dict[str, CachedResult]] = {}
        # Store history (LRU order): {dataset_version: deque[str]} (run keys)
        self._histories: dict[str, deque[str]] = {}

    def get(self, run_key: str, dataset_version: str) -> CachedResult | None:
        """
        Get cached result for run key and dataset version.

        Args:
            run_key: Run key identifier
            dataset_version: Dataset version identifier

        Returns:
            CachedResult if found, None otherwise
        """
        dataset_results = self._results.get(dataset_version)
        if dataset_results is None:
            return None

        result = dataset_results.get(run_key)
        if result is None:
            return None

        # Update LRU: move accessed key to end (most recent)
        history = self._histories.get(dataset_version)
        if history is not None and run_key in history:
            history.remove(run_key)
            history.append(run_key)

        return result

    def put(self, cached_result: CachedResult) -> None:
        """
        Store result in cache with LRU eviction.

        Auto-evicts oldest result if max_size reached for dataset.

        Args:
            cached_result: Result to cache
        """
        dataset_version = cached_result.dataset_version
        run_key = cached_result.run_key

        # Initialize dataset storage if needed
        if dataset_version not in self._results:
            self._results[dataset_version] = {}
        if dataset_version not in self._histories:
            self._histories[dataset_version] = deque(maxlen=self._max_size)

        history = self._histories[dataset_version]

        # Capture what will be evicted BEFORE any modifications
        evicted_key = None
        if len(history) == self._max_size and run_key not in history:
            evicted_key = history[0]  # Oldest will be evicted

        # De-dupe: move existing key to end (LRU behavior)
        if run_key in history:
            history.remove(run_key)
        history.append(run_key)

        # Store result
        self._results[dataset_version][run_key] = cached_result

        # Delete evicted result immediately (O(1) instead of O(n) scan)
        if evicted_key:
            dataset_results = self._results[dataset_version]
            if evicted_key in dataset_results:
                del dataset_results[evicted_key]

    def evict_oldest(self, dataset_version: str) -> None:
        """
        Evict oldest result for dataset version.

        Args:
            dataset_version: Dataset version identifier
        """
        history = self._histories.get(dataset_version)
        if history is None or len(history) == 0:
            return

        evicted_key = history.popleft()
        dataset_results = self._results.get(dataset_version)
        if dataset_results is not None and evicted_key in dataset_results:
            del dataset_results[evicted_key]

    def clear(self, dataset_version: str | None = None) -> None:
        """
        Clear results for dataset version or all datasets.

        Args:
            dataset_version: Dataset version to clear, or None to clear all
        """
        if dataset_version is None:
            # Clear all datasets
            self._results = {}
            self._histories = {}
        else:
            # Clear specific dataset
            if dataset_version in self._results:
                del self._results[dataset_version]
            if dataset_version in self._histories:
                del self._histories[dataset_version]

    def get_history(self, dataset_version: str) -> list[str]:
        """
        Get run key history in LRU order (oldest to newest).

        Args:
            dataset_version: Dataset version identifier

        Returns:
            List of run keys in LRU order
        """
        history = self._histories.get(dataset_version)
        if history is None:
            return []
        # Return as list (deque not serializable)
        return list(history)

    def serialize(self) -> dict[str, Any]:
        """
        Serialize cache state to dict for persistence.

        Returns:
            Serializable dict representation
        """
        return {
            "results": {
                dataset_version: {
                    run_key: {
                        "run_key": result.run_key,
                        "query": result.query,
                        "result": result.result,
                        "timestamp": result.timestamp.isoformat(),
                        "dataset_version": result.dataset_version,
                    }
                    for run_key, result in dataset_results.items()
                }
                for dataset_version, dataset_results in self._results.items()
            },
            "histories": {dataset_version: list(history) for dataset_version, history in self._histories.items()},
            "max_size": self._max_size,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "ResultCache":
        """
        Deserialize cache state from dict.

        Args:
            data: Serializable dict representation

        Returns:
            ResultCache instance with restored state
        """
        max_size = data.get("max_size", 50)
        cache = cls(max_size=max_size)

        # Restore results
        for dataset_version, dataset_results in data.get("results", {}).items():
            cache._results[dataset_version] = {}
            for run_key, result_data in dataset_results.items():
                cached_result = CachedResult(
                    run_key=result_data["run_key"],
                    query=result_data["query"],
                    result=result_data["result"],
                    timestamp=datetime.fromisoformat(result_data["timestamp"]),
                    dataset_version=result_data["dataset_version"],
                )
                cache._results[dataset_version][run_key] = cached_result

        # Restore histories
        for dataset_version, history_list in data.get("histories", {}).items():
            cache._histories[dataset_version] = deque(history_list, maxlen=max_size)

        return cache
