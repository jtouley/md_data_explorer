"""Tests for FastAPI dependency injection providers (api/dependencies.py).

Covers the per-dataset SemanticLayer cache: creation, reuse, error mapping
(dataset-not-found -> 404, unexpected failure propagates), cache-poisoning on
failed load, and invalidation.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from clinical_analytics.api import dependencies


@pytest.fixture(autouse=True)
def _clear_semantic_layer_cache():
    """Isolate the module-level cache between tests."""
    dependencies._semantic_layer_cache.clear()
    yield
    dependencies._semantic_layer_cache.clear()


class TestGetSemanticLayer:
    """Tests for get_semantic_layer's caching and error-mapping behavior."""

    def test_unit_firstCall_createsAndCachesInstance(self):
        # Arrange
        mock_dataset = MagicMock()
        mock_semantic_layer = MagicMock()
        mock_dataset.get_semantic_layer.return_value = mock_semantic_layer

        # Act
        with patch.object(
            dependencies.UploadedDatasetFactory, "create_dataset", return_value=mock_dataset
        ) as mock_create:
            result = dependencies.get_semantic_layer("dataset_1")

        # Assert
        assert result is mock_semantic_layer
        mock_create.assert_called_once_with("dataset_1")
        mock_dataset.load.assert_called_once()
        assert dependencies._semantic_layer_cache["dataset_1"] is mock_semantic_layer

    def test_unit_secondCall_reusesCache_doesNotReload(self):
        # Arrange
        mock_dataset = MagicMock()
        mock_semantic_layer = MagicMock()
        mock_dataset.get_semantic_layer.return_value = mock_semantic_layer

        # Act
        with patch.object(
            dependencies.UploadedDatasetFactory, "create_dataset", return_value=mock_dataset
        ) as mock_create:
            first = dependencies.get_semantic_layer("dataset_1")
            second = dependencies.get_semantic_layer("dataset_1")

        # Assert
        assert first is second is mock_semantic_layer
        mock_create.assert_called_once()  # not called again on cache hit

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("Upload data not found: dataset_x"),
            ValueError("Upload dataset_x not found"),
        ],
        ids=["file_not_found", "no_metadata_value_error"],
    )
    def test_unit_datasetNotFound_raises404(self, error):
        # Act / Assert
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset", side_effect=error):
            with pytest.raises(HTTPException) as exc_info:
                dependencies.get_semantic_layer("missing_dataset")

        assert exc_info.value.status_code == 404
        assert "missing_dataset" not in dependencies._semantic_layer_cache

    def test_unit_unexpectedFailure_propagatesUnwrapped(self):
        """Bugs are not mapped to an HTTPException whose detail would leak internals."""
        # Act / Assert
        with patch.object(
            dependencies.UploadedDatasetFactory,
            "create_dataset",
            side_effect=RuntimeError("unexpected /private/path"),
        ):
            with pytest.raises(RuntimeError, match="unexpected"):
                dependencies.get_semantic_layer("dataset_1")

        assert "dataset_1" not in dependencies._semantic_layer_cache

    def test_unit_failedLoad_doesNotPoisonCache(self):
        """A failed load must not leave a broken/partial entry blocking retry."""
        # Arrange
        mock_dataset = MagicMock()
        mock_dataset.load.side_effect = FileNotFoundError("gone")

        # Act
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset", return_value=mock_dataset):
            with pytest.raises(HTTPException):
                dependencies.get_semantic_layer("dataset_1")

        # Assert: retry is possible, nothing cached from the failed attempt
        assert "dataset_1" not in dependencies._semantic_layer_cache


class TestGetSemanticLayerConcurrency:
    def test_unit_concurrentColdRequests_loadOnce(self):
        """Concurrent first requests for one dataset build a single SemanticLayer."""
        # Arrange
        import threading
        import time

        mock_dataset = MagicMock()
        mock_dataset.get_semantic_layer.side_effect = lambda: (time.sleep(0.1), MagicMock())[1]
        results = []

        def request():
            results.append(dependencies.get_semantic_layer("dataset_1"))

        threads = [threading.Thread(target=request) for _ in range(4)]

        # Act
        with patch.object(
            dependencies.UploadedDatasetFactory, "create_dataset", return_value=mock_dataset
        ) as mock_create:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        # Assert
        mock_create.assert_called_once_with("dataset_1")
        assert all(layer is results[0] for layer in results)


class TestInvalidateSemanticLayerCache:
    """Tests for invalidate_semantic_layer_cache."""

    def test_unit_cachedDataset_removedFromCache(self):
        # Arrange
        dependencies._semantic_layer_cache["dataset_1"] = MagicMock()

        # Act
        dependencies.invalidate_semantic_layer_cache("dataset_1")

        # Assert
        assert "dataset_1" not in dependencies._semantic_layer_cache

    def test_unit_uncachedDataset_isNoOp(self):
        # Act / Assert: must not raise for a dataset_id that was never cached
        dependencies.invalidate_semantic_layer_cache("never_cached")
        assert "never_cached" not in dependencies._semantic_layer_cache


def _layer_factory(layers: dict[str, MagicMock]):
    """create_dataset stand-in whose dataset builds (and records) one mock layer per id."""

    def create(dataset_id: str) -> MagicMock:
        dataset = MagicMock()
        layer = MagicMock(name=f"layer_{dataset_id}")
        layers[dataset_id] = layer
        dataset.get_semantic_layer.return_value = layer
        return dataset

    return create


class TestSemanticLayerCacheBound:
    """The cache keeps at most SEMANTIC_LAYER_CACHE_SIZE layers and closes the ones it drops."""

    def test_unit_cacheFull_leastRecentlyUsedEvictedAndClosed(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(dependencies.nl_query_config, "SEMANTIC_LAYER_CACHE_SIZE", 2)
        layers: dict[str, MagicMock] = {}
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset", side_effect=_layer_factory(layers)):
            dependencies.get_semantic_layer("a")
            dependencies.get_semantic_layer("b")
            dependencies.get_semantic_layer("a")  # a is now the most recently used

            # Act
            dependencies.get_semantic_layer("c")

        # Assert
        assert list(dependencies._semantic_layer_cache) == ["a", "c"]
        layers["b"].con.disconnect.assert_called_once()
        layers["a"].con.disconnect.assert_not_called()
        layers["c"].con.disconnect.assert_not_called()

    def test_unit_invalidate_closesDroppedLayer(self):
        # Arrange
        layer = MagicMock()
        dependencies._semantic_layer_cache["dataset_1"] = layer

        # Act
        dependencies.invalidate_semantic_layer_cache("dataset_1")

        # Assert
        layer.con.disconnect.assert_called_once()

    def test_unit_newLayer_duckdbLimitsApplied(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(dependencies.nl_query_config, "DUCKDB_MEMORY_LIMIT", "1GB")
        monkeypatch.setattr(dependencies.nl_query_config, "DUCKDB_THREADS", 2)
        layers: dict[str, MagicMock] = {}

        # Act
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset", side_effect=_layer_factory(layers)):
            dependencies.get_semantic_layer("a")

        # Assert
        statements = [call.args[0] for call in layers["a"].con.con.execute.call_args_list]
        assert statements == ["SET memory_limit = '1GB'", "SET threads = 2"]

    @pytest.mark.parametrize(
        ("memory_limit", "threads", "message"),
        [
            ("lots", 2, "duckdb_memory_limit"),
            ("1GB'; DROP TABLE x; --", 2, "duckdb_memory_limit"),
            ("", 2, "duckdb_memory_limit"),
            ("1GB", -1, "duckdb_threads"),
        ],
    )
    def test_unit_badDuckdbSetting_raisesValueErrorNotNotFound(self, monkeypatch, memory_limit, threads, message):
        # Arrange
        monkeypatch.setattr(dependencies.nl_query_config, "DUCKDB_MEMORY_LIMIT", memory_limit)
        monkeypatch.setattr(dependencies.nl_query_config, "DUCKDB_THREADS", threads)

        # Act / Assert
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset", side_effect=_layer_factory({})):
            with pytest.raises(ValueError, match=message):
                dependencies.get_semantic_layer("a")
        assert "a" not in dependencies._semantic_layer_cache

    @pytest.mark.parametrize("size", [0, -3])
    def test_unit_cacheSizeBelowOne_raisesValueError(self, monkeypatch, size):
        # Arrange
        monkeypatch.setattr(dependencies.nl_query_config, "SEMANTIC_LAYER_CACHE_SIZE", size)

        # Act / Assert
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset", side_effect=_layer_factory({})):
            with pytest.raises(ValueError, match="semantic_layer_cache_size"):
                dependencies.get_semantic_layer("a")

    @pytest.mark.parametrize(("memory_limit", "threads"), [("20%", 0), ("2GB", 3), ("512MiB", 1), ("1.5 GB", 8)])
    def test_unit_duckdbSettings_realConnection_accepted(self, monkeypatch, memory_limit, threads):
        # Arrange
        import ibis

        monkeypatch.setattr(dependencies.nl_query_config, "DUCKDB_MEMORY_LIMIT", memory_limit)
        monkeypatch.setattr(dependencies.nl_query_config, "DUCKDB_THREADS", threads)
        layer = MagicMock()
        layer.con = ibis.duckdb.connect()

        # Act
        dependencies._configure_duckdb(layer)

        # Assert
        applied = layer.con.con.execute("SELECT current_setting('threads')").fetchone()[0]
        assert applied == dependencies.duckdb_settings()["threads"] >= 1
        layer.con.disconnect()
