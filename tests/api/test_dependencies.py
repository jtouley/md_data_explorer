"""Tests for FastAPI dependency injection providers (api/dependencies.py).

Covers leasing a dataset's SemanticLayer from the bounded cache: creation, reuse, error mapping
(dataset-not-found -> 404, unexpected failure propagates), no cache poisoning on a failed load,
invalidation, and the route dependency releasing its lease.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from clinical_analytics.api import dependencies

pytestmark = pytest.mark.usefixtures("fresh_layer_cache")


class TestLeaseSemanticLayer:
    """Tests for lease_semantic_layer's caching and error-mapping behavior."""

    def test_unit_firstCall_createsAndCachesInstance(self, fresh_layer_cache):
        # Arrange
        mock_dataset = MagicMock()
        mock_semantic_layer = MagicMock()
        mock_dataset.get_semantic_layer.return_value = mock_semantic_layer

        # Act
        with patch.object(
            dependencies.UploadedDatasetFactory, "create_dataset", return_value=mock_dataset
        ) as mock_create:
            with dependencies.lease_semantic_layer("dataset_1") as lease:
                result = lease.layer

        # Assert
        assert result is mock_semantic_layer
        mock_create.assert_called_once_with("dataset_1")
        mock_dataset.load.assert_called_once()
        assert fresh_layer_cache.cached_layer("dataset_1") is mock_semantic_layer

    def test_unit_secondCall_reusesCache_doesNotReload(self):
        # Arrange
        mock_dataset = MagicMock()
        mock_semantic_layer = MagicMock()
        mock_dataset.get_semantic_layer.return_value = mock_semantic_layer

        # Act
        with patch.object(
            dependencies.UploadedDatasetFactory, "create_dataset", return_value=mock_dataset
        ) as mock_create:
            with dependencies.lease_semantic_layer("dataset_1") as first:
                first_layer = first.layer
            with dependencies.lease_semantic_layer("dataset_1") as second:
                second_layer = second.layer

        # Assert
        assert first_layer is second_layer is mock_semantic_layer
        mock_create.assert_called_once()  # not called again on cache hit

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("Upload data not found: dataset_x"),
            ValueError("Upload dataset_x not found"),
        ],
        ids=["file_not_found", "no_metadata_value_error"],
    )
    def test_unit_datasetNotFound_raises404(self, error, fresh_layer_cache):
        # Act / Assert
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset", side_effect=error):
            with pytest.raises(HTTPException) as exc_info:
                dependencies.lease_semantic_layer("missing_dataset")

        assert exc_info.value.status_code == 404
        assert fresh_layer_cache.cached_layer("missing_dataset") is None

    def test_unit_unexpectedFailure_propagatesUnwrapped(self, fresh_layer_cache):
        """Bugs are not mapped to an HTTPException whose detail would leak internals."""
        # Act / Assert
        with patch.object(
            dependencies.UploadedDatasetFactory,
            "create_dataset",
            side_effect=RuntimeError("unexpected /private/path"),
        ):
            with pytest.raises(RuntimeError, match="unexpected"):
                dependencies.lease_semantic_layer("dataset_1")

        assert fresh_layer_cache.cached_layer("dataset_1") is None

    def test_unit_failedLoad_doesNotPoisonCache(self, fresh_layer_cache):
        """A failed load must not leave a broken/partial entry blocking retry."""
        # Arrange
        mock_dataset = MagicMock()
        mock_dataset.load.side_effect = FileNotFoundError("gone")

        # Act
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset", return_value=mock_dataset):
            with pytest.raises(HTTPException):
                dependencies.lease_semantic_layer("dataset_1")

        # Assert: retry is possible, nothing cached from the failed attempt
        assert fresh_layer_cache.cached_ids() == []

    def test_unit_datasetBeingDeleted_raises409WithoutBuilding(self):
        # Arrange
        with patch.object(dependencies.UploadedDatasetFactory, "create_dataset") as mock_create:
            with dependencies.dataset_deletion_guard("dataset_1"):
                # Act / Assert
                with pytest.raises(HTTPException) as exc_info:
                    dependencies.lease_semantic_layer("dataset_1")

        assert exc_info.value.status_code == 409
        mock_create.assert_not_called()


class TestLeaseSemanticLayerConcurrency:
    def test_unit_concurrentColdRequests_loadOnce(self):
        """Concurrent first requests for one dataset build a single SemanticLayer."""
        # Arrange
        mock_dataset = MagicMock()
        mock_dataset.get_semantic_layer.side_effect = lambda: (time.sleep(0.1), MagicMock())[1]
        results = []

        def request():
            with dependencies.lease_semantic_layer("dataset_1") as lease:
                results.append(lease.layer)

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
        assert len(results) == 4
        assert all(layer is results[0] for layer in results)


class TestLeasedSemanticLayerDependency:
    def test_unit_routeDependency_releasesLeaseWhenRequestEnds(self, seed_layer_cache, fresh_layer_cache):
        # Arrange
        layer = seed_layer_cache("dataset_1", MagicMock())
        dependency = dependencies.leased_semantic_layer("dataset_1")

        # Act
        served = next(dependency)
        dependencies.invalidate_semantic_layer_cache("dataset_1")
        fresh_layer_cache.wait_for_closes()
        closed_during_request = layer.close.called
        dependency.close()
        fresh_layer_cache.wait_for_closes()

        # Assert
        assert served is layer
        assert closed_during_request is False
        layer.close.assert_called_once()


class TestInvalidateSemanticLayerCache:
    """Tests for invalidate_semantic_layer_cache."""

    def test_unit_cachedDataset_removedFromCacheAndClosed(self, seed_layer_cache, fresh_layer_cache):
        # Arrange
        layer = seed_layer_cache("dataset_1", MagicMock())

        # Act
        dependencies.invalidate_semantic_layer_cache("dataset_1")
        fresh_layer_cache.wait_for_closes()

        # Assert
        assert fresh_layer_cache.cached_layer("dataset_1") is None
        layer.close.assert_called_once()

    def test_unit_uncachedDataset_isNoOp(self, fresh_layer_cache):
        # Act / Assert: must not raise for a dataset_id that was never cached
        dependencies.invalidate_semantic_layer_cache("never_cached")
        assert fresh_layer_cache.cached_layer("never_cached") is None
