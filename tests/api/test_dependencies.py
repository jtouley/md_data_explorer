"""Tests for FastAPI dependency injection providers (api/dependencies.py).

Covers the per-dataset SemanticLayer cache: creation, reuse, error mapping
(dataset-not-found -> 404, unexpected failure -> 500), cache-poisoning on
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

    def test_unit_unexpectedFailure_raises500(self):
        # Act / Assert
        with patch.object(
            dependencies.UploadedDatasetFactory,
            "create_dataset",
            side_effect=RuntimeError("unexpected"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                dependencies.get_semantic_layer("dataset_1")

        assert exc_info.value.status_code == 500

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
