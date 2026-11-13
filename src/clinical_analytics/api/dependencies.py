# SPDX-License-Identifier: MIT
"""FastAPI dependency injection providers.

Provides reusable dependencies for routes:
- Database session injection
- Semantic layers, leased from a bounded cache (replaces @st.cache_resource)
- Authentication (future)

Reference: docs/architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md
"""

import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import Depends, HTTPException, Path, status

from clinical_analytics.api.layer_cache import LayerCache, LayerCacheError, LayerLease
from clinical_analytics.core import nl_query_config
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.datasets.uploaded.definition import SemanticLayerBuildError, UploadedDatasetFactory
from clinical_analytics.storage.datastore import STORE_BUSY_MESSAGE, StoreBusyError

# ============================================================================
# Semantic Layers (bounded cache, leased per request)
# ============================================================================

# Datasets being deleted: requests that would load their rows get 409 until the delete ends.
_deleting_datasets: set[str] = set()

DATASET_BEING_DELETED = "This dataset is being deleted."
DATASET_CHANGED_WHILE_LOADING = "This dataset changed while it was loading. Try again."


def reject_if_deleting(dataset_id: str) -> None:
    """Raise 409 while ``dataset_id`` is being deleted."""
    if dataset_id in _deleting_datasets:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=DATASET_BEING_DELETED)


def _build_semantic_layer(dataset_id: str) -> SemanticLayer:
    """Load the dataset and build its layer (DuckDB + Ibis). Runs under the dataset's build lock.

    Raises:
        HTTPException: 409 while the dataset is being deleted, 503 when the store is busy, 404 when
            the dataset does not exist, 500 when its query view cannot be built (other failures
            propagate).
    """
    # Checked again under the build lock: the delete guard takes the same lock.
    reject_if_deleting(dataset_id)
    try:
        dataset = UploadedDatasetFactory.create_dataset(dataset_id)
        dataset.load()
        layer: SemanticLayer = dataset.get_semantic_layer()
        return layer
    except StoreBusyError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=STORE_BUSY_MESSAGE) from e
    except SemanticLayerBuildError as e:
        # The dataset exists but its query view could not be built: say so, never "not found".
        # Nothing is cached, so the next request tries again.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset '{dataset_id}' could not be loaded for questions. Upload it again.",
        ) from e
    except (FileNotFoundError, ValueError) as e:
        # UploadedDataset.__init__ raises ValueError when no metadata exists for
        # dataset_id at all; UploadedDataset.load() raises FileNotFoundError when
        # metadata exists but the backing file is missing. Both mean "not found".
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            # No exception text: it carries absolute paths.
            detail=f"Dataset '{dataset_id}' not found",
        ) from e
    # Any other failure is a bug, not a client error: it propagates so FastAPI logs the
    # traceback and returns a generic 500 without leaking paths or internals.


# At most SEMANTIC_LAYER_CACHE_SIZE layers stay open; each gets budget / size of DuckDB memory.
# A layer is used only through a lease, and closed (on a background thread) once it has left
# the cache and its last lease is released. See api/layer_cache.py.
_layer_cache: LayerCache[SemanticLayer] = LayerCache(nl_query_config.SEMANTIC_LAYER_CACHE_SIZE, _build_semantic_layer)


def lease_semantic_layer(dataset_id: str) -> LayerLease[SemanticLayer]:
    """Lease the dataset's layer, building it if needed. Release the lease in ``finally``.

    Blocking (a cold build reads the store): call it off the event loop.

    Raises:
        HTTPException: as _build_semantic_layer; 503 if the dataset kept changing while it loaded.
    """
    reject_if_deleting(dataset_id)
    try:
        return _layer_cache.lease(dataset_id)
    except LayerCacheError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=DATASET_CHANGED_WHILE_LOADING
        ) from e


def leased_semantic_layer(dataset_id: Annotated[str, Path(..., description="Dataset ID")]) -> Iterator[SemanticLayer]:
    """Route dependency: the dataset's layer, held for the request and released after it.

    Example:
        @app.get("/api/datasets/{dataset_id}/info")
        def get_dataset_info(semantic_layer: SemanticLayerDep):
            return semantic_layer.get_dataset_info()
    """
    with lease_semantic_layer(dataset_id) as lease:
        yield lease.layer


@contextmanager
def dataset_deletion_guard(dataset_id: str) -> Iterator[None]:
    """Mark ``dataset_id`` as being deleted and drop its cached layer, atomically w.r.t. builds."""
    with _layer_cache.build_lock(dataset_id):
        if dataset_id in _deleting_datasets:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=DATASET_BEING_DELETED)
        _deleting_datasets.add(dataset_id)
        _layer_cache.invalidate(dataset_id)
    try:
        yield
    finally:
        with _layer_cache.build_lock(dataset_id):
            _layer_cache.invalidate(dataset_id)
            _deleting_datasets.discard(dataset_id)


def invalidate_semantic_layer_cache(dataset_id: str) -> None:
    """Drop the dataset's cached layer: it was deleted, uploaded again or reconfigured.

    Never blocks, so the async upload route may call it on the event loop: a layer in use is
    closed once its last lease is released, an idle one on the cache's background thread. The
    next question builds a new layer, and its AsyncQueryService rebinds to it (keeping results,
    events and history).

    Args:
        dataset_id: Dataset identifier to invalidate
    """
    _layer_cache.invalidate(dataset_id)


# One lock per dataset_id around writes of derived files (contracts, quality reports) and the
# purge that deletes them, so a slow backfill cannot write after a concurrent delete purged.
# Weak values: an entry lives only while a request holds or waits for it, so the map stays small.
_dataset_files_locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_dataset_files_locks_guard = threading.Lock()


@contextmanager
def dataset_files_lock(dataset_id: str) -> Iterator[None]:
    """Hold the per-dataset lock for derived-file writes and the delete purge (blocking)."""
    with _dataset_files_locks_guard:
        lock = _dataset_files_locks.get(dataset_id)
        if lock is None:
            lock = threading.Lock()
            _dataset_files_locks[dataset_id] = lock
    with lock:
        yield


# ============================================================================
# Type Aliases for Route Injection
# ============================================================================

# Semantic layer dependency annotation: leased for the request, released when it ends.
SemanticLayerDep = Annotated[SemanticLayer, Depends(leased_semantic_layer)]
