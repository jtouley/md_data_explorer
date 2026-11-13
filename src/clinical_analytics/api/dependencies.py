# SPDX-License-Identifier: MIT
"""FastAPI dependency injection providers.

Provides reusable dependencies for routes:
- Database session injection
- Semantic layer singleton (replaces @st.cache_resource)
- Authentication (future)

Reference: docs/architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md
"""

import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import Depends, HTTPException, Path, status

from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.storage.datastore import STORE_BUSY_MESSAGE, StoreBusyError

# ============================================================================
# Semantic Layer Dependency (Singleton Pattern)
# ============================================================================

# Module-level cache: one SemanticLayer instance per dataset_id
# Replaces Streamlit's @st.cache_resource pattern
# Non-picklable objects (DuckDB connections, Ibis expressions) stored directly
_semantic_layer_cache: dict[str, SemanticLayer] = {}
# One lock per dataset_id so concurrent cold requests build a single SemanticLayer
# (and a single DuckDB connection) instead of racing and orphaning the loser.
_semantic_layer_locks: dict[str, threading.Lock] = {}
_semantic_layer_locks_guard = threading.Lock()


# Datasets being deleted: requests that would load their rows get 409 until the delete ends.
_deleting_datasets: set[str] = set()

DATASET_BEING_DELETED = "This dataset is being deleted."


def reject_if_deleting(dataset_id: str) -> None:
    """Raise 409 while ``dataset_id`` is being deleted."""
    if dataset_id in _deleting_datasets:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=DATASET_BEING_DELETED)


@contextmanager
def dataset_deletion_guard(dataset_id: str) -> Iterator[None]:
    """Mark ``dataset_id`` as being deleted and drop its cached layer, atomically w.r.t. builds."""
    with _semantic_layer_lock(dataset_id):
        if dataset_id in _deleting_datasets:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=DATASET_BEING_DELETED)
        _deleting_datasets.add(dataset_id)
        _semantic_layer_cache.pop(dataset_id, None)
    try:
        yield
    finally:
        with _semantic_layer_lock(dataset_id):
            _semantic_layer_cache.pop(dataset_id, None)
            _deleting_datasets.discard(dataset_id)


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


def _semantic_layer_lock(dataset_id: str) -> threading.Lock:
    with _semantic_layer_locks_guard:
        return _semantic_layer_locks.setdefault(dataset_id, threading.Lock())


def get_semantic_layer(dataset_id: Annotated[str, Path(..., description="Dataset ID")]) -> SemanticLayer:
    """Get or create cached semantic layer instance for a dataset.

    Replaces Streamlit's @st.cache_resource pattern:
    - Single instance per dataset_id (singleton)
    - Stored in module-level dict (no pickling required)
    - Reused across all requests for same dataset
    - Thread-safe (DuckDB handles concurrent access)

    Args:
        dataset_id: Dataset identifier for cache key

    Returns:
        SemanticLayer: Cached or newly created semantic layer instance

    Raises:
        HTTPException: 404 if dataset not found (other load failures propagate)

    Example:
        @app.get("/api/datasets/{dataset_id}/info")
        async def get_dataset_info(
            semantic_layer: Annotated[SemanticLayer, Depends(get_semantic_layer)]
        ):
            return semantic_layer.get_dataset_info()
    """
    reject_if_deleting(dataset_id)
    cached = _semantic_layer_cache.get(dataset_id)
    if cached is not None:
        return cached

    with _semantic_layer_lock(dataset_id):
        # Checked again under the lock: the delete guard takes the same lock.
        reject_if_deleting(dataset_id)
        # Re-check: another request may have filled the cache while this one waited.
        if dataset_id not in _semantic_layer_cache:
            try:
                # Create dataset instance and load data
                dataset = UploadedDatasetFactory.create_dataset(dataset_id)
                dataset.load()

                # Get semantic layer (DuckDB + Ibis)
                _semantic_layer_cache[dataset_id] = dataset.get_semantic_layer()

            except StoreBusyError as e:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=STORE_BUSY_MESSAGE) from e
            except (FileNotFoundError, ValueError) as e:
                # UploadedDataset.__init__ raises ValueError when no metadata exists for
                # dataset_id at all; UploadedDataset.load() raises FileNotFoundError when
                # metadata exists but the backing file is missing. Both mean "not found".
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    # No exception text: it carries absolute paths.
                    detail=f"Dataset '{dataset_id}' not found",
                ) from e
            # Any other failure is a bug, not a client error: let it propagate so FastAPI logs
            # the traceback and returns a generic 500 without leaking paths or internals.

        return _semantic_layer_cache[dataset_id]


def invalidate_semantic_layer_cache(dataset_id: str) -> None:
    """Invalidate semantic layer cache for a dataset.

    Call this when:
    - Dataset is deleted
    - Dataset is updated (new upload)
    - Dataset configuration changes

    The next get_query_service_for_dataset() sees a different layer and rebuilds its
    AsyncQueryService, so dropping the layer here is enough to refresh queries.

    Args:
        dataset_id: Dataset identifier to invalidate
    """
    with _semantic_layer_lock(dataset_id):
        _semantic_layer_cache.pop(dataset_id, None)


# ============================================================================
# Type Aliases for Route Injection
# ============================================================================

# Semantic layer dependency annotation
SemanticLayerDep = Annotated[SemanticLayer, Depends(get_semantic_layer)]
