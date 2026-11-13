# SPDX-License-Identifier: MIT
"""FastAPI dependency injection providers.

Provides reusable dependencies for routes:
- Database session injection
- Semantic layer singleton (replaces @st.cache_resource)
- Authentication (future)

Reference: docs/architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md
"""

import os
import re
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

import duckdb
import structlog
from fastapi import Depends, HTTPException, Path, status

from clinical_analytics.core import nl_query_config
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.datasets.uploaded.definition import SemanticLayerBuildError, UploadedDatasetFactory
from clinical_analytics.storage.datastore import STORE_BUSY_MESSAGE, StoreBusyError

# ============================================================================
# Semantic Layer Dependency (Singleton Pattern)
# ============================================================================

logger = structlog.get_logger(__name__)

# Module-level cache: one SemanticLayer instance per dataset_id, least recently used first.
# Each layer holds its dataset in an in-memory DuckDB, so the cache is bounded
# (nl_query_config.SEMANTIC_LAYER_CACHE_SIZE) and an evicted layer's connection is closed.
# Non-picklable objects (DuckDB connections, Ibis expressions) stored directly
_semantic_layer_cache: OrderedDict[str, SemanticLayer] = OrderedDict()
# Guards reads and writes of the cache's order; the per-dataset locks below guard builds.
_semantic_layer_cache_guard = threading.Lock()
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
        dropped = _pop_layer(dataset_id)
    _close_layers([dropped])
    try:
        yield
    finally:
        with _semantic_layer_lock(dataset_id):
            dropped = _pop_layer(dataset_id)
            _deleting_datasets.discard(dataset_id)
        _close_layers([dropped])


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


_MEMORY_LIMIT = re.compile(r"^\d+(?:\.\d+)?\s*(?:%|[KMGT]i?B)$", re.IGNORECASE)


def _cache_size() -> int:
    size = nl_query_config.SEMANTIC_LAYER_CACHE_SIZE
    if size < 1:
        raise ValueError(f"semantic_layer_cache_size must be at least 1, got {size}")
    return size


def duckdb_settings() -> dict[str, str | int]:
    """The DuckDB ``memory_limit`` and ``threads`` each semantic layer runs with, validated.

    Raises:
        ValueError: a memory limit that is not a size or percentage, or a negative thread count.
    """
    memory_limit = nl_query_config.DUCKDB_MEMORY_LIMIT.strip()
    if not _MEMORY_LIMIT.fullmatch(memory_limit):
        raise ValueError(
            f"duckdb_memory_limit must be a size such as '2GB' or a share such as '20%', got {memory_limit!r}"
        )
    threads = nl_query_config.DUCKDB_THREADS
    if threads < 0:
        raise ValueError(f"duckdb_threads must be 0 (automatic) or more, got {threads}")
    if threads == 0:
        threads = max(1, (os.cpu_count() or 2) // 2)
    return {"memory_limit": memory_limit, "threads": threads}


def _configure_duckdb(layer: SemanticLayer) -> None:
    """Apply the configured memory limit and thread count to the layer's own connection."""
    settings = duckdb_settings()
    raw = layer.con.con
    # Both values are validated above (a size pattern and an int), so formatting is safe.
    raw.execute(f"SET memory_limit = '{settings['memory_limit']}'")
    raw.execute(f"SET threads = {int(settings['threads'])}")


def _pop_layer(dataset_id: str) -> SemanticLayer | None:
    with _semantic_layer_cache_guard:
        return _semantic_layer_cache.pop(dataset_id, None)


def _cached_layer(dataset_id: str) -> SemanticLayer | None:
    """The cached layer, marked most recently used."""
    with _semantic_layer_cache_guard:
        layer = _semantic_layer_cache.get(dataset_id)
        if layer is not None:
            _semantic_layer_cache.move_to_end(dataset_id)
        return layer


def _store_layer(dataset_id: str, layer: SemanticLayer) -> list[SemanticLayer]:
    """Cache ``layer`` as most recently used; return the layers evicted to stay in bounds."""
    evicted: list[SemanticLayer] = []
    with _semantic_layer_cache_guard:
        _semantic_layer_cache[dataset_id] = layer
        _semantic_layer_cache.move_to_end(dataset_id)
        while len(_semantic_layer_cache) > _cache_size():
            old_id, old_layer = _semantic_layer_cache.popitem(last=False)
            logger.info("semantic_layer_evicted", dataset_id=old_id, cache_size=_cache_size())
            evicted.append(old_layer)
    return evicted


def _close_layers(layers: list[SemanticLayer | None]) -> None:
    """Close each dropped layer's DuckDB connection, after any question running on it ends.

    A question holds the layer's connection lock for its whole run, so closing under that lock
    never cuts a running query short. Call without holding a dataset lock.
    """
    for layer in layers:
        if layer is None:
            continue
        lock = getattr(layer, "_con_lock", None)
        try:
            if isinstance(lock, type(threading.Lock())):
                with lock:
                    layer.con.disconnect()
            else:
                layer.con.disconnect()
        except duckdb.Error as error:
            # The layer is already out of the cache; a failed close leaks memory, never an answer.
            logger.warning("semantic_layer_close_failed", error_type=type(error).__name__)


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
    cached = _cached_layer(dataset_id)
    if cached is not None:
        return cached

    evicted: list[SemanticLayer] = []
    with _semantic_layer_lock(dataset_id):
        # Checked again under the lock: the delete guard takes the same lock.
        reject_if_deleting(dataset_id)
        # Re-check: another request may have filled the cache while this one waited.
        layer = _cached_layer(dataset_id)
        if layer is None:
            try:
                # Create dataset instance and load data
                dataset = UploadedDatasetFactory.create_dataset(dataset_id)
                dataset.load()

                # Get semantic layer (DuckDB + Ibis)
                layer = dataset.get_semantic_layer()

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
            # Any other failure is a bug, not a client error: let it propagate so FastAPI logs
            # the traceback and returns a generic 500 without leaking paths or internals.
            # Outside the try: a bad limit in the config is a 500, never "not found".
            _configure_duckdb(layer)
            evicted = _store_layer(dataset_id, layer)
    _close_layers(list(evicted))
    return layer


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
        dropped = _pop_layer(dataset_id)
    _close_layers([dropped])


# ============================================================================
# Type Aliases for Route Injection
# ============================================================================

# Semantic layer dependency annotation
SemanticLayerDep = Annotated[SemanticLayer, Depends(get_semantic_layer)]
