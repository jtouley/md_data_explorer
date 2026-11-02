"""FastAPI dependency injection providers.

Provides reusable dependencies for routes:
- Database session injection
- Semantic layer singleton (replaces @st.cache_resource)
- Authentication (future)

Reference: docs/architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md
"""

import threading
from typing import Annotated

from fastapi import Depends, HTTPException, Path, status

from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory

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
    cached = _semantic_layer_cache.get(dataset_id)
    if cached is not None:
        return cached

    with _semantic_layer_lock(dataset_id):
        # Re-check: another request may have filled the cache while this one waited.
        if dataset_id not in _semantic_layer_cache:
            try:
                # Create dataset instance and load data
                dataset = UploadedDatasetFactory.create_dataset(dataset_id)
                dataset.load()

                # Get semantic layer (DuckDB + Ibis)
                _semantic_layer_cache[dataset_id] = dataset.get_semantic_layer()

            except (FileNotFoundError, ValueError) as e:
                # UploadedDataset.__init__ raises ValueError when no metadata exists for
                # dataset_id at all; UploadedDataset.load() raises FileNotFoundError when
                # metadata exists but the backing file is missing. Both mean "not found".
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dataset '{dataset_id}' not found: {e}",
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
