"""FastAPI dependency injection providers.

Provides reusable dependencies for routes:
- Database session injection
- Semantic layer singleton (replaces @st.cache_resource)
- Authentication (future)

Reference: docs/architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Path, status

from clinical_analytics.core.semantic_layer import SemanticLayer
from clinical_analytics.datasets.uploaded_dataset_factory import UploadedDatasetFactory

# ============================================================================
# Semantic Layer Dependency (Singleton Pattern)
# ============================================================================

# Module-level cache: one SemanticLayer instance per dataset_id
# Replaces Streamlit's @st.cache_resource pattern
# Non-picklable objects (DuckDB connections, Ibis expressions) stored directly
_semantic_layer_cache: dict[str, SemanticLayer] = {}


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
        HTTPException: 404 if dataset not found

    Example:
        @app.get("/api/datasets/{dataset_id}/info")
        async def get_dataset_info(
            semantic_layer: Annotated[SemanticLayer, Depends(get_semantic_layer)]
        ):
            return semantic_layer.get_dataset_info()
    """
    if dataset_id not in _semantic_layer_cache:
        try:
            # Create dataset instance and load data
            dataset = UploadedDatasetFactory.create_dataset(dataset_id)
            dataset.load()

            # Get semantic layer (DuckDB + Ibis)
            _semantic_layer_cache[dataset_id] = dataset.get_semantic_layer()

        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset '{dataset_id}' not found: {e}",
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load dataset '{dataset_id}': {e}",
            ) from e

    return _semantic_layer_cache[dataset_id]


def invalidate_semantic_layer_cache(dataset_id: str) -> None:
    """Invalidate semantic layer cache for a dataset.

    Call this when:
    - Dataset is deleted
    - Dataset is updated (new upload)
    - Dataset configuration changes

    Args:
        dataset_id: Dataset identifier to invalidate
    """
    if dataset_id in _semantic_layer_cache:
        del _semantic_layer_cache[dataset_id]


# ============================================================================
# Type Aliases for Route Injection
# ============================================================================

# Semantic layer dependency annotation
SemanticLayerDep = Annotated[SemanticLayer, Depends(get_semantic_layer)]
