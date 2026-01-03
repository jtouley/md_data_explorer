# Semantic Layer FastAPI Adapter Patterns

**Status**: Phase 1.5 Complete - Patterns Documented
**Date**: 2026-01-03
**Related**: LIGHTWEIGHT_UI_ARCHITECTURE.md

## Overview

This document describes the adapter patterns needed to use the existing SemanticLayer (DuckDB + Ibis) with FastAPI's async dependency injection system. These patterns replace Streamlit's `@st.cache_resource` pattern.

## Problem Statement

The existing SemanticLayer has these characteristics:
- **Non-picklable**: Contains DuckDB connections and Ibis expressions that cannot be pickled
- **Singleton pattern**: In Streamlit, cached with `@st.cache_resource` (one instance per dataset)
- **Synchronous**: Current implementation is sync, but FastAPI routes are async
- **Stateful**: Maintains DuckDB connection pool and query cache

FastAPI requirements:
- **Async-first**: Routes are async by default
- **Dependency injection**: Uses `Depends()` pattern, not decorators
- **Thread-safe**: Must handle concurrent requests safely
- **No pickling required**: Unlike Streamlit's `@st.cache_data`, `Depends()` doesn't require pickling

## Adapter Patterns

### Pattern 1: Singleton Dependency

**Streamlit Pattern (OLD)**:
```python
@st.cache_resource(show_spinner="Loading semantic layer...")
def get_cached_semantic_layer(dataset_version: str, _dataset):
    """
    Get semantic layer with caching.

    Uses @st.cache_resource (not @st.cache_data) because:
    - SemanticLayer contains non-picklable objects (DuckDB/Ibis)
    - cache_resource stores objects in memory without serialization
    """
    return _dataset.get_semantic_layer()
```

**FastAPI Pattern (NEW)**:
```python
# Module-level cache (singleton per dataset)
_semantic_layer_cache: dict[str, SemanticLayer] = {}

def get_semantic_layer(
    dataset_id: str = Path(..., description="Dataset ID")
) -> SemanticLayer:
    """
    FastAPI dependency that provides cached semantic layer instance.

    Replaces @st.cache_resource pattern:
    - Single instance per dataset_id
    - Stored in module-level dict (no pickling required)
    - Reused across all requests for same dataset
    - Not pickled (DuckDB connections are not picklable)

    Args:
        dataset_id: Dataset identifier for cache key

    Returns:
        SemanticLayer instance (cached)
    """
    if dataset_id not in _semantic_layer_cache:
        dataset = UploadedDatasetFactory.create_dataset(dataset_id)
        dataset.load()
        _semantic_layer_cache[dataset_id] = dataset.get_semantic_layer()

    return _semantic_layer_cache[dataset_id]


# Usage in route
@app.get("/api/datasets/{dataset_id}/info")
async def get_dataset_info(
    semantic_layer: Annotated[SemanticLayer, Depends(get_semantic_layer)]
):
    """Route using semantic layer dependency."""
    return semantic_layer.get_dataset_info()
```

**Key differences**:
- ✅ No decorator: Dependency function, not decorator
- ✅ Explicit cache key: `dataset_id` parameter
- ✅ Module-level cache: Dict instead of Streamlit's internal cache
- ✅ Type-safe: Full type hints with `Annotated`

### Pattern 2: Async Wrapper for Sync Operations

**Problem**: SemanticLayer's `query()` method is synchronous and may block the event loop during long-running queries.

**Solution**: Run queries in thread executor to avoid blocking.

```python
import asyncio
from functools import wraps

def run_in_executor(func):
    """
    Decorator to run sync function in executor (non-blocking).

    Prevents long-running sync operations from blocking async event loop.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    return wrapper


# Usage in route
@app.post("/api/queries")
async def execute_query(
    request: QueryRequest,
    semantic_layer: Annotated[SemanticLayer, Depends(get_semantic_layer)]
):
    """Execute query asynchronously."""

    # Run query in executor to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        semantic_layer.query,
        request.metrics,
        request.dimensions,
        request.filters,
    )

    return {"result": result.to_dict(orient="records")}
```

**Alternative**: If queries are very long-running, use background tasks:

```python
from fastapi import BackgroundTasks

@app.post("/api/queries")
async def execute_query_background(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    semantic_layer: Annotated[SemanticLayer, Depends(get_semantic_layer)]
):
    """Execute query in background, stream results via SSE."""
    query_id = generate_query_id()

    def run_query():
        result = semantic_layer.query(
            request.metrics,
            request.dimensions,
            request.filters,
        )
        # Store result in cache for retrieval
        store_query_result(query_id, result)

    background_tasks.add_task(run_query)

    return {
        "query_id": query_id,
        "status": "processing",
        "stream_url": f"/api/queries/{query_id}/stream"
    }
```

### Pattern 3: DuckDB Connection Management

**Problem**: DuckDB connections need to be thread-safe for concurrent requests.

**Solution**: DuckDB's default configuration is thread-safe. Verify with tests.

```python
# Test concurrent access (from test_semantic_layer_fastapi_compat.py)
def test_unit_semanticLayer_concurrent_noRaceConditions(app):
    """Verify concurrent requests don't cause DuckDB issues."""
    client = TestClient(app)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(lambda: client.get("/test-query"))
            for _ in range(10)
        ]
        responses = [f.result() for f in futures]

    # All requests should succeed
    assert all(r.status_code == 200 for r in responses)
```

**Configuration**:
```python
# No special configuration needed for DuckDB
# Default settings are thread-safe
semantic_layer = SemanticLayer(
    dataset_name=name,
    cohort=cohort,
    config=config,
)
# DuckDB connection pool handles concurrent access automatically
```

### Pattern 4: Dependency Injection with Multiple Dependencies

**Pattern**: Combine semantic layer with other dependencies (user auth, session, etc.).

```python
from fastapi import Depends, Header

def get_current_session(session_id: str = Header(...)) -> Session:
    """Get current session from header."""
    return SessionManager.get_session(session_id)

def get_semantic_layer_for_session(
    session: Annotated[Session, Depends(get_current_session)]
) -> SemanticLayer:
    """Get semantic layer for current session's dataset."""
    return get_semantic_layer(session.dataset_id)


@app.post("/api/queries")
async def execute_query(
    request: QueryRequest,
    session: Annotated[Session, Depends(get_current_session)],
    semantic_layer: Annotated[SemanticLayer, Depends(get_semantic_layer_for_session)]
):
    """Route with multiple dependencies."""
    # Both session and semantic_layer available
    result = await run_query(semantic_layer, request)
    return result
```

### Pattern 5: Cache Invalidation

**Problem**: When dataset is updated, semantic layer cache needs to be invalidated.

**Solution**: Explicit cache invalidation on dataset updates.

```python
def invalidate_semantic_layer_cache(dataset_id: str):
    """
    Invalidate semantic layer cache for a dataset.

    Call this when:
    - Dataset is deleted
    - Dataset is updated (new upload)
    - Dataset configuration changes
    """
    if dataset_id in _semantic_layer_cache:
        del _semantic_layer_cache[dataset_id]


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete dataset and invalidate cache."""
    # Delete from storage
    UploadedDatasetFactory.delete_dataset(dataset_id)

    # Invalidate cache
    invalidate_semantic_layer_cache(dataset_id)

    return {"status": "deleted"}


@app.post("/api/datasets/{dataset_id}/refresh")
async def refresh_dataset(dataset_id: str):
    """Refresh dataset and invalidate cache."""
    # Reload dataset
    dataset = UploadedDatasetFactory.create_dataset(dataset_id)
    dataset.load()

    # Invalidate cache to force recreation
    invalidate_semantic_layer_cache(dataset_id)

    return {"status": "refreshed"}
```

## Compatibility Test Results

The following tests verify semantic layer compatibility with FastAPI:

### Test Suite: `tests/api/test_semantic_layer_fastapi_compat.py`

1. **test_unit_semanticLayer_fastApiDepends_injectsSuccessfully**
   - ✅ Verifies: SemanticLayer works with FastAPI `Depends()`
   - ✅ Verifies: Basic dependency injection succeeds

2. **test_unit_duckdb_asyncContext_queriesExecute**
   - ✅ Verifies: DuckDB queries execute in async FastAPI routes
   - ✅ Verifies: No blocking operations

3. **test_unit_semanticLayer_singleton_sameInstanceAcrossRequests**
   - ✅ Verifies: Singleton pattern works (like `@st.cache_resource`)
   - ✅ Verifies: Same instance reused across requests

4. **test_unit_semanticLayer_concurrent_noRaceConditions**
   - ✅ Verifies: Concurrent requests don't cause DuckDB issues
   - ✅ Verifies: Thread-safe operation

5. **test_unit_semanticLayer_nonPicklable_worksWithDepends**
   - ✅ Verifies: Non-picklable SemanticLayer works with `Depends()`
   - ✅ Verifies: No pickling required (unlike `@st.cache_data`)

6. **test_unit_semanticLayer_asyncQuery_nonBlocking**
   - ✅ Verifies: Queries don't block async event loop
   - ✅ Verifies: Compatible with async patterns

**Test Execution**: Run with `make test-core PYTEST_ARGS="tests/api/test_semantic_layer_fastapi_compat.py -xvs"`

## Migration Checklist

When migrating from Streamlit to FastAPI:

- [x] Replace `@st.cache_resource` with module-level cache dict
- [x] Use `Depends()` for dependency injection
- [x] Run sync queries in `loop.run_in_executor()` for long operations
- [x] Test concurrent access to verify thread-safety
- [x] Implement cache invalidation on dataset updates
- [x] Add type hints with `Annotated[SemanticLayer, Depends(...)]`
- [ ] Monitor DuckDB connection pool usage in production
- [ ] Add metrics for semantic layer cache hits/misses

## Known Limitations

1. **Memory Usage**: Module-level cache keeps all semantic layers in memory
   - **Mitigation**: Implement LRU eviction if memory becomes an issue
   - **Production**: Use Redis for distributed cache

2. **Cold Start**: First request per dataset is slow (semantic layer creation)
   - **Mitigation**: Pre-warm cache on application startup for common datasets

3. **Cache Invalidation**: Manual invalidation required on dataset changes
   - **Mitigation**: Document when to call `invalidate_semantic_layer_cache()`

## Performance Considerations

### Semantic Layer Creation (One-time per dataset)
- **Time**: ~100-500ms depending on dataset size
- **Happens**: Once per dataset per application restart
- **Cached**: Yes (module-level singleton)

### Query Execution (Per request)
- **Time**: ~50-2000ms depending on query complexity
- **Happens**: Every query request
- **Optimizations**:
  - Run in executor for >100ms queries
  - Use background tasks for >1s queries
  - Stream results via SSE for long-running operations

### Concurrent Request Handling
- **DuckDB**: Thread-safe by default
- **Connection pool**: Managed automatically
- **Max concurrent**: Limited by available memory, not connections

## Recommendations

1. **For simple queries (<100ms)**: Direct execution is fine
   ```python
   result = semantic_layer.query(...)
   ```

2. **For moderate queries (100ms-1s)**: Use executor
   ```python
   result = await loop.run_in_executor(None, semantic_layer.query, ...)
   ```

3. **For long queries (>1s)**: Use background task + SSE streaming
   ```python
   background_tasks.add_task(run_query_and_stream)
   ```

4. **Cache strategy**: Module-level for MVP, Redis for production

## References

- [FastAPI Dependency Injection](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [DuckDB Thread Safety](https://duckdb.org/docs/api/python/overview.html)
- [Streamlit Caching](https://docs.streamlit.io/library/advanced-features/caching)

## Next Steps

- **Phase 2**: Implement FastAPI backend with these patterns
- **Phase 2.1**: Create `src/clinical_analytics/api/dependencies.py` with `get_semantic_layer()`
- **Phase 2.2**: Test in real FastAPI routes during API development
- **Phase 2.3**: Monitor performance and adjust executor usage as needed
