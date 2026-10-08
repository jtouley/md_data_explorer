"""Query execution API routes with SSE streaming.

Endpoints:
- POST /api/queries - Submit query for async execution
- GET /api/queries/{query_id} - Get query status/result
- GET /api/queries/{query_id}/stream - SSE stream for real-time updates
"""

import json
from typing import Annotated, Any

import polars as pl
import structlog
from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import StreamingResponse

from clinical_analytics.api.dependencies import get_semantic_layer
from clinical_analytics.api.models.schemas import QueryRequest, QueryResponse, QueryResult
from clinical_analytics.api.services.query_service import AsyncQueryService
from clinical_analytics.core.semantic import SemanticLayer

router = APIRouter()
logger = structlog.get_logger()


class DataFrameEncoder(json.JSONEncoder):
    """JSON encoder that handles Polars DataFrames."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, pl.DataFrame):
            return {"columns": obj.columns, "rows": obj.to_dicts()}
        return super().default(obj)


# Cache for dataset query services (keyed by dataset_id). Semantic layer
# caching itself lives in api.dependencies.get_semantic_layer — shared with
# any other route that needs a per-dataset SemanticLayer, rather than each
# route keeping its own divergent cache.
_query_services: dict[str, AsyncQueryService] = {}


def get_query_service_for_dataset(dataset_id: str) -> AsyncQueryService:
    """Get or create AsyncQueryService for a specific dataset.

    Raises:
        HTTPException: 404 if the dataset doesn't exist, 500 on other load failures
            (raised by get_semantic_layer).
    """
    if dataset_id in _query_services:
        return _query_services[dataset_id]

    semantic_layer: SemanticLayer = get_semantic_layer(dataset_id)
    service = AsyncQueryService(semantic_layer)
    _query_services[dataset_id] = service
    logger.info("query_service_created", dataset_id=dataset_id)
    return service


# ============================================================================
# POST /api/queries - Submit Query
# ============================================================================


@router.post("/queries", response_model=QueryResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_query(
    request: QueryRequest,
) -> QueryResponse:
    """Submit a natural language query for async execution.

    Returns immediately with query_id and stream URL for tracking progress.

    Args:
        request: Query request with session_id, dataset_id, query_text

    Returns:
        QueryResponse: Query ID and stream URL

    Raises:
        HTTPException: 404 if dataset not found, 400 if query invalid

    Example:
        POST /api/queries
        {
            "session_id": "sess_abc123",
            "dataset_id": "upload_xyz789",
            "query_text": "What is the average age of patients?"
        }

        Response (202):
        {
            "query_id": "qry_a1b2c3d4",
            "status": "processing",
            "stream_url": "/api/queries/qry_a1b2c3d4/stream"
        }
    """
    logger.info(
        "query_submit_request",
        session_id=request.session_id,
        dataset_id=request.dataset_id,
        query_length=len(request.query_text),
    )

    try:
        query_service = get_query_service_for_dataset(request.dataset_id)

        query_id = await query_service.submit_query(
            query=request.query_text,
            dataset_id=request.dataset_id,
            session_id=request.session_id,
        )

        return QueryResponse(
            query_id=query_id,
            status="processing",
            stream_url=f"/api/queries/{query_id}/stream",
        )

    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            ) from e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        ) from e


# ============================================================================
# GET /api/queries/{query_id} - Get Query Status/Result
# ============================================================================


async def find_query_result(query_id: str) -> tuple[AsyncQueryService, Any] | None:
    """Find a query result across all services.

    Returns tuple of (service, result) if found, None otherwise.
    """
    for service in _query_services.values():
        result = await service.get_result(query_id)
        if result is not None:
            return (service, result)
    return None


@router.get("/queries/{query_id}", response_model=QueryResult)
async def get_query_result(
    query_id: Annotated[str, Path(..., description="Query ID to retrieve")],
) -> QueryResult:
    """Get query status and result.

    Args:
        query_id: Query identifier

    Returns:
        QueryResult: Query status and result data

    Raises:
        HTTPException: 404 if query not found

    Example:
        GET /api/queries/qry_a1b2c3d4

        Response (200):
        {
            "query_id": "qry_a1b2c3d4",
            "status": "completed",
            "intent": "DESCRIBE",
            "result_data": {"mean": 45.5, "std": 12.3},
            "confidence": 0.95
        }
    """
    found = await find_query_result(query_id)

    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' not found",
        )

    _service, result = found

    # Map AsyncQueryResult to QueryResult schema
    return QueryResult(
        query_id=result.query_id,
        intent=result.intent_type or "UNKNOWN",
        status=result.status,  # type: ignore[arg-type]
        confidence=result.confidence,
        result_data=result.result,
        interpretation=result.interpretation,
        follow_up_suggestions=result.follow_ups,
        error=result.error,
        execution_time_ms=None,
    )


# ============================================================================
# GET /api/queries/{query_id}/stream - SSE Stream
# ============================================================================


async def generate_sse_events(
    query_id: str,
    query_service: AsyncQueryService,
) -> Any:
    """Generate SSE events for query progress.

    Yields events in SSE format until query completes.
    Uses anonymous events (no event: line) with event type in data for
    compatibility with Electron contextBridge proxy.
    """
    try:
        async for event in query_service.stream_events(query_id):
            # Include event type in data for onmessage handler
            data_with_event = {"event": event.event, **event.data}
            data_line = f"data: {json.dumps(data_with_event, cls=DataFrameEncoder)}\n\n"
            yield data_line

    except Exception as e:
        logger.error("sse_stream_error", query_id=query_id, error=str(e))
        error_data = {"event": "query_failed", "error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


@router.get("/queries/{query_id}/stream")
async def stream_query_events(
    query_id: Annotated[str, Path(..., description="Query ID to stream")],
) -> StreamingResponse:
    """Stream query progress via Server-Sent Events.

    Args:
        query_id: Query identifier

    Returns:
        StreamingResponse: SSE stream with progress events

    Raises:
        HTTPException: 404 if query not found

    Example:
        GET /api/queries/qry_a1b2c3d4/stream

        SSE Response:
        event: query_started
        data: {"query_id": "qry_a1b2c3d4"}

        event: query_progress
        data: {"stage": "parsing", "message": "Analyzing query..."}

        event: query_completed
        data: {"query_id": "qry_a1b2c3d4", "intent_type": "DESCRIBE"}
    """
    # Check if query exists and find the service
    found = await find_query_result(query_id)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' not found",
        )

    query_service, _result = found
    logger.info("sse_stream_started", query_id=query_id)

    return StreamingResponse(
        generate_sse_events(query_id, query_service),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
