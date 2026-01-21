"""Query execution API routes with SSE streaming.

Endpoints:
- POST /api/queries - Submit query for async execution
- GET /api/queries/{query_id} - Get query status/result
- GET /api/queries/{query_id}/stream - SSE stream for real-time updates
"""

import json
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Path, status
from fastapi.responses import StreamingResponse

from clinical_analytics.api.models.schemas import QueryRequest, QueryResponse, QueryResult
from clinical_analytics.api.services.query_service import AsyncQueryService
from clinical_analytics.core.semantic import SemanticLayer

router = APIRouter()
logger = structlog.get_logger()

# Global service instance (lazy initialized)
_query_service: AsyncQueryService | None = None


def get_semantic_layer() -> SemanticLayer:
    """Get or create SemanticLayer instance.

    This is a placeholder - in production, this would be properly configured
    based on the dataset being queried.
    """
    # For now, return a minimal semantic layer with a default dataset name
    # This will be enhanced when integrated with dataset management
    from clinical_analytics.core.semantic import SemanticLayer

    return SemanticLayer(dataset_name="default")


def get_query_service() -> AsyncQueryService:
    """Get or create AsyncQueryService instance."""
    global _query_service
    if _query_service is None:
        semantic_layer = get_semantic_layer()
        _query_service = AsyncQueryService(semantic_layer)
    return _query_service


# ============================================================================
# POST /api/queries - Submit Query
# ============================================================================


@router.post("/queries", response_model=QueryResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_query(
    request: QueryRequest,
    query_service: Annotated[AsyncQueryService, Depends(get_query_service)],
) -> QueryResponse:
    """Submit a natural language query for async execution.

    Returns immediately with query_id and stream URL for tracking progress.

    Args:
        request: Query request with session_id, dataset_id, query_text
        query_service: Query service (injected)

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


@router.get("/queries/{query_id}", response_model=QueryResult)
async def get_query_result(
    query_id: Annotated[str, Path(..., description="Query ID to retrieve")],
    query_service: Annotated[AsyncQueryService, Depends(get_query_service)],
) -> QueryResult:
    """Get query status and result.

    Args:
        query_id: Query identifier
        query_service: Query service (injected)

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
    result = await query_service.get_result(query_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' not found",
        )

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
    """
    try:
        async for event in query_service.stream_events(query_id):
            # Format as SSE
            event_line = f"event: {event.event}\n"
            data_line = f"data: {json.dumps(event.data)}\n"
            yield f"{event_line}{data_line}\n"

    except Exception as e:
        logger.error("sse_stream_error", query_id=query_id, error=str(e))
        error_event = f"event: query_failed\ndata: {json.dumps({'error': str(e)})}\n\n"
        yield error_event


@router.get("/queries/{query_id}/stream")
async def stream_query_events(
    query_id: Annotated[str, Path(..., description="Query ID to stream")],
    query_service: Annotated[AsyncQueryService, Depends(get_query_service)],
) -> StreamingResponse:
    """Stream query progress via Server-Sent Events.

    Args:
        query_id: Query identifier
        query_service: Query service (injected)

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
    # Check if query exists
    result = await query_service.get_result(query_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' not found",
        )

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
