"""Query execution API routes with SSE streaming.

Endpoints:
- POST /api/queries - Submit query for async execution
- GET /api/queries/{query_id} - Get query status/result
- GET /api/queries/{query_id}/stream - SSE stream for real-time updates
"""

import json
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import StreamingResponse

from clinical_analytics.api.models.schemas import QueryRequest, QueryResponse, QueryResult
from clinical_analytics.api.services.query_service import AsyncQueryService
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory

router = APIRouter()
logger = structlog.get_logger()

# Cache for dataset semantic layers (keyed by dataset_id)
_semantic_layers: dict[str, SemanticLayer] = {}
_query_services: dict[str, AsyncQueryService] = {}


def get_semantic_layer_for_dataset(dataset_id: str) -> SemanticLayer:
    """Get or create SemanticLayer for a specific dataset.

    Uses the same pattern as the Streamlit UI:
    1. UploadedDatasetFactory.create_dataset(upload_id)
    2. dataset.load()
    3. dataset.get_semantic_layer()

    Args:
        dataset_id: Dataset identifier (upload_id for uploaded datasets)

    Returns:
        SemanticLayer configured for the dataset

    Raises:
        ValueError: If dataset not found
    """
    if dataset_id in _semantic_layers:
        logger.debug("semantic_layer_cache_hit", dataset_id=dataset_id)
        return _semantic_layers[dataset_id]

    logger.info("semantic_layer_loading", dataset_id=dataset_id)

    try:
        dataset = UploadedDatasetFactory.create_dataset(dataset_id)
        dataset.load()
        semantic_layer: SemanticLayer = dataset.get_semantic_layer()

        _semantic_layers[dataset_id] = semantic_layer
        logger.info("semantic_layer_loaded", dataset_id=dataset_id, dataset_name=dataset.name)
        return semantic_layer

    except ValueError as e:
        logger.error("dataset_not_found", dataset_id=dataset_id, error=str(e))
        raise
    except FileNotFoundError as e:
        logger.error("dataset_file_not_found", dataset_id=dataset_id, error=str(e))
        raise ValueError(f"Dataset file not found: {e}") from e
    except Exception as e:
        logger.error("semantic_layer_init_failed", dataset_id=dataset_id, error=str(e))
        raise ValueError(f"Failed to load dataset '{dataset_id}': {e}") from e


def get_query_service_for_dataset(dataset_id: str) -> AsyncQueryService:
    """Get or create AsyncQueryService for a specific dataset."""
    if dataset_id in _query_services:
        return _query_services[dataset_id]

    semantic_layer = get_semantic_layer_for_dataset(dataset_id)
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
