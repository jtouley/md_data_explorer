# SPDX-License-Identifier: MIT
"""Query execution API routes with SSE streaming.

Endpoints:
- POST /api/queries - Submit query for async execution
- GET /api/queries/{query_id} - Get query status/result
- GET /api/queries/{query_id}/stream - SSE stream for real-time updates
"""

import asyncio
import contextvars
import functools
import json
import threading
from typing import Annotated, Any

import polars as pl
import structlog
from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import StreamingResponse

from clinical_analytics.api.dependencies import lease_semantic_layer
from clinical_analytics.api.layer_cache import LayerLease
from clinical_analytics.api.models.schemas import QueryRequest, QueryResponse, QueryResult
from clinical_analytics.api.services.query_service import (
    UNEXPECTED_FAILURE_MESSAGE,
    AsyncQueryService,
)
from clinical_analytics.core.log_redaction import error_fields
from clinical_analytics.core.semantic import SemanticLayer

router = APIRouter()
logger = structlog.get_logger()


class DataFrameEncoder(json.JSONEncoder):
    """JSON encoder that handles Polars DataFrames."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, pl.DataFrame):
            return {"columns": obj.columns, "rows": obj.to_dicts()}
        return super().default(obj)


# One query service per dataset (keyed by dataset_id): its results, SSE events and conversation
# history, and nothing else. It holds no layer: each question runs on the core of the layer it
# leased, which the layer cache drops when it closes the layer. So a service is never dropped
# when its layer leaves the cache, and a poll, a stream or a follow-up still finds its state.
# Only a dataset delete forgets it.
_query_services: dict[str, AsyncQueryService] = {}
_query_services_lock = threading.Lock()


def lease_query_service(dataset_id: str) -> tuple[AsyncQueryService, LayerLease[SemanticLayer]]:
    """The dataset's query service and a lease on the dataset's current layer.

    The caller owns the lease: pass it to submit_query, or release it. Blocking (a cold layer
    build reads the store): call it off the event loop, through ``_lease_off_loop``.

    Raises:
        HTTPException: 404 if the dataset doesn't exist (raised by lease_semantic_layer).
    """
    lease = lease_semantic_layer(dataset_id)
    with _query_services_lock:
        service = _query_services.get(dataset_id)
        if service is None:
            service = _query_services[dataset_id] = AsyncQueryService()
            logger.info("query_service_created", dataset_id=dataset_id)
    return service, lease


def _release_leased(future: "asyncio.Future[tuple[AsyncQueryService, LayerLease[SemanticLayer]]]") -> None:
    """Release the lease a lease_query_service call took for a request that is gone."""
    if future.cancelled() or future.exception() is not None:
        return
    _service, lease = future.result()
    lease.release()


async def _lease_off_loop(dataset_id: str) -> tuple[AsyncQueryService, LayerLease[SemanticLayer]]:
    """lease_query_service in a worker thread. If the request is cancelled while it runs, the
    thread still takes the lease; it is then released from the thread's result, never leaked."""
    # As asyncio.to_thread does: the thread sees this request's context (its bound log fields).
    call = functools.partial(contextvars.copy_context().run, lease_query_service, dataset_id)
    future = asyncio.get_running_loop().run_in_executor(None, call)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        future.add_done_callback(_release_leased)
        raise


def forget_query_service(dataset_id: str) -> bool:
    """Drop a dataset's query service and with it every in-memory result, event and history."""
    with _query_services_lock:
        return _query_services.pop(dataset_id, None) is not None


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
        query_id = await _submit_leased(request)

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


async def _submit_leased(request: QueryRequest) -> str:
    """Lease the dataset's layer and submit the question on it; the question's worker thread
    releases the lease when it is done with the layer."""
    # Building a semantic layer reads the store (and may wait for its build): not on the loop.
    query_service, lease = await _lease_off_loop(request.dataset_id)
    try:
        return await query_service.submit_query(
            query=request.query_text,
            dataset_id=request.dataset_id,
            lease=lease,
            session_id=request.session_id,
            context=request.context,
            confirmed=request.confirmed,
            confirm_plan_line=request.confirm_plan_line,
            confirm_plan_hash=request.confirm_plan_hash,
        )
    except BaseException:
        # submit_query releases a lease it rejects; this covers anything raised before it
        # took it. Release is idempotent, so the lease is never counted twice.
        lease.release()
        raise


# ============================================================================
# GET /api/queries/{query_id} - Get Query Status/Result
# ============================================================================


async def find_query_result(query_id: str) -> tuple[AsyncQueryService, Any] | None:
    """Find a query result across all services.

    Returns tuple of (service, result) if found, None otherwise.
    """
    with _query_services_lock:  # a snapshot: questions add services from worker threads
        services = list(_query_services.values())
    for service in services:
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
        claim=result.claim,
        requires_confirmation=result.requires_confirmation,
        failure_reason=result.failure_reason,
        plan_line=result.plan_line,
        plan_hash=result.plan_hash,
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
        logger.error("sse_stream_error", query_id=query_id, **error_fields(e))
        error_data = {"event": "query_failed", "error": UNEXPECTED_FAILURE_MESSAGE}
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
