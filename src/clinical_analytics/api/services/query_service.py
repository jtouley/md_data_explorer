"""
AsyncQueryService - Async wrapper around core QueryService for API usage.

Provides async interface with SSE streaming support for real-time query progress.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from clinical_analytics.api.models.schemas import SSEEvent
from clinical_analytics.core.query_service import QueryResult as CoreQueryResult
from clinical_analytics.core.query_service import QueryService as CoreQueryService
from clinical_analytics.core.semantic import SemanticLayer

logger = structlog.get_logger()


@dataclass
class AsyncQueryResult:
    """Result of async query execution."""

    query_id: str
    status: str  # "pending", "processing", "completed", "failed"
    intent_type: str | None = None
    result: dict[str, Any] | None = None
    interpretation: str | None = None
    follow_ups: list[str] = field(default_factory=list)
    error: str | None = None
    confidence: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class AsyncQueryService:
    """
    Async wrapper around core QueryService for API usage.

    Supports async execution with SSE streaming of progress events.
    """

    def __init__(self, semantic_layer: SemanticLayer) -> None:
        """
        Initialize async query service with semantic layer.

        Args:
            semantic_layer: SemanticLayer instance for query parsing and execution
        """
        self._semantic_layer = semantic_layer
        self._core = CoreQueryService(semantic_layer)
        self._queries: dict[str, AsyncQueryResult] = {}
        self._events: dict[str, list[SSEEvent]] = {}

    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        return f"qry_{uuid.uuid4().hex[:12]}"

    async def submit_query(
        self,
        query: str,
        dataset_id: str,
        session_id: str | None = None,
    ) -> str:
        """
        Submit query for async execution, returns query_id immediately.

        Args:
            query: Natural language query text
            dataset_id: Dataset to query
            session_id: Optional session ID for conversation context

        Returns:
            Unique query ID for tracking

        Raises:
            ValueError: If query is empty
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query_id = self._generate_query_id()

        logger.info("query_submitted", query_id=query_id, dataset_id=dataset_id)

        # Initialize query state
        self._queries[query_id] = AsyncQueryResult(
            query_id=query_id,
            status="pending",
        )
        self._events[query_id] = []

        # Start query execution in background
        asyncio.create_task(self._execute_query_background(query_id, query, dataset_id, session_id))

        return query_id

    async def _execute_query_background(
        self,
        query_id: str,
        query: str,
        dataset_id: str,
        session_id: str | None = None,
    ) -> None:
        """Execute query in background and emit SSE events."""
        try:
            # Emit started event
            self._emit_event(
                query_id,
                "query_started",
                {"query_id": query_id, "dataset_id": dataset_id},
            )

            # Update status
            self._queries[query_id].status = "processing"

            # Emit progress event
            self._emit_event(
                query_id,
                "query_progress",
                {"stage": "parsing", "message": "Analyzing query..."},
            )

            # Execute query synchronously in thread pool
            result: CoreQueryResult = await asyncio.to_thread(
                self._core.ask,
                question=query,
                dataset_id=dataset_id,
                upload_id=session_id,
            )

            # Emit progress event
            self._emit_event(
                query_id,
                "query_progress",
                {"stage": "executing", "message": "Running analysis..."},
            )

            # Determine intent type from context
            intent_type = None
            if result.context and result.context.inferred_intent:
                intent_type = result.context.inferred_intent.value

            # Check for errors
            if result.issues:
                errors = [i["message"] for i in result.issues if i.get("severity") == "error"]
                if errors:
                    self._queries[query_id].status = "failed"
                    self._queries[query_id].error = "; ".join(errors)
                    self._emit_event(
                        query_id,
                        "query_failed",
                        {"error": "; ".join(errors), "details": result.issues},
                    )
                    logger.warning("query_failed", query_id=query_id, error=errors[0])
                    return

            # Success - update query result (serialize DataFrames)
            self._queries[query_id].status = "completed"
            self._queries[query_id].intent_type = intent_type
            self._queries[query_id].result = self._serialize_result(result.result)
            self._queries[query_id].confidence = result.confidence
            self._queries[query_id].completed_at = datetime.now(UTC)

            # Emit completed event
            self._emit_event(
                query_id,
                "query_completed",
                {
                    "query_id": query_id,
                    "intent_type": intent_type,
                    "result_preview": self._get_result_preview(result.result),
                },
            )

            logger.info(
                "query_completed",
                query_id=query_id,
                intent_type=intent_type,
            )

        except Exception as e:
            self._queries[query_id].status = "failed"
            self._queries[query_id].error = str(e)
            self._emit_event(
                query_id,
                "query_failed",
                {"error": str(e), "details": None},
            )
            logger.warning("query_failed", query_id=query_id, error=str(e))

    def _emit_event(self, query_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Emit SSE event for query."""
        event = SSEEvent(
            event=event_type,  # type: ignore[arg-type]
            data=data,
            timestamp=datetime.now(UTC),
        )
        self._events.setdefault(query_id, []).append(event)

    def _serialize_result(self, result: Any | None) -> dict[str, Any] | None:
        """Convert result to JSON-serializable dict, handling DataFrames recursively."""
        if result is None:
            return None

        import polars as pl

        # Handle Polars DataFrame directly
        if isinstance(result, pl.DataFrame):
            return {
                "table": {
                    "columns": result.columns,
                    "rows": result.to_dicts(),
                },
                "row_count": len(result),
            }

        # Handle dict with potential DataFrame values (recursive)
        if isinstance(result, dict):
            serialized: dict[str, Any] = {}
            for k, v in result.items():
                if isinstance(v, pl.DataFrame):
                    serialized[k] = {
                        "table": {
                            "columns": v.columns,
                            "rows": v.to_dicts(),
                        },
                        "row_count": len(v),
                    }
                elif isinstance(v, dict):
                    # Recursively serialize nested dicts
                    serialized[k] = self._serialize_result(v)
                elif isinstance(v, list):
                    # Handle lists (may contain dicts with DataFrames)
                    serialized[k] = [
                        self._serialize_result(item) if isinstance(item, dict | pl.DataFrame) else item for item in v
                    ]
                elif isinstance(v, str | int | float | bool | type(None)):
                    serialized[k] = v
                else:
                    # Convert other types to string
                    serialized[k] = str(v)
            return serialized

        # Fallback for other types
        return {"value": str(result)}

    def _get_result_preview(self, result: Any | None) -> dict[str, Any] | None:
        """Get preview of result for SSE event, converting DataFrames to dicts.

        Uses _serialize_result to ensure all DataFrames are converted, then limits
        to first 3 keys and 5 rows per table for preview.
        """
        if result is None:
            return None

        # First fully serialize (handles nested DataFrames)
        serialized = self._serialize_result(result)
        if serialized is None:
            return None

        # Check if this is a direct table result (from _serialize_result of a DataFrame)
        if "table" in serialized and "rows" in serialized.get("table", {}):
            return {
                "table": {
                    "columns": serialized["table"]["columns"],
                    "rows": serialized["table"]["rows"][:5],
                },
                "row_count": serialized.get("row_count", len(serialized["table"]["rows"])),
            }

        # Limit to first 3 keys and truncate row counts for preview
        preview: dict[str, Any] = {}
        for i, (k, v) in enumerate(serialized.items()):
            if i >= 3:
                break
            # Truncate tables to 5 rows for preview
            if isinstance(v, dict) and "table" in v and "rows" in v.get("table", {}):
                preview[k] = {
                    "table": {
                        "columns": v["table"]["columns"],
                        "rows": v["table"]["rows"][:5],
                    },
                    "row_count": v.get("row_count", len(v["table"]["rows"])),
                }
            else:
                preview[k] = v
        return preview

    async def execute_query(
        self,
        query: str,
        dataset_id: str,
        session_id: str | None = None,
    ) -> AsyncQueryResult:
        """
        Execute query synchronously (blocking) and return result.

        For use when SSE streaming is not needed.

        Args:
            query: Natural language query text
            dataset_id: Dataset to query
            session_id: Optional session ID for conversation context

        Returns:
            AsyncQueryResult with query results

        Raises:
            ValueError: If query is empty or dataset not found
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query_id = self._generate_query_id()

        logger.info("query_submitted", query_id=query_id, dataset_id=dataset_id)

        # Execute synchronously in thread pool
        result: CoreQueryResult = await asyncio.to_thread(
            self._core.ask,
            question=query,
            dataset_id=dataset_id,
            upload_id=session_id,
        )

        # Check for errors
        if result.issues:
            errors = [i["message"] for i in result.issues if i.get("severity") == "error"]
            if errors:
                # Check if it's a "dataset not found" type error
                error_msg = "; ".join(errors)
                if "not found" in error_msg.lower():
                    raise ValueError(error_msg)

        # Determine intent type
        intent_type = None
        if result.context and result.context.inferred_intent:
            intent_type = result.context.inferred_intent.value

        return AsyncQueryResult(
            query_id=query_id,
            status="completed",
            intent_type=intent_type,
            result=self._serialize_result(result.result),
            confidence=result.confidence,
            completed_at=datetime.now(UTC),
        )

    async def get_result(self, query_id: str) -> AsyncQueryResult | None:
        """
        Get query result by ID.

        Args:
            query_id: Query identifier

        Returns:
            AsyncQueryResult if found, None otherwise
        """
        return self._queries.get(query_id)

    async def stream_events(self, query_id: str) -> AsyncIterator[SSEEvent]:
        """
        Stream SSE events for query progress.

        Yields events as they become available, completing when query finishes.

        Args:
            query_id: Query identifier to stream

        Yields:
            SSEEvent for each progress update
        """
        event_index = 0

        while True:
            # Yield any new events
            events = self._events.get(query_id, [])
            while event_index < len(events):
                yield events[event_index]
                event_index += 1

            # Check if query is complete
            query_result = self._queries.get(query_id)
            if query_result and query_result.status in ("completed", "failed"):
                break

            # Wait for more events
            await asyncio.sleep(0.05)
