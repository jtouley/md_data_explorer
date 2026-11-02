"""
AsyncQueryService - Async wrapper around core QueryService for API usage.

Provides async interface with SSE streaming support for real-time query progress.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from clinical_analytics.api.models.schemas import SSEEvent
from clinical_analytics.core.query_service import QueryResult as CoreQueryResult
from clinical_analytics.core.query_service import QueryService as CoreQueryService
from clinical_analytics.core.result_presentation import present_analysis, to_json_safe
from clinical_analytics.core.semantic import SemanticLayer

logger = structlog.get_logger()

# Turns kept per session for refinement parsing ("now by site", "remove the n/a").
MAX_CONVERSATION_TURNS = 20


def context_to_turn(context: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Map a client-supplied QueryRequest.context to one conversation_history entry.

    Accepts history-shaped keys (query, intent, group_by, metric, filters_applied) or the
    schema's previous_* keys; previous_variables is read as [metric, group_by].
    """
    if not context:
        return None
    variables = context.get("previous_variables") or []
    turn = {
        "query": context.get("query") or context.get("previous_query"),
        "intent": context.get("intent") or context.get("previous_intent"),
        "metric": context.get("metric")
        or context.get("previous_metric")
        or (variables[0] if len(variables) > 0 else None),
        "group_by": context.get("group_by")
        or context.get("previous_group_by")
        or (variables[1] if len(variables) > 1 else None),
        "filters_applied": context.get("filters_applied") or context.get("previous_filters") or [],
    }
    if not any(turn[k] for k in ("query", "intent", "metric", "group_by")):
        return None
    return turn


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
        self._history: dict[str, list[dict[str, Any]]] = {}

    def conversation_history(
        self, session_id: str | None, context: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Prior turns for a session; falls back to the client's context when there are none."""
        history = list(self._history.get(session_id, [])) if session_id else []
        if history:
            return history
        turn = context_to_turn(context)
        return [turn] if turn else []

    def _record_turn(self, session_id: str | None, query: str, result: CoreQueryResult) -> None:
        """Remember a completed turn so the session's next question can refine it."""
        if not session_id or result.plan is None:
            return
        plan = result.plan
        turns = self._history.setdefault(session_id, [])
        turns.append(
            {
                "query": query,
                "intent": plan.intent,
                "metric": plan.metric,
                "group_by": plan.group_by,
                "filters_applied": [asdict(f) for f in plan.filters],
            }
        )
        del turns[:-MAX_CONVERSATION_TURNS]

    @property
    def semantic_layer(self) -> SemanticLayer:
        """Semantic layer this service executes against."""
        return self._semantic_layer

    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        return f"qry_{uuid.uuid4().hex[:12]}"

    async def submit_query(
        self,
        query: str,
        dataset_id: str,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
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
        asyncio.create_task(self._execute_query_background(query_id, query, dataset_id, session_id, context))

        return query_id

    async def _execute_query_background(
        self,
        query_id: str,
        query: str,
        dataset_id: str,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
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
            # session_id identifies the conversation, not the upload: it selects the history
            # passed to the parser, while dataset_id stays the data key.
            result: CoreQueryResult = await asyncio.to_thread(
                self._core.ask,
                question=query,
                dataset_id=dataset_id,
                conversation_history=self.conversation_history(session_id, context),
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
            payload = self._analysis_payload(result)
            self._queries[query_id].result = payload
            self._record_turn(session_id, query, result)
            self._queries[query_id].confidence = result.confidence
            self._queries[query_id].completed_at = datetime.now(UTC)

            # Emit completed event
            self._emit_event(
                query_id,
                "query_completed",
                {
                    "query_id": query_id,
                    "intent_type": intent_type,
                    "result_preview": self._preview_of(payload),
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

    def _analysis_payload(self, result: CoreQueryResult) -> dict[str, Any] | None:
        """Claim, statistics table and full analysis for a completed query.

        Falls back to the raw execution rows only when no analysis was produced.
        """
        if result.analysis is None:
            return self._serialize_result(result.result)
        presentation = present_analysis(result.analysis)
        table = presentation["table"]
        return {
            "analysis_type": result.analysis.get("type"),
            "claim": presentation["claim"],
            "table": table,
            "row_count": len(table["rows"]) if table else 0,
            "chart_spec": to_json_safe(result.analysis.get("chart_spec")),
            "analysis": to_json_safe(result.analysis),
        }

    def _preview_of(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """SSE preview: the claim-first payload without the full analysis dict."""
        if payload is None or "claim" not in payload:
            return self._get_result_preview(payload)
        return {k: v for k, v in payload.items() if k != "analysis"}

    def _table_payload(self, frame: Any) -> dict[str, Any] | None:
        """JSON table for a Polars or pandas frame. Ibis execute() returns pandas."""
        import pandas as pd
        import polars as pl

        if isinstance(frame, pl.DataFrame):
            rows = frame.to_dicts()
            columns = list(frame.columns)
        elif isinstance(frame, pd.DataFrame):
            rows = frame.to_dict(orient="records")
            columns = [str(column) for column in frame.columns]
        else:
            return None
        return {
            "table": {"columns": columns, "rows": rows},
            "row_count": len(rows),
        }

    def _serialize_result(self, result: Any | None) -> dict[str, Any] | None:
        """Convert result to JSON-serializable dict, handling DataFrames recursively."""
        if result is None:
            return None

        import polars as pl

        table = self._table_payload(result)
        if table is not None:
            return table

        # Handle dict with potential DataFrame values (recursive)
        if isinstance(result, dict):
            serialized: dict[str, Any] = {}
            for k, v in result.items():
                nested = self._table_payload(v)
                if nested is not None:
                    serialized[k] = nested
                elif isinstance(v, dict):
                    serialized[k] = self._serialize_result(v)
                elif isinstance(v, list):
                    serialized[k] = [
                        self._serialize_result(item) if isinstance(item, dict | pl.DataFrame) else item for item in v
                    ]
                elif isinstance(v, str | int | float | bool | type(None)):
                    serialized[k] = v
                else:
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
            conversation_history=self.conversation_history(session_id),
        )

        # Check for errors
        if result.issues:
            errors = [i["message"] for i in result.issues if i.get("severity") == "error"]
            if errors:
                # Check if it's a "dataset not found" type error
                error_msg = "; ".join(errors)
                if "not found" in error_msg.lower():
                    raise ValueError(error_msg)

        if not any(i.get("severity") == "error" for i in result.issues):
            self._record_turn(session_id, query, result)

        # Determine intent type
        intent_type = None
        if result.context and result.context.inferred_intent:
            intent_type = result.context.inferred_intent.value

        return AsyncQueryResult(
            query_id=query_id,
            status="completed",
            intent_type=intent_type,
            result=self._analysis_payload(result),
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
        Sends a stream_end event to signal client should close connection.

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
                # Send explicit stream_end signal for client to close connection
                yield SSEEvent(
                    event="stream_end",
                    data={"query_id": query_id, "status": query_result.status},
                    timestamp=datetime.now(UTC),
                )
                break

            # Wait for more events
            await asyncio.sleep(0.05)
