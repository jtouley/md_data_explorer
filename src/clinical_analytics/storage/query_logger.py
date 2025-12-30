"""
QueryLogger - JSONL Conversation History

Logs query parsing, execution, and results to JSONL files for audit trail.
JSONL format enables streaming-friendly, append-only logging.

Architecture:
- One JSONL file per upload_id: {upload_id}_queries.jsonl
- Append-only (never rewrite entire file)
- Three event types: query, execution, result
- Scoped by upload_id (different datasets have separate logs)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class QueryLogger:
    """
    Logs query events to JSONL files for audit trail and debugging.

    Each upload gets its own JSONL file: {upload_id}_queries.jsonl
    Events are appended in order (streaming-friendly).

    Event Types:
    - query: User submitted a natural language query
    - execution: Query was parsed and executed
    - result: Results were computed and returned

    Example Usage:
        >>> logger = QueryLogger(Path("data/query_logs"))
        >>> logger.log_query("upload_123", "How many patients?", "2025-12-30T10:00:00Z")
        >>> logger.log_execution("upload_123", "q1", {"intent": "count"}, 150.5, "2025-12-30T10:00:01Z")
        >>> logger.log_result("upload_123", "q1", "count", {"total": 42}, "2025-12-30T10:00:02Z")
    """

    def __init__(self, log_dir: Path | str):
        """
        Initialize QueryLogger with log directory.

        Args:
            log_dir: Directory to store JSONL log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"QueryLogger initialized with log directory: {self.log_dir}")

    def log_query(
        self,
        upload_id: str,
        query_text: str,
        timestamp: str | None = None,
        query_id: str | None = None,
    ) -> None:
        """
        Log a user query (natural language input).

        Args:
            upload_id: Upload identifier (scopes log file)
            query_text: Natural language query from user
            timestamp: ISO 8601 timestamp (auto-generated if not provided)
            query_id: Optional query identifier for correlation
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        entry = {
            "event_type": "query",
            "upload_id": upload_id,
            "query_text": query_text,
            "timestamp": timestamp,
        }

        if query_id:
            entry["query_id"] = query_id

        self._append_entry(upload_id, entry)

    def log_execution(
        self,
        upload_id: str,
        query_id: str,
        query_plan: dict[str, Any],
        execution_time_ms: float,
        timestamp: str | None = None,
    ) -> None:
        """
        Log query execution details (parsing + execution).

        Args:
            upload_id: Upload identifier
            query_id: Query identifier (correlates with log_query)
            query_plan: Parsed query plan (intent, variables, filters, etc.)
            execution_time_ms: Execution time in milliseconds
            timestamp: ISO 8601 timestamp (auto-generated if not provided)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        entry = {
            "event_type": "execution",
            "upload_id": upload_id,
            "query_id": query_id,
            "query_plan": query_plan,
            "execution_time_ms": execution_time_ms,
            "timestamp": timestamp,
        }

        self._append_entry(upload_id, entry)

    def log_result(
        self,
        upload_id: str,
        query_id: str,
        result_type: str,
        result_summary: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """
        Log query result metadata.

        Args:
            upload_id: Upload identifier
            query_id: Query identifier (correlates with log_query/log_execution)
            result_type: Type of result (count, descriptive, comparison, etc.)
            result_summary: Summary of results (counts, statistics, etc.)
            timestamp: ISO 8601 timestamp (auto-generated if not provided)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        entry = {
            "event_type": "result",
            "upload_id": upload_id,
            "query_id": query_id,
            "result_type": result_type,
            "result_summary": result_summary,
            "timestamp": timestamp,
        }

        self._append_entry(upload_id, entry)

    def get_query_history(self, upload_id: str) -> list[dict[str, Any]]:
        """
        Retrieve all query events for an upload.

        Args:
            upload_id: Upload identifier

        Returns:
            List of event dicts in chronological order

        Example:
            >>> history = logger.get_query_history("upload_123")
            >>> for event in history:
            ...     print(event["event_type"], event["timestamp"])
        """
        log_file = self._get_log_file(upload_id)

        if not log_file.exists():
            return []

        entries = []
        with open(log_file) as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    entries.append(json.loads(line))

        return entries

    def get_latest_queries(self, upload_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve N most recent query events.

        Args:
            upload_id: Upload identifier
            limit: Maximum number of queries to return

        Returns:
            List of query event dicts (newest first)

        Example:
            >>> latest = logger.get_latest_queries("upload_123", limit=5)
            >>> print(f"Last 5 queries: {[e['query_text'] for e in latest]}")
        """
        history = self.get_query_history(upload_id)

        # Filter to only "query" events
        queries = [e for e in history if e["event_type"] == "query"]

        # Return latest N (reversed for newest first)
        return list(reversed(queries[-limit:]))

    def _append_entry(self, upload_id: str, entry: dict[str, Any]) -> None:
        """
        Append a JSONL entry to the log file.

        Args:
            upload_id: Upload identifier (determines log file)
            entry: Event dict to append
        """
        log_file = self._get_log_file(upload_id)

        # Append to JSONL file (one JSON object per line)
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.debug(f"Logged {entry['event_type']} event to {log_file.name}")

    def _get_log_file(self, upload_id: str) -> Path:
        """
        Get log file path for an upload.

        Args:
            upload_id: Upload identifier

        Returns:
            Path to JSONL log file
        """
        return self.log_dir / f"{upload_id}_queries.jsonl"
