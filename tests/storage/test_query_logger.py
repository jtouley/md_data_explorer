"""
Tests for QueryLogger (JSONL conversation history).

Tests follow AAA pattern and verify:
- JSONL logging of queries, executions, results
- Streaming-friendly append-only format
- Per-upload logging (scoped by upload_id)
- Log rotation and retrieval
"""

import json

import pytest


@pytest.fixture
def query_logger(tmp_path):
    """Create QueryLogger with temporary log directory."""
    from clinical_analytics.storage.query_logger import QueryLogger

    log_dir = tmp_path / "query_logs"
    return QueryLogger(log_dir)


class TestQueryLoggerBasics:
    """Test basic QueryLogger functionality."""

    def test_query_logger_creates_log_directory(self, tmp_path):
        """QueryLogger should create log directory if it doesn't exist."""
        # Arrange
        from clinical_analytics.storage.query_logger import QueryLogger

        log_dir = tmp_path / "new_logs"
        assert not log_dir.exists()

        # Act
        QueryLogger(log_dir)

        # Assert
        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_log_query_writes_jsonl_entry(self, query_logger, tmp_path):
        """log_query should write a JSONL entry to the log file."""
        # Arrange
        upload_id = "test_upload_001"
        query_text = "How many patients have diabetes?"

        # Act
        query_logger.log_query(
            upload_id=upload_id,
            query_text=query_text,
            timestamp="2025-12-30T10:00:00Z",
        )

        # Assert: JSONL file created
        log_file = tmp_path / "query_logs" / f"{upload_id}_queries.jsonl"
        assert log_file.exists()

        # Read JSONL entry
        with open(log_file) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["event_type"] == "query"
        assert entry["query_text"] == query_text
        assert entry["upload_id"] == upload_id
        assert entry["timestamp"] == "2025-12-30T10:00:00Z"

    def test_log_execution_writes_jsonl_entry(self, query_logger, tmp_path):
        """log_execution should write execution details to JSONL."""
        # Arrange
        upload_id = "test_upload_002"
        query_id = "query_123"

        # Act
        query_logger.log_execution(
            upload_id=upload_id,
            query_id=query_id,
            query_plan={"intent": "count", "filters": []},
            execution_time_ms=150.5,
            timestamp="2025-12-30T10:01:00Z",
        )

        # Assert
        log_file = tmp_path / "query_logs" / f"{upload_id}_queries.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["event_type"] == "execution"
        assert entry["query_id"] == query_id
        assert entry["execution_time_ms"] == 150.5
        assert "query_plan" in entry

    def test_log_result_writes_jsonl_entry(self, query_logger, tmp_path):
        """log_result should write result metadata to JSONL."""
        # Arrange
        upload_id = "test_upload_003"
        query_id = "query_456"

        # Act
        query_logger.log_result(
            upload_id=upload_id,
            query_id=query_id,
            result_type="count",
            result_summary={"total_count": 42},
            timestamp="2025-12-30T10:02:00Z",
        )

        # Assert
        log_file = tmp_path / "query_logs" / f"{upload_id}_queries.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["event_type"] == "result"
        assert entry["query_id"] == query_id
        assert entry["result_type"] == "count"
        assert entry["result_summary"]["total_count"] == 42


class TestQueryLoggerStreaming:
    """Test streaming/append behavior."""

    def test_multiple_queries_append_to_same_file(self, query_logger, tmp_path):
        """Multiple queries should append to JSONL file (streaming-friendly)."""
        # Arrange
        upload_id = "test_upload_004"

        # Act: Log 3 queries
        for i in range(3):
            query_logger.log_query(
                upload_id=upload_id,
                query_text=f"Query {i}",
                timestamp=f"2025-12-30T10:0{i}:00Z",
            )

        # Assert: All 3 entries in file
        log_file = tmp_path / "query_logs" / f"{upload_id}_queries.jsonl"
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["query_text"] == f"Query {i}"

    def test_different_uploads_use_separate_files(self, query_logger, tmp_path):
        """Different uploads should use separate JSONL files."""
        # Arrange & Act
        query_logger.log_query("upload_001", "Query 1", "2025-12-30T10:00:00Z")
        query_logger.log_query("upload_002", "Query 2", "2025-12-30T10:01:00Z")

        # Assert: Separate files created
        log_file_1 = tmp_path / "query_logs" / "upload_001_queries.jsonl"
        log_file_2 = tmp_path / "query_logs" / "upload_002_queries.jsonl"

        assert log_file_1.exists()
        assert log_file_2.exists()


class TestQueryLoggerRetrieval:
    """Test log retrieval and querying."""

    def test_get_query_history_returns_all_entries(self, query_logger):
        """get_query_history should return all entries for an upload."""
        # Arrange: Log multiple events
        upload_id = "test_upload_005"
        query_logger.log_query(upload_id, "Query 1", "2025-12-30T10:00:00Z")
        query_logger.log_execution(upload_id, "q1", {"intent": "count"}, 100, "2025-12-30T10:00:01Z")
        query_logger.log_result(upload_id, "q1", "count", {"total": 10}, "2025-12-30T10:00:02Z")

        # Act
        history = query_logger.get_query_history(upload_id)

        # Assert
        assert len(history) == 3
        assert history[0]["event_type"] == "query"
        assert history[1]["event_type"] == "execution"
        assert history[2]["event_type"] == "result"

    def test_get_query_history_returns_empty_for_nonexistent_upload(self, query_logger):
        """get_query_history should return empty list for non-existent upload."""
        # Act
        history = query_logger.get_query_history("nonexistent_upload")

        # Assert
        assert history == []

    def test_get_latest_queries_returns_recent_entries(self, query_logger):
        """get_latest_queries should return N most recent queries."""
        # Arrange: Log 5 queries
        upload_id = "test_upload_006"
        for i in range(5):
            query_logger.log_query(upload_id, f"Query {i}", f"2025-12-30T10:0{i}:00Z")

        # Act: Get latest 3
        latest = query_logger.get_latest_queries(upload_id, limit=3)

        # Assert: Returns last 3 (newest first)
        assert len(latest) == 3
        assert latest[0]["query_text"] == "Query 4"  # Newest
        assert latest[1]["query_text"] == "Query 3"
        assert latest[2]["query_text"] == "Query 2"
