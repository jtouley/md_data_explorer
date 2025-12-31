"""
Tests for enhanced query logging (Phase 7.1).

Ensures:
- Comprehensive query context logging (matched vars, confidence, execution path)
- Execution details logging (run_key, QueryPlan JSON, warnings)
- Failure logging with suggested fixes
"""


import pytest

from clinical_analytics.storage.query_logger import QueryLogger


class TestEnhancedQueryContextLogging:
    """Test comprehensive query context logging."""

    @pytest.fixture
    def query_logger(self, tmp_path):
        """Create QueryLogger with temp directory."""
        return QueryLogger(tmp_path / "query_logs")

    def test_log_query_with_full_context(self, query_logger):
        """log_query should support comprehensive context (matched vars, confidence, execution path)."""
        # Arrange: Query with full context
        upload_id = "test_upload_123"
        query_text = "how many patients with LDL > 100?"
        context = {
            "matched_variables": ["LDL mg/dL"],
            "confidence": 0.85,
            "execution_path": "pattern_match",
            "aliasing": {"LDL": "LDL mg/dL"},
        }

        # Act: Log query with context
        query_logger.log_query(
            upload_id=upload_id,
            query_text=query_text,
            query_id="q1",
            context=context,
        )

        # Assert: Context should be logged
        history = query_logger.get_query_history(upload_id)
        assert len(history) == 1

        entry = history[0]
        assert entry["event_type"] == "query"
        assert entry["query_text"] == query_text
        assert "context" in entry
        assert entry["context"]["matched_variables"] == ["LDL mg/dL"]
        assert entry["context"]["confidence"] == 0.85
        assert entry["context"]["execution_path"] == "pattern_match"
        assert entry["context"]["aliasing"] == {"LDL": "LDL mg/dL"}

    def test_log_execution_with_warnings(self, query_logger):
        """log_execution should support warnings list."""
        # Arrange: Execution with warnings
        upload_id = "test_upload_123"
        query_plan = {
            "intent": "COUNT",
            "metric": None,
            "group_by": "status",
            "filters": [],
        }
        warnings = [
            "Low confidence (0.65 < 0.8)",
            "Missing required field: entity_key",
        ]

        # Act: Log execution with warnings
        query_logger.log_execution(
            upload_id=upload_id,
            query_id="q1",
            query_plan=query_plan,
            execution_time_ms=150.5,
            warnings=warnings,
            run_key="run_abc123",
        )

        # Assert: Warnings and run_key should be logged
        history = query_logger.get_query_history(upload_id)
        assert len(history) == 1

        entry = history[0]
        assert entry["event_type"] == "execution"
        assert "warnings" in entry
        assert len(entry["warnings"]) == 2
        assert "Low confidence" in entry["warnings"][0]
        assert "run_key" in entry
        assert entry["run_key"] == "run_abc123"

    def test_log_result_with_row_count(self, query_logger):
        """log_result should include result row count."""
        # Arrange: Result with row count
        upload_id = "test_upload_123"
        result_summary = {
            "total": 42,
            "row_count": 5,  # Number of rows in result DataFrame
        }

        # Act: Log result
        query_logger.log_result(
            upload_id=upload_id,
            query_id="q1",
            result_type="count",
            result_summary=result_summary,
        )

        # Assert: Row count should be logged
        history = query_logger.get_query_history(upload_id)
        assert len(history) == 1

        entry = history[0]
        assert entry["event_type"] == "result"
        assert entry["result_summary"]["row_count"] == 5


class TestFailureLogging:
    """Test failure logging with suggested fixes."""

    @pytest.fixture
    def query_logger(self, tmp_path):
        """Create QueryLogger with temp directory."""
        return QueryLogger(tmp_path / "query_logs")

    def test_log_failure_with_details(self, query_logger):
        """QueryLogger should support logging failures with parsing tier and suggested fixes."""
        # Arrange: Failed query
        upload_id = "test_upload_123"
        query_text = "show me BMI for diabetics"
        failure_details = {
            "parsing_tier": "pattern_match",
            "failure_reason": "Column 'BMI' not found in dataset",
            "suggested_fixes": [
                "Check if column is named differently (e.g., 'BMI kg/m2', 'body_mass_index')",
                "Verify column exists in uploaded dataset",
            ],
        }

        # Act: Log failure (using log_query with failure context)
        query_logger.log_query(
            upload_id=upload_id,
            query_text=query_text,
            query_id="q1",
            context={"failure": failure_details},
        )

        # Assert: Failure details should be logged
        history = query_logger.get_query_history(upload_id)
        assert len(history) == 1

        entry = history[0]
        assert entry["event_type"] == "query"
        assert "context" in entry
        assert "failure" in entry["context"]
        assert entry["context"]["failure"]["parsing_tier"] == "pattern_match"
        assert "Column 'BMI' not found" in entry["context"]["failure"]["failure_reason"]
        assert len(entry["context"]["failure"]["suggested_fixes"]) == 2

    def test_log_failure_using_dedicated_method(self, query_logger):
        """QueryLogger should have dedicated log_failure() method."""
        # Arrange: Failed query
        upload_id = "test_upload_123"
        query_text = "average cholesterol by gender"
        failure_reason = "Column 'cholesterol' not found in dataset"
        suggested_fixes = [
            "Use 'LDL mg/dL' or 'HDL mg/dL' instead",
            "Check available columns in dataset summary",
        ]

        # Act: Log failure using dedicated method
        query_logger.log_failure(
            upload_id=upload_id,
            query_text=query_text,
            query_id="q2",
            failure_reason=failure_reason,
            parsing_tier="llm_fallback",
            suggested_fixes=suggested_fixes,
        )

        # Assert: Failure should be logged
        history = query_logger.get_query_history(upload_id)
        assert len(history) == 1

        entry = history[0]
        assert entry["event_type"] == "failure"
        assert entry["query_text"] == query_text
        assert entry["failure_reason"] == failure_reason
        assert entry["parsing_tier"] == "llm_fallback"
        assert len(entry["suggested_fixes"]) == 2


class TestBackwardCompatibility:
    """Test backward compatibility with existing log_query/log_execution methods."""

    @pytest.fixture
    def query_logger(self, tmp_path):
        """Create QueryLogger with temp directory."""
        return QueryLogger(tmp_path / "query_logs")

    def test_log_query_without_context_still_works(self, query_logger):
        """log_query should work without context parameter (backward compatible)."""
        # Arrange: Simple query (no context)
        upload_id = "test_upload_123"
        query_text = "how many patients?"

        # Act: Log query without context
        query_logger.log_query(
            upload_id=upload_id,
            query_text=query_text,
            query_id="q1",
        )

        # Assert: Query should be logged successfully
        history = query_logger.get_query_history(upload_id)
        assert len(history) == 1
        assert history[0]["query_text"] == query_text

    def test_log_execution_without_warnings_still_works(self, query_logger):
        """log_execution should work without warnings parameter (backward compatible)."""
        # Arrange: Simple execution (no warnings)
        upload_id = "test_upload_123"
        query_plan = {"intent": "COUNT"}

        # Act: Log execution without warnings
        query_logger.log_execution(
            upload_id=upload_id,
            query_id="q1",
            query_plan=query_plan,
            execution_time_ms=100.0,
        )

        # Assert: Execution should be logged successfully
        history = query_logger.get_query_history(upload_id)
        assert len(history) == 1
        assert history[0]["query_plan"]["intent"] == "COUNT"
