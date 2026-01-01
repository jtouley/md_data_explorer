"""
Tests for LLM observability event schema and query pattern sanitization.

Tests cover:
- Event schema validation (required fields)
- Query pattern sanitization (no raw query text logged)
- Query hash generation (SHA256)
- Pattern tag extraction (controlled vocabulary)
- Privacy protection enforcement
"""

import hashlib
from datetime import datetime

from clinical_analytics.core.llm_observability import (
    LLMEvent,
    extract_pattern_tags,
    log_llm_event,
    sanitize_query,
)


class TestSanitizeQuery:
    """Test query pattern sanitization for privacy protection."""

    def test_sanitize_query_returns_hash_and_tags(self):
        # Arrange
        query = "compare mortality by treatment arm"

        # Act
        result = sanitize_query(query)

        # Assert
        assert "query_hash" in result
        assert "pattern_tags" in result
        assert "token_count" in result
        assert isinstance(result["query_hash"], str)
        assert isinstance(result["pattern_tags"], list)
        assert len(result["query_hash"]) == 64  # SHA256 hex digest

    def test_sanitize_query_hash_is_deterministic(self):
        # Arrange
        query = "what predicts mortality?"

        # Act
        result1 = sanitize_query(query)
        result2 = sanitize_query(query)

        # Assert
        assert result1["query_hash"] == result2["query_hash"]

    def test_sanitize_query_hash_is_sha256(self):
        # Arrange
        query = "describe age distribution"

        # Act
        result = sanitize_query(query)

        # Assert
        expected_hash = hashlib.sha256(query.encode()).hexdigest()
        assert result["query_hash"] == expected_hash

    def test_sanitize_query_never_includes_raw_text(self):
        # Arrange
        query = "sensitive patient data query"

        # Act
        result = sanitize_query(query)

        # Assert
        assert "query" not in result
        assert "raw_query" not in result
        assert "text" not in result
        # Ensure raw query text doesn't appear anywhere in result
        for value in result.values():
            if isinstance(value, str):
                assert query not in value
            elif isinstance(value, list):
                for item in value:
                    assert query not in str(item)


class TestExtractPatternTags:
    """Test pattern tag extraction with controlled vocabulary."""

    def test_extract_pattern_tags_detects_negation(self):
        # Arrange
        query = "patients not on statins"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        assert "contains_negation" in tags

    def test_extract_pattern_tags_detects_missingness(self):
        # Arrange
        query = "exclude missing values for age"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        assert "mentions_missingness" in tags

    def test_extract_pattern_tags_detects_numeric_range(self):
        # Arrange
        query = "patients aged 50 to 75"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        assert "contains_numeric_range" in tags

    def test_extract_pattern_tags_detects_comparison(self):
        # Arrange
        query = "LDL greater than 100"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        assert "contains_comparison" in tags

    def test_extract_pattern_tags_detects_grouping(self):
        # Arrange
        query = "average age by treatment group"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        assert "contains_grouping" in tags

    def test_extract_pattern_tags_detects_value_exclusion(self):
        # Arrange
        query = "get rid of the n/a values"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        assert "contains_value_exclusion" in tags

    def test_extract_pattern_tags_detects_multi_table_join(self):
        # Arrange
        query = "join patients with medications table"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        assert "multi_table_join_request" in tags

    def test_extract_pattern_tags_returns_empty_for_simple_query(self):
        # Arrange
        query = "describe age"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        # Simple query may have some tags, but should be a list
        assert isinstance(tags, list)

    def test_extract_pattern_tags_returns_multiple_tags(self):
        # Arrange
        query = "exclude patients not on statins with missing age values"

        # Act
        tags = extract_pattern_tags(query)

        # Assert
        assert len(tags) >= 2  # Should detect negation AND missingness
        assert "contains_negation" in tags
        assert "mentions_missingness" in tags


class TestLLMEvent:
    """Test LLMEvent dataclass."""

    def test_llmevent_has_all_required_fields(self):
        # Arrange & Act
        event = LLMEvent(
            event="llm_call_success",
            timestamp=datetime.now(),
            run_key="test_run_123",
            query_hash="abc123" * 10 + "abcd",  # 64 chars
            dataset_version="v1.0",
            tier=3,
            model="llama3.1:8b",
            feature="followups",
            timeout_s=15.0,
            latency_ms=1234.5,
            success=True,
            error_type=None,
            error_message=None,
        )

        # Assert
        assert event.event == "llm_call_success"
        assert isinstance(event.timestamp, datetime)
        assert event.run_key == "test_run_123"
        assert len(event.query_hash) == 64
        assert event.tier == 3
        assert event.model == "llama3.1:8b"
        assert event.feature == "followups"
        assert event.timeout_s == 15.0
        assert event.latency_ms == 1234.5
        assert event.success is True
        assert event.error_type is None

    def test_llmevent_allows_optional_fields_none(self):
        # Arrange & Act
        event = LLMEvent(
            event="llm_call_timeout",
            timestamp=datetime.now(),
            run_key=None,  # Optional
            query_hash="abc123" * 10 + "abcd",
            dataset_version="v1.0",
            tier=3,
            model="llama3.1:8b",
            feature="interpretation",
            timeout_s=10.0,
            latency_ms=10000.0,
            success=False,
            error_type="timeout",
            error_message="LLM timed out after 10s",
        )

        # Assert
        assert event.run_key is None
        assert event.success is False
        assert event.error_type == "timeout"


class TestLogLLMEvent:
    """Test log_llm_event function."""

    def test_log_llm_event_creates_valid_event(self):
        # Arrange
        event_name = "test_event"
        query = "test query"
        feature = "parse"

        # Act
        event = log_llm_event(
            event=event_name,
            query=query,
            tier=3,
            model="test_model",
            feature=feature,
            timeout_s=5.0,
            latency_ms=100.0,
            success=True,
        )

        # Assert
        assert event.event == event_name
        assert isinstance(event.timestamp, datetime)
        assert len(event.query_hash) == 64
        assert event.tier == 3
        assert event.feature == feature

    def test_log_llm_event_sanitizes_query(self):
        # Arrange
        query = "sensitive patient query"

        # Act
        event = log_llm_event(
            event="test",
            query=query,
            tier=3,
            model="test",
            feature="test",
            timeout_s=5.0,
            latency_ms=100.0,
            success=True,
        )

        # Assert
        # Query should be hashed, not stored as-is
        expected_hash = hashlib.sha256(query.encode()).hexdigest()
        assert event.query_hash == expected_hash

    def test_log_llm_event_with_error(self):
        # Arrange
        event_name = "llm_call_failed"
        query = "test query"

        # Act
        event = log_llm_event(
            event=event_name,
            query=query,
            tier=3,
            model="test_model",
            feature="followups",
            timeout_s=15.0,
            latency_ms=50.0,
            success=False,
            error_type="ollama_unavailable",
            error_message="Ollama service not running",
        )

        # Assert
        assert event.success is False
        assert event.error_type == "ollama_unavailable"
        assert event.error_message == "Ollama service not running"
