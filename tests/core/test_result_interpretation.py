"""
Tests for result interpretation with LLM (ADR009 Phase 3).

Tests cover:
- interpret_result_with_llm() function existence and basic behavior
- Graceful degradation when LLM unavailable
- Privacy-preserving (no raw data in prompts)
- Result structure validation
"""

from unittest.mock import patch

from clinical_analytics.core.llm_feature import LLMCallResult
from clinical_analytics.core.result_interpretation import interpret_result_with_llm


class TestInterpretResultWithLLM:
    """Test result interpretation with LLM."""

    def test_interpret_result_returns_interpretation_on_success(self):
        # Arrange
        result = {
            "intent": "DESCRIBE",
            "metric": "age",
            "summary": {"mean": 45.5, "median": 44.0, "std": 12.3},
        }

        mock_llm_result = LLMCallResult(
            raw_text='{"interpretation": "The average age is 45.5 years"}',
            payload={"interpretation": "The average age is 45.5 years"},
            latency_ms=150.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.result_interpretation.call_llm", return_value=mock_llm_result):
            interpretation = interpret_result_with_llm(result)

        # Assert
        assert interpretation == "The average age is 45.5 years"

    def test_interpret_result_returns_none_on_llm_failure(self):
        # Arrange
        result = {
            "intent": "DESCRIBE",
            "metric": "age",
            "summary": {"mean": 45.5},
        }

        mock_llm_result = LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=0.0,
            timed_out=True,
            error="Timeout",
        )

        # Act
        with patch("clinical_analytics.core.result_interpretation.call_llm", return_value=mock_llm_result):
            interpretation = interpret_result_with_llm(result)

        # Assert
        assert interpretation is None

    def test_interpret_result_returns_none_on_malformed_json(self):
        # Arrange
        result = {
            "intent": "DESCRIBE",
            "metric": "age",
            "summary": {"mean": 45.5},
        }

        mock_llm_result = LLMCallResult(
            raw_text="not json",
            payload=None,
            latency_ms=100.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.result_interpretation.call_llm", return_value=mock_llm_result):
            interpretation = interpret_result_with_llm(result)

        # Assert
        assert interpretation is None

    def test_interpret_result_returns_none_on_missing_interpretation_field(self):
        # Arrange
        result = {
            "intent": "DESCRIBE",
            "metric": "age",
            "summary": {"mean": 45.5},
        }

        mock_llm_result = LLMCallResult(
            raw_text='{"other_field": "value"}',
            payload={"other_field": "value"},
            latency_ms=100.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.result_interpretation.call_llm", return_value=mock_llm_result):
            interpretation = interpret_result_with_llm(result)

        # Assert
        assert interpretation is None

    def test_interpret_result_uses_correct_timeout(self):
        # Arrange
        result = {"intent": "DESCRIBE", "metric": "age", "summary": {"mean": 45.5}}

        mock_llm_result = LLMCallResult(
            raw_text='{"interpretation": "Test"}',
            payload={"interpretation": "Test"},
            latency_ms=100.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.result_interpretation.call_llm", return_value=mock_llm_result) as mock_call:
            interpret_result_with_llm(result)

        # Assert
        # Verify timeout parameter was passed (should be LLM_TIMEOUT_RESULT_INTERPRETATION_S from config)
        call_args = mock_call.call_args
        assert call_args.kwargs["timeout_s"] == 20.0  # From nl_query_config.py

    def test_interpret_result_sanitizes_large_result_data(self):
        # Arrange - Create result with large data table
        large_table = [{"col1": i, "col2": i * 2} for i in range(1000)]
        result = {
            "intent": "DESCRIBE",
            "metric": "age",
            "summary": {"mean": 45.5},
            "data_table": large_table,  # Large data should be sanitized
        }

        mock_llm_result = LLMCallResult(
            raw_text='{"interpretation": "Test"}',
            payload={"interpretation": "Test"},
            latency_ms=100.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.result_interpretation.call_llm", return_value=mock_llm_result) as mock_call:
            interpret_result_with_llm(result)

        # Assert
        # Verify that prompt doesn't include raw data table
        user_prompt = mock_call.call_args.kwargs["user"]
        # Should include summary stats but not raw data
        assert "mean" in user_prompt or "45.5" in user_prompt
        # Should NOT include all 1000 rows
        assert len(user_prompt) < 5000  # Reasonable size check
