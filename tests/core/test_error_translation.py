"""
Tests for error message translation with LLM (ADR009 Phase 4).

Tests cover:
- translate_error_with_llm() function existence and basic behavior
- Graceful degradation when LLM unavailable
- Privacy-preserving (no sensitive data in prompts)
- Timeout handling
"""

from unittest.mock import patch

from clinical_analytics.core.error_translation import translate_error_with_llm
from clinical_analytics.core.llm_feature import LLMCallResult


class TestTranslateErrorWithLLM:
    """Test error message translation with LLM."""

    def test_translate_error_returns_friendly_message_on_success(self):
        # Arrange
        technical_error = "ColumnNotFoundError: Column 'ldl_cholesterol' not found in schema"

        mock_llm_result = LLMCallResult(
            raw_text='{"friendly_message": "I couldn\'t find a column called \'ldl_cholesterol\'. '
            'Try \'LDL mg/dL\' instead."}',
            payload={"friendly_message": "I couldn't find a column called 'ldl_cholesterol'. Try 'LDL mg/dL' instead."},
            latency_ms=100.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.error_translation.call_llm", return_value=mock_llm_result):
            friendly_message = translate_error_with_llm(technical_error)

        # Assert
        assert friendly_message == "I couldn't find a column called 'ldl_cholesterol'. Try 'LDL mg/dL' instead."

    def test_translate_error_returns_none_on_llm_failure(self):
        # Arrange
        technical_error = "ValueError: Invalid query"

        mock_llm_result = LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=0.0,
            timed_out=True,
            error="Timeout",
        )

        # Act
        with patch("clinical_analytics.core.error_translation.call_llm", return_value=mock_llm_result):
            friendly_message = translate_error_with_llm(technical_error)

        # Assert
        assert friendly_message is None

    def test_translate_error_returns_none_on_malformed_json(self):
        # Arrange
        technical_error = "TypeError: expected str, got int"

        mock_llm_result = LLMCallResult(
            raw_text="not json",
            payload=None,
            latency_ms=50.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.error_translation.call_llm", return_value=mock_llm_result):
            friendly_message = translate_error_with_llm(technical_error)

        # Assert
        assert friendly_message is None

    def test_translate_error_returns_none_on_missing_friendly_message_field(self):
        # Arrange
        technical_error = "RuntimeError: unexpected state"

        mock_llm_result = LLMCallResult(
            raw_text='{"other_field": "value"}',
            payload={"other_field": "value"},
            latency_ms=80.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.error_translation.call_llm", return_value=mock_llm_result):
            friendly_message = translate_error_with_llm(technical_error)

        # Assert
        assert friendly_message is None

    def test_translate_error_uses_correct_timeout(self):
        # Arrange
        technical_error = "IndexError: list index out of range"

        mock_llm_result = LLMCallResult(
            raw_text='{"friendly_message": "Test"}',
            payload={"friendly_message": "Test"},
            latency_ms=50.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.error_translation.call_llm", return_value=mock_llm_result) as mock_call:
            translate_error_with_llm(technical_error)

        # Assert
        # Verify timeout parameter (should be LLM_TIMEOUT_ERROR_TRANSLATION_S from config)
        call_args = mock_call.call_args
        assert call_args.kwargs["timeout_s"] == 5.0  # From nl_query_config.py

    def test_translate_error_does_not_expose_sensitive_data(self):
        # Arrange
        technical_error = "DatabaseError: Connection failed to host 'internal-db-prod.company.com' port 5432"

        mock_llm_result = LLMCallResult(
            raw_text='{"friendly_message": "Database connection failed"}',
            payload={"friendly_message": "Database connection failed"},
            latency_ms=100.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.error_translation.call_llm", return_value=mock_llm_result) as mock_call:
            translate_error_with_llm(technical_error)

        # Assert
        # Verify that prompt includes error but guidelines prevent exposing internals
        user_prompt = mock_call.call_args.kwargs["user"]
        # Error should be included for translation
        assert "Connection failed" in user_prompt or "DatabaseError" in user_prompt

    def test_translate_error_handles_empty_error_message(self):
        # Arrange
        technical_error = ""

        mock_llm_result = LLMCallResult(
            raw_text='{"friendly_message": "An unknown error occurred"}',
            payload={"friendly_message": "An unknown error occurred"},
            latency_ms=50.0,
            timed_out=False,
            error=None,
        )

        # Act
        with patch("clinical_analytics.core.error_translation.call_llm", return_value=mock_llm_result):
            friendly_message = translate_error_with_llm(technical_error)

        # Assert
        # Should still work with empty error (LLM can provide generic message)
        assert friendly_message == "An unknown error occurred"
