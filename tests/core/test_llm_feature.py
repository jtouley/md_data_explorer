"""
Tests for LLMFeature enum and unified call_llm() wrapper.

Tests cover:
- LLMFeature enum values
- Unified call_llm() wrapper with consistent logging
- Timeout handling
- Error handling and graceful degradation
- JSON parsing integration
- Latency tracking
"""

from unittest.mock import MagicMock, patch

from clinical_analytics.core.llm_feature import (
    LLMCallResult,
    LLMFeature,
    call_llm,
)


class TestLLMFeature:
    """Test LLMFeature enum."""

    def test_llmfeature_has_all_expected_values(self):
        # Arrange & Act & Assert
        assert LLMFeature.PARSE
        assert LLMFeature.FOLLOWUPS
        assert LLMFeature.INTERPRETATION
        assert LLMFeature.RESULT_INTERPRETATION
        assert LLMFeature.ERROR_TRANSLATION
        assert LLMFeature.FILTER_EXTRACTION
        assert LLMFeature.QUESTION_GENERATION

    def test_llmfeature_validation_values_exist(self):
        """Test that validation layer LLMFeature values exist."""
        # Arrange & Act & Assert
        assert LLMFeature.DBA_VALIDATION
        assert LLMFeature.ANALYST_VALIDATION
        assert LLMFeature.MANAGER_APPROVAL
        assert LLMFeature.VALIDATION_RETRY

    def test_llmfeature_values_are_strings(self):
        # Arrange & Act & Assert
        assert LLMFeature.PARSE.value == "parse"
        assert LLMFeature.FOLLOWUPS.value == "followups"
        assert LLMFeature.INTERPRETATION.value == "interpretation"
        assert LLMFeature.RESULT_INTERPRETATION.value == "result_interpretation"
        assert LLMFeature.ERROR_TRANSLATION.value == "error_translation"
        assert LLMFeature.FILTER_EXTRACTION.value == "filter_extraction"
        assert LLMFeature.QUESTION_GENERATION.value == "question_generation"

    def test_llmfeature_validation_values_are_strings(self):
        """Test that validation layer LLMFeature values are correct strings."""
        # Arrange & Act & Assert
        assert LLMFeature.DBA_VALIDATION.value == "dba_validation"
        assert LLMFeature.ANALYST_VALIDATION.value == "analyst_validation"
        assert LLMFeature.MANAGER_APPROVAL.value == "manager_approval"
        assert LLMFeature.VALIDATION_RETRY.value == "validation_retry"


class TestCallLLM:
    """Test unified call_llm() wrapper."""

    @patch("clinical_analytics.core.llm_feature.OllamaClient")
    @patch("clinical_analytics.core.llm_feature.parse_json_response")
    def test_call_llm_success_returns_result(self, mock_parse, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"intent": "DESCRIBE"}'
        mock_parse.return_value = {"intent": "DESCRIBE"}

        # Act
        result = call_llm(
            feature=LLMFeature.PARSE,
            system="You are a helpful assistant",
            user="Describe age distribution",
            timeout_s=10.0,
        )

        # Assert
        assert result.raw_text == '{"intent": "DESCRIBE"}'
        assert result.payload == {"intent": "DESCRIBE"}
        assert result.timed_out is False
        assert result.error is None
        assert result.latency_ms > 0

    @patch("clinical_analytics.core.llm_feature.OllamaClient")
    def test_call_llm_ollama_unavailable_returns_error(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_available.return_value = False

        # Act
        result = call_llm(
            feature=LLMFeature.FOLLOWUPS,
            system="Generate follow-ups",
            user="Query context",
            timeout_s=15.0,
        )

        # Assert
        assert result.raw_text is None
        assert result.payload is None
        assert result.timed_out is False
        assert result.error == "ollama_unavailable"
        assert result.latency_ms > 0

    @patch("clinical_analytics.core.llm_feature.OllamaClient")
    def test_call_llm_timeout_sets_timed_out_flag(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = None  # Simulates timeout

        # Act
        result = call_llm(
            feature=LLMFeature.INTERPRETATION,
            system="Interpret query",
            user="Query context",
            timeout_s=5.0,
        )

        # Assert
        assert result.raw_text is None
        assert result.payload is None
        assert result.timed_out is True
        assert result.error == "timeout"

    @patch("clinical_analytics.core.llm_feature.OllamaClient")
    @patch("clinical_analytics.core.llm_feature.parse_json_response")
    def test_call_llm_malformed_json_returns_raw_text_only(self, mock_parse, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Not valid JSON"
        mock_parse.return_value = None  # Parse fails

        # Act
        result = call_llm(
            feature=LLMFeature.ERROR_TRANSLATION,
            system="Translate error",
            user="Error context",
            timeout_s=5.0,
        )

        # Assert
        assert result.raw_text == "Not valid JSON"
        assert result.payload is None
        assert result.timed_out is False
        assert result.error == "json_parse_failed"

    @patch("clinical_analytics.core.llm_feature.OllamaClient")
    def test_call_llm_respects_timeout_parameter(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_available.return_value = True

        timeout_s = 20.0

        # Act
        call_llm(
            feature=LLMFeature.RESULT_INTERPRETATION,
            system="Interpret results",
            user="Result context",
            timeout_s=timeout_s,
        )

        # Assert
        # Verify OllamaClient was created with correct timeout
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert call_args[1]["timeout"] == timeout_s

    @patch("clinical_analytics.core.llm_feature.OllamaClient")
    @patch("clinical_analytics.core.llm_feature.parse_json_response")
    def test_call_llm_tracks_latency(self, mock_parse, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"result": "ok"}'
        mock_parse.return_value = {"result": "ok"}

        # Act
        result = call_llm(
            feature=LLMFeature.FILTER_EXTRACTION,
            system="Extract filters",
            user="Query text",
            timeout_s=10.0,
        )

        # Assert
        assert result.latency_ms > 0
        assert isinstance(result.latency_ms, float)

    @patch("clinical_analytics.core.llm_feature.OllamaClient")
    @patch("clinical_analytics.core.llm_feature.parse_json_response")
    def test_call_llm_uses_json_mode(self, mock_parse, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"test": "value"}'
        mock_parse.return_value = {"test": "value"}

        # Act
        call_llm(
            feature=LLMFeature.PARSE,
            system="Parse query",
            user="Test query",
            timeout_s=5.0,
        )

        # Assert
        mock_client.generate.assert_called_once()
        call_args = mock_client.generate.call_args
        assert call_args[1]["json_mode"] is True


class TestLLMCallResult:
    """Test LLMCallResult dataclass."""

    def test_llmcallresult_success_state(self):
        # Arrange & Act
        result = LLMCallResult(
            raw_text='{"key": "value"}',
            payload={"key": "value"},
            latency_ms=123.45,
            timed_out=False,
            error=None,
        )

        # Assert
        assert result.raw_text == '{"key": "value"}'
        assert result.payload == {"key": "value"}
        assert result.latency_ms == 123.45
        assert result.timed_out is False
        assert result.error is None

    def test_llmcallresult_timeout_state(self):
        # Arrange & Act
        result = LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=5000.0,
            timed_out=True,
            error="timeout",
        )

        # Assert
        assert result.raw_text is None
        assert result.payload is None
        assert result.latency_ms == 5000.0
        assert result.timed_out is True
        assert result.error == "timeout"

    def test_llmcallresult_error_state(self):
        # Arrange & Act
        result = LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=50.0,
            timed_out=False,
            error="ollama_unavailable",
        )

        # Assert
        assert result.error == "ollama_unavailable"
        assert result.timed_out is False
