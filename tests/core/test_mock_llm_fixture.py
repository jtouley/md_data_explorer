"""Tests for mock_llm_calls fixture."""


def test_mock_llm_calls_fixture_mocks_parse_feature(mock_llm_calls):
    """Test that mock_llm_calls fixture mocks PARSE feature correctly."""
    # Arrange - fixture is applied via parameter
    # Import after fixture is applied so patch works
    from clinical_analytics.core.llm_feature import LLMCallResult, LLMFeature, call_llm

    # Act
    result = call_llm(
        feature=LLMFeature.PARSE,
        system="Parse query",
        user="count patients",
        timeout_s=10.0,
    )

    # Assert
    assert isinstance(result, LLMCallResult)
    assert result.raw_text == '{"intent": "DESCRIBE", "confidence": 0.8}'
    assert result.payload == {"intent": "DESCRIBE", "confidence": 0.8}
    assert result.latency_ms == 10.0
    assert result.timed_out is False
    assert result.error is None


def test_mock_llm_calls_fixture_mocks_filter_extraction_feature(mock_llm_calls):
    """Test that mock_llm_calls fixture mocks FILTER_EXTRACTION feature correctly."""
    # Arrange - fixture is applied via parameter
    from clinical_analytics.core.llm_feature import LLMCallResult, LLMFeature, call_llm

    # Act
    result = call_llm(
        feature=LLMFeature.FILTER_EXTRACTION,
        system="Extract filters",
        user="age > 50",
        timeout_s=10.0,
    )

    # Assert
    assert isinstance(result, LLMCallResult)
    assert result.raw_text == '{"filters": []}'
    assert result.payload == {"filters": []}
    assert result.latency_ms == 10.0
    assert result.timed_out is False
    assert result.error is None


def test_mock_llm_calls_fixture_mocks_followups_feature(mock_llm_calls):
    """Test that mock_llm_calls fixture mocks FOLLOWUPS feature correctly."""
    # Arrange - fixture is applied via parameter
    from clinical_analytics.core.llm_feature import LLMCallResult, LLMFeature, call_llm

    # Act
    result = call_llm(
        feature=LLMFeature.FOLLOWUPS,
        system="Generate follow-ups",
        user="Query context",
        timeout_s=10.0,
    )

    # Assert
    assert isinstance(result, LLMCallResult)
    assert result.raw_text == '{"follow_ups": []}'
    assert result.payload == {"follow_ups": []}
    assert result.latency_ms == 10.0
    assert result.timed_out is False
    assert result.error is None


def test_mock_llm_calls_fixture_mocks_result_interpretation_feature(mock_llm_calls):
    """Test that mock_llm_calls fixture mocks RESULT_INTERPRETATION feature correctly."""
    # Arrange - fixture is applied via parameter
    from clinical_analytics.core.llm_feature import LLMCallResult, LLMFeature, call_llm

    # Act
    result = call_llm(
        feature=LLMFeature.RESULT_INTERPRETATION,
        system="Interpret results",
        user="Results data",
        timeout_s=10.0,
    )

    # Assert
    assert isinstance(result, LLMCallResult)
    assert result.raw_text == '{"interpretation": "Test interpretation"}'
    assert result.payload == {"interpretation": "Test interpretation"}
    assert result.latency_ms == 10.0
    assert result.timed_out is False
    assert result.error is None


def test_mock_llm_calls_fixture_mocks_error_translation_feature(mock_llm_calls):
    """Test that mock_llm_calls fixture mocks ERROR_TRANSLATION feature correctly."""
    # Arrange - fixture is applied via parameter
    from clinical_analytics.core.llm_feature import LLMCallResult, LLMFeature, call_llm

    # Act
    result = call_llm(
        feature=LLMFeature.ERROR_TRANSLATION,
        system="Translate error",
        user="Error message",
        timeout_s=10.0,
    )

    # Assert
    assert isinstance(result, LLMCallResult)
    assert result.raw_text == '{"translation": "Test error translation"}'
    assert result.payload == {"translation": "Test error translation"}
    assert result.latency_ms == 10.0
    assert result.timed_out is False
    assert result.error is None


def test_mock_llm_calls_fixture_returns_fast(mock_llm_calls):
    """Test that mocked LLM calls return quickly (<1s)."""
    # Arrange
    import time

    from clinical_analytics.core.llm_feature import LLMFeature, call_llm

    # Act
    start = time.perf_counter()
    result = call_llm(
        feature=LLMFeature.PARSE,
        system="Test",
        user="Test",
        timeout_s=10.0,
    )
    duration = time.perf_counter() - start

    # Assert
    assert duration < 0.1, f"Mocked call should be fast, took {duration:.3f}s"
    assert result is not None
