"""
LLMFeature enum and unified call_llm() wrapper.

This module provides a single choke point for all LLM calls, ensuring:
- Consistent logging and observability
- Standardized timeout handling
- Uniform error handling
- Automatic JSON parsing
- Latency tracking

All LLM features must use call_llm() instead of calling OllamaClient directly.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from clinical_analytics.core.llm_client import OllamaClient
from clinical_analytics.core.llm_json import parse_json_response
from clinical_analytics.core.nl_query_config import OLLAMA_DEFAULT_MODEL

logger = structlog.get_logger()


class LLMFeature(Enum):
    """
    Enumeration of all LLM features in the system.

    Each feature represents a distinct use case for LLM invocation:
    - PARSE: Query parsing (Tier 3 fallback)
    - FOLLOWUPS: Context-aware follow-up question generation
    - INTERPRETATION: Query interpretation and confidence explanation
    - RESULT_INTERPRETATION: Clinical insights from results
    - ERROR_TRANSLATION: User-friendly error messages
    - FILTER_EXTRACTION: Complex filter pattern extraction
    - QUESTION_GENERATION: Proactive question generation (upload-time and query-time)
    - DBA_VALIDATION: LLM-based DBA validation (type safety, schema checks)
    - VALIDATION_RETRY: LLM-based retry with DBA feedback
    """

    PARSE = "parse"
    FOLLOWUPS = "followups"
    INTERPRETATION = "interpretation"
    RESULT_INTERPRETATION = "result_interpretation"
    ERROR_TRANSLATION = "error_translation"
    FILTER_EXTRACTION = "filter_extraction"
    QUESTION_GENERATION = "question_generation"
    DBA_VALIDATION = "dba_validation"
    VALIDATION_RETRY = "validation_retry"


@dataclass
class LLMCallResult:
    """
    Result of a unified LLM call.

    Attributes:
        raw_text: Raw text response from LLM (None if unavailable/timeout)
        payload: Parsed JSON payload (None if parsing failed or N/A)
        latency_ms: Time taken for LLM call in milliseconds
        timed_out: Whether the call timed out
        error: Error type if call failed (None on success)
    """

    raw_text: str | None
    payload: dict[str, Any] | list[Any] | None
    latency_ms: float
    timed_out: bool
    error: str | None


def call_llm(
    feature: LLMFeature,
    system: str,
    user: str,
    timeout_s: float,
    model: str | None = None,
) -> LLMCallResult:
    """
    Unified LLM call wrapper with consistent logging and error handling.

    This is the single entry point for all LLM calls in the system. It provides:
    - Automatic timeout handling
    - JSON parsing integration
    - Latency tracking
    - Standardized error handling
    - Consistent observability logging

    Args:
        feature: LLMFeature indicating what this call is for
        system: System prompt
        user: User prompt
        timeout_s: Timeout in seconds (must be <= LLM_TIMEOUT_MAX_S)
        model: Optional model override (defaults to OLLAMA_DEFAULT_MODEL)

    Returns:
        LLMCallResult with raw_text, parsed payload, latency, timeout/error flags

    Examples:
        >>> result = call_llm(
        ...     feature=LLMFeature.FOLLOWUPS,
        ...     system="Generate follow-up questions",
        ...     user="Query: compare mortality by treatment",
        ...     timeout_s=15.0,
        ... )
        >>> if result.payload:
        ...     follow_ups = result.payload.get("follow_ups", [])
    """
    start_time = time.perf_counter()

    # Use default model if not specified
    if model is None:
        model = OLLAMA_DEFAULT_MODEL

    # Create client with specified timeout
    client = OllamaClient(model=model, timeout=timeout_s)

    # Check availability
    if not client.is_available():
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "llm_call_unavailable",
            feature=feature.value,
            model=model,
            timeout_s=timeout_s,
            latency_ms=latency_ms,
        )
        return LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=latency_ms,
            timed_out=False,
            error="ollama_unavailable",
        )

    # Generate response
    raw_text = client.generate(
        prompt=user,
        system_prompt=system,
        json_mode=True,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Handle timeout
    if raw_text is None:
        logger.warning(
            "llm_call_timeout",
            feature=feature.value,
            model=model,
            timeout_s=timeout_s,
            latency_ms=latency_ms,
        )
        return LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=latency_ms,
            timed_out=True,
            error="timeout",
        )

    # Parse JSON
    payload = parse_json_response(raw_text)

    # Handle JSON parse failure
    if payload is None:
        logger.warning(
            "llm_call_json_parse_failed",
            feature=feature.value,
            model=model,
            timeout_s=timeout_s,
            latency_ms=latency_ms,
            raw_length=len(raw_text),
        )
        return LLMCallResult(
            raw_text=raw_text,
            payload=None,
            latency_ms=latency_ms,
            timed_out=False,
            error="json_parse_failed",
        )

    # Success
    logger.info(
        "llm_call_success",
        feature=feature.value,
        model=model,
        timeout_s=timeout_s,
        latency_ms=latency_ms,
        payload_keys=list(payload.keys()) if isinstance(payload, dict) else f"array[{len(payload)}]",
    )

    return LLMCallResult(
        raw_text=raw_text,
        payload=payload,
        latency_ms=latency_ms,
        timed_out=False,
        error=None,
    )
