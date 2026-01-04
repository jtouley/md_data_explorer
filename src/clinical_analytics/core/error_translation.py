"""
Error message translation with LLM (ADR009 Phase 4).

This module provides LLM-powered translation of technical error messages
into user-friendly explanations that help users understand what went wrong
and how to fix it.

Key functions:
- translate_error_with_llm: Convert technical errors to friendly messages
"""

import structlog

from clinical_analytics.core.llm_feature import LLMFeature, call_llm
from clinical_analytics.core.nl_query_config import LLM_TIMEOUT_ERROR_TRANSLATION_S

logger = structlog.get_logger()


def translate_error_with_llm(technical_error: str, model: str | None = None) -> str | None:
    """
    Translate technical error message into user-friendly explanation using LLM.

    Converts error messages like "ColumnNotFoundError: Column 'foo' not found"
    into helpful messages like "I couldn't find a column called 'foo'. Did you mean 'bar'?"

    Args:
        technical_error: Raw error message from exception
        model: Optional model override (defaults to config)

    Returns:
        User-friendly error message if successful, None if LLM fails

    Examples:
        >>> error = "ColumnNotFoundError: Column 'ldl' not found"
        >>> friendly = translate_error_with_llm(error)
        >>> print(friendly)
        "I couldn't find a column called 'ldl'. Try 'LDL mg/dL' instead."
    """
    # Build prompt
    system_prompt = """You are a helpful assistant that translates technical error messages \
into user-friendly explanations.

Generate a clear, concise explanation in 1-2 sentences that:
- Explains what went wrong in plain language
- Suggests how to fix it (if applicable)
- Avoids technical jargon
- Does NOT expose internal system details (database hosts, file paths, etc.)

Return JSON with a single field:
- friendly_message: String with user-friendly explanation

Example:
{"friendly_message": "I couldn't find a column called 'cholesterol'. \
Try 'LDL mg/dL' or 'Total Cholesterol mg/dL' instead."}"""

    user_prompt = f"Translate this error message:\n\n{technical_error}"

    # Call LLM with timeout
    llm_result = call_llm(
        feature=LLMFeature.ERROR_TRANSLATION,
        system=system_prompt,
        user=user_prompt,
        timeout_s=LLM_TIMEOUT_ERROR_TRANSLATION_S,
        model=model,
    )

    # Handle failures gracefully
    if llm_result.error or llm_result.timed_out or llm_result.payload is None:
        logger.debug(
            "error_translation_failed",
            error=llm_result.error,
            timed_out=llm_result.timed_out,
            latency_ms=llm_result.latency_ms,
        )
        return None

    # Extract friendly message from payload
    if isinstance(llm_result.payload, dict) and "friendly_message" in llm_result.payload:
        friendly_message = llm_result.payload["friendly_message"]
        logger.debug(
            "error_translation_success",
            latency_ms=llm_result.latency_ms,
            message_length=len(str(friendly_message)),
        )
        return str(friendly_message) if friendly_message is not None else None

    # Payload malformed or missing friendly_message field
    logger.debug(
        "error_translation_malformed_payload",
        payload_keys=list(llm_result.payload.keys()) if isinstance(llm_result.payload, dict) else "not_dict",
    )
    return None
