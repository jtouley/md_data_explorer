"""
Result interpretation with LLM (ADR009 Phase 3).

This module provides LLM-powered interpretation of analysis results,
generating clinical insights and explanations for end users.

Key functions:
- interpret_result_with_llm: Generate human-readable interpretation of results
- _sanitize_result_for_prompt: Privacy-preserving result sanitization
"""

from typing import Any

import structlog

from clinical_analytics.core.llm_feature import LLMFeature, call_llm
from clinical_analytics.core.nl_query_config import LLM_TIMEOUT_RESULT_INTERPRETATION_S

logger = structlog.get_logger()


def _sanitize_result_for_prompt(result: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize result for LLM prompt (privacy-preserving).

    Removes large data tables and sensitive fields, keeping only
    summary statistics and metadata needed for interpretation.

    Args:
        result: Raw analysis result dict

    Returns:
        Sanitized dict safe for LLM prompt
    """
    sanitized = {}

    # Include intent and metric for context
    if "intent" in result:
        sanitized["intent"] = result["intent"]
    if "metric" in result:
        sanitized["metric"] = result["metric"]
    if "group_by" in result:
        sanitized["group_by"] = result["group_by"]

    # Include summary statistics (not raw data)
    if "summary" in result:
        sanitized["summary"] = result["summary"]

    # Include headline if present
    if "headline" in result:
        sanitized["headline"] = result["headline"]

    # Include statistical test results (p-values, test names)
    if "statistical_test" in result:
        sanitized["statistical_test"] = result["statistical_test"]

    # Include group comparisons (summary level, not raw data)
    if "group_summaries" in result:
        sanitized["group_summaries"] = result["group_summaries"]

    # EXCLUDE: data_table, raw_data, cohort, any large arrays
    # These would bloat the prompt and potentially leak PII

    return sanitized


def interpret_result_with_llm(result: dict[str, Any], model: str | None = None) -> str | None:
    """
    Generate human-readable interpretation of analysis result using LLM.

    Provides clinical insights and explanations for end users, helping them
    understand what the analysis found and what it means.

    Args:
        result: Analysis result dict (from compute_analysis_by_type)
        model: Optional model override (defaults to config)

    Returns:
        Interpretation string if successful, None if LLM fails

    Examples:
        >>> result = {"intent": "DESCRIBE", "metric": "age", "summary": {"mean": 45.5}}
        >>> interpretation = interpret_result_with_llm(result)
        >>> print(interpretation)
        "The average age is 45.5 years, indicating a middle-aged cohort."
    """
    # Sanitize result for prompt (remove large data, keep summary stats)
    sanitized_result = _sanitize_result_for_prompt(result)

    # Build prompt
    system_prompt = """You are a medical data analyst helping users understand analysis results.

Generate a clear, concise interpretation of the analysis result in 1-3 sentences.

Guidelines:
- Focus on what the data shows, not on clinical recommendations
- Explain statistical significance if present
- Use plain language, avoid jargon
- Be objective and evidence-based
- Do NOT make clinical recommendations or diagnoses

Return JSON with a single field:
- interpretation: String with 1-3 sentence explanation

Example:
{"interpretation": "The average LDL cholesterol is 120 mg/dL, which is within the recommended range. \
The comparison between treatment groups shows a statistically significant difference (p=0.02)."}"""

    user_prompt = f"Interpret this analysis result:\n\n{sanitized_result}"

    # Call LLM with timeout
    llm_result = call_llm(
        feature=LLMFeature.RESULT_INTERPRETATION,
        system=system_prompt,
        user=user_prompt,
        timeout_s=LLM_TIMEOUT_RESULT_INTERPRETATION_S,
        model=model,
    )

    # Handle failures gracefully
    if llm_result.error or llm_result.timed_out or llm_result.payload is None:
        logger.debug(
            "result_interpretation_failed",
            error=llm_result.error,
            timed_out=llm_result.timed_out,
            latency_ms=llm_result.latency_ms,
        )
        return None

    # Extract interpretation from payload
    if isinstance(llm_result.payload, dict) and "interpretation" in llm_result.payload:
        interpretation = llm_result.payload["interpretation"]
        logger.debug(
            "result_interpretation_success",
            latency_ms=llm_result.latency_ms,
            interpretation_length=len(str(interpretation)),
        )
        return str(interpretation) if interpretation is not None else None

    # Payload malformed or missing interpretation field
    logger.debug(
        "result_interpretation_malformed_payload",
        payload_keys=list(llm_result.payload.keys()) if isinstance(llm_result.payload, dict) else "not_dict",
    )
    return None
