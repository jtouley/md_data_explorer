"""
Automated golden question generation (ADR009 Phase 6).

This module provides automated generation, analysis, and maintenance of golden questions
(canonical query examples) from query logs using LLM assistance.

Key functions:
- generate_golden_questions_from_logs: Generate new golden questions from logs
- analyze_golden_question_coverage: Find coverage gaps in golden questions
- maintain_golden_questions_automatically: Auto-maintain golden question set
- validate_golden_question: Validate golden question schema
"""

from typing import Any

import structlog

from clinical_analytics.core.llm_feature import LLMFeature, call_llm
from clinical_analytics.core.nl_query_config import LLM_TIMEOUT_MAX_S, LLM_TIMEOUT_PARSE_S

logger = structlog.get_logger()

# Valid intent types for golden questions
VALID_INTENTS = ["COUNT", "DESCRIBE", "COMPARE_GROUPS", "FIND_PREDICTORS", "CORRELATIONS"]


def validate_golden_question(question: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate a golden question against schema.

    Validates:
    - Required fields: question, intent
    - Intent is one of VALID_INTENTS
    - Question is non-empty string

    Args:
        question: Dictionary with question and intent fields

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if question is valid
        - error_message: None if valid, error description if invalid
    """
    if "question" not in question:
        return False, "Missing 'question' field"

    if "intent" not in question:
        return False, "Missing 'intent' field"

    question_text = question.get("question")
    intent = question.get("intent")

    if not isinstance(question_text, str) or not question_text.strip():
        return False, "Question must be non-empty string"

    if intent not in VALID_INTENTS:
        return False, f"Invalid intent '{intent}'. Must be one of {VALID_INTENTS}"

    return True, None


def generate_golden_questions_from_logs(
    query_logs: list[dict[str, Any]],
    semantic_layer: Any,
    min_confidence: float = 0.75,
    min_frequency: int = 1,
) -> list[dict[str, Any]]:
    """
    Generate golden questions from query logs using LLM (ADR009 Phase 6).

    Analyzes successful queries from logs and generates canonical golden questions
    that represent common analysis patterns.

    Args:
        query_logs: List of query log entries with query, intent, confidence, success
        semantic_layer: SemanticLayer instance for context
        min_confidence: Minimum confidence threshold for included queries (default 0.75)
        min_frequency: Minimum frequency for pattern extraction (default 1)

    Returns:
        List of generated golden questions with schema validation
        Returns empty list if LLM unavailable (graceful degradation)

    Example:
        >>> query_logs = [
        ...     {"query": "average LDL?", "intent": "DESCRIBE", "confidence": 0.85, "success": True},
        ...     {"query": "compare by treatment", "intent": "COMPARE_GROUPS", "confidence": 0.90, "success": True},
        ... ]
        >>> questions = generate_golden_questions_from_logs(query_logs, semantic_layer)
        >>> # Returns [{"question": "What is the average LDL?", "intent": "DESCRIBE"}, ...]
    """
    # Filter successful, high-confidence queries
    candidate_queries = [q for q in query_logs if q.get("success") and q.get("confidence", 0) >= min_confidence]

    if not candidate_queries:
        logger.debug("golden_question_generation_no_candidates", log_count=len(query_logs))
        return []

    # Build prompt with query patterns
    system_prompt = """You are an expert data analysis coach. Analyze query logs and generate
canonical "golden questions" that represent common analysis patterns.

Golden questions should be:
- Clear, complete questions (not fragments)
- Representative of actual user queries
- Diverse across different analysis types
- Actionable and self-contained

Return JSON with structure:
{
  "golden_questions": [
    {"question": "Full question text?", "intent": "DESCRIBE|COMPARE_GROUPS|FIND_PREDICTORS|CORRELATIONS|COUNT"},
    ...
  ]
}

Generate 3-5 representative golden questions from the patterns you observe."""

    query_text = "\n".join(
        f"- {q.get('query')} (intent: {q.get('intent')}, confidence: {q.get('confidence'):.2f})"
        for q in candidate_queries[:20]  # Limit for prompt size
    )

    user_prompt = f"""Analyze these successful queries and generate representative golden questions:

{query_text}

Generate canonical questions that capture the most important analysis patterns."""

    # Call LLM
    timeout_s = min(LLM_TIMEOUT_PARSE_S * 2, LLM_TIMEOUT_MAX_S)  # Double timeout for generation
    result = call_llm(
        feature=LLMFeature.PARSE,  # Reuse PARSE feature for golden question generation
        system=system_prompt,
        user=user_prompt,
        timeout_s=timeout_s,
    )

    # Handle LLM failures (graceful degradation)
    if result.error or result.payload is None:
        logger.debug(
            "golden_question_generation_llm_failed",
            error=result.error,
            timed_out=result.timed_out,
        )
        return []

    # Extract and validate questions
    questions_data = result.payload.get("golden_questions", [])
    if not isinstance(questions_data, list):
        logger.warning("golden_question_generation_invalid_payload")
        return []

    valid_questions = []
    for q in questions_data:
        is_valid, error_msg = validate_golden_question(q)
        if is_valid:
            valid_questions.append(q)
        else:
            logger.debug("golden_question_validation_failed", error=error_msg)

    logger.debug(
        "golden_question_generation_completed",
        generated_count=len(questions_data),
        valid_count=len(valid_questions),
    )

    return valid_questions


def analyze_golden_question_coverage(
    golden_questions: list[dict[str, Any]],
    query_logs: list[dict[str, Any]],
    semantic_layer: Any,
) -> dict[str, Any]:
    """
    Analyze coverage gaps in golden questions (ADR009 Phase 6).

    Compares golden questions against query logs to identify missing patterns,
    high-frequency missing queries, and edge cases.

    Args:
        golden_questions: List of existing golden questions
        query_logs: List of query log entries
        semantic_layer: SemanticLayer instance for context

    Returns:
        Dictionary with coverage analysis results:
        - gaps: List of identified coverage gaps
        - coverage: Coverage metrics by intent
        - high_frequency_missing: Patterns appearing in logs but not in golden questions
    """
    system_prompt = """You are an expert data analyst. Compare a set of golden questions
against query logs to identify coverage gaps and missing patterns.

Return JSON with structure:
{
  "gaps": [
    {"intent": "FIND_PREDICTORS", "description": "No predictor-finding questions"},
    ...
  ],
  "coverage_by_intent": {
    "DESCRIBE": 80,
    "COMPARE_GROUPS": 60,
    ...
  }
}"""

    golden_text = "\n".join(f"- {q.get('question')} ({q.get('intent')})" for q in golden_questions)
    log_text = "\n".join(
        f"- {q.get('query')} ({q.get('intent')})"
        for q in query_logs[:20]  # Sample for prompt size
    )

    user_prompt = f"""Golden questions:
{golden_text}

Query logs (sample):
{log_text}

Analyze coverage gaps and identify missing patterns."""

    # Call LLM
    result = call_llm(
        feature=LLMFeature.PARSE,
        system=system_prompt,
        user=user_prompt,
        timeout_s=min(LLM_TIMEOUT_PARSE_S * 1.5, LLM_TIMEOUT_MAX_S),
    )

    # Handle failures (graceful degradation)
    if result.error or result.payload is None:
        logger.debug("golden_question_coverage_analysis_failed", error=result.error)
        return {"gaps": [], "coverage_by_intent": {}, "error": result.error}

    return result.payload or {}


def maintain_golden_questions_automatically(
    query_logs: list[dict[str, Any]],
    existing_golden_questions: list[dict[str, Any]],
    semantic_layer: Any,
    dry_run: bool = True,
) -> dict[str, Any]:
    """
    Automatically maintain golden questions from query logs (ADR009 Phase 6).

    Generates new candidates, analyzes coverage, and recommends updates
    to the golden question set.

    Args:
        query_logs: Recent query logs
        existing_golden_questions: Current golden questions
        semantic_layer: SemanticLayer instance
        dry_run: If True, return recommendations without modifying

    Returns:
        Dictionary with maintenance results:
        - new_questions: Recommended new golden questions
        - removed_questions: Questions recommended for removal
        - coverage_analysis: Coverage gap analysis
        - action: "keep" | "add" | "remove" | "replace"
    """
    logger.debug(
        "golden_question_maintenance_started",
        existing_count=len(existing_golden_questions),
        log_count=len(query_logs),
        dry_run=dry_run,
    )

    # Generate candidates from logs
    candidates = generate_golden_questions_from_logs(query_logs, semantic_layer)

    # Analyze coverage
    coverage = analyze_golden_question_coverage(existing_golden_questions + candidates, query_logs, semantic_layer)

    # Prepare recommendations
    results = {
        "new_questions": candidates,
        "removed_questions": [],
        "coverage_analysis": coverage,
        "action": "add" if candidates else "keep",
        "dry_run": dry_run,
    }

    logger.debug(
        "golden_question_maintenance_completed",
        new_count=len(candidates),
        dry_run=dry_run,
    )

    return results
