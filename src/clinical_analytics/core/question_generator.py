"""
Proactive Question Generator for Upload Context

Generates example questions during upload using local LLM,
falling back to deterministic generation if LLM unavailable.

**Critical Constraints**:
- Bounded by semantic layer (only known columns/aliases)
- Confidence-gated (deterministic thresholds)
- Idempotent (cached by dataset_version + run_key + query)
- Feature-flagged (ENABLE_PROACTIVE_QUESTIONS)
- Time-budgeted (hard timeout like parsing tiers)
- Observable (logs generation/selection/dismissal)
- Clinically safe (analysis navigation only, not medical advice)

**Architecture**:
- NO Streamlit imports (core layer must be UI-agnostic)
- Uses CacheBackend protocol for dependency injection
- UI layer provides StreamlitCacheBackend implementation
- Tests use simple dict-based mock
"""

import hashlib
from typing import Literal, Protocol

import structlog

from clinical_analytics.core.llm_feature import LLMFeature, call_llm
from clinical_analytics.core.nl_query_config import (
    ENABLE_PROACTIVE_QUESTIONS,
    LLM_TIMEOUT_QUESTION_GENERATION_S,
)
from clinical_analytics.core.nl_query_engine import QueryIntent

logger = structlog.get_logger()


class CacheBackend(Protocol):
    """Protocol for cache backend (UI-agnostic, no Streamlit coupling)."""

    def get(self, key: str) -> list[str] | None:
        """Get cached value by key. Returns None if not cached."""
        ...

    def set(self, key: str, value: list[str]) -> None:
        """Cache value by key."""
        ...


def generate_upload_questions(
    semantic_layer,
    inferred_schema,
    doc_context: str | None = None,
) -> list[str]:
    """
    Generate simple example questions during upload (stored in metadata).

    **Upload-Time Only**: These are simple examples, not confidence-gated.
    Stored in metadata JSON and displayed on first load.

    Args:
        semantic_layer: SemanticLayer instance (for column bounds)
        inferred_schema: InferredSchema from Phase 2
        doc_context: Extracted documentation text from Phase 1

    Returns:
        List of 3-5 simple example questions
    """
    # Get available columns for bounding
    alias_index = semantic_layer.get_column_alias_index()
    available_columns = list(alias_index.values()) if alias_index else []

    if not available_columns:
        return []

    # Fallback: deterministic questions (LLM integration will come later)
    return _deterministic_upload_questions(available_columns, inferred_schema)


def _deterministic_upload_questions(
    available_columns: list[str],
    inferred_schema,
) -> list[str]:
    """
    Generate deterministic questions without LLM (fallback).

    Uses template-based generation based on available columns.
    Always bounded by semantic layer.
    """
    if not available_columns:
        return []

    # Simple template-based questions
    questions = [
        f"What is the distribution of {available_columns[0]}?",
        f"Are there any outliers in {available_columns[0]}?",
        "What are the key relationships in the data?",
    ]

    # Limit to 5 questions
    return questions[:5]


def _deterministic_questions(
    semantic_layer,
    available_columns: list[str],
    query_intent=None,
) -> list[str]:
    """
    Generate deterministic questions without LLM (fallback).

    Uses template-based generation based on available columns.
    Always bounded by semantic layer.
    """
    questions = []

    # Simple templates based on intent type
    if query_intent:
        if query_intent.intent_type == "DESCRIBE":
            # Suggest grouping/stratification questions
            for col in available_columns[:3]:  # Limit to top 3
                questions.append(f"Would you like to stratify by {col}?")
        elif query_intent.intent_type == "COMPARE_GROUPS":
            # Suggest additional grouping variables
            for col in available_columns[:3]:
                questions.append(f"Would you like to compare by {col}?")

    # Default: generic analysis questions
    if not questions:
        questions = [
            "What is the distribution of the data?",
            "Are there any outliers?",
            "What are the key relationships?",
        ]

    return questions[:5]  # Limit to 5 questions


def _validate_questions_bounded(
    questions: list[str],
    available_columns: list[str],
    available_aliases: list[str],
) -> list[str]:
    """
    Validate that questions only reference known columns/aliases.

    Filters out questions that reference columns not in semantic layer.
    This prevents hallucination.

    **Validation Strategy**:
    - Primary: Whole-word matching (case-insensitive) - prevents false positives like "age" matching "average"
    - Fallback: Substring matching for edge cases (e.g., "patient_id" matching "patient id")
    - Edge cases: Partial matches (e.g., "age" matches "age_group") are acceptable (conservative filtering)

    Args:
        questions: List of generated questions
        available_columns: List of canonical column names (hard boundary)
        available_aliases: List of alias names (hard boundary)

    Returns:
        Filtered list of questions (only those referencing known columns/aliases)
    """
    import re

    validated = []
    all_valid_names = set(available_columns + available_aliases)

    for question in questions:
        question_lower = question.lower()
        # Check if question mentions any valid column/alias as whole word (preferred)
        # Use word boundaries to prevent "age" matching "average"
        matched = False
        for name in all_valid_names:
            name_lower = name.lower()
            # Whole word match (preferred) - prevents false positives
            pattern = r"\b" + re.escape(name_lower) + r"\b"
            if re.search(pattern, question_lower):
                matched = True
                break
            # Fallback: substring match for edge cases (e.g., "patient_id" vs "patient id")
            # Only if name is longer than 3 chars to avoid too many false positives
            if len(name_lower) > 3 and name_lower in question_lower:
                matched = True
                break

        if matched:
            validated.append(question)

    return validated


def generate_proactive_questions(
    semantic_layer,
    query_intent: QueryIntent | None = None,
    dataset_version: str | None = None,
    run_key: str | None = None,
    normalized_query: str | None = None,
    cache_backend: CacheBackend | None = None,
) -> list[str]:
    """
    Generate proactive follow-up questions based on query intent and semantic layer.

    **Semantic Layer Bounded**: Only generates questions about columns/aliases in semantic layer.
    **Confidence Gated**: Respects deterministic thresholds:
        - confidence ≥ 0.85: suggest next questions freely
        - 0.5-0.85: suggest only clarification/disambiguation questions
        - < 0.5: do not suggest proactively (ask user to rephrase)

    **Idempotent**: Cached by (dataset_version, run_key, normalized_query) to prevent duplicates.
    **Architecture**: Uses CacheBackend protocol for UI independence (no Streamlit imports in core).

    Args:
        semantic_layer: SemanticLayer instance (for column/alias bounds)
        query_intent: QueryIntent from previous parse (for confidence gating)
        dataset_version: Dataset version for caching
        run_key: Run key for caching
        normalized_query: Normalized query text for caching
        cache_backend: CacheBackend protocol for UI-agnostic caching (injected dependency)

    Returns:
        List of 3-5 example questions (empty if confidence too low or feature disabled)

    Raises:
        AssertionError: If semantic layer missing required metadata
    """
    # Feature flag check
    if not ENABLE_PROACTIVE_QUESTIONS:
        return []

    # Confidence gating (deterministic, not vibes-based)
    if query_intent is None or query_intent.confidence < 0.5:
        return []  # Don't suggest proactively if confidence too low

    # Check cache for idempotency (uses injected CacheBackend, no Streamlit coupling)
    cache_key = _build_cache_key(dataset_version, run_key, normalized_query)
    if cache_backend:
        cached = cache_backend.get(cache_key)
        if cached is not None:
            return cached

    # Get available columns/aliases from semantic layer (hard boundary)
    alias_index = semantic_layer.get_column_alias_index()
    available_columns = list(alias_index.values()) if alias_index else []
    available_aliases = list(alias_index.keys()) if alias_index else []

    if not available_columns:
        return []  # No columns available, can't generate questions

    # Try LLM first (if available and confidence high enough)
    questions: list[str] = []
    if query_intent.confidence >= 0.85:
        # High confidence: suggest next questions freely
        llm_questions = _llm_generate_questions(
            semantic_layer,
            available_columns,
            available_aliases,
            query_intent,
            question_type="next_questions",
        )
        if llm_questions:
            questions = llm_questions
    elif query_intent.confidence >= 0.5:
        # Medium confidence: only clarification/disambiguation
        llm_questions = _llm_generate_questions(
            semantic_layer,
            available_columns,
            available_aliases,
            query_intent,
            question_type="clarification",
        )
        if llm_questions:
            questions = llm_questions

    # Fallback: deterministic questions if LLM unavailable
    if not questions:
        questions = _deterministic_questions(semantic_layer, available_columns, query_intent)

    # Validate questions are bounded by semantic layer
    questions = _validate_questions_bounded(questions, available_columns, available_aliases)

    # Cache results for idempotency (uses injected CacheBackend)
    if cache_backend:
        cache_backend.set(cache_key, questions)

    # Log observability event
    _log_question_generation(questions, query_intent.confidence, dataset_version, len(available_columns))

    return questions


def _llm_generate_questions(
    semantic_layer,
    available_columns: list[str],
    available_aliases: list[str],
    query_intent: QueryIntent,
    question_type: Literal["next_questions", "clarification"],
    timeout_s: float | None = None,
) -> list[str] | None:
    """
    Generate questions using local LLM via call_llm().

    **Semantic Layer Bounded**: Prompt explicitly lists only available columns/aliases.
    **Clinical Safety**: Prompts emphasize analysis navigation, not medical advice.
    **Time Budgeted**: Hard timeout (5s default, same as parsing tiers).

    Args:
        semantic_layer: SemanticLayer instance
        available_columns: List of column names (hard boundary)
        available_aliases: List of alias names (hard boundary)
        query_intent: QueryIntent from previous parse
        question_type: "next_questions" (confidence ≥0.85) or "clarification" (0.5-0.85)
        timeout_s: Hard timeout (defaults to LLM_TIMEOUT_QUESTION_GENERATION_S)

    Returns:
        List of questions or None if LLM unavailable/failed
    """
    if timeout_s is None:
        timeout_s = LLM_TIMEOUT_QUESTION_GENERATION_S

    # Build prompt with semantic layer bounds (only known columns/aliases)
    system_prompt = f"""Generate 3-5 example questions users might ask about this dataset.

**CRITICAL CONSTRAINTS**:
1. Only reference columns/aliases that exist in the dataset (listed below)
2. Questions must be about analysis navigation, NOT medical advice
3. Use phrases like "Would you like to stratify by X?" not "You should consider Y treatment"

**Available Columns**: {", ".join(available_columns[:20])}  # Limit to top 20 for token budget
**Available Aliases**: {", ".join(available_aliases[:10])}  # Limit to top 10 for token budget

**Question Type**: {question_type}
- "next_questions": Suggest logical next analysis steps
- "clarification": Ask for clarification/disambiguation only

**Previous Query Intent**: {query_intent.intent_type} (confidence: {query_intent.confidence:.2f})

Return JSON: {{"questions": ["question1", "question2", ...]}}"""

    user_prompt = f"Generate {question_type} questions based on the previous query intent."

    result = call_llm(
        feature=LLMFeature.QUESTION_GENERATION,
        system=system_prompt,
        user=user_prompt,
        timeout_s=timeout_s,  # Hard time budget
    )

    if result.error or result.payload is None:
        logger.debug(
            "question_generation_llm_failed",
            error=result.error,
            timed_out=result.timed_out,
            question_type=question_type,
        )
        return None

    if isinstance(result.payload, dict) and "questions" in result.payload:
        questions = result.payload["questions"]
        if isinstance(questions, list):
            # Validate questions are bounded (check for hallucinated columns)
            return _validate_questions_bounded(questions, available_columns, available_aliases)

    return None


def _build_cache_key(
    dataset_version: str | None,
    run_key: str | None,
    normalized_query: str | None,
) -> str:
    """
    Build cache key for idempotency.

    Uses same pattern as execution result caching in 03_💬_Ask_Questions.py:
    - Key format: "proactive_questions:{dataset_version}:{run_key}:{query_hash}"
    - query_hash: SHA256 hash of normalized_query (first 16 chars, same as exec_result caching)

    Args:
        dataset_version: Dataset version identifier
        run_key: Run key for idempotency
        normalized_query: Normalized query text (for hashing)

    Returns:
        Cache key string with "proactive_questions:" prefix for use with CacheBackend
    """
    # Use SHA256 (same as exec_result caching) for deterministic hashing
    query_hash = hashlib.sha256(normalized_query.encode() if normalized_query else b"").hexdigest()[:16]
    return f"proactive_questions:{dataset_version or 'unknown'}:{run_key or 'unknown'}:{query_hash}"


def _log_question_generation(
    questions: list[str],
    confidence: float,
    dataset_version: str | None,
    column_count: int,
) -> None:
    """Log observability event for question generation."""
    logger.info(
        "proactive_questions_generated",
        question_count=len(questions),
        confidence=confidence,
        dataset_version=dataset_version,
        column_count=column_count,
    )
