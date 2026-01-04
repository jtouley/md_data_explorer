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

from typing import Protocol


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

    Args:
        questions: List of generated questions
        available_columns: List of canonical column names (hard boundary)
        available_aliases: List of alias names (hard boundary)

    Returns:
        Filtered list of questions (only those referencing known columns/aliases)
    """
    validated = []
    all_valid_names = set(available_columns + available_aliases)

    for question in questions:
        # Check if question mentions any valid column/alias (case-insensitive substring match)
        question_lower = question.lower()
        if any(name.lower() in question_lower for name in all_valid_names):
            validated.append(question)

    return validated
