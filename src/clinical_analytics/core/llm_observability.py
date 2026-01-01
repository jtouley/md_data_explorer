"""
LLM Observability Event Schema and Query Pattern Sanitization.

This module defines the observability event schema for all LLM operations and provides
privacy-preserving query sanitization.

Key principles:
- NEVER log raw query text (privacy protection)
- Log only query_hash (SHA256) and pattern_tags (controlled vocabulary)
- Required fields for ALL LLM events (enables downstream metrics)
- Sanitization enforced by code, not good intentions

Required fields for all LLM events:
- event: str (event name)
- timestamp: datetime
- run_key: str | None
- query_hash: str (SHA256, never raw query)
- dataset_version: str
- tier: int (1|2|3)
- model: str (e.g., "llama3.1:8b")
- feature: str (followups|interpretation|result_interpretation|error_translation|filter_extraction)
- timeout_s: float
- latency_ms: float
- success: bool
- error_type: str | None
- error_message: str | None (sanitized, no PII)
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

import structlog

logger = structlog.get_logger()

# Controlled vocabulary for pattern tags (privacy-preserving)
PATTERN_TAGS_VOCABULARY = [
    "contains_negation",
    "mentions_missingness",
    "multi_table_join_request",
    "contains_numeric_range",
    "contains_value_exclusion",
    "contains_comparison",
    "contains_grouping",
]


@dataclass
class LLMEvent:
    """
    Structured event for LLM operations (observability).

    All required fields must be populated for every LLM event.
    This enables downstream metric computation (e.g., tier3_rate_rolling_100).

    Privacy Protection:
        - query_hash is SHA256 hash of raw query (never log raw query text)
        - error_message must be sanitized (no PII)
    """

    event: str  # Event name (e.g., "llm_call_success", "llm_call_timeout")
    timestamp: datetime
    run_key: str | None  # Deterministic key for idempotent execution
    query_hash: str  # SHA256 hash of query (never raw query text)
    dataset_version: str
    tier: int  # 1|2|3
    model: str  # e.g., "llama3.1:8b"
    feature: str  # followups|interpretation|result_interpretation|error_translation|filter_extraction
    timeout_s: float
    latency_ms: float
    success: bool
    error_type: str | None
    error_message: str | None  # Sanitized, no PII


def sanitize_query(query: str) -> dict[str, str | list[str] | int]:
    """
    Sanitize query for privacy-preserving logging.

    CRITICAL: This function NEVER returns raw query text.
    Returns only query_hash (SHA256) and pattern_tags (controlled vocabulary).

    Args:
        query: Raw query text (NEVER logged)

    Returns:
        Dict with:
            - query_hash: SHA256 hex digest (64 chars)
            - pattern_tags: List of pattern tags from controlled vocabulary
            - token_count: Number of tokens (approximate)

    Examples:
        >>> result = sanitize_query("compare mortality by treatment")
        >>> len(result["query_hash"])
        64
        >>> "query" not in result  # Raw query never in result
        True
        >>> isinstance(result["pattern_tags"], list)
        True
    """
    # Generate deterministic hash
    query_hash = hashlib.sha256(query.encode()).hexdigest()

    # Extract pattern tags
    pattern_tags = extract_pattern_tags(query)

    # Approximate token count (simple whitespace split)
    token_count = len(query.split())

    return {
        "query_hash": query_hash,
        "pattern_tags": pattern_tags,
        "token_count": token_count,
    }


def extract_pattern_tags(query: str) -> list[str]:
    """
    Extract pattern tags from query using controlled vocabulary.

    This function identifies query patterns without logging raw text.
    Only returns tags from PATTERN_TAGS_VOCABULARY.

    Args:
        query: Raw query text

    Returns:
        List of pattern tags (controlled vocabulary)

    Examples:
        >>> extract_pattern_tags("patients not on statins")
        ['contains_negation']
        >>> extract_pattern_tags("exclude missing values")
        ['mentions_missingness']
        >>> extract_pattern_tags("aged 50 to 75")
        ['contains_numeric_range']
    """
    tags: list[str] = []
    query_lower = query.lower()

    # Check for negation patterns
    negation_patterns = [r"\bnot\b", r"\bno\b", r"\bexclude\b", r"\bwithout\b", r"\bneither\b", r"\bnor\b"]
    if any(re.search(pattern, query_lower) for pattern in negation_patterns):
        tags.append("contains_negation")

    # Check for missingness mentions
    missingness_patterns = [
        r"\bmissing\b",
        r"\bnull\b",
        r"\bn/a\b",
        r"\bna\b",
        r"\bempty\b",
        r"\bblank\b",
        r"\bunknown\b",
    ]
    if any(re.search(pattern, query_lower) for pattern in missingness_patterns):
        tags.append("mentions_missingness")

    # Check for numeric ranges
    range_patterns = [r"\d+\s*to\s*\d+", r"\d+\s*-\s*\d+", r"\bbetween\b.*\band\b"]
    if any(re.search(pattern, query_lower) for pattern in range_patterns):
        tags.append("contains_numeric_range")

    # Check for comparisons
    comparison_patterns = [
        r"\bgreater\s+than\b",
        r"\bless\s+than\b",
        r"\babove\b",
        r"\bbelow\b",
        r"\bhigher\b",
        r"\blower\b",
        r">",
        r"<",
        r">=",
        r"<=",
    ]
    if any(re.search(pattern, query_lower) for pattern in comparison_patterns):
        tags.append("contains_comparison")

    # Check for grouping
    grouping_patterns = [r"\bby\b.*\bgroup\b", r"\bby\b", r"\bgrouped\b", r"\bstratif", r"\bbreakdown\b"]
    if any(re.search(pattern, query_lower) for pattern in grouping_patterns):
        tags.append("contains_grouping")

    # Check for value exclusion
    exclusion_patterns = [
        r"\bremove\b",
        r"\bget\s+rid\s+of\b",
        r"\bdrop\b",
        r"\beliminate\b",
        r"\bexclude\b",
        r"\bfilter\s+out\b",
    ]
    if any(re.search(pattern, query_lower) for pattern in exclusion_patterns):
        tags.append("contains_value_exclusion")

    # Check for multi-table join requests
    join_patterns = [r"\bjoin\b", r"\bmerge\b", r"\bcombine\b.*\btable", r"\blink\b.*\btable"]
    if any(re.search(pattern, query_lower) for pattern in join_patterns):
        tags.append("multi_table_join_request")

    return tags


def log_llm_event(
    event: str,
    query: str,
    tier: int,
    model: str,
    feature: str,
    timeout_s: float,
    latency_ms: float,
    success: bool,
    run_key: str | None = None,
    dataset_version: str = "unknown",
    error_type: str | None = None,
    error_message: str | None = None,
) -> LLMEvent:
    """
    Log LLM event with required observability schema.

    This function creates a structured LLM event and logs it.
    Query is automatically sanitized (NEVER logs raw query text).

    Args:
        event: Event name (e.g., "llm_call_success")
        query: Raw query text (will be sanitized, never logged)
        tier: Tier (1|2|3)
        model: Model name (e.g., "llama3.1:8b")
        feature: Feature name (followups|interpretation|etc)
        timeout_s: Timeout in seconds
        latency_ms: Latency in milliseconds
        success: Whether call succeeded
        run_key: Optional run key for idempotency
        dataset_version: Dataset version
        error_type: Optional error type
        error_message: Optional error message (sanitized, no PII)

    Returns:
        LLMEvent instance

    Examples:
        >>> event = log_llm_event(
        ...     event="llm_call_success",
        ...     query="compare mortality by treatment",
        ...     tier=3,
        ...     model="llama3.1:8b",
        ...     feature="followups",
        ...     timeout_s=15.0,
        ...     latency_ms=1234.5,
        ...     success=True,
        ... )
        >>> event.success
        True
        >>> len(event.query_hash)
        64
    """
    # Sanitize query (NEVER log raw query text)
    sanitized = sanitize_query(query)

    # Create event
    event_obj = LLMEvent(
        event=event,
        timestamp=datetime.now(),
        run_key=run_key,
        query_hash=sanitized["query_hash"],
        dataset_version=dataset_version,
        tier=tier,
        model=model,
        feature=feature,
        timeout_s=timeout_s,
        latency_ms=latency_ms,
        success=success,
        error_type=error_type,
        error_message=error_message,
    )

    # Log structured event
    logger.info(
        event,
        timestamp=event_obj.timestamp.isoformat(),
        run_key=run_key,
        query_hash=sanitized["query_hash"],
        pattern_tags=sanitized["pattern_tags"],
        token_count=sanitized["token_count"],
        dataset_version=dataset_version,
        tier=tier,
        model=model,
        feature=feature,
        timeout_s=timeout_s,
        latency_ms=latency_ms,
        success=success,
        error_type=error_type,
        error_message=error_message,
    )

    return event_obj
