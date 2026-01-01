"""
Natural Language Query Engine for parsing free-form clinical research questions.

This module implements a three-tier architecture for parsing natural language queries:
- Tier 1: Fast regex pattern matching for common patterns
- Tier 2: Semantic embeddings using sentence-transformers
- Tier 3: LLM fallback (optional) for complex queries

Example:
    >>> from clinical_analytics.core.nl_query_engine import NLQueryEngine
    >>> engine = NLQueryEngine(semantic_layer)
    >>> intent = engine.parse_query("compare mortality by treatment")
    >>> print(intent.intent_type)
    'COMPARE_GROUPS'
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path

import structlog

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

logger = structlog.get_logger()

# Valid intent types (single source of truth)
VALID_INTENT_TYPES = [
    "DESCRIBE",
    "COMPARE_GROUPS",
    "FIND_PREDICTORS",
    "SURVIVAL",
    "CORRELATIONS",
    "COUNT",
]


def _stable_hash(s: str) -> str:
    """
    Stable hash for metrics (SHA256, not Python's randomized hash()).

    Args:
        s: String to hash

    Returns:
        First 12 chars of SHA256 hex digest
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


@dataclass
class QueryIntent:
    """
    Parsed intent from natural language query.

    Attributes:
        intent_type: Analysis type (DESCRIBE, COMPARE_GROUPS, FIND_PREDICTORS, SURVIVAL, CORRELATIONS)
        primary_variable: Main outcome or variable of interest
        grouping_variable: Variable to group/stratify by
        predictor_variables: List of predictor variables for regression
        time_variable: Time column for survival analysis
        event_variable: Event indicator for survival analysis
        filters: List of FilterSpec objects for filter conditions
        confidence: Confidence score 0-1 for the parse
    """

    intent_type: str
    primary_variable: str | None = None
    grouping_variable: str | None = None
    predictor_variables: list[str] = field(default_factory=list)
    time_variable: str | None = None
    event_variable: str | None = None
    filters: list[FilterSpec] = field(default_factory=list)
    confidence: float = 0.0
    parsing_tier: str | None = None  # "pattern_match", "semantic_match", "llm_fallback"
    parsing_attempts: list[dict] = field(default_factory=list)  # What was tried
    failure_reason: str | None = None  # Why it failed
    suggestions: list[str] = field(default_factory=list)  # How to improve query
    # ADR009 Phase 1: LLM-generated follow-up questions
    follow_ups: list[str] = field(default_factory=list)  # Context-aware follow-up questions
    follow_up_explanation: str = ""  # Why these follow-ups are relevant
    # ADR009 Phase 2: Query interpretation and confidence explanation
    interpretation: str = ""  # Human-readable explanation of what the query is asking
    confidence_explanation: str = ""  # Why the confidence score is what it is

    def __post_init__(self):
        """Validate intent_type."""
        if self.intent_type not in VALID_INTENT_TYPES:
            raise ValueError(f"Invalid intent_type: {self.intent_type}. Must be one of {VALID_INTENT_TYPES}")


class NLQueryEngine:
    """
    Natural language query engine with three-tier parsing.

    Architecture:
        Tier 1: Pattern matching (regex for common queries)
        Tier 2: Semantic embeddings (sentence-transformers)
        Tier 3: LLM fallback (structured prompt with RAG) - optional

    Args:
        semantic_layer: SemanticLayer instance for metadata access
        embedding_model: sentence-transformers model name (default: all-MiniLM-L6-v2)

    Example:
        >>> engine = NLQueryEngine(semantic_layer)
        >>> intent = engine.parse_query("what predicts mortality?")
        >>> print(f"{intent.intent_type}: {intent.primary_variable}")
        'FIND_PREDICTORS: mortality'
    """

    def __init__(self, semantic_layer, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize query engine.

        Args:
            semantic_layer: SemanticLayer instance for metadata access
            embedding_model: sentence-transformers model name
        """
        self.semantic_layer = semantic_layer
        self.embedding_model_name = embedding_model
        self.encoder = None  # Lazy load
        self.template_embeddings = None  # Lazy load

        # Overlay cache (mtime-based hot reload)
        self._overlay_cache_text = ""
        self._overlay_cache_mtime_ns = 0

        # Build query templates from metadata
        self._build_query_templates()

    def _prompt_overlay_path(self) -> Path:
        """
        Get overlay file path (configurable via env var).

        Defaults to /tmp/nl_query_learning/prompt_overlay.txt to keep
        learning artifacts out of source tree.

        Returns:
            Path to overlay file
        """
        # Prefer explicit override
        p = os.getenv("NL_PROMPT_OVERLAY_PATH")
        if p:
            return Path(p)

        # Default: same directory as self-improve logs
        # (keeps artifacts out of source tree)
        return Path("/tmp/nl_query_learning/prompt_overlay.txt")

    def _load_prompt_overlay(self) -> str:
        """
        Load prompt overlay from disk with mtime-based caching.

        Only re-reads file if modified since last load (hot reload).

        Returns:
            Overlay text to append to system prompt, or empty string
        """
        p = self._prompt_overlay_path()

        try:
            st = p.stat()
        except FileNotFoundError:
            self._overlay_cache_text = ""
            self._overlay_cache_mtime_ns = 0
            return ""

        # Cache hit: file unchanged since last load
        if st.st_mtime_ns == self._overlay_cache_mtime_ns:
            return self._overlay_cache_text

        # Cache miss: file changed, reload
        try:
            text = p.read_text(encoding="utf-8").strip()
            self._overlay_cache_text = text
            self._overlay_cache_mtime_ns = st.st_mtime_ns
            if text:
                logger.info("prompt_overlay_loaded", path=str(p), length=len(text))
            return text
        except Exception as e:
            logger.warning("prompt_overlay_load_failed", path=str(p), error=str(e))
            return ""

    def _build_query_templates(self):
        """Build query templates from semantic layer metadata."""
        self.query_templates = [
            {
                "template": "compare {outcome} by {group}",
                "intent": "COMPARE_GROUPS",
                "slots": ["outcome", "group"],
            },
            {
                "template": "compare {outcome} between {group}",
                "intent": "COMPARE_GROUPS",
                "slots": ["outcome", "group"],
            },
            {
                "template": "difference in {outcome} by {group}",
                "intent": "COMPARE_GROUPS",
                "slots": ["outcome", "group"],
            },
            {
                "template": "what predicts {outcome}",
                "intent": "FIND_PREDICTORS",
                "slots": ["outcome"],
            },
            {
                "template": "risk factors for {outcome}",
                "intent": "FIND_PREDICTORS",
                "slots": ["outcome"],
            },
            {
                "template": "predictors of {outcome}",
                "intent": "FIND_PREDICTORS",
                "slots": ["outcome"],
            },
            {"template": "survival analysis", "intent": "SURVIVAL", "slots": []},
            {"template": "kaplan meier", "intent": "SURVIVAL", "slots": []},
            {"template": "time to event", "intent": "SURVIVAL", "slots": []},
            {
                "template": "correlation between {var1} and {var2}",
                "intent": "CORRELATIONS",
                "slots": ["var1", "var2"],
            },
            {
                "template": "relationship between {var1} and {var2}",
                "intent": "CORRELATIONS",
                "slots": ["var1", "var2"],
            },
            {"template": "association", "intent": "CORRELATIONS", "slots": []},
            {"template": "descriptive statistics", "intent": "DESCRIBE", "slots": []},
            {"template": "summary statistics", "intent": "DESCRIBE", "slots": []},
            {"template": "describe", "intent": "DESCRIBE", "slots": []},
        ]

    def parse_query(
        self,
        query: str,
        dataset_id: str | None = None,
        upload_id: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> QueryIntent:
        """
        Parse natural language query into structured intent with conversation context.

        Supports conversational refinements (ADR009 Phase 6): When conversation_history
        is provided, the LLM can detect refinement queries like "remove the n/a" that
        modify previous queries, and intelligently merge them.

        Args:
            query: User's question (e.g., "compare survival by treatment arm")
            dataset_id: Optional dataset identifier for logging
            upload_id: Optional upload identifier for logging
            conversation_history: Optional list of previous queries for context
                Each entry should contain: query, intent, group_by, filters_applied

        Returns:
            QueryIntent with extracted intent type and variables

        Raises:
            ValueError: If query is empty

        Example:
            >>> intent = engine.parse_query("compare mortality by treatment")
            >>> assert intent.intent_type == "COMPARE_GROUPS"
            >>> assert intent.confidence > 0.9

        Example (with conversation context):
            >>> history = [{"query": "count by statin", "intent": "COUNT", "group_by": "statin"}]
            >>> intent = engine.parse_query("remove the n/a", conversation_history=history)
            >>> assert intent.intent_type == "COUNT"  # Inherited from previous
            >>> assert len(intent.filters) > 0  # Added filter from refinement
        """
        if not query or not query.strip():
            logger.error(
                "query_parse_failed",
                error_type="empty_query",
                query=query,
                dataset_id=dataset_id,
                upload_id=upload_id,
            )
            raise ValueError("Query cannot be empty")

        query = query.strip()

        # Import config constants
        from clinical_analytics.core.nl_query_config import (
            TIER_1_PATTERN_MATCH_THRESHOLD,
            TIER_2_SEMANTIC_MATCH_THRESHOLD,
        )

        # Log query parsing start
        log_context = {
            "query": query,
            "dataset_id": dataset_id,
            "upload_id": upload_id,
        }
        logger.info("query_parse_start", **log_context)

        # Track parsing attempts for diagnostics
        parsing_attempts = []

        # Tier 1: Pattern matching
        pattern_intent = self._pattern_match(query)
        attempt = {
            "tier": "pattern_match",
            "result": "success"
            if pattern_intent and pattern_intent.confidence >= TIER_1_PATTERN_MATCH_THRESHOLD
            else "failed",
            "confidence": pattern_intent.confidence if pattern_intent else 0.0,
        }
        parsing_attempts.append(attempt)

        if pattern_intent and pattern_intent.confidence >= TIER_1_PATTERN_MATCH_THRESHOLD:
            pattern_intent.parsing_tier = "pattern_match"
            pattern_intent.parsing_attempts = parsing_attempts
            matched_vars = self._get_matched_variables(pattern_intent)
            logger.info(
                "query_parse_success",
                intent=pattern_intent.intent_type,
                confidence=pattern_intent.confidence,
                matched_vars=matched_vars,
                tier="pattern_match",
                **log_context,
            )
            intent = pattern_intent  # Set for post-processing
        else:
            # Pattern match found something but below threshold - try semantic match
            # but keep pattern match as fallback if semantic match is worse
            # Tier 2: Semantic embeddings
            semantic_intent = self._semantic_match(query)
            attempt = {
                "tier": "semantic_match",
                "result": "success"
                if semantic_intent and semantic_intent.confidence >= TIER_2_SEMANTIC_MATCH_THRESHOLD
                else "failed",
                "confidence": semantic_intent.confidence if semantic_intent else 0.0,
            }
            parsing_attempts.append(attempt)

            # Choose best intent: prefer semantic if it meets threshold, otherwise use pattern if available
            if semantic_intent and semantic_intent.confidence >= TIER_2_SEMANTIC_MATCH_THRESHOLD:
                intent = semantic_intent
                intent.parsing_tier = "semantic_match"
                intent.parsing_attempts = parsing_attempts
                matched_vars = self._get_matched_variables(intent)
                logger.info(
                    "query_parse_success",
                    intent=intent.intent_type,
                    confidence=intent.confidence,
                    matched_vars=matched_vars,
                    tier="semantic_match",
                    **log_context,
                )
            elif pattern_intent and pattern_intent.intent_type != "DESCRIBE":
                # Use pattern match result even if below threshold, if it's more specific than DESCRIBE
                # Also prefer pattern match if it's better than semantic match
                if semantic_intent is None or (
                    pattern_intent.confidence > semantic_intent.confidence and pattern_intent.intent_type != "DESCRIBE"
                ):
                    intent = pattern_intent
                    intent.parsing_tier = "pattern_match"
                    intent.parsing_attempts = parsing_attempts
                    logger.info(
                        "query_parse_partial_pattern_match",
                        intent=intent.intent_type,
                        confidence=intent.confidence,
                        reason="pattern_match_below_threshold_but_better_than_semantic",
                        **log_context,
                    )
                else:
                    intent = semantic_intent  # Use semantic if it's better
            elif semantic_intent:
                intent = semantic_intent  # Use semantic even if below threshold
            else:
                intent = pattern_intent  # Fallback to pattern match if available

        # Tier 3: LLM fallback (stub for now) - only if we don't have a good intent yet
        if not intent or (intent.confidence < 0.5 and intent.intent_type == "DESCRIBE"):
            llm_intent = self._llm_parse(query, conversation_history=conversation_history)
            attempt = {
                "tier": "llm_fallback",
                "result": "success" if llm_intent else "failed",
                "confidence": llm_intent.confidence if llm_intent else 0.0,
            }
            parsing_attempts.append(attempt)

            if llm_intent:
                intent = llm_intent
                intent.parsing_tier = "llm_fallback"
                intent.parsing_attempts = parsing_attempts
                matched_vars = self._get_matched_variables(intent)
                logger.info(
                    "query_parse_success",
                    intent=intent.intent_type,
                    confidence=intent.confidence,
                    matched_vars=matched_vars,
                    tier="llm_fallback",
                    **log_context,
                )

        # If we still don't have a good intent, set failure diagnostics
        if not intent or (intent.confidence < 0.3 and intent.intent_type == "DESCRIBE"):
            if intent is None:
                intent = QueryIntent(
                    intent_type="DESCRIBE",
                    confidence=0.0,
                    failure_reason="All parsing tiers failed",
                    suggestions=self._generate_suggestions(query),
                    parsing_attempts=parsing_attempts,
                )
            else:
                intent.failure_reason = "All parsing tiers failed"
                intent.suggestions = self._generate_suggestions(query)
                intent.parsing_attempts = parsing_attempts

            logger.warning(
                "query_parse_failed",
                error_type="no_intent_found",
                query=query,
                dataset_id=dataset_id,
                upload_id=upload_id,
            )

        # Post-process: Extract and assign variables if missing (runs in src, not UI)
        if intent and intent.intent_type in ["COMPARE_GROUPS", "FIND_PREDICTORS", "CORRELATIONS"]:
            # Extract variables from query if not already set
            matched_vars: list[str] = []  # Initialize for logging
            if not intent.primary_variable or not intent.grouping_variable:
                # For COMPARE_GROUPS: prioritize pattern-based extraction for "which X had lowest Y"
                if intent.intent_type == "COMPARE_GROUPS":
                    # First, try to extract directly from "which X had lowest Y" pattern
                    match = re.search(
                        r"(?:which|what)\s+(\w+(?:\s+\w+)*?)\s+had\s+the\s+(lowest|highest)\s+(\w+(?:\s+\w+)*)",
                        query.lower(),
                    )
                    if match:
                        group_term = match.group(1).strip()
                        primary_term = match.group(3).strip()

                        # Try fuzzy matching with improved matching
                        group_var, group_conf, _ = self._fuzzy_match_variable(group_term)
                        primary_var, primary_conf, _ = self._fuzzy_match_variable(primary_term)

                        if group_var and primary_var:
                            # Both matched - use them
                            intent.grouping_variable = group_var
                            intent.primary_variable = primary_var
                            logger.info(
                                "variables_extracted_post_parse",
                                intent_type=intent.intent_type,
                                matched_vars=[primary_var, group_var],
                                primary_variable=intent.primary_variable,
                                grouping_variable=intent.grouping_variable,
                            )
                        elif group_var or primary_var:
                            # At least one matched - use what we have
                            if group_var:
                                intent.grouping_variable = group_var
                            if primary_var:
                                intent.primary_variable = primary_var
                            logger.info(
                                "variables_extracted_post_parse",
                                intent_type=intent.intent_type,
                                matched_vars=[v for v in [primary_var, group_var] if v],
                                primary_variable=intent.primary_variable,
                                grouping_variable=intent.grouping_variable,
                            )
                        else:
                            # Pattern extraction failed - fall back to general extraction
                            matched_vars, _ = self._extract_variables_from_query(query)
                            if len(matched_vars) >= 2:
                                # For "which X had lowest Y", X is grouping, Y is primary
                                # But _extract_variables_from_query might return in different order
                                # Try to match based on query position
                                query_lower = query.lower()
                                first_pos = query_lower.find(matched_vars[0].lower())
                                second_pos = query_lower.find(matched_vars[1].lower())

                                if first_pos < second_pos:
                                    # First variable appears first in query - likely the grouping variable
                                    intent.grouping_variable = matched_vars[0]
                                    intent.primary_variable = matched_vars[1]
                                else:
                                    # Second variable appears first - use reverse order
                                    intent.grouping_variable = matched_vars[1]
                                    intent.primary_variable = matched_vars[0]
                            elif len(matched_vars) == 1:
                                # Only one variable - try to infer from query structure
                                match = re.search(r"(?:which|what)\s+(\w+(?:\s+\w+)*?)\s+had", query.lower())
                                if match:
                                    group_term = match.group(1).strip()
                                    group_var, _, _ = self._fuzzy_match_variable(group_term)
                                    if group_var:
                                        intent.grouping_variable = group_var
                                        intent.primary_variable = matched_vars[0]
                                    else:
                                        intent.primary_variable = matched_vars[0]
                                else:
                                    intent.primary_variable = matched_vars[0]
                    else:
                        # No "which X had lowest Y" pattern - use general extraction
                        matched_vars, _ = self._extract_variables_from_query(query)
                        if len(matched_vars) >= 2:
                            intent.grouping_variable = matched_vars[0]
                            intent.primary_variable = matched_vars[1]
                        elif len(matched_vars) == 1:
                            intent.primary_variable = matched_vars[0]

                # For FIND_PREDICTORS: first variable is outcome
                elif intent.intent_type == "FIND_PREDICTORS":
                    if not intent.primary_variable and len(matched_vars) >= 1:
                        intent.primary_variable = matched_vars[0]

                # For CORRELATIONS: extract ALL variables mentioned (not just first 2)
                # CORRELATIONS queries often mention multiple variables (e.g., "how does X, Y relate to Z and W")
                # Put ALL extracted variables in predictor_variables for correlation analysis
                elif intent.intent_type == "CORRELATIONS":
                    # For CORRELATIONS, use predictor_variables to hold all variables
                    if matched_vars:
                        intent.predictor_variables = matched_vars
                        # Also set primary/grouping for backward compatibility
                        if len(matched_vars) >= 1:
                            intent.primary_variable = matched_vars[0]
                        if len(matched_vars) >= 2:
                            intent.grouping_variable = matched_vars[1]

                if matched_vars or (intent.primary_variable or intent.grouping_variable):
                    logger.info(
                        "variables_extracted_post_parse",
                        intent_type=intent.intent_type,
                        matched_vars=matched_vars,
                        primary_variable=intent.primary_variable,
                        grouping_variable=intent.grouping_variable,
                    )

        # Extract filters from query (applies to all intent types)
        if intent:
            # Extract regex-based filters
            regex_filters = self._extract_filters(query, grouping_variable=intent.grouping_variable)

            # CRITICAL FIX: Validate all regex-extracted filters before applying
            # Phase 5 filter extraction provides validation, but regex extraction doesn't
            # We must filter out invalid filters (e.g., string "n/a" for float column)
            valid_regex_filters = []
            if regex_filters:
                from clinical_analytics.core.filter_extraction import _validate_filter

                invalid_count = 0
                for f in regex_filters:
                    is_valid, error_msg = _validate_filter(
                        {"column": f.column, "operator": f.operator, "value": f.value}, self.semantic_layer
                    )
                    if is_valid:
                        valid_regex_filters.append(f)
                    else:
                        invalid_count += 1
                        logger.debug("regex_filter_validation_failed", filter=f, error=error_msg)

                if invalid_count > 0:
                    intent.confidence = max(0.6, intent.confidence - 0.1 * invalid_count)
                    logger.debug(
                        "regex_filters_invalidated",
                        query=query,
                        invalid_count=invalid_count,
                        valid_count=len(valid_regex_filters),
                        confidence=intent.confidence,
                    )

            # Merge regex filters with existing intent filters (e.g., from LLM parse)
            # Avoid duplicates
            if valid_regex_filters:
                existing_filter_keys = {
                    (f.column, f.operator, str(f.value) if not isinstance(f.value, list) else tuple(sorted(f.value)))
                    for f in intent.filters
                }

                for rf in valid_regex_filters:
                    filter_key = (
                        rf.column,
                        rf.operator,
                        str(rf.value) if not isinstance(rf.value, list) else tuple(sorted(rf.value)),
                    )

                    if filter_key not in existing_filter_keys:
                        intent.filters.append(rf)
                        existing_filter_keys.add(filter_key)

            if intent.filters:
                logger.debug(
                    "filters_extracted_in_parse",
                    query=query,
                    filter_count=len(intent.filters),
                    intent_type=intent.intent_type,
                )

            # Extract grouping variable from compound queries (e.g., "which X was most Y")
            # This handles queries like "how many patients were on statins and which statin was most prescribed?"
            if intent.intent_type == "COUNT" and not intent.grouping_variable:
                grouping_var = self._extract_grouping_from_compound_query(query)
                if grouping_var:
                    intent.grouping_variable = grouping_var
                    logger.info(
                        "grouping_extracted_from_compound",
                        query=query,
                        grouping_variable=grouping_var,
                        intent_type=intent.intent_type,
                    )
                else:
                    logger.debug(
                        "grouping_extraction_failed",
                        query=query,
                        intent_type=intent.intent_type,
                    )

        return intent

    def _get_matched_variables(self, intent: QueryIntent) -> list[str]:
        """Extract matched variables from intent for logging."""
        vars_list = []
        if intent.primary_variable:
            vars_list.append(intent.primary_variable)
        if intent.grouping_variable:
            vars_list.append(intent.grouping_variable)
        if intent.predictor_variables:
            vars_list.extend(intent.predictor_variables)
        if intent.time_variable:
            vars_list.append(intent.time_variable)
        if intent.event_variable:
            vars_list.append(intent.event_variable)
        return vars_list

    def _pattern_match(self, query: str) -> QueryIntent | None:
        """
        Tier 1: Regex pattern matching for common queries.

        Args:
            query: User's question

        Returns:
            QueryIntent if pattern matches, None otherwise
        """
        query_lower = query.lower()

        # Pattern: "compare X by Y" or "compare X between Y"
        match = re.search(r"compare\s+(\w+)\s+(?:by|between|across)\s+(\w+)", query_lower)
        if match:
            primary_var, _, _ = self._fuzzy_match_variable(match.group(1))
            group_var, _, _ = self._fuzzy_match_variable(match.group(2))

            if primary_var and group_var:
                return QueryIntent(
                    intent_type="COMPARE_GROUPS",
                    primary_variable=primary_var,
                    grouping_variable=group_var,
                    confidence=0.95,
                )

        # Pattern: "what predicts X" or "predictors of X"
        match = re.search(r"(?:what predicts|predictors of|predict|risk factors for)\s+(\w+)", query_lower)
        if match:
            outcome_var, _, _ = self._fuzzy_match_variable(match.group(1))

            if outcome_var:
                return QueryIntent(intent_type="FIND_PREDICTORS", primary_variable=outcome_var, confidence=0.95)

        # Pattern: "survival" or "time to event"
        if re.search(r"\b(survival|time to event|kaplan|cox)\b", query_lower):
            return QueryIntent(intent_type="SURVIVAL", confidence=0.9)

        # Pattern: "correlation" or "relationship" or "relate" or "association"
        # Matches: "correlate", "correlation", "relationship", "relate", "relates", "associated", "association"
        if re.search(r"\b(correlat|relationship|relate|associat)\b", query_lower):
            # Try to extract variables from query
            variables, _ = self._extract_variables_from_query(query)
            if len(variables) >= 2:
                # For CORRELATIONS with multiple variables, put ALL in predictor_variables
                return QueryIntent(
                    intent_type="CORRELATIONS",
                    primary_variable=variables[0],  # First variable for backward compatibility
                    grouping_variable=variables[1],  # Second for backward compatibility
                    predictor_variables=variables,  # ALL variables for correlation analysis
                    confidence=0.9,
                )
            elif len(variables) == 1:
                # Single variable found - still CORRELATIONS but with one variable
                return QueryIntent(
                    intent_type="CORRELATIONS",
                    primary_variable=variables[0],
                    predictor_variables=variables,  # Include in predictor_variables too
                    confidence=0.85,
                )
            else:
                # No variables extracted but relationship keyword found
                return QueryIntent(intent_type="CORRELATIONS", confidence=0.85)

        # Pattern: "how many" or "count" or "number of" (COUNT intent)
        count_patterns = [
            r"how many",
            r"\bcount\b",
            r"number of",
        ]
        if any(re.search(pattern, query_lower) for pattern in count_patterns):
            return QueryIntent(intent_type="COUNT", confidence=0.9)

        # Pattern: "what X were/was Y on" - COUNT with grouping
        # Examples: "what statins were patients on?", "what treatments were they on?"
        # This asks for a breakdown/distribution, so it's a COUNT intent
        what_were_on = re.search(r"what\s+(\w+(?:\s+\w+)?)\s+(?:were|was)\s+(?:\w+\s+)?on", query_lower)
        if what_were_on:
            variable_term = what_were_on.group(1).strip()
            matched_var, _, _ = self._fuzzy_match_variable(variable_term)
            if matched_var:
                return QueryIntent(
                    intent_type="COUNT",
                    grouping_variable=matched_var,
                    confidence=0.9,
                )

        # Pattern: "which X was most Y" or "what was the most Y" - COUNT with grouping
        # This pattern asks for the top result by count, so it's a COUNT intent with grouping
        # More flexible pattern to handle "which was the most Y", "what was the most Y",
        # "what was the most common X", and "excluding X, which was the most Y"
        if (
            re.search(r"which\s+(?:\w+\s+)?(?:was|is)\s+the?\s+most\s+\w+", query_lower)
            or re.search(r"which\s+\w+(?:\s+\w+)*?\s+was\s+most\s+\w+", query_lower)
            or re.search(
                r"what\s+was\s+the\s+most\s+(?:common|prescribed|frequent)\s+\w+", query_lower
            )  # "what was the most common X"
            or re.search(r"what\s+was\s+the\s+most\s+\w+", query_lower)  # "what was the most X" (fallback)
        ):
            return QueryIntent(intent_type="COUNT", confidence=0.9)

        # Pattern: "what is the average/mean X" - DESCRIBE with variable extraction
        # Examples: "what is the average age?", "what is the mean BMI?", "what is the mean and median age?"
        what_is_match = re.search(
            r"what\s+is\s+the\s+(?:(?:average|mean|median)(?:\s+and\s+(?:average|mean|median))*\s+)(\w+(?:\s+\w+)*?)(?:\?|$)",
            query_lower,
        )
        if what_is_match:
            variable_term = what_is_match.group(1).strip()
            # Remove common trailing words
            variable_term = re.sub(r"\s+(patients|subjects|individuals|people|cases|all|the)$", "", variable_term)
            variable_term = variable_term.strip()

            # Try to match the variable
            matched_var, var_conf, _ = self._fuzzy_match_variable(variable_term)
            if matched_var:
                logger.debug(
                    "pattern_match_what_is_average",
                    variable_term=variable_term,
                    matched_var=matched_var,
                    confidence=var_conf,
                )
                return QueryIntent(
                    intent_type="DESCRIBE",
                    primary_variable=matched_var,
                    confidence=0.9,
                )

        # Pattern: "average X" or "mean X" or "avg X" - DESCRIBE with variable extraction
        # Examples: "average BMI of patients", "mean age", "avg ldl", "average ldl of all patients"
        # Match: "average/mean/avg" + optional "of" + variable + stop at grouping keywords like "by"
        avg_match = re.search(
            r"\b(average|mean|avg)\s+(?:of\s+)?(\w+(?:\s+\w+)*?)(?:\s+of|\s+in|\s+for|\s+by|\s+across|\s+between|\s+all|\s+the|$)",
            query_lower,
        )
        if avg_match:
            variable_term = avg_match.group(2).strip()
            # Remove common trailing words that might be captured
            variable_term = re.sub(r"\s+(patients|subjects|individuals|people|cases|all|the)$", "", variable_term)
            variable_term = variable_term.strip()

            # Try to match the variable
            matched_var, var_conf, _ = self._fuzzy_match_variable(variable_term)
            if matched_var:
                # Check for grouping pattern "by X" or "across X"
                grouping_match = re.search(r"(?:by|across)\s+(\w+(?:\s+\w+)?)", query_lower)
                group_var = None
                if grouping_match:
                    group_term = grouping_match.group(1).strip()
                    group_var, _, _ = self._fuzzy_match_variable(group_term)

                logger.debug(
                    "pattern_match_average_with_variable",
                    variable_term=variable_term,
                    matched_var=matched_var,
                    confidence=var_conf,
                    grouping=group_var,
                )
                return QueryIntent(
                    intent_type="DESCRIBE",
                    primary_variable=matched_var,
                    grouping_variable=group_var,
                    confidence=0.9,
                )
            else:
                # Still return DESCRIBE intent, variable will be extracted later
                logger.debug(
                    "pattern_match_average_no_variable_match",
                    variable_term=variable_term,
                    reason="fuzzy_match_failed_but_pattern_matched",
                )
                return QueryIntent(intent_type="DESCRIBE", confidence=0.85)

        # Pattern: "describe X" or "summary of X" - DESCRIBE with variable extraction
        describe_match = re.search(
            r"\b(describe|summarize?|overview of)\s+(\w+(?:\s+\w+)*?)"
            r"(?:\s+statistics|\s+levels|\s+values|\s+distribution|\s+for|\s+by|\s+across|$)",
            query_lower,
        )
        if describe_match:
            variable_term = describe_match.group(2).strip()
            # Remove common trailing words
            variable_term = re.sub(r"\s+(patients|subjects|individuals|people|cases)$", "", variable_term)
            variable_term = variable_term.strip()

            # Try to match the variable
            matched_var, var_conf, _ = self._fuzzy_match_variable(variable_term)
            if matched_var:
                logger.debug(
                    "pattern_match_describe_with_variable",
                    variable_term=variable_term,
                    matched_var=matched_var,
                    confidence=var_conf,
                )
                return QueryIntent(
                    intent_type="DESCRIBE",
                    primary_variable=matched_var,
                    confidence=0.9,
                )

        # Pattern: "describe" or "summary" (no variable extracted)
        if re.search(r"\b(describe|summary|overview|statistics)\b", query_lower):
            return QueryIntent(intent_type="DESCRIBE", confidence=0.9)

        # Pattern: "compare X across/between Y" - COMPARE_GROUPS
        # Examples: "compare age across different statuses", "compare LDL between treatment groups"
        compare_match = re.search(
            r"\bcompare\s+(\w+(?:\s+\w+)*?)\s+(?:across|between)\s+(?:different\s+)?(\w+(?:\s+\w+)*?)(?:\s+and|$)",
            query_lower,
        )
        if compare_match:
            primary_term = compare_match.group(1).strip()
            group_term = compare_match.group(2).strip()

            # Remove common trailing words
            group_term = re.sub(r"\s+(groups?|categories|types?)$", "", group_term)
            group_term = group_term.strip()

            primary_var, _, _ = self._fuzzy_match_variable(primary_term)
            group_var, _, _ = self._fuzzy_match_variable(group_term)

            if primary_var and group_var:
                logger.debug(
                    "pattern_match_compare_across",
                    primary_term=primary_term,
                    group_term=group_term,
                    primary_var=primary_var,
                    group_var=group_var,
                )
                return QueryIntent(
                    intent_type="COMPARE_GROUPS",
                    primary_variable=primary_var,
                    grouping_variable=group_var,
                    confidence=0.95,
                )

        # Pattern: "difference" implies comparison
        match = re.search(r"difference\s+(?:in|of)\s+(\w+)\s+(?:by|between)\s+(\w+)", query_lower)
        if match:
            primary_var, _, _ = self._fuzzy_match_variable(match.group(1))
            group_var, _, _ = self._fuzzy_match_variable(match.group(2))

            if primary_var and group_var:
                return QueryIntent(
                    intent_type="COMPARE_GROUPS",
                    primary_variable=primary_var,
                    grouping_variable=group_var,
                    confidence=0.95,
                )

        # Pattern: "which X had the lowest/highest Y" or "what X had the lowest/highest Y"
        match = re.search(
            r"(?:which|what)\s+(\w+(?:\s+\w+)*?)\s+had\s+the\s+(lowest|highest)\s+(\w+(?:\s+\w+)*)", query_lower
        )
        if match:
            group_term = match.group(1).strip()
            primary_term = match.group(3).strip()

            group_var, group_conf, _ = self._fuzzy_match_variable(group_term)
            primary_var, primary_conf, _ = self._fuzzy_match_variable(primary_term)

            # Log for debugging
            logger.debug(
                "pattern_match_which_x_had_y",
                group_term=group_term,
                primary_term=primary_term,
                group_var=group_var,
                primary_var=primary_var,
                group_conf=group_conf,
                primary_conf=primary_conf,
            )

            if group_var and primary_var:
                return QueryIntent(
                    intent_type="COMPARE_GROUPS",
                    primary_variable=primary_var,
                    grouping_variable=group_var,
                    confidence=0.95,
                )
            # If fuzzy matching failed, still return COMPARE_GROUPS but with lower confidence
            # Variables will be extracted later via _extract_variables_from_query
            elif group_term or primary_term:
                logger.info(
                    "pattern_match_partial",
                    group_term=group_term,
                    primary_term=primary_term,
                    reason="fuzzy_match_failed_but_terms_extracted",
                )
                # Return COMPARE_GROUPS intent - variables will be filled by semantic extraction
                return QueryIntent(
                    intent_type="COMPARE_GROUPS",
                    confidence=0.85,  # Lower confidence since variables not matched yet
                )

        return None

    def _semantic_match(self, query: str) -> QueryIntent | None:
        """
        Tier 2: Semantic embedding similarity matching.

        Args:
            query: User's question

        Returns:
            QueryIntent if good match found, None otherwise
        """
        try:
            # Lazy load encoder
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer(self.embedding_model_name)

            # Lazy compute template embeddings
            if self.template_embeddings is None:
                template_texts = [t["template"] for t in self.query_templates]
                self.template_embeddings = self.encoder.encode(template_texts)

            # Encode query
            query_embedding = self.encoder.encode([query])

            # Compute similarity with all templates
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(query_embedding, self.template_embeddings)[0]

            # Get best match
            best_idx = similarities.argmax()
            best_score = similarities[best_idx]

            if best_score > 0.7:  # Threshold
                best_template = self.query_templates[best_idx]

                # Extract slot values using fuzzy matching
                intent = QueryIntent(intent_type=best_template["intent"], confidence=float(best_score))

                # Extract variables mentioned in query
                variables = self._extract_variables_from_query(query)

                # Assign variables to slots based on template
                if "outcome" in best_template["slots"] and variables:
                    intent.primary_variable = variables[0]
                if "group" in best_template["slots"] and len(variables) > 1:
                    intent.grouping_variable = variables[1]
                if "var1" in best_template["slots"] and variables:
                    intent.primary_variable = variables[0]
                if "var2" in best_template["slots"] and len(variables) > 1:
                    intent.grouping_variable = variables[1]

                return intent

        except Exception as e:
            # If sentence-transformers fails, fall through to Tier 3
            logger.warning(
                "semantic_match_failed",
                error_type="semantic_matching_exception",
                error=str(e),
                query=query,
            )
            pass

        return None

    def _get_ollama_client(self):
        """Get or create Ollama client via OllamaManager (lazy initialization)."""
        if not hasattr(self, "_ollama_client"):
            from clinical_analytics.core.ollama_manager import get_ollama_manager

            manager = get_ollama_manager()
            self._ollama_client = manager.get_client()

        return self._ollama_client

    def _build_rag_context(self, query: str) -> dict:
        """
        Build RAG context from semantic layer metadata and golden questions.

        Uses golden questions as RAG corpus - retrieves similar examples
        to help LLM pattern-match instead of hallucinate.

        Args:
            query: User's question

        Returns:
            Dict with columns, aliases, examples, and query
        """
        # Extract columns from semantic layer base view
        base_view = self.semantic_layer.get_base_view()
        columns = list(base_view.columns)

        # Get alias mappings
        alias_index = self.semantic_layer.get_column_alias_index()

        # RAG: Load golden questions as corpus
        golden_examples = self._load_golden_questions_rag()

        # Find top 3 most similar examples
        relevant_examples = self._find_similar_examples(query, golden_examples, top_k=3)

        return {
            "columns": columns,
            "aliases": alias_index,
            "examples": relevant_examples,
            "query": query,
        }

    def _load_golden_questions_rag(self) -> list[dict]:
        """Load golden questions for RAG retrieval."""
        try:
            from pathlib import Path

            import yaml

            golden_path = Path(__file__).parent.parent.parent / "tests" / "eval" / "golden_questions.yaml"
            if not golden_path.exists():
                return []

            with open(golden_path) as f:
                data = yaml.safe_load(f)
            return data.get("golden_questions", [])
        except Exception as e:
            logger.warning("failed_to_load_golden_questions_for_rag", error=str(e))
            return []

    def _find_similar_examples(self, query: str, examples: list[dict], top_k: int = 3) -> list[str]:
        """Find most similar examples using keyword matching."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score each example by keyword overlap
        scored = []
        for ex in examples:
            ex_query = ex.get("query", "").lower()
            ex_words = set(ex_query.split())

            # Jaccard similarity
            overlap = len(query_words & ex_words)
            union = len(query_words | ex_words)
            score = overlap / union if union > 0 else 0

            # Boost if refinement phrases match
            refinement_phrases = ["remove", "exclude", "without", "only", "get rid of"]
            if any(phrase in query_lower for phrase in refinement_phrases):
                if any(phrase in ex_query for phrase in refinement_phrases):
                    score += 0.5

            scored.append((score, ex))

        # Sort by score and take top k
        scored.sort(reverse=True, key=lambda x: x[0])
        top_examples = scored[:top_k]

        # Format as examples
        formatted = []
        for score, ex in top_examples:
            intent = ex.get("expected_intent", "DESCRIBE")
            metric = ex.get("expected_metric")
            group_by = ex.get("expected_group_by")

            example_text = f'Q: "{ex.get("query", "")}"\n'
            example_text += f"   Intent: {intent}"
            if metric:
                example_text += f", Metric: {metric}"
            if group_by:
                example_text += f", GroupBy: {group_by}"

            formatted.append(example_text)

        return (
            formatted
            if formatted
            else [
                'Q: "What is the average age?"\n   Intent: DESCRIBE, Metric: age',
                'Q: "How many patients?"\n   Intent: COUNT',
                'Q: "Compare by treatment"\n   Intent: COMPARE_GROUPS, GroupBy: treatment',
            ]
        )

    def _build_llm_prompt(
        self,
        query: str,
        context: dict,
        conversation_history: list[dict] | None = None,
    ) -> tuple[str, str]:
        """
        Build structured prompts for LLM with conversation context.

        Phase 5.1: Request QueryPlan JSON schema instead of legacy QueryIntent.
        Phase 6: Support conversational refinements (ADR009).

        Args:
            query: User's question
            context: RAG context with columns, aliases, examples
            conversation_history: Optional list of previous queries for refinement detection

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Build conversation context section if history provided
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get most recent query (limit to last 1 for simplicity)
            recent = conversation_history[-1]
            conversation_context = f"""

=== CONVERSATION CONTEXT (CRITICAL FOR REFINEMENTS) ===

Previous Query: "{recent.get("query", "N/A")}"
Previous Intent: {recent.get("intent", "N/A")}
Previous Group By: {recent.get("group_by", "null")}
Previous Metric: {recent.get("metric", "null")}
Previous Filters: {recent.get("filters_applied", [])}

=== REFINEMENT DETECTION RULES ===

**IS THIS A REFINEMENT?** Check if current query modifies the previous query:
- Refinement phrases: "remove", "exclude", "without", "only", "just", "also",
  "actually", "get rid of", "drop"
- Examples: "remove the n/a", "exclude missing", "only active", "actually over 65"

**IF YES - THIS IS A REFINEMENT:**
1. **COPY the previous intent EXACTLY** - Do NOT invent new intents
   - If previous was COUNT → use COUNT
   - If previous was DESCRIBE → use DESCRIBE
   - If previous was COMPARE_GROUPS → use COMPARE_GROUPS
2. **COPY previous group_by and metric** - Preserve what user was analyzing
3. **ADD/UPDATE filters only** - This is what's being refined
4. **Set confidence >= 0.7** - You have clear context
5. **In explanation**: Say "Refining previous query to [what changed]"

**IF NO - THIS IS A NEW QUERY:**
- Parse independently
- Do NOT use previous intent/group_by/metric
- Set confidence based on query clarity

=== REFINEMENT EXAMPLES (COPY THESE PATTERNS) ===

Example 1:
Previous: {{"intent": "COUNT", "group_by": "statin", "metric": null}}
Current: "remove the n/a"
Result: {{"intent": "COUNT", "group_by": "statin", "filters": [{{"column": "statin", "operator": "!=", "value": 0}}]}}

Example 2:
Previous: {{"intent": "DESCRIBE", "metric": "cholesterol", "group_by": null}}
Current: "exclude missing values"
Result: {{"intent": "DESCRIBE", "metric": "cholesterol",
  "filters": [{{"column": "cholesterol", "operator": "!=", "value": 0}}]}}

Example 3:
Previous: {{"intent": "COUNT", "filters": [{{"column": "age", "operator": ">", "value": 50}}]}}
Current: "actually over 65"
Result: {{"intent": "COUNT", "filters": [{{"column": "age", "operator": ">", "value": 65}}]}}

**REMEMBER**: NEVER create intents like "REMOVE_NA", "FILTER_OUT", "EXCLUDE" - these are INVALID.
Only use: COUNT, DESCRIBE, COMPARE_GROUPS, FIND_PREDICTORS, CORRELATIONS"""

        system_prompt = """You are a medical data query parser. Extract structured query intent from natural language.

Return JSON matching the QueryPlan schema with these REQUIRED fields:
- intent: One of ["COUNT", "DESCRIBE", "COMPARE_GROUPS", "FIND_PREDICTORS", "CORRELATIONS"]
- metric: Main variable to analyze (string or null)
- group_by: Variable to group by (string or null)
- filters: List of filter objects (empty list if none)
- confidence: Your confidence 0.0-1.0
- explanation: Brief explanation of what the query asks for

OPTIONAL fields for enhanced UX (ADR009):
- follow_ups: Array of 2-3 context-aware follow-up questions (as suggestions, not endorsements)
- follow_up_explanation: Brief explanation of why these follow-ups are relevant
- interpretation: Human-readable explanation of what the query is asking (helps user understand parsing)
- confidence_explanation: Brief explanation of why the confidence score is what it is

CRITICAL: Your JSON response must be FLAT with these fields at the TOP LEVEL.
NEVER create nested objects like {{ "query": {{ ... }} }} or {{ "action": {{ ... }} }}.

INVALID EXAMPLES (DO NOT DO THIS):
❌ {{ "query": {{ "remove": ["n/a"], "recalc": true }} }}
❌ {{ "action": {{ "type": "exclude", "value": "n/a" }} }}
❌ {{ "refinement": {{ "previous": "...", "change": "..." }} }}

VALID EXAMPLE:
✅ {{
  "intent": "COUNT",
  "metric": null,
  "group_by": "statin_used",
  "filters": [{{"column": "statin_used", "operator": "!=", "value": 0}}],
  "confidence": 0.85,
  "explanation": "Count by statin type, excluding n/a"
}}

Available columns: {columns}
Aliases: {aliases}

Examples:
{examples}

Follow-up generation guidelines:
- Provide 2-3 exploratory questions that build on the current query
- Questions should be helpful suggestions, not authoritative recommendations
- Avoid clinical advice territory - focus on data exploration
- Examples: "What predicts X?", "Compare by Y group", "Are there outliers?"

IMPORTANT: Use exact field names from QueryPlan schema (intent, metric, group_by),
not legacy names (intent_type, primary_variable, grouping_variable).

Filter extraction (ADR009 Phase 5):
- Extract filter conditions from queries like "get rid of the n/a", "exclude missing", "remove 0"
- For exclusion patterns ("get rid of", "exclude", "remove"), use operator "!=" with value 0 (n/a code)
- For coded columns, use numeric codes (0=n/a, 1=first value, etc.)
- Examples:
  * "get rid of the n/a" → {{"column": "treatment_group", "operator": "!=", "value": 0}}
  * "exclude missing values" → {{"column": "treatment_group", "operator": "!=", "value": 0}}
  * "patients on statins" → {{
      "column": "statin_prescribed", "operator": "==", "value": 1
    }}{conversation_context}""".format(
            columns=", ".join(context["columns"]),
            aliases=str(context["aliases"]),
            examples="\n".join(f"- {ex}" for ex in context["examples"]),
            conversation_context=conversation_context,
        )

        user_prompt = f"Parse this query: {query}"

        # Load and append overlay (auto-generated fixes from self-improvement)
        overlay = self._load_prompt_overlay()
        if overlay:
            system_prompt = system_prompt + "\n\n" + overlay

        return (system_prompt, user_prompt)

    def _extract_query_intent_from_llm_response(self, response: str, max_retries: int = 3) -> QueryIntent | None:
        """
        Extract QueryIntent from LLM JSON response with retries.

        Phase 5.1: Parse QueryPlan JSON schema and validate using QueryPlan.from_dict()

        Args:
            response: JSON string from LLM
            max_retries: Maximum retry attempts (not used in this version)

        Returns:
            QueryIntent if valid, None if parsing fails
        """

        try:
            data = json.loads(response)

            # Phase 5.1: Validate using QueryPlan.from_dict() (raises on invalid schema)
            try:
                query_plan = QueryPlan.from_dict(data)

                # Convert validated QueryPlan back to QueryIntent for backward compatibility
                # ADR009 Phase 1: Preserve follow_ups fields
                # ADR009 Phase 2: Preserve interpretation fields
                return QueryIntent(
                    intent_type=query_plan.intent,  # type: ignore[arg-type]
                    primary_variable=query_plan.metric,
                    grouping_variable=query_plan.group_by,
                    confidence=query_plan.confidence,
                    parsing_tier="llm_fallback",
                    filters=query_plan.filters,  # Preserve validated filters
                    follow_ups=query_plan.follow_ups,  # Preserve LLM-generated follow-ups
                    follow_up_explanation=query_plan.follow_up_explanation,
                    interpretation=query_plan.interpretation,  # Preserve LLM-generated interpretation
                    confidence_explanation=query_plan.confidence_explanation,  # Preserve confidence explanation
                )

            except (ValueError, KeyError) as validation_error:
                # If QueryPlan validation fails, try legacy QueryIntent format for backward compatibility
                logger.warning(
                    "llm_queryplan_validation_failed_trying_legacy_format",
                    error=str(validation_error),
                    response=response[:100],
                )

                # Legacy format fallback
                if "intent_type" not in data:
                    logger.warning("llm_response_missing_required_fields", response=response[:100])
                    return None

                # Extract fields with defaults (legacy format)
                intent_type = data.get("intent_type", "DESCRIBE")
                primary_variable = data.get("primary_variable")
                grouping_variable = data.get("grouping_variable")
                confidence = float(data.get("confidence", 0.5))

                # Clamp confidence to valid range
                confidence = max(0.0, min(1.0, confidence))

                return QueryIntent(
                    intent_type=intent_type,
                    primary_variable=primary_variable,
                    grouping_variable=grouping_variable,
                    confidence=confidence,
                    parsing_tier="llm_fallback",
                )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(
                "llm_response_parse_failed",
                error=str(e),
                response=response[:100],
            )
            return None

    def _llm_parse(self, query: str, conversation_history: list[dict] | None = None) -> QueryIntent:
        """
        Tier 3: LLM fallback with RAG context from semantic layer.

        This is the blast shield - catches all exceptions and always returns a QueryIntent.
        Never crashes, always provides a fallback (confidence=0.3 stub).

        Privacy-preserving: Uses local Ollama only, no external API calls.

        Supports conversational refinements (ADR009 Phase 6): When conversation_history
        is provided, LLM can detect refinement queries and merge with previous context.

        Args:
            query: User's question
            conversation_history: Optional list of previous queries for context

        Returns:
            QueryIntent with confidence >= 0.5 on success, or 0.3 stub on failure
        """
        from clinical_analytics.core.nl_query_config import TIER_3_MIN_CONFIDENCE

        try:
            # Step 1: Get Ollama client (lazy init)
            client = self._get_ollama_client()

            # Step 2: Check if Ollama is available
            if not client.is_available():
                logger.info("ollama_not_available_fallback_to_stub", query=query)
                return QueryIntent(intent_type="DESCRIBE", confidence=0.3, parsing_tier="llm_fallback")

            # Step 3: Build RAG context
            context = self._build_rag_context(query)

            # Step 4: Build structured prompts
            system_prompt, user_prompt = self._build_llm_prompt(query, context, conversation_history)

            # Step 5: Call Ollama with JSON mode
            response = client.generate(user_prompt, system_prompt=system_prompt, json_mode=True)

            if response is None:
                logger.info("ollama_generate_failed_fallback_to_stub", query=query)
                return QueryIntent(intent_type="DESCRIBE", confidence=0.3, parsing_tier="llm_fallback")

            # Step 6: Extract QueryIntent from response
            intent = self._extract_query_intent_from_llm_response(response)

            if intent is None:
                logger.info("llm_parse_extraction_failed_fallback_to_stub", query=query)
                return QueryIntent(intent_type="DESCRIBE", confidence=0.3, parsing_tier="llm_fallback")

            # ADR009 Phase 5: Extract filters using LLM (for complex patterns)
            from clinical_analytics.core.filter_extraction import _extract_filters_with_llm

            llm_filters, confidence_delta, validation_failures = _extract_filters_with_llm(
                query, self.semantic_layer, current_confidence=intent.confidence
            )

            # Merge LLM-extracted filters with filters from main parse
            # Deduplicate: if same column+operator+value exists, keep only one
            existing_filter_keys = {
                (f.column, f.operator, str(f.value) if not isinstance(f.value, list) else tuple(sorted(f.value)))
                for f in intent.filters
            }
            for llm_filter in llm_filters:
                filter_key = (
                    llm_filter.column,
                    llm_filter.operator,
                    str(llm_filter.value)
                    if not isinstance(llm_filter.value, list)
                    else tuple(sorted(llm_filter.value)),
                )
                if filter_key not in existing_filter_keys:
                    intent.filters.append(llm_filter)
                    existing_filter_keys.add(filter_key)

            # Update confidence based on filter validation results
            if confidence_delta < 0:
                intent.confidence = max(0.6, intent.confidence + confidence_delta)  # Cap at 0.6 minimum
                if validation_failures:
                    # Add validation failures to confidence explanation
                    if intent.confidence_explanation:
                        intent.confidence_explanation += (
                            f" Filter validation issues: {len(validation_failures)} invalid filter(s)."
                        )
                    else:
                        intent.confidence_explanation = (
                            f"Filter validation issues: {len(validation_failures)} invalid filter(s)."
                        )

            # Step 7: Validate confidence meets minimum threshold
            if intent.confidence < TIER_3_MIN_CONFIDENCE:
                logger.info(
                    "llm_parse_low_confidence_fallback_to_stub",
                    confidence=intent.confidence,
                    threshold=TIER_3_MIN_CONFIDENCE,
                    query=query,
                )
                return QueryIntent(intent_type="DESCRIBE", confidence=0.3, parsing_tier="llm_fallback")

            # Success!
            logger.info(
                "llm_parse_success",
                intent_type=intent.intent_type,
                confidence=intent.confidence,
                query=query,
            )
            return intent

        except Exception as e:
            # Blast shield: catch everything, log, return stub
            # This is the only place where broad exception catching is correct
            logger.warning(
                "llm_parse_exception_fallback_to_stub",
                error_type=type(e).__name__,
                error=str(e),
                query=query,
            )
            return QueryIntent(intent_type="DESCRIBE", confidence=0.3, parsing_tier="llm_fallback")

    def _extract_variables_from_query(self, query: str) -> tuple[list[str], dict[str, list[str]]]:
        """
        Extract variables with collision suggestions using n-gram matching.

        Args:
            query: User's question

        Returns:
            Tuple of (matched_variables, collision_suggestions)
            - matched_variables: List of matched column names
            - collision_suggestions: Dict mapping query_term -> list of canonical_names

        Example:
            >>> vars, suggestions = engine._extract_variables_from_query("compare dexa scan by treatment")
            >>> vars
            ['dexa_scan_result', 'treatment_arm']
            >>> suggestions
            {'dexa': ['dexa_scan_result', 'dexa_bone_density']}  # If collision detected
        """
        words = query.lower().split()
        matched_vars = []
        collision_suggestions: dict[str, list[str]] = {}

        # Try n-grams (3-word, 2-word, 1-word) in order
        for n in [3, 2, 1]:
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])

                # Skip if phrase contains _USED_ marker (already matched)
                if "_USED_" in phrase:
                    continue

                # Match phrase (returns canonical_name, confidence, suggestions)
                matched, conf, suggestions = self._fuzzy_match_variable(phrase)

                if matched and matched not in matched_vars:
                    matched_vars.append(matched)
                    # Mark words as used
                    words[i : i + n] = ["_USED_"] * n

                # Store collision suggestions
                if suggestions:
                    collision_suggestions[phrase] = suggestions

        return matched_vars, collision_suggestions

    def _fuzzy_match_variable(self, query_term: str) -> tuple[str | None, float, list[str] | None]:
        """
        Match variable with collision awareness.

        Args:
            query_term: Single term or phrase (not entire query)

        Returns:
            Tuple of (matched_canonical_name, confidence, suggestions)
            - matched_canonical_name: Canonical column name if matched, None otherwise
            - confidence: Confidence score 0.0-1.0
            - suggestions: List of canonical names if collision detected, None otherwise
        """
        # Get alias index from semantic layer
        alias_index = self.semantic_layer.get_column_alias_index()

        # Try multiple normalization strategies for better matching
        # Strategy 1: Standard normalization (spaces preserved)
        normalized_query = self.semantic_layer._normalize_alias(query_term)

        # Strategy 2: Replace spaces with underscores (common in aliases)
        normalized_query_underscore = normalized_query.replace(" ", "_")

        # Strategy 3: Just lowercase (for exact matches)
        normalized_query_lower = query_term.lower().strip()

        # Check if this alias was dropped due to collision
        suggestions = self.semantic_layer.get_collision_suggestions(query_term)
        if suggestions:
            # Collision detected - return suggestions
            return None, 0.2, suggestions

        # Try direct matches with all normalization strategies
        for norm_query in [normalized_query, normalized_query_underscore, normalized_query_lower]:
            if norm_query in alias_index:
                collisions = self.semantic_layer.get_collision_warnings()
                if norm_query in collisions:
                    # Collision warning (shouldn't happen if we dropped it, but check anyway)
                    return alias_index[norm_query], 0.4, None
                return alias_index[norm_query], 0.9, None

        # Fuzzy match using difflib - try all normalization strategies
        for norm_query in [normalized_query, normalized_query_underscore, normalized_query_lower]:
            matches = get_close_matches(
                norm_query,
                alias_index.keys(),
                n=1,
                cutoff=0.6,  # Lower cutoff for better matching
            )

            if matches:
                matched_alias = matches[0]
                collisions = self.semantic_layer.get_collision_warnings()
                if matched_alias in collisions:
                    return alias_index[matched_alias], 0.4, None
                return alias_index[matched_alias], 0.7, None

        # Last resort: Try substring matching on canonical names
        # This helps with columns like "Current Regimen     1: Biktarvy..." matching "current regimen"
        query_lower = query_term.lower().strip()
        for alias, canonical in alias_index.items():
            # Check if query is a substring of the alias or vice versa
            if query_lower in alias.lower() or alias.lower() in query_lower:
                # Prefer longer matches
                if len(query_lower) >= 3 and len(alias.lower()) >= 3:
                    collisions = self.semantic_layer.get_collision_warnings()
                    if alias in collisions:
                        return canonical, 0.5, None
                    return canonical, 0.6, None

        return None, 0.0, None

    def _is_coded_column(self, column_name: str, alias_name: str | None = None) -> bool:
        """
        Determine if a column is coded (numeric values with labels in alias).

        Uses metadata when available (more reliable), falls back to alias parsing.
        This is a generic, extensible check that looks for multiple indicators:
        1. Metadata check: column has "numeric": true and categorical/type metadata
        2. Alias contains code patterns (e.g., "1: Yes 2: No", "0: n/a 1: Atorvastatin")
        3. Column name suggests coded format

        Args:
            column_name: Canonical column name
            alias_name: Optional alias name (if None, will look it up)

        Returns:
            True if column appears to be coded (numeric with label mappings)
        """
        # Strategy 1: Check metadata first (most reliable)
        # This uses the variable_types metadata we capture during upload
        column_metadata = self.semantic_layer.get_column_metadata(column_name)
        if column_metadata:
            var_type = column_metadata.get("type")
            metadata_info = column_metadata.get("metadata", {})
            is_numeric = metadata_info.get("numeric", False)

            # Coded columns are categorical/binary with numeric values
            if var_type in ("categorical", "binary") and is_numeric:
                return True

        # Strategy 2: Fall back to alias parsing (for datasets without metadata)
        # Get alias if not provided
        if alias_name is None:
            alias_index = self.semantic_layer.get_column_alias_index()
            for alias, canonical in alias_index.items():
                if canonical == column_name:
                    alias_name = alias
                    break

        if not alias_name:
            return False

        # Indicator 1: Contains code pattern (digits followed by colon and label)
        # Pattern: "1: Yes", "0: n/a 1: Atorvastatin", etc.
        has_code_pattern = ":" in alias_name and any(char.isdigit() for char in alias_name)

        # Indicator 2: Contains common coded column indicators
        alias_lower = alias_name.lower()
        has_coded_indicators = (
            any(
                indicator in alias_lower
                for indicator in [
                    "prescribed",
                    "used",
                    "type",
                    "category",
                    "class",
                    "status",
                    "level",
                ]
            )
            and ":" in alias_name
        )

        return has_code_pattern or has_coded_indicators

    def _extract_filters(self, query: str, grouping_variable: str | None = None) -> list[FilterSpec]:
        """
        Extract filter conditions from query text.

        Patterns to detect:
        - "those that had X" / "patients with X" → categorical filter
        - "scores below/above X" → numeric range filter
        - "with X" / "without X" → presence filter
        - "on statins" / "were on statins" → categorical filter (value matching)
        - "don't want X" / "exclude X" → exclusion filter

        Args:
            query: User's natural language query
            grouping_variable: Optional grouping variable from query intent (for follow-up queries)

        Returns:
            List of FilterSpec objects

        Example:
            >>> filters = engine._extract_filters("how many patients were on statins")
            >>> # Returns [FilterSpec(column="Statin Used", operator="IN", value=[1,2,3,...])]
        """
        filters = []
        query_lower = query.lower()

        # Pattern 1: Categorical filters with explicit values
        # "those that had X", "patients with X", "with X", "on X"
        # "excluding those not on X" / "excluding X" - exclusion filters
        # Stop at common query continuation words: and, or, which, what, how, where, when
        categorical_patterns = [
            r"(?:those|patients|subjects|people)\s+(?:that|who)\s+(?:had|have|were|are)\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            r"(?:patients|subjects|people)\s+with\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            r"with\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            r"on\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",  # "on statins", "on treatment"
            r"were\s+on\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",  # "were on statins"
        ]

        # Pattern 1b: Exclusion filters - "excluding those not on X" or "excluding X"
        # Order matters: more specific patterns first
        # Handle commas and continuation words
        exclusion_patterns = [
            # "excluding those not on X" - most specific pattern
            # Handles: "excluding those not on X", "excluding patients who were not on X", etc.
            r"excluding\s+(?:those|patients|subjects|people)\s+(?:(?:that|who)\s+)?(?:(?:were|are)\s+)?not\s+on\s+(\w+)",
            # "excluding X" (fallback, handles commas) - must stop at continuation words
            r"excluding\s+([^,\.\?]+?)(?:\s*(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            # Phase 4.1: Add "exclude" variant (not just "excluding")
            # Handles: "exclude n/a", "exclude 0", etc.
            r"exclude\s+([^,\.\?]+?)(?:\s*(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            # Phase 4.1: Add "remove" pattern for value exclusion
            # Handles: "remove 0", "remove n/a", etc.
            r"remove\s+([^,\.\?]+?)(?:\s*(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            # Add "don't want" / "do not want" / "i don't want" patterns
            # Handles: "i don't want the 0 results", "don't want 0", "do not want n/a", etc.
            r"(?:i\s+)?(?:don'?t|do\s+not)\s+want\s+(?:the\s+)?([^,\.\?]+?)(?:\s+(?:results|values|rows|entries)?(?:\s*(?:and|or|which|what|how|where|when|,|\.|\?)|$)|$)",
        ]

        # Track which value phrases were already processed by exclusion patterns
        processed_value_phrases = set()

        # Process exclusion patterns first (they have higher priority)
        for pattern in exclusion_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                value_phrase = match.group(1).strip()

                # For "excluding those not on X", we want to exclude code 0 (n/a)
                # Find the column and create a filter to exclude 0
                # Skip if this value_phrase looks like it contains multiple words (likely wrong match)
                # Exception: "those not on X" from fallback pattern is OK if it's 3 words
                if len(value_phrase.split()) > 3:
                    continue

                column_name, conf, _ = self._fuzzy_match_variable(value_phrase)

                # Phase 4.1: Handle direct value exclusion (e.g., "exclude n/a", "remove 0")
                # If fuzzy matching fails (value_phrase is a value, not a column name),
                # try to infer the column from context
                if not column_name or conf <= 0.5:
                    # Value phrases like "n/a", "0", "na" indicate value exclusion
                    # Try to infer column from earlier "on X" pattern in query
                    inferred_column = None

                    # Extract actual value from phrases like "the n/a (0)" or "n/a (0)" or "the 0"
                    # Try to find parenthesized number or standalone number/value
                    # Strip "the" prefix if present
                    value_phrase_clean = re.sub(r"^the\s+", "", value_phrase, flags=re.IGNORECASE).strip()
                    value_str = value_phrase_clean
                    parenthesized_num = re.search(r"\((\d+)\)", value_phrase_clean)
                    if parenthesized_num:
                        # Found "(0)" or "(1)" etc. - use that as the value
                        value_str = parenthesized_num.group(1)
                    elif "n/a" in value_phrase_clean.lower() or "na" in value_phrase_clean.lower():
                        # Contains "n/a" or "na" - normalize to "n/a"
                        value_str = "n/a"
                    elif any(char.isdigit() for char in value_phrase_clean):
                        # Contains digits - extract just the digits
                        digits_match = re.search(r"(\d+)", value_phrase_clean)
                        if digits_match:
                            value_str = digits_match.group(1)

                    # Check if value_str looks like a coded value (n/a, 0, etc.)
                    is_value = (
                        value_str.lower() in ["n/a", "na", "none", "unknown"]
                        or value_str.isdigit()
                        or (value_str.startswith("0") and ":" in value_str)  # Coded value like "0: n/a"
                    )

                    if is_value:
                        inferred_column = None
                        inferred_conf = 0.0

                        # Strategy 1: Try to find "on X" pattern earlier in query to infer column
                        on_pattern = r"on\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)"
                        on_matches = list(re.finditer(on_pattern, query_lower))
                        if on_matches:
                            # Use the last "on X" match (most recent context)
                            inferred_term = on_matches[-1].group(1).strip()
                            inferred_column, inferred_conf, _ = self._fuzzy_match_variable(inferred_term)

                        # Strategy 2: If no "on X" pattern, try to find grouping variable from query
                        # Look for "by X", "per X", "broken down by X" patterns
                        if not inferred_column or inferred_conf <= 0.5:
                            grouping_patterns = [
                                r"by\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
                                r"per\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
                                r"broken\s+down\s+by\s+(?:count\s+of\s+)?(?:\w+\s+)*?per\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
                                r"broken\s+down\s+by\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
                            ]
                            for gp in grouping_patterns:
                                gp_match = re.search(gp, query_lower)
                                if gp_match:
                                    group_term = gp_match.group(1).strip()
                                    inferred_column, inferred_conf, _ = self._fuzzy_match_variable(group_term)
                                    if inferred_column and inferred_conf > 0.5:
                                        break

                        # Strategy 3: If still no column, use grouping_variable from query intent
                        # This helps with follow-up queries like "i don't want the 0 results"
                        # where the grouping variable from the previous query is available
                        if (not inferred_column or inferred_conf <= 0.5) and grouping_variable:
                            inferred_column = grouping_variable
                            inferred_conf = 0.8  # High confidence since it's from the query intent

                        # Strategy 4: If still no column, skip creating a filter
                        # This is better than creating a wrong filter
                        if not inferred_column or inferred_conf <= 0.5:
                            continue

                        if inferred_column and inferred_conf > 0.5:
                            # Map value to code
                            # For "n/a", "na", "none" → code 0
                            # For numeric strings → parse as int
                            # Use value_str (extracted value) instead of value_phrase
                            if value_str.lower() in ["n/a", "na", "none", "unknown"]:
                                exclusion_value = 0
                            elif value_str.isdigit():
                                exclusion_value = int(value_str)
                            else:
                                # Unknown value, skip
                                continue

                            # Create exclusion filter with inferred column
                            filters.append(
                                FilterSpec(
                                    column=inferred_column,
                                    operator="!=",
                                    value=exclusion_value,
                                    exclude_nulls=True,
                                )
                            )
                            processed_value_phrases.add(value_phrase)
                            continue

                if column_name and conf > 0.5:
                    # Normalize column name: if _fuzzy_match_variable returned full alias string,
                    # look up the canonical name from alias_index
                    if ":" in column_name and any(char.isdigit() for char in column_name):
                        # This looks like a full alias string, not a canonical name
                        # Look up the canonical name from alias_index
                        alias_index = self.semantic_layer.get_column_alias_index()
                        canonical_name = None
                        for alias, canonical in alias_index.items():
                            if alias == column_name or column_name in alias:
                                canonical_name = canonical
                                break
                        if canonical_name:
                            column_name = canonical_name

                    # Check if this is a coded column
                    if self._is_coded_column(column_name):
                        # For exclusion filters on coded columns, exclude code 0 (n/a)
                        filters.append(
                            FilterSpec(
                                column=column_name,
                                operator="!=",
                                value=0,
                                exclude_nulls=True,
                            )
                        )
                        # Mark this value phrase as processed
                        processed_value_phrases.add(value_phrase)
                        # Also mark any related phrases (e.g., "statins" if we matched "those not on statins")
                        if " " in value_phrase:
                            # Extract the last word (the actual value)
                            last_word = value_phrase.split()[-1]
                            processed_value_phrases.add(last_word)
                        continue
                    else:
                        # For non-coded columns, use NOT_IN or != based on context
                        # Normalize column name if needed (same as above)
                        if ":" in column_name and any(char.isdigit() for char in column_name):
                            alias_index = self.semantic_layer.get_column_alias_index()
                            canonical_name = None
                            for alias, canonical in alias_index.items():
                                if alias == column_name or column_name in alias:
                                    canonical_name = canonical
                                    break
                            if canonical_name:
                                column_name = canonical_name

                        filters.append(
                            FilterSpec(
                                column=column_name,
                                operator="!=",
                                value=value_phrase,
                                exclude_nulls=True,
                            )
                        )
                        # Mark as processed
                        processed_value_phrases.add(value_phrase)
                        if " " in value_phrase:
                            last_word = value_phrase.split()[-1]
                            processed_value_phrases.add(last_word)
                        continue

        for pattern in categorical_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                value_phrase = match.group(1).strip()

                # Skip if this value phrase was already processed by exclusion patterns
                if value_phrase in processed_value_phrases:
                    continue

                # Strategy 1: Try to match phrase as column name (for direct column filters)
                # BUT: Skip this for coded columns - we need Strategy 2 to extract proper numeric codes
                # Strategy 1 only makes sense for exact column name matches with string values
                column_name, conf, _ = self._fuzzy_match_variable(value_phrase)
                matched_column_from_strategy1 = None
                matched_alias_from_strategy1 = None

                if column_name and conf > 0.5:
                    # Check if this is a coded column using generic detection
                    if self._is_coded_column(column_name):
                        # This is a coded column - Strategy 2 will extract proper numeric codes
                        # Store the matched column for Strategy 2 to use
                        matched_column_from_strategy1 = column_name
                        # Find the alias for this column
                        # Note: _fuzzy_match_variable might return the full alias string as canonical
                        # Check if column_name itself looks like an alias (contains ":" and digits)
                        if ":" in column_name and any(char.isdigit() for char in column_name):
                            # column_name is already the full alias string - use it directly
                            matched_alias_from_strategy1 = column_name
                        else:
                            # Look up the alias from alias_index
                            alias_index = self.semantic_layer.get_column_alias_index()
                            for alias, canonical in alias_index.items():
                                if canonical == column_name:
                                    matched_alias_from_strategy1 = alias
                                    break
                            # If not found, try using column_name as alias (fallback)
                            if not matched_alias_from_strategy1:
                                matched_alias_from_strategy1 = column_name
                    else:
                        # Not a coded column - use Strategy 1 for direct column matching
                        filters.append(
                            FilterSpec(
                                column=column_name,
                                operator="==",
                                value=value_phrase,
                                exclude_nulls=True,
                            )
                        )
                        continue

                # Strategy 2: Try to find a related column (e.g., "statins" → "Statin Prescribed?")
                # Always search for matching columns and prioritize binary columns, even if Strategy 1 found something
                # This ensures we select the best match (binary > multi-value) for "on X" queries
                alias_index = self.semantic_layer.get_column_alias_index()
                value_lower = value_phrase.lower()

                # Find columns that might contain this value
                # Example: For "statins", look for columns with "statin" in the name (generic pattern matching)
                matching_columns = []
                for alias, canonical in alias_index.items():
                    alias_lower = alias.lower()
                    # Check if value phrase is related to column name
                    if value_lower in alias_lower or alias_lower in value_lower:
                        matching_columns.append((canonical, alias))
                    # Also check if they share a root word
                    value_words = set(value_lower.split())
                    alias_words = set(alias_lower.split())
                    if value_words & alias_words:  # Intersection
                        if (canonical, alias) not in matching_columns:
                            matching_columns.append((canonical, alias))

                if matching_columns:
                    # Prioritize binary yes/no columns for "on X" queries
                    # These typically have patterns like "1: Yes 2: No" or "prescribed"
                    prioritized_columns = []
                    other_columns = []

                    for canonical, alias in matching_columns:
                        alias_lower = alias.lower()
                        # Check for binary yes/no pattern indicators (higher priority)
                        # Look for: "1:" followed by "yes" or "prescribed" in the name
                        has_binary_pattern = (
                            "1:" in alias
                            and ("yes" in alias_lower or "prescribed" in alias_lower or "no" in alias_lower)
                        ) or (
                            # Alternative pattern: column name suggests binary (prescribed, yes/no, etc.)
                            any(term in alias_lower for term in ["prescribed", "yes", "no"])
                            and ":" in alias
                            and any(char.isdigit() for char in alias)
                        )

                        if has_binary_pattern:
                            prioritized_columns.append((canonical, alias))
                        else:
                            other_columns.append((canonical, alias))

                    # Use prioritized columns first, then others
                    # If Strategy 1 found a match and it's in prioritized, use it; otherwise use best match
                    if prioritized_columns:
                        column_name, alias_name = prioritized_columns[0]
                    elif matched_column_from_strategy1 and matched_alias_from_strategy1:
                        # Fallback to Strategy 1 match if no binary column found
                        column_name = matched_column_from_strategy1
                        alias_name = matched_alias_from_strategy1
                    else:
                        column_name, alias_name = matching_columns[0]
                elif matched_column_from_strategy1 and matched_alias_from_strategy1:
                    # No matching columns found, but Strategy 1 found something - use it
                    column_name = matched_column_from_strategy1
                    alias_name = matched_alias_from_strategy1
                else:
                    # No matching columns found - skip this filter
                    continue

                # For "on statins" / "were on statins", determine the filter value
                # Check if column name suggests it's a coded variable
                # (e.g., "Statin Prescribed? 1: Yes 2: No" or "Statin Used: 0: n/a 1: Atorvastatin...")
                if ":" in alias_name and any(char.isdigit() for char in alias_name):
                    import re as re_module

                    # Extract codes like "1:", "2:", etc. with their labels
                    code_pattern = r"(\d+):\s*([^0-9]+?)(?=\s+\d+:|$)"
                    codes = re_module.findall(code_pattern, alias_name)

                    if codes:
                        # Check if this is a binary yes/no column (e.g., "1: Yes 2: No")
                        # For "on X" queries, we want the "Yes" value
                        alias_lower = alias_name.lower()
                        is_binary_yes_no = (
                            len(codes) == 2  # Binary = exactly 2 codes
                            and ("yes" in alias_lower or "prescribed" in alias_lower)
                            and any("yes" in label.lower() or "no" in label.lower() for _, label in codes)
                        )

                        if is_binary_yes_no:
                            # Binary yes/no column - find the "Yes" code
                            yes_code = None
                            for code, label in codes:
                                label_lower = label.lower()
                                if "yes" in label_lower or ("prescribed" in alias_lower and int(code) != 0):
                                    yes_code = int(code)
                                    break

                            if yes_code is not None:
                                filters.append(
                                    FilterSpec(
                                        column=column_name,
                                        operator="==",
                                        value=yes_code,
                                        exclude_nulls=True,
                                    )
                                )
                            else:
                                # Fallback: use code 1 if it exists (common pattern)
                                if any(int(code) == 1 for code, _ in codes):
                                    filters.append(
                                        FilterSpec(
                                            column=column_name,
                                            operator="==",
                                            value=1,
                                            exclude_nulls=True,
                                        )
                                    )
                        else:
                            # Coded column (not binary prescribed) - filter for non-zero values
                            # For "on statins" with "Statin Used", filter for != 0 or IN [1,2,3,4,5]
                            non_zero_codes = [int(code) for code, _ in codes if int(code) != 0]
                            if non_zero_codes:
                                filters.append(
                                    FilterSpec(
                                        column=column_name,
                                        operator="IN",
                                        value=non_zero_codes,
                                        exclude_nulls=True,
                                    )
                                )
                            else:
                                # Fallback: just exclude zero
                                filters.append(
                                    FilterSpec(
                                        column=column_name,
                                        operator="!=",
                                        value=0,
                                        exclude_nulls=True,
                                    )
                                )
                    else:
                        # No codes found, but column matches - use != 0 as default
                        filters.append(
                            FilterSpec(
                                column=column_name,
                                operator="!=",
                                value=0,
                                exclude_nulls=True,
                            )
                        )
                else:
                    # Not a coded column - use equality with the phrase
                    filters.append(
                        FilterSpec(
                            column=column_name,
                            operator="==",
                            value=value_phrase,
                            exclude_nulls=True,
                        )
                    )

        # Pattern 2: Numeric range filters
        # "below X", "above X", "less than X", "greater than X", "> X", "< X"
        # "patients over X" / "patients under X" → age filter (medical context)
        numeric_patterns = [
            (r"(?:patients|subjects|people)\s+over\s+([0-9]+)", ">", "age"),  # "patients over 50"
            (r"(?:patients|subjects|people)\s+under\s+([0-9]+)", "<", "age"),  # "patients under 30"
            (r"(\w+(?:\s+\w+)*)\s+below\s+([0-9]+\.?[0-9]*)", "<", None),
            (r"(\w+(?:\s+\w+)*)\s+above\s+([0-9]+\.?[0-9]*)", ">", None),
            (r"(\w+(?:\s+\w+)*)\s+less\s+than\s+([0-9]+\.?[0-9]*)", "<", None),
            (r"(\w+(?:\s+\w+)*)\s+greater\s+than\s+([0-9]+\.?[0-9]*)", ">", None),
            (r"(\w+(?:\s+\w+)*)\s+<=\s+([0-9]+\.?[0-9]*)", "<=", None),
            (r"(\w+(?:\s+\w+)*)\s+>=\s+([0-9]+\.?[0-9]*)", ">=", None),
            (r"(\w+(?:\s+\w+)*)\s+<\s+([0-9]+\.?[0-9]*)", "<", None),
            (r"(\w+(?:\s+\w+)*)\s+>\s+([0-9]+\.?[0-9]*)", ">", None),
            (r"scores?\s+below\s+([0-9\-]+\.?[0-9]*)", "<", "score"),  # "scores below -2.5"
            (r"scores?\s+above\s+([0-9\-]+\.?[0-9]*)", ">", "score"),
        ]

        for pattern_info in numeric_patterns:
            if len(pattern_info) == 3:
                pattern, operator, default_column = pattern_info
            else:
                pattern, operator = pattern_info
                default_column = None

            matches = re.finditer(pattern, query_lower)
            for match in matches:
                if len(match.groups()) == 2:
                    column_phrase = match.group(1).strip()
                    value_str = match.group(2).strip()
                elif len(match.groups()) == 1:
                    # Pattern with default column (e.g., "patients over 50" → age)
                    value_str = match.group(1).strip()
                    column_phrase = default_column if default_column else "score"
                else:
                    # Pattern like "scores below -2.5" (no column name, just value)
                    # Try to find a score column
                    value_str = match.group(1).strip()
                    column_phrase = default_column if default_column else "score"

                # Try to match column phrase to actual column
                column_name, conf, _ = self._fuzzy_match_variable(column_phrase)
                if column_name and conf > 0.5:
                    try:
                        # Try to parse as float first, then int
                        value = float(value_str)
                        if value.is_integer():
                            value = int(value)
                        filters.append(
                            FilterSpec(
                                column=column_name,
                                operator=operator,
                                value=value,
                                exclude_nulls=True,
                            )
                        )
                    except ValueError:
                        # Couldn't parse as number, skip
                        pass

        # Pattern 3: "without X" → inverse filter (NOT_IN or !=)
        without_pattern = r"without\s+([^,\.]+)"
        matches = re.finditer(without_pattern, query_lower)
        for match in matches:
            value_phrase = match.group(1).strip()
            column_name, conf, _ = self._fuzzy_match_variable(value_phrase)
            if column_name and conf > 0.5:
                filters.append(
                    FilterSpec(
                        column=column_name,
                        operator="!=",
                        value=value_phrase,
                        exclude_nulls=True,
                    )
                )

        # Pattern 4: Specific value mentions (e.g., "osteoporosis", "statin")
        # This is a fallback for queries like "how many patients were on statins"
        # where "statins" might not match a column but should match a value
        # We'll look for common medical terms that might be values
        # This is heuristic and will be improved with better value matching

        # Deduplicate filters (same column + operator + value)
        seen_filters = set()
        deduplicated_filters = []
        for f in filters:
            # Create a unique key for this filter
            filter_key = (
                f.column,
                f.operator,
                str(f.value) if not isinstance(f.value, list) else tuple(sorted(f.value)),
            )
            if filter_key not in seen_filters:
                seen_filters.add(filter_key)
                deduplicated_filters.append(f)

        logger.debug(
            "filters_extracted",
            query=query,
            filter_count=len(deduplicated_filters),
            filters=[{"column": f.column, "operator": f.operator, "value": f.value} for f in deduplicated_filters],
        )

        return deduplicated_filters

    def _extract_grouping_from_compound_query(self, query: str) -> str | None:
        """
        Extract grouping variable from compound queries with "which X" and "broken down by" patterns.

        Handles patterns like:
        - "which statin was most prescribed"
        - "which treatment was most common"
        - "and which X"
        - "broken down by count of patients per statin"
        - "broken down by X"
        - "by X" (simple pattern at end)
        - "per X" (at end of query)

        Args:
            query: User's natural language query

        Returns:
            Canonical column name if found, None otherwise
        """
        query_lower = query.lower()

        # Pattern 1: "broken down by X" or "by X" or "per X" (prioritize explicit grouping phrases)
        # These patterns are semantic - they capture grouping intent without hardcoding specific words
        broken_down_patterns = [
            # "broken down by count of patients per X" - captures grouping after "per"
            r"broken\s+down\s+by\s+(?:count\s+of\s+)?(?:\w+\s+)*?per\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
            # "broken down by X" - captures grouping variable directly
            r"broken\s+down\s+by\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
            # "by X" at end of query - simple grouping pattern
            r"by\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
            # "per X" at end of query - frequency/rate grouping pattern
            r"per\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
        ]

        for pattern in broken_down_patterns:
            match = re.search(pattern, query_lower)
            if match:
                group_phrase = match.group(1).strip()
                # Try to match this phrase to a column name using semantic layer
                column_name, conf, _ = self._fuzzy_match_variable(group_phrase)
                if column_name and conf > 0.5:
                    logger.debug(
                        "grouping_extracted_broken_down",
                        query=query,
                        group_phrase=group_phrase,
                        column_name=column_name,
                        confidence=conf,
                    )
                    return column_name
                # If fuzzy match failed, try partial matching (uses semantic layer alias index)
                if not column_name or conf <= 0.5:
                    column_name = self._find_column_by_partial_match(group_phrase)
                    if column_name:
                        logger.debug(
                            "grouping_extracted_broken_down_partial",
                            query=query,
                            group_phrase=group_phrase,
                            column_name=column_name,
                        )
                        return column_name

        # Pattern 2: "which X was most Y" or "what was the most Y" - semantic pattern
        # Uses "most" + any word pattern to be flexible with domain-specific terms
        # Also handles "which was the most Y", "what was the most Y", and "excluding X, which was the most Y"
        patterns = [
            # "which X was most Y" - captures X, Y can be any word (prescribed, common, used, frequent, etc.)
            r"which\s+(\w+(?:\s+\w+)*?)\s+was\s+most\s+\w+",
            # "which was the most Y" - captures the grouping variable from context
            # (e.g., "which was the most prescribed statin?")
            r"which\s+was\s+the?\s+most\s+(\w+)",
            # "what was the most Y" - captures grouping variable (e.g., "what was the most common Current Regimen")
            r"what\s+was\s+the\s+most\s+\w+\s+(\w+(?:\s+\w+)*?)(?:\?|$)",
            # "and which X was most Y" - same pattern with conjunction
            r"and\s+which\s+(\w+(?:\s+\w+)*?)\s+was\s+most\s+\w+",
            # "excluding X, which was the most Y" - handles exclusion + grouping
            r"(?:excluding|excluding\s+those\s+not\s+on)\s+[^,]+,?\s*which\s+was\s+the?\s+most\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                group_phrase = match.group(1).strip()
                # For "which was the most Y", we need to extract the grouping variable from context
                # If group_phrase is just one word (like "prescribed"), we need to find the related column
                # by looking for columns that contain this word or are related to the query context
                if len(group_phrase.split()) == 1:
                    # Single word like "prescribed" or "common" - try to find related column from context
                    # Look for columns that contain this word or are related
                    column_name = self._find_column_by_partial_match(group_phrase)
                    if not column_name:
                        # Try to find column by looking for the subject from earlier in the query
                        # For "excluding those not on statins, which was the most prescribed statin?"
                        # We want to find "statin" related column
                        # Also handle "what was the most common Current Regimen" - look for "Current Regimen" in query
                        subject_match = re.search(
                            r"(?:excluding\s+(?:those|patients)\s+not\s+on|on|were\s+on|most\s+\w+\s+)(\w+(?:\s+\w+)*?)(?:\s*,|\s+which|\s+what|\?|$)",
                            query_lower,
                        )
                        if subject_match:
                            subject_phrase = subject_match.group(1).strip()
                            # Skip common words
                            if subject_phrase not in ["the", "a", "an", "those", "patients", "common", "prescribed"]:
                                column_name = self._find_column_by_partial_match(subject_phrase)
                else:
                    # Multi-word phrase - try direct matching
                    column_name, conf, _ = self._fuzzy_match_variable(group_phrase)
                    if not column_name or conf <= 0.5:
                        column_name = self._find_column_by_partial_match(group_phrase)

                if column_name:
                    logger.debug(
                        "grouping_extracted",
                        query=query,
                        group_phrase=group_phrase,
                        column_name=column_name,
                        confidence=0.7,
                    )
                    return column_name

        # Pattern 3: "which X" at the end of query (fallback - most general pattern)
        # This is semantic - captures any "which X" without hardcoding what follows
        match = re.search(r"which\s+(\w+(?:\s+\w+)*?)(?:\?|$)", query_lower)
        if match:
            group_phrase = match.group(1).strip()
            # Use semantic layer for matching (not hardcoded keywords)
            column_name, conf, _ = self._fuzzy_match_variable(group_phrase)
            if column_name and conf > 0.5:
                logger.debug(
                    "grouping_extracted_fallback",
                    query=query,
                    group_phrase=group_phrase,
                    column_name=column_name,
                    confidence=conf,
                )
                return column_name
            # If fuzzy match failed, try partial matching (uses semantic layer alias index)
            if not column_name or conf <= 0.5:
                column_name = self._find_column_by_partial_match(group_phrase)
                if column_name:
                    logger.debug(
                        "grouping_extracted_fallback_partial",
                        query=query,
                        group_phrase=group_phrase,
                        column_name=column_name,
                    )
                    return column_name

        return None

    def _find_column_by_partial_match(self, search_term: str) -> str | None:
        """
        Find column by partial match (e.g., "statin" matches "Statin Used").

        This is more lenient than fuzzy matching and useful for grouping variables
        where the user might use a shortened form.

        Args:
            search_term: Term to search for (e.g., "statin")

        Returns:
            Canonical column name if found, None otherwise
        """
        search_term_lower = search_term.lower().strip()
        alias_index = self.semantic_layer.get_column_alias_index()

        # Alias index is dict[normalized_alias, canonical_name]
        # We need to check if any normalized alias contains/starts with the search term
        # Priority: exact match > starts with > contains
        best_match = None
        best_score = 0.0

        # Normalize search term to match alias index format
        normalized_search = self.semantic_layer._normalize_alias(search_term)

        for normalized_alias, canonical_name in alias_index.items():
            # Exact match (after normalization)
            if normalized_alias == normalized_search:
                return canonical_name
            # Starts with search term
            if normalized_alias.startswith(normalized_search):
                score = len(normalized_search) / len(normalized_alias)
                if score > best_score:
                    best_match = canonical_name
                    best_score = score
            # Contains search term (lower priority)
            elif normalized_search in normalized_alias:
                score = len(normalized_search) / len(normalized_alias) * 0.7
                if score > best_score:
                    best_match = canonical_name
                    best_score = score
            # Also check original search term (not normalized) against normalized alias
            elif search_term_lower in normalized_alias:
                score = len(search_term_lower) / len(normalized_alias) * 0.5
                if score > best_score:
                    best_match = canonical_name
                    best_score = score

        # Return if we found a reasonable match (score > 0.3)
        if best_match and best_score > 0.3:
            return best_match

        return None

    def _intent_to_plan(self, intent: QueryIntent, dataset_version: str) -> QueryPlan:
        """
        Convert QueryIntent to QueryPlan.

        Phase 1.1.5: run_key is NOT set here - semantic layer will generate it deterministically
        using _generate_run_key() which includes normalized query text. This ensures all execution
        paths produce the same run_key.

        Args:
            intent: QueryIntent from parse_query()
            dataset_version: Dataset version identifier (upload_id or dataset_id)

        Returns:
            QueryPlan with run_key=None (semantic layer will generate it)
        """
        # Create QueryPlan from intent
        plan = QueryPlan(
            intent=intent.intent_type,  # type: ignore[arg-type]  # QueryPlan expects Literal, but we validate in QueryIntent
            metric=intent.primary_variable,
            group_by=intent.grouping_variable,
            filters=intent.filters,  # Already FilterSpec objects
            confidence=intent.confidence,
            explanation="",  # Will be populated from intent if available
            run_key=None,  # Phase 1.1.5: Semantic layer will generate run_key deterministically
        )

        logger.debug(
            "intent_converted_to_plan",
            intent_type=intent.intent_type,
            filter_count=len(plan.filters),
        )

        return plan

    def _generate_suggestions(self, query: str) -> list[str]:
        """
        Generate suggestions for improving query when all tiers fail.

        Args:
            query: User's query that failed to parse

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Query length suggestions
        if len(query.split()) < 3:
            suggestions.append("Try adding more details to your question")
        elif len(query.split()) > 20:
            suggestions.append("Try simplifying your question")

        # Suggest available columns if semantic layer has them
        try:
            alias_index = self.semantic_layer.get_column_alias_index()
            if alias_index:
                sample_columns = list(alias_index.keys())[:5]
                suggestions.append(f"Try mentioning specific variable names (e.g., {', '.join(sample_columns)})")
        except Exception:
            pass  # If semantic layer unavailable, skip this suggestion

        # Generic suggestions
        if not suggestions:
            suggestions.append("Try using phrases like 'compare X by Y' or 'what predicts X'")
            suggestions.append("Mention specific variable names from your dataset")

        return suggestions
