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

import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

import structlog

logger = structlog.get_logger()

# Valid intent types (single source of truth)
VALID_INTENT_TYPES = [
    "DESCRIBE",
    "COMPARE_GROUPS",
    "FIND_PREDICTORS",
    "SURVIVAL",
    "CORRELATIONS",
]


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
        filters: Dictionary of filter conditions
        confidence: Confidence score 0-1 for the parse
    """

    intent_type: str
    primary_variable: str | None = None
    grouping_variable: str | None = None
    predictor_variables: list[str] = field(default_factory=list)
    time_variable: str | None = None
    event_variable: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    parsing_tier: str | None = None  # "pattern_match", "semantic_match", "llm_fallback"
    parsing_attempts: list[dict] = field(default_factory=list)  # What was tried
    failure_reason: str | None = None  # Why it failed
    suggestions: list[str] = field(default_factory=list)  # How to improve query

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

        # Build query templates from metadata
        self._build_query_templates()

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

    def parse_query(self, query: str, dataset_id: str | None = None, upload_id: str | None = None) -> QueryIntent:
        """
        Parse natural language query into structured intent.

        Args:
            query: User's question (e.g., "compare survival by treatment arm")
            dataset_id: Optional dataset identifier for logging
            upload_id: Optional upload identifier for logging

        Returns:
            QueryIntent with extracted intent type and variables

        Raises:
            ValueError: If query is empty

        Example:
            >>> intent = engine.parse_query("compare mortality by treatment")
            >>> assert intent.intent_type == "COMPARE_GROUPS"
            >>> assert intent.confidence > 0.9
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
        intent = self._pattern_match(query)
        attempt = {
            "tier": "pattern_match",
            "result": "success" if intent and intent.confidence >= TIER_1_PATTERN_MATCH_THRESHOLD else "failed",
            "confidence": intent.confidence if intent else 0.0,
        }
        parsing_attempts.append(attempt)

        if intent and intent.confidence >= TIER_1_PATTERN_MATCH_THRESHOLD:
            intent.parsing_tier = "pattern_match"
            intent.parsing_attempts = parsing_attempts
            matched_vars = self._get_matched_variables(intent)
            logger.info(
                "query_parse_success",
                intent=intent.intent_type,
                confidence=intent.confidence,
                matched_vars=matched_vars,
                tier="pattern_match",
                **log_context,
            )
            return intent

        # Tier 2: Semantic embeddings
        intent = self._semantic_match(query)
        attempt = {
            "tier": "semantic_match",
            "result": "success" if intent and intent.confidence >= TIER_2_SEMANTIC_MATCH_THRESHOLD else "failed",
            "confidence": intent.confidence if intent else 0.0,
        }
        parsing_attempts.append(attempt)

        if intent and intent.confidence >= TIER_2_SEMANTIC_MATCH_THRESHOLD:
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
            return intent

        # Tier 3: LLM fallback (stub for now)
        intent = self._llm_parse(query)
        attempt = {
            "tier": "llm_fallback",
            "result": "success" if intent else "failed",
            "confidence": intent.confidence if intent else 0.0,
        }
        parsing_attempts.append(attempt)

        if intent:
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
        else:
            # All tiers failed - set failure diagnostics
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

        # Pattern: "correlation" or "relationship"
        if re.search(r"\b(correlat|relationship|associat)\b", query_lower):
            # Try to extract two variables
            variables, _ = self._extract_variables_from_query(query)
            if len(variables) >= 2:
                return QueryIntent(
                    intent_type="CORRELATIONS",
                    primary_variable=variables[0],
                    grouping_variable=variables[1],
                    confidence=0.9,
                )
            else:
                return QueryIntent(intent_type="CORRELATIONS", confidence=0.85)

        # Pattern: "describe" or "summary"
        if re.search(r"\b(describe|summary|overview|statistics)\b", query_lower):
            return QueryIntent(intent_type="DESCRIBE", confidence=0.9)

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
            group_var, _, _ = self._fuzzy_match_variable(match.group(1))
            primary_var, _, _ = self._fuzzy_match_variable(match.group(3))

            if group_var and primary_var:
                return QueryIntent(
                    intent_type="COMPARE_GROUPS",
                    primary_variable=primary_var,
                    grouping_variable=group_var,
                    confidence=0.95,
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

    def _llm_parse(self, query: str) -> QueryIntent:
        """
        Tier 3: LLM fallback with RAG context from semantic layer.

        NOTE: This is a stub implementation. Full implementation would:
        1. Build context from semantic layer metadata
        2. Create structured prompt with available variables
        3. Call LLM API (or local model) with JSON schema
        4. Parse response into QueryIntent

        Args:
            query: User's question

        Returns:
            QueryIntent with low confidence (fallback to structured input)
        """
        # For now, return a low-confidence DESCRIBE intent as safe default
        # This triggers the "ask clarifying questions" flow in the UI
        return QueryIntent(intent_type="DESCRIBE", confidence=0.3)

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
        normalized_query = self.semantic_layer._normalize_alias(query_term)

        # Check if this alias was dropped due to collision
        suggestions = self.semantic_layer.get_collision_suggestions(query_term)
        if suggestions:
            # Collision detected - return suggestions
            return None, 0.2, suggestions

        # Direct match
        if normalized_query in alias_index:
            collisions = self.semantic_layer.get_collision_warnings()
            if normalized_query in collisions:
                # Collision warning (shouldn't happen if we dropped it, but check anyway)
                return alias_index[normalized_query], 0.4, None
            return alias_index[normalized_query], 0.9, None

        # Fuzzy match using difflib
        matches = get_close_matches(
            normalized_query,
            alias_index.keys(),
            n=1,
            cutoff=0.7,
        )

        if matches:
            matched_alias = matches[0]
            collisions = self.semantic_layer.get_collision_warnings()
            if matched_alias in collisions:
                return alias_index[matched_alias], 0.4, None
            return alias_index[matched_alias], 0.7, None

        return None, 0.0, None

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
