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
import re
from dataclasses import dataclass, field
from difflib import get_close_matches

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
            llm_intent = self._llm_parse(query)
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

                # For CORRELATIONS: assign first two variables
                elif intent.intent_type == "CORRELATIONS":
                    if not intent.primary_variable and len(matched_vars) >= 1:
                        intent.primary_variable = matched_vars[0]
                    if not intent.grouping_variable and len(matched_vars) >= 2:
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
            intent.filters = self._extract_filters(query)
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

        # Pattern: "how many" or "count" or "number of" (COUNT intent)
        count_patterns = [
            r"how many",
            r"\bcount\b",
            r"number of",
        ]
        if any(re.search(pattern, query_lower) for pattern in count_patterns):
            return QueryIntent(intent_type="COUNT", confidence=0.9)

        # Pattern: "which X was most Y" - COUNT with grouping (e.g., "which statin was most prescribed?")
        # This pattern asks for the top result by count, so it's a COUNT intent with grouping
        if re.search(r"which\s+\w+(?:\s+\w+)*?\s+was\s+most\s+\w+", query_lower):
            return QueryIntent(intent_type="COUNT", confidence=0.9)

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

    def _extract_filters(self, query: str) -> list[FilterSpec]:
        """
        Extract filter conditions from query text.

        Patterns to detect:
        - "those that had X" / "patients with X" → categorical filter
        - "scores below/above X" → numeric range filter
        - "with X" / "without X" → presence filter
        - "on statins" / "were on statins" → categorical filter (value matching)

        Args:
            query: User's natural language query

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
        # Stop at common query continuation words: and, or, which, what, how, where, when
        categorical_patterns = [
            r"(?:those|patients|subjects|people)\s+(?:that|who)\s+(?:had|have|were|are)\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            r"(?:patients|subjects|people)\s+with\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            r"with\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",
            r"on\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",  # "on statins", "on treatment"
            r"were\s+on\s+([^,\.\?]+?)(?:\s+(?:and|or|which|what|how|where|when|,|\.|\?)|$)",  # "were on statins"
        ]

        for pattern in categorical_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                value_phrase = match.group(1).strip()

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
        numeric_patterns = [
            (r"(\w+(?:\s+\w+)*)\s+below\s+([0-9]+\.?[0-9]*)", "<"),
            (r"(\w+(?:\s+\w+)*)\s+above\s+([0-9]+\.?[0-9]*)", ">"),
            (r"(\w+(?:\s+\w+)*)\s+less\s+than\s+([0-9]+\.?[0-9]*)", "<"),
            (r"(\w+(?:\s+\w+)*)\s+greater\s+than\s+([0-9]+\.?[0-9]*)", ">"),
            (r"(\w+(?:\s+\w+)*)\s+<=\s+([0-9]+\.?[0-9]*)", "<="),
            (r"(\w+(?:\s+\w+)*)\s+>=\s+([0-9]+\.?[0-9]*)", ">="),
            (r"(\w+(?:\s+\w+)*)\s+<\s+([0-9]+\.?[0-9]*)", "<"),
            (r"(\w+(?:\s+\w+)*)\s+>\s+([0-9]+\.?[0-9]*)", ">"),
            (r"scores?\s+below\s+([0-9\-]+\.?[0-9]*)", "<"),  # "scores below -2.5"
            (r"scores?\s+above\s+([0-9\-]+\.?[0-9]*)", ">"),
        ]

        for pattern, operator in numeric_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                if len(match.groups()) == 2:
                    column_phrase = match.group(1).strip()
                    value_str = match.group(2).strip()
                else:
                    # Pattern like "scores below -2.5" (no column name, just value)
                    # Try to find a score column
                    value_str = match.group(1).strip()
                    column_phrase = "score"  # Default to "score"

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

        # Pattern 2: "which X was most Y" - semantic pattern (captures any superlative, not hardcoded words)
        # Uses "most" + any word pattern to be flexible with domain-specific terms
        patterns = [
            # "which X was most Y" - captures X, Y can be any word (prescribed, common, used, frequent, etc.)
            r"which\s+(\w+(?:\s+\w+)*?)\s+was\s+most\s+\w+",
            # "and which X was most Y" - same pattern with conjunction
            r"and\s+which\s+(\w+(?:\s+\w+)*?)\s+was\s+most\s+\w+",
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                group_phrase = match.group(1).strip()
                # Try to match this phrase to a column name
                column_name, conf, _ = self._fuzzy_match_variable(group_phrase)
                if column_name and conf > 0.5:
                    logger.debug(
                        "grouping_extracted",
                        query=query,
                        group_phrase=group_phrase,
                        column_name=column_name,
                        confidence=conf,
                    )
                    return column_name
                # If fuzzy match failed, try partial matching
                if not column_name or conf <= 0.5:
                    column_name = self._find_column_by_partial_match(group_phrase)
                    if column_name:
                        logger.debug(
                            "grouping_extracted_partial",
                            query=query,
                            group_phrase=group_phrase,
                            column_name=column_name,
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
        Convert QueryIntent to QueryPlan with deterministic run_key.

        Args:
            intent: QueryIntent from parse_query()
            dataset_version: Dataset version identifier (upload_id or dataset_id)

        Returns:
            QueryPlan with deterministic run_key for idempotent execution
        """
        # Create QueryPlan from intent
        plan = QueryPlan(
            intent=intent.intent_type,  # type: ignore[arg-type]  # QueryPlan expects Literal, but we validate in QueryIntent
            metric=intent.primary_variable,
            group_by=intent.grouping_variable,
            filters=intent.filters,  # Already FilterSpec objects
            confidence=intent.confidence,
            explanation="",  # Will be populated from intent if available
            run_key=None,  # Will be set below
        )

        # Generate deterministic run_key: hash of (dataset_version, normalized_plan)
        normalized_plan = {
            "intent": plan.intent,
            "metric": plan.metric,
            "group_by": plan.group_by,
            "filters": [
                {
                    "column": f.column,
                    "operator": f.operator,
                    "value": f.value,
                    "exclude_nulls": f.exclude_nulls,
                }
                for f in plan.filters
            ],
        }
        plan_hash = hashlib.sha256(json.dumps(normalized_plan, sort_keys=True).encode()).hexdigest()[:16]
        plan.run_key = f"{dataset_version}_{plan_hash}"

        logger.debug(
            "intent_converted_to_plan",
            intent_type=intent.intent_type,
            run_key=plan.run_key,
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
