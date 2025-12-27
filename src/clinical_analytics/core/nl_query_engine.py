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

    def parse_query(self, query: str) -> QueryIntent:
        """
        Parse natural language query into structured intent.

        Args:
            query: User's question (e.g., "compare survival by treatment arm")

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
            raise ValueError("Query cannot be empty")

        query = query.strip()

        # Tier 1: Pattern matching
        intent = self._pattern_match(query)
        if intent and intent.confidence > 0.9:
            return intent

        # Tier 2: Semantic embeddings
        intent = self._semantic_match(query)
        if intent and intent.confidence > 0.75:
            return intent

        # Tier 3: LLM fallback (stub for now)
        intent = self._llm_parse(query)
        return intent

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
            primary_var = self._fuzzy_match_variable(match.group(1))
            group_var = self._fuzzy_match_variable(match.group(2))

            if primary_var and group_var:
                return QueryIntent(
                    intent_type="COMPARE_GROUPS",
                    primary_variable=primary_var,
                    grouping_variable=group_var,
                    confidence=0.95,
                )

        # Pattern: "what predicts X" or "predictors of X"
        match = re.search(
            r"(?:what predicts|predictors of|predict|risk factors for)\s+(\w+)", query_lower
        )
        if match:
            outcome_var = self._fuzzy_match_variable(match.group(1))

            if outcome_var:
                return QueryIntent(
                    intent_type="FIND_PREDICTORS", primary_variable=outcome_var, confidence=0.95
                )

        # Pattern: "survival" or "time to event"
        if re.search(r"\b(survival|time to event|kaplan|cox)\b", query_lower):
            return QueryIntent(intent_type="SURVIVAL", confidence=0.9)

        # Pattern: "correlation" or "relationship"
        if re.search(r"\b(correlat|relationship|associat)\b", query_lower):
            # Try to extract two variables
            variables = self._extract_variables_from_query(query)
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
            primary_var = self._fuzzy_match_variable(match.group(1))
            group_var = self._fuzzy_match_variable(match.group(2))

            if primary_var and group_var:
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
                intent = QueryIntent(
                    intent_type=best_template["intent"], confidence=float(best_score)
                )

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
            print(f"Tier 2 semantic matching failed: {e}")
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

    def _fuzzy_match_variable(self, query_term: str) -> str | None:
        """
        Fuzzy match query term to actual column name in dataset.

        Handles:
        - Synonyms (e.g., "age" → "age_years", "died" → "mortality")
        - Partial matches (e.g., "treat" → "treatment_arm")
        - Case insensitivity

        Args:
            query_term: Term from user's query

        Returns:
            Matched column name, or None if no good match

        Example:
            >>> engine._fuzzy_match_variable("age")
            'age_years'
        """
        # Get all columns from semantic layer
        view = self.semantic_layer.get_base_view()
        available_columns = view.columns

        # Direct match (case-insensitive)
        for col in available_columns:
            if col.lower() == query_term.lower():
                return col

        # Check synonyms
        synonyms = {
            "age": ["age_years", "patient_age", "age_at_admission"],
            "mortality": ["died", "death", "deceased", "expired", "dead"],
            "treatment": ["tx", "therapy", "intervention", "treatment_arm"],
            "outcome": ["result", "endpoint", "event"],
            "icu": ["intensive_care", "icu_admission"],
            "hospital": ["hospitalized", "hospitalization", "hosp"],
        }

        query_term_lower = query_term.lower()
        for canonical, synonym_list in synonyms.items():
            if query_term_lower == canonical or query_term_lower in synonym_list:
                # Find any column matching canonical or synonyms
                for col in available_columns:
                    col_lower = col.lower()
                    if canonical in col_lower or any(syn in col_lower for syn in synonym_list):
                        return col

        # Fuzzy match
        matches = get_close_matches(
            query_term.lower(), [c.lower() for c in available_columns], n=1, cutoff=0.6
        )

        if matches:
            # Find original casing
            for col in available_columns:
                if col.lower() == matches[0]:
                    return col

        return None

    def _extract_variables_from_query(self, query: str) -> list[str]:
        """
        Extract all potential variable names from query.

        Args:
            query: User's question

        Returns:
            List of matched column names

        Example:
            >>> engine._extract_variables_from_query("compare age by treatment")
            ['age_years', 'treatment_arm']
        """
        words = query.lower().split()

        # Match against all available columns
        view = self.semantic_layer.get_base_view()
        available_columns = view.columns

        matched_vars = []
        for word in words:
            # Remove punctuation
            word_clean = word.strip(",.?!")

            # Check if word is a column name (fuzzy)
            matched = self._fuzzy_match_variable(word_clean)
            if matched and matched not in matched_vars:
                matched_vars.append(matched)

        return matched_vars
