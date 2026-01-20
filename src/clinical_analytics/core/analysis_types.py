"""
Analysis Types - Pure Python data structures for analysis context.

Extracted from ui/components/question_engine.py to enable UI-agnostic usage.
Zero Streamlit dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnalysisIntent(Enum):
    """Inferred analysis intentions (hidden from user)."""

    DESCRIBE = "describe"
    COMPARE_GROUPS = "compare_groups"
    FIND_PREDICTORS = "find_predictors"
    EXAMINE_SURVIVAL = "examine_survival"
    EXPLORE_RELATIONSHIPS = "explore_relationships"
    COUNT = "count"
    UNKNOWN = "unknown"


@dataclass
class AnalysisContext:
    """
    Tracks the current state of analysis configuration.

    This is built up through user responses to questions.
    Pure Python dataclass - no UI dependencies.
    """

    # What the user wants to know
    research_question: str | None = None

    # Variables
    primary_variable: str | None = None
    grouping_variable: str | None = None
    predictor_variables: list[str] = field(default_factory=list)
    time_variable: str | None = None
    event_variable: str | None = None

    # Analysis configuration
    compare_groups: bool | None = None
    find_predictors: bool | None = None
    time_to_event: bool | None = None

    # Inferred intent (hidden from user)
    inferred_intent: AnalysisIntent = AnalysisIntent.UNKNOWN

    # Filters
    filters: list[Any] = field(default_factory=list)  # List of FilterSpec objects

    # QueryPlan (structured plan from NLU)
    query_plan: Any = None  # QueryPlan | None - will be set after QueryIntent conversion

    # Original query text (for "most" detection, etc.)
    query_text: str | None = None

    # Metadata
    variable_types: dict[str, str] = field(default_factory=dict)
    match_suggestions: dict[str, list[str]] = field(default_factory=dict)  # {query_term: [canonical_names]}
    confidence: float = 0.0  # Confidence from NL query parsing (for auto-execution logic)

    def is_complete_for_intent(self) -> bool:
        """Check if we have enough information for the inferred analysis."""
        if self.inferred_intent == AnalysisIntent.DESCRIBE:
            return True  # Just needs data

        elif self.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
            return self.primary_variable is not None and self.grouping_variable is not None

        elif self.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
            return self.primary_variable is not None and len(self.predictor_variables) > 0

        elif self.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
            return self.time_variable is not None and self.event_variable is not None

        elif self.inferred_intent == AnalysisIntent.EXPLORE_RELATIONSHIPS:
            # Complete if we have at least 2 predictor variables OR primary + grouping variables
            # (NLU may extract variables as primary/grouping instead of predictor_variables)
            # For CORRELATIONS, predictor_variables should contain ALL variables mentioned
            has_predictors = len(self.predictor_variables) >= 2
            has_primary_grouping = self.primary_variable is not None and self.grouping_variable is not None
            # Also check if we have primary + grouping that can be used
            return has_predictors or has_primary_grouping

        elif self.inferred_intent == AnalysisIntent.COUNT:
            return True  # Just needs data (can optionally filter by grouping_variable)

        return False

    def get_missing_info(self) -> list[str]:
        """Return list of missing information needed to complete analysis."""
        missing: list[str] = []

        if self.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
            if not self.primary_variable:
                missing.append("what you want to measure or compare")
            if not self.grouping_variable:
                missing.append("which groups to compare")

        elif self.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
            if not self.primary_variable:
                missing.append("what outcome you want to predict")
            if not self.predictor_variables:
                missing.append("which variables might predict the outcome")

        elif self.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
            if not self.time_variable:
                missing.append("how you measure time")
            if not self.event_variable:
                missing.append("what event you're tracking")

        elif self.inferred_intent == AnalysisIntent.EXPLORE_RELATIONSHIPS:
            # Check if we have enough variables (either as predictors or primary+grouping)
            has_predictors = len(self.predictor_variables) >= 2
            has_primary_grouping = self.primary_variable is not None and self.grouping_variable is not None
            if not has_predictors and not has_primary_grouping:
                missing.append("at least 2 variables to examine relationships")

        elif self.inferred_intent == AnalysisIntent.COUNT:
            # COUNT doesn't require any additional info
            pass

        return missing
