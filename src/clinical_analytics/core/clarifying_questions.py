"""Clarifying Questions Engine for refining low-confidence NL queries.

Interactive questions to help users refine their queries when confidence is low.
Leverages semantic layer metadata to provide context-aware suggestions.
"""

from typing import TYPE_CHECKING

import streamlit as st
import structlog

if TYPE_CHECKING:
    from clinical_analytics.core.nl_query_engine import QueryIntent

logger = structlog.get_logger()


class ClarifyingQuestionsEngine:
    """Interactive clarifying questions to refine low-confidence queries.

    Leverages existing semantic layer infrastructure:
    - get_column_alias_index() for available columns
    - get_collision_suggestions() for ambiguous variables
    - get_available_metrics() and get_available_dimensions() for context
    - get_data_quality_warnings() for data quality context

    Structured logging: Logs all user interactions for debugging.
    """

    @staticmethod
    def ask_clarifying_questions(
        intent: "QueryIntent",
        semantic_layer,
        available_columns: list[str],  # Just column names, not DataFrame
    ) -> "QueryIntent":
        """Ask targeted questions to refine intent using semantic layer metadata.

        Args:
            intent: Low-confidence QueryIntent to refine
            semantic_layer: SemanticLayer instance for metadata access
            available_columns: List of column names (extracted from cohort before calling)

        Returns:
            Refined QueryIntent with higher confidence

        Raises:
            ValueError: If semantic layer missing required metadata
        """
        from clinical_analytics.core.column_parser import parse_column_name
        from clinical_analytics.core.nl_query_config import ENABLE_CLARIFYING_QUESTIONS
        from clinical_analytics.core.nl_query_engine import VALID_INTENT_TYPES

        if not ENABLE_CLARIFYING_QUESTIONS:
            return intent  # Feature flag disabled, return original

        log = logger.bind(intent_type=intent.intent_type, confidence=intent.confidence)
        log.info("clarifying_questions_start")

        try:
            # 1. If intent_type is ambiguous or missing, ask user to select
            if not intent.intent_type or intent.confidence < 0.3:
                st.subheader("What type of analysis do you want?")
                intent_type = st.selectbox(
                    "Analysis Type",
                    VALID_INTENT_TYPES,  # Use constant, not hardcoded
                    help="Select the type of analysis you're looking for",
                )
                intent.intent_type = intent_type
                intent.confidence = max(intent.confidence, 0.6)

            # 2. If primary_variable missing, show available columns
            if not intent.primary_variable:
                # Get display names using parse_column_name
                column_options = {}
                for canonical_name in available_columns:
                    meta = parse_column_name(canonical_name)
                    display_name = meta.display_name or canonical_name
                    column_options[display_name] = canonical_name

                if column_options:
                    st.subheader("Which variable are you interested in?")
                    selected_display = st.selectbox(
                        "Primary Variable",
                        list(column_options.keys()),
                        help="Select the main outcome or variable you want to analyze",
                    )
                    intent.primary_variable = column_options[selected_display]
                    intent.confidence = max(intent.confidence, 0.7)

            # 3. If grouping_variable missing for COMPARE_GROUPS
            if intent.intent_type == "COMPARE_GROUPS" and not intent.grouping_variable:
                available_dims = semantic_layer.get_available_dimensions()
                if available_dims:
                    st.subheader("How do you want to group the data?")
                    dim_options = list(available_dims.keys())
                    selected_dim = st.selectbox("Grouping Variable", dim_options)
                    intent.grouping_variable = selected_dim
                    intent.confidence = max(intent.confidence, 0.7)

            # 4. Handle collisions (ACTUAL IMPLEMENTATION)
            collision_suggestions = {}
            if intent.primary_variable:
                suggestions = semantic_layer.get_collision_suggestions(intent.primary_variable)
                if suggestions:
                    collision_suggestions["primary_variable"] = suggestions

            if collision_suggestions:
                st.warning("⚠️ Some terms matched multiple columns. Please select:")
                for var_name, options in collision_suggestions.items():
                    # Show display names
                    display_options = {parse_column_name(opt).display_name or opt: opt for opt in options}
                    selected_display = st.selectbox(
                        f"Which '{var_name}' did you mean?",
                        list(display_options.keys()),
                    )
                    # Update intent with selected canonical name
                    if var_name == "primary_variable":
                        intent.primary_variable = display_options[selected_display]
                        intent.confidence = max(intent.confidence, 0.8)

            # 5. Surface quality warnings
            quality_warnings = semantic_layer.get_data_quality_warnings()
            if quality_warnings and intent.primary_variable:
                # Filter warnings relevant to selected variable
                relevant_warnings = [w for w in quality_warnings if w.get("column") == intent.primary_variable]
                if relevant_warnings:
                    st.warning(f"⚠️ Note: {relevant_warnings[0].get('message', '')}")

            log.info("clarifying_questions_complete", refined_confidence=intent.confidence)
            return intent

        except (RuntimeError, AttributeError) as e:
            # Handle Streamlit widget failures (browser close, session expired)
            logger.warning("clarifying_questions_aborted", error=str(e))
            return intent  # Return original intent gracefully
