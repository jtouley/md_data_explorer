"""
Question Engine - Conversational analysis configuration through questions.

Guides users through analysis setup by asking natural questions,
infers the appropriate statistical test, and dynamically configures analysis.
Supports free-form natural language queries only.
"""

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import streamlit as st

from clinical_analytics.core.nl_query_engine import QueryIntent


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
    filters: list = field(default_factory=list)  # List of FilterSpec objects

    # QueryPlan (structured plan from NLU)
    query_plan = None  # QueryPlan | None - will be set after QueryIntent conversion (type: ignore for forward ref)

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
            has_primary_grouping = (
                self.primary_variable is not None and self.grouping_variable is not None
            )
            # Also check if we have primary + grouping that can be used
            return has_predictors or has_primary_grouping

        elif self.inferred_intent == AnalysisIntent.COUNT:
            return True  # Just needs data (can optionally filter by grouping_variable)

        return False

    def get_missing_info(self) -> list[str]:
        """Return list of missing information needed to complete analysis."""
        missing = []

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
            has_primary_grouping = (
                self.primary_variable is not None and self.grouping_variable is not None
            )
            if not has_predictors and not has_primary_grouping:
                missing.append("at least 2 variables to examine relationships")

        elif self.inferred_intent == AnalysisIntent.COUNT:
            # COUNT doesn't require any additional info
            pass

        return missing


class QuestionEngine:
    """
    Manages conversational flow for analysis configuration.

    Asks questions, tracks answers, infers intent, prompts for missing info.
    """

    @staticmethod
    def infer_intent(context: AnalysisContext) -> AnalysisIntent:
        """
        Infer analysis intent from current context.

        This happens behind the scenes - user never sees analysis type names.
        """
        # Explicit indicators
        if context.time_to_event:
            return AnalysisIntent.EXAMINE_SURVIVAL

        if context.find_predictors:
            return AnalysisIntent.FIND_PREDICTORS

        if context.compare_groups:
            return AnalysisIntent.COMPARE_GROUPS

        # Infer from variable configuration
        if context.time_variable and context.event_variable:
            return AnalysisIntent.EXAMINE_SURVIVAL

        if context.primary_variable and len(context.predictor_variables) > 1:
            return AnalysisIntent.FIND_PREDICTORS

        if context.primary_variable and context.grouping_variable:
            return AnalysisIntent.COMPARE_GROUPS

        if len(context.predictor_variables) >= 2:
            return AnalysisIntent.EXPLORE_RELATIONSHIPS

        # Default to describe if nothing specific
        if not context.primary_variable and not context.predictor_variables:
            return AnalysisIntent.DESCRIBE

        return AnalysisIntent.UNKNOWN

    @staticmethod
    def select_primary_variable(
        df: pd.DataFrame,
        context: AnalysisContext,
        prompt: str = "What do you want to measure or analyze?",
    ) -> str | None:
        """
        Ask user to select their primary variable of interest.

        Uses plain language prompt based on context.
        """
        available_cols = [c for c in df.columns if c not in ["patient_id", "time_zero"]]

        st.markdown(f"### {prompt}")

        # Provide context-specific help text
        if context.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
            help_text = "This is your outcome - what you want to predict or explain"
        elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
            help_text = "This is what you want to compare between groups"
        else:
            help_text = "Select the main variable you're interested in"

        variable = st.selectbox("Variable:", ["(Choose one)"] + available_cols, help=help_text)

        return None if variable == "(Choose one)" else variable

    @staticmethod
    def select_grouping_variable(df: pd.DataFrame, exclude: list[str] = None) -> str | None:
        """Ask user to select groups to compare."""
        available_cols = [c for c in df.columns if c not in ["patient_id", "time_zero"]]
        if exclude:
            available_cols = [c for c in available_cols if c not in exclude]

        st.markdown("### Which groups do you want to compare?")

        variable = st.selectbox(
            "Grouping variable:",
            ["(Choose one)"] + available_cols,
            help="This splits your data into groups (e.g., treatment arm, sex, age group)",
        )

        return None if variable == "(Choose one)" else variable

    @staticmethod
    def select_predictor_variables(df: pd.DataFrame, exclude: list[str] = None, min_vars: int = 1) -> list[str]:
        """Ask user to select predictor variables."""
        available_cols = [c for c in df.columns if c not in ["patient_id", "time_zero"]]
        if exclude:
            available_cols = [c for c in available_cols if c not in exclude]

        st.markdown("### Which variables might affect or predict this?")

        variables = st.multiselect(
            "Predictor variables:",
            available_cols,
            default=[],
            help=f"Select at least {min_vars} variable(s) that might influence the outcome",
        )

        return variables

    @staticmethod
    def select_time_variables(df: pd.DataFrame) -> tuple[str | None, str | None]:
        """Ask user to select time and event variables for survival analysis."""
        available_cols = [c for c in df.columns if c not in ["patient_id", "time_zero"]]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### How do you measure time?")
            time_var = st.selectbox(
                "Time variable:",
                ["(Choose one)"] + available_cols,
                help="Time from start until event or censoring (e.g., days, months)",
            )

        with col2:
            st.markdown("### What event are you tracking?")
            remaining_cols = [c for c in available_cols if c != time_var]
            event_var = st.selectbox(
                "Event variable:",
                ["(Choose one)"] + remaining_cols,
                help="Binary: did the event happen? (1=yes, 0=no/censored)",
            )

        time_var = None if time_var == "(Choose one)" else time_var
        event_var = None if event_var == "(Choose one)" else event_var

        return time_var, event_var

    @staticmethod
    def render_progress_indicator(context: AnalysisContext):
        """Show user how much information we still need."""
        missing = context.get_missing_info()

        if missing:
            st.info(f"ℹ️ I still need to know: {', '.join(missing)}")
        # Removed "I have everything I need" message - bad UX design

    @staticmethod
    def _show_progressive_feedback(nl_engine, query: str) -> QueryIntent | None:
        """Parse query with progressive feedback showing each tier.

        Uses timeout to prevent long waits on LLM fallback.
        Uses constants from nl_query_config (not magic values).

        Args:
            nl_engine: NLQueryEngine instance
            query: User's query string

        Returns:
            QueryIntent if parsing succeeds, None otherwise
        """
        from contextlib import contextmanager

        from clinical_analytics.core.nl_query_config import (
            ENABLE_PROGRESSIVE_FEEDBACK,
            TIER_1_PATTERN_MATCH_THRESHOLD,
            TIER_2_SEMANTIC_MATCH_THRESHOLD,
            TIER_TIMEOUT_SECONDS,
        )

        if not ENABLE_PROGRESSIVE_FEEDBACK:
            # Fallback to simple parsing without feedback
            return nl_engine.parse_query(query)

        @contextmanager
        def timeout_context(seconds: float):
            """Timeout using threading (cross-platform, thread-safe)."""
            import threading

            timed_out = {"value": False}

            def set_timeout():
                timed_out["value"] = True

            timer = threading.Timer(seconds, set_timeout)
            timer.start()
            try:
                yield timed_out
            finally:
                timer.cancel()

        import structlog

        logger = structlog.get_logger()
        parsing_attempts = []
        final_intent = None  # Don't reuse variable name across tiers
        best_partial_intent = None  # Track best partial result for fallback

        def track_attempt(tier: str, tier_intent: QueryIntent | None, threshold: float) -> dict:
            """Track parsing attempt (DRY helper)."""
            return {
                "tier": tier,
                "result": "success" if tier_intent and tier_intent.confidence >= threshold else "failed",
                "confidence": tier_intent.confidence if tier_intent else 0.0,
            }

        with st.status("🔍 Analyzing your question...", expanded=True) as status:
            # Tier 1: Pattern matching (fast, no timeout needed)
            status.update(label="Trying pattern matching...")
            try:
                tier1_intent = nl_engine._pattern_match(query)
                attempt = track_attempt("pattern_match", tier1_intent, TIER_1_PATTERN_MATCH_THRESHOLD)
                parsing_attempts.append(attempt)

                if tier1_intent and tier1_intent.confidence >= TIER_1_PATTERN_MATCH_THRESHOLD:
                    tier1_intent.parsing_tier = "pattern_match"
                    tier1_intent.parsing_attempts = parsing_attempts
                    status.update(label=f"✅ Matched via pattern matching (confidence: {tier1_intent.confidence:.0%})")
                    return tier1_intent
                elif tier1_intent and tier1_intent.intent_type != "DESCRIBE":
                    # Save as potential fallback if it's more specific than DESCRIBE
                    best_partial_intent = tier1_intent
                    best_partial_intent.parsing_tier = "pattern_match_partial"
                    logger.debug(
                        "pattern_match_saved_as_fallback",
                        intent_type=tier1_intent.intent_type,
                        confidence=tier1_intent.confidence,
                    )
            except Exception as e:
                logger.warning("pattern_match_failed", error=str(e))
                parsing_attempts.append({"tier": "pattern_match", "result": "error", "error": str(e)})

            # Tier 2: Semantic search (may be slow, but usually < 1s)
            status.update(label="Trying semantic search...")
            try:
                with timeout_context(TIER_TIMEOUT_SECONDS) as timeout_state:
                    tier2_intent = nl_engine._semantic_match(query)
                    if timeout_state["value"]:
                        raise TimeoutError("Semantic match exceeded timeout")
                attempt = track_attempt("semantic_match", tier2_intent, TIER_2_SEMANTIC_MATCH_THRESHOLD)
                parsing_attempts.append(attempt)

                if tier2_intent and tier2_intent.confidence >= TIER_2_SEMANTIC_MATCH_THRESHOLD:
                    tier2_intent.parsing_tier = "semantic_match"
                    tier2_intent.parsing_attempts = parsing_attempts
                    status.update(label=f"✅ Matched via semantic search (confidence: {tier2_intent.confidence:.0%})")
                    return tier2_intent
            except TimeoutError:
                status.update(label="⏱️ Semantic search timed out, trying advanced parsing...")
                logger.warning("semantic_match_timeout", query=query)
                parsing_attempts.append({"tier": "semantic_match", "result": "timeout"})
            except Exception as e:
                logger.warning("semantic_match_failed", error=str(e))
                parsing_attempts.append({"tier": "semantic_match", "result": "error", "error": str(e)})

            # Tier 3: LLM fallback (can be very slow - 3-5s)
            status.update(label="Trying advanced parsing...")
            try:
                with timeout_context(TIER_TIMEOUT_SECONDS) as timeout_state:
                    tier3_intent = nl_engine._llm_parse(query)
                    if timeout_state["value"]:
                        raise TimeoutError("LLM parse exceeded timeout")
                attempt = track_attempt("llm_fallback", tier3_intent, 0.0)  # LLM has no threshold
                parsing_attempts.append(attempt)

                if tier3_intent:
                    tier3_intent.parsing_tier = "llm_fallback"
                    tier3_intent.parsing_attempts = parsing_attempts
                    status.update(label=f"✅ Matched via advanced parsing (confidence: {tier3_intent.confidence:.0%})")
                    return tier3_intent
                final_intent = tier3_intent  # May be None
            except TimeoutError:
                status.update(label="❌ Advanced parsing timed out")
                logger.warning("llm_parse_timeout", query=query)
                parsing_attempts.append({"tier": "llm_fallback", "result": "timeout"})
            except Exception as e:
                logger.warning("llm_parse_failed", error=str(e))
                parsing_attempts.append({"tier": "llm_fallback", "result": "error", "error": str(e)})

            status.update(label="❌ Could not understand query")
            # Return intent with diagnostics even if all tiers failed
            # BUT: Use best partial result if it's more specific than DESCRIBE
            if best_partial_intent and best_partial_intent.intent_type != "DESCRIBE":
                # Use the partial pattern match - it's better than nothing
                best_partial_intent.parsing_attempts = parsing_attempts
                status.update(
                    label=f"⚠️ Using partial match: {best_partial_intent.intent_type} "
                    f"(confidence: {best_partial_intent.confidence:.0%})"
                )
                logger.info(
                    "using_partial_pattern_match",
                    intent_type=best_partial_intent.intent_type,
                    confidence=best_partial_intent.confidence,
                    query=query,
                )
                return best_partial_intent
            elif final_intent is None:
                # Create a failure intent if all tiers failed
                from clinical_analytics.core.nl_query_engine import QueryIntent

                final_intent = QueryIntent(
                    intent_type="DESCRIBE",
                    confidence=0.0,
                    parsing_attempts=parsing_attempts,
                    failure_reason="All parsing tiers failed",
                    suggestions=nl_engine._generate_suggestions(query),
                )
            else:
                final_intent.parsing_attempts = parsing_attempts
                final_intent.failure_reason = "All parsing tiers failed"
                final_intent.suggestions = nl_engine._generate_suggestions(query)
            return final_intent

    @staticmethod
    def _format_diagnostic_error(intent: QueryIntent) -> str:
        """Format structured diagnostics into user-friendly message.

        Engine returns structured diagnostics, UI formats for display.
        This avoids duplication - single formatting layer.

        Args:
            intent: QueryIntent with diagnostics (parsing_attempts, failure_reason, suggestions)

        Returns:
            Formatted error message with actionable suggestions
        """
        parts = ["❌ Could not understand your query."]

        if intent.suggestions:
            parts.append("\n💡 Suggestions:")
            for suggestion in intent.suggestions:
                parts.append(f"  • {suggestion}")

        if intent.parsing_attempts:
            parts.append("\n🔍 What I tried:")
            for attempt in intent.parsing_attempts:
                tier_name = attempt.get("tier", "unknown")
                result = attempt.get("result", "failed")
                parts.append(f"  • {tier_name}: {result}")

        if intent.failure_reason:
            parts.append(f"\n⚠️ Reason: {intent.failure_reason}")

        return "\n".join(parts)

    @staticmethod
    def ask_free_form_question(
        semantic_layer, dataset_id: str | None = None, upload_id: str | None = None, dataset_version: str | None = None
    ) -> AnalysisContext | None:
        """
        Ask user to type their question in natural language.

        Uses NLQueryEngine to parse the query and convert to AnalysisContext.
        Shows confidence and interpretation, asks for clarification if needed.

        Args:
            semantic_layer: SemanticLayer instance for NL parsing
            dataset_id: Optional dataset identifier for logging
            upload_id: Optional upload identifier for logging

        Returns:
            AnalysisContext if query successfully parsed, None otherwise
        """
        st.markdown("## 💬 Ask your question")

        st.markdown("""
        Just type what you want to know in plain English. I'll figure out the right analysis.

        **Examples:**
        - "Compare survival by treatment arm"
        - "What predicts mortality?"
        - "Show me correlation between age and outcome"
        - "Descriptive statistics for all patients"
        """)

        # Text input for query
        query = st.text_input(
            "Your question:",
            placeholder="e.g., compare survival by treatment arm",
            help="Ask in plain English - I'll figure out the right analysis",
            key="nl_query_input",
        )

        if not query:
            return None

        # Log query entry
        import structlog

        logger = structlog.get_logger()
        logger.info(
            "nl_query_entered",
            query=query,
            dataset_id=dataset_id,
            upload_id=upload_id,
            query_length=len(query),
        )

        # Parse query with NLQueryEngine
        try:
            from clinical_analytics.core.nl_query_config import ENABLE_PROGRESSIVE_FEEDBACK
            from clinical_analytics.core.nl_query_engine import NLQueryEngine

            # Initialize NL query engine
            nl_engine = NLQueryEngine(semantic_layer)

            logger.debug("nl_query_engine_initialized", dataset_id=dataset_id, upload_id=upload_id)

            # Use progressive feedback if enabled, otherwise simple parsing
            if ENABLE_PROGRESSIVE_FEEDBACK:
                query_intent = QuestionEngine._show_progressive_feedback(nl_engine, query)
            else:
                # Fallback to simple parsing without progressive feedback
                with st.spinner("Understanding your question..."):
                    query_intent = nl_engine.parse_query(query, dataset_id=dataset_id, upload_id=upload_id)

            # Extract variables with collision suggestions (for collision handling only)
            # Note: Variables are already extracted and assigned in parse_query() for COMPARE_GROUPS, etc.
            matched_vars, collision_suggestions = nl_engine._extract_variables_from_query(query)

            logger.info(
                "variables_extracted",
                query=query,
                matched_vars=matched_vars,
                collision_suggestions=list(collision_suggestions.keys()) if collision_suggestions else [],
                intent_type=query_intent.intent_type if query_intent else None,
                confidence=query_intent.confidence if query_intent else 0.0,
            )

            # Pre-populate primary_variable from matched_vars for DESCRIBE intents
            # This ensures the clarifying questions use the right default instead of first column
            if (
                query_intent
                and query_intent.intent_type == "DESCRIBE"
                and not query_intent.primary_variable
                and matched_vars
            ):
                # Find the variable that best matches the query terms
                # e.g., "what was the average t score" should match "DEXA Score (T score)"
                query_lower = query.lower()
                best_match = None
                best_score = 0

                for var in matched_vars:
                    var_lower = var.lower()
                    # Check how many query words appear in the variable name
                    query_words = query_lower.split()
                    score = sum(1 for word in query_words if word in var_lower and len(word) > 2)

                    # Also check for compound terms like "tscore" matching "t score"
                    compound_terms = ["tscore", "zscore", "t-score", "z-score"]
                    for term in compound_terms:
                        if term in query_lower.replace(" ", "") and term[0] + " score" in var_lower:
                            score += 5  # Strong boost for score type matches

                    if score > best_score:
                        best_score = score
                        best_match = var

                # Use best match if found, otherwise fall back to first
                query_intent.primary_variable = best_match or matched_vars[0]
                logger.info(
                    "primary_variable_inferred",
                    query=query,
                    primary_variable=query_intent.primary_variable,
                    match_score=best_score,
                    from_matched_vars=matched_vars,
                )

            # Ask clarifying questions if confidence is low
            from clinical_analytics.core.nl_query_config import (
                CLARIFYING_QUESTIONS_THRESHOLD,
                ENABLE_CLARIFYING_QUESTIONS,
            )

            if (
                query_intent
                and query_intent.confidence < CLARIFYING_QUESTIONS_THRESHOLD
                and ENABLE_CLARIFYING_QUESTIONS
            ):
                # Get available columns from semantic layer for clarifying questions
                alias_index = semantic_layer.get_column_alias_index()
                available_columns = list(alias_index.values()) if alias_index else []

                from clinical_analytics.core.clarifying_questions import ClarifyingQuestionsEngine

                query_intent = ClarifyingQuestionsEngine.ask_clarifying_questions(
                    query_intent, semantic_layer, available_columns
                )

            # Show confidence (only if progressive feedback didn't already show it)
            if not ENABLE_PROGRESSIVE_FEEDBACK:
                if query_intent and query_intent.confidence > 0.75:
                    st.success(f"✅ I understand! (Confidence: {query_intent.confidence:.0%})")
                elif query_intent and query_intent.confidence > 0.5:
                    st.warning(f"⚠️ I think I understand, but please verify (Confidence: {query_intent.confidence:.0%})")

            # Show diagnostic info if parsing failed
            if query_intent is None or (query_intent and query_intent.confidence < 0.5 and query_intent.failure_reason):
                if query_intent:
                    # Use formatted diagnostic error message
                    error_message = QuestionEngine._format_diagnostic_error(query_intent)
                    st.error(error_message)
                elif query_intent is None:
                    st.error("❌ Could not understand your query. Please try rephrasing.")
                return None  # Could not parse query

            # Show interpretation (only if we have a valid intent)
            if query_intent:
                with st.expander("🔍 How I interpreted your question", expanded=(query_intent.confidence < 0.85)):
                    intent_names = {
                        "DESCRIBE": "Descriptive Statistics",
                        "COMPARE_GROUPS": "Compare Groups",
                        "FIND_PREDICTORS": "Find Risk Factors/Predictors",
                        "SURVIVAL": "Survival Analysis",
                        "CORRELATIONS": "Correlation Analysis",
                    }

                    st.write(
                        f"**Analysis Type**: {intent_names.get(query_intent.intent_type, query_intent.intent_type)}"
                    )

                    if query_intent.primary_variable:
                        st.write(f"**Primary Variable**: {query_intent.primary_variable}")
                    if query_intent.grouping_variable:
                        st.write(f"**Grouping Variable**: {query_intent.grouping_variable}")
                    if query_intent.predictor_variables:
                        st.write(f"**Predictor Variables**: {', '.join(query_intent.predictor_variables)}")
                    if query_intent.time_variable:
                        st.write(f"**Time Variable**: {query_intent.time_variable}")
                    if query_intent.event_variable:
                        st.write(f"**Event Variable**: {query_intent.event_variable}")

                    # Allow user to correct
                    if query_intent.confidence < 0.85:
                        correct = st.radio(
                            "Is this what you meant?",
                            ["Yes, that's correct", "No, let me clarify"],
                            key="nl_query_confirm",
                        )

                        if "No" in correct:
                            st.info("💡 Try rephrasing your question.")
                            return None

                # Convert QueryIntent to AnalysisContext
                context = AnalysisContext()

                # Map intent type
                intent_map = {
                    "DESCRIBE": AnalysisIntent.DESCRIBE,
                    "COMPARE_GROUPS": AnalysisIntent.COMPARE_GROUPS,
                    "FIND_PREDICTORS": AnalysisIntent.FIND_PREDICTORS,
                    "SURVIVAL": AnalysisIntent.EXAMINE_SURVIVAL,
                    "CORRELATIONS": AnalysisIntent.EXPLORE_RELATIONSHIPS,
                    "COUNT": AnalysisIntent.COUNT,
                }
                context.inferred_intent = intent_map.get(query_intent.intent_type, AnalysisIntent.UNKNOWN)

                # Map variables
                context.research_question = query
                context.primary_variable = query_intent.primary_variable
                context.grouping_variable = query_intent.grouping_variable
                context.predictor_variables = query_intent.predictor_variables
                context.time_variable = query_intent.time_variable
                context.event_variable = query_intent.event_variable

                # Copy filters from QueryIntent to AnalysisContext
                context.filters = query_intent.filters

                # Convert QueryIntent to QueryPlan and store in context
                if dataset_version:
                    query_plan = nl_engine._intent_to_plan(query_intent, dataset_version)
                    context.query_plan = query_plan
                    # Use QueryPlan confidence (may differ from QueryIntent)
                    context.confidence = query_plan.confidence
                else:
                    # Fallback: use QueryIntent confidence if dataset_version not available
                    context.confidence = query_intent.confidence

                # Propagate collision suggestions to context
                context.match_suggestions = collision_suggestions

                # Set flags based on intent
                context.compare_groups = query_intent.intent_type == "COMPARE_GROUPS"
                context.find_predictors = query_intent.intent_type == "FIND_PREDICTORS"
                context.time_to_event = query_intent.intent_type == "SURVIVAL"

                logger.info(
                    "analysis_context_created",
                    query=query,
                    intent_type=context.inferred_intent.value,
                    primary_variable=context.primary_variable,
                    grouping_variable=context.grouping_variable,
                    confidence=context.confidence,
                    is_complete=context.is_complete_for_intent(),
                    dataset_id=dataset_id,
                    upload_id=upload_id,
                )

                return context

        except Exception as e:
            st.error(f"❌ Error parsing query: {str(e)}")
            st.info("💡 Please try rephrasing your question.")
            return None
