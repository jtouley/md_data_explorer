"""
Question Engine - Conversational analysis configuration through questions.

Guides users through analysis setup by asking natural questions,
infers the appropriate statistical test, and dynamically configures analysis.
Supports both free-form natural language queries and structured questions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
import streamlit as st


class AnalysisIntent(Enum):
    """Inferred analysis intentions (hidden from user)."""

    DESCRIBE = "describe"
    COMPARE_GROUPS = "compare_groups"
    FIND_PREDICTORS = "find_predictors"
    EXAMINE_SURVIVAL = "examine_survival"
    EXPLORE_RELATIONSHIPS = "explore_relationships"
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

    # Metadata
    variable_types: dict[str, str] = field(default_factory=dict)
    match_suggestions: dict[str, list[str]] = field(default_factory=dict)  # {query_term: [canonical_names]}

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
            return len(self.predictor_variables) >= 2

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
            if len(self.predictor_variables) < 2:
                missing.append("at least 2 variables to examine relationships")

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
    def ask_initial_question(df: pd.DataFrame) -> str | None:
        """
        Ask the first question: What do you want to know?

        Returns the user's selection as a string intent signal.
        """
        st.markdown("## 💬 What do you want to know about your data?")

        st.markdown("""
        Choose what you'd like to understand. Don't worry about statistical tests -
        I'll figure out the right approach based on your question.
        """)

        question_type = st.radio(
            "I want to...",
            [
                "📊 See what's in my data (describe patient characteristics)",
                "📈 Compare something between groups (e.g., treatment vs control)",
                "🎯 Find what predicts or causes an outcome",
                "⏱️ Analyze time until an event happens (survival analysis)",
                "🔗 See how variables relate to each other",
                "💭 I'm not sure - help me figure it out",
            ],
            label_visibility="collapsed",
        )

        # Map to intent signals
        intent_map = {
            "📊 See what's in my data": "describe",
            "📈 Compare something between groups": "compare",
            "🎯 Find what predicts or causes an outcome": "predict",
            "⏱️ Analyze time until an event happens": "survival",
            "🔗 See how variables relate to each other": "relationships",
            "💭 I'm not sure": "help",
        }

        for key, value in intent_map.items():
            if key in question_type:
                return value

        return None

    @staticmethod
    def ask_help_questions(df: pd.DataFrame) -> dict[str, Any]:
        """
        If user selects 'help me figure it out', ask clarifying questions.

        Returns answers that help infer intent.
        """
        st.markdown("### Let me ask a few questions to understand what you need:")

        answers = {}

        # Question 1: Do they have an outcome?
        has_outcome = st.radio(
            "Do you have a specific outcome or result you're interested in?",
            [
                "Yes, I want to understand what affects a specific outcome",
                "No, I just want to explore and describe my data",
            ],
        )
        answers["has_outcome"] = "yes" in has_outcome.lower()

        if answers["has_outcome"]:
            # Question 2: Groups or predictors?
            approach = st.radio(
                "What do you want to know about this outcome?",
                [
                    "Compare it between groups (e.g., does treatment affect outcome?)",
                    "Find what variables predict or cause it",
                ],
            )
            answers["approach"] = "compare" if "compare" in approach.lower() else "predict"

        # Question 3: Time element?
        has_time = st.radio(
            "Does your question involve time?",
            [
                "Yes - I want to know how long until something happens",
                "No - time isn't important for my question",
            ],
        )
        answers["has_time"] = "yes" in has_time.lower()

        return answers

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
    def build_context_from_intent(intent_signal: str, df: pd.DataFrame) -> AnalysisContext:
        """
        Build initial context based on user's intent signal.

        Args:
            intent_signal: One of 'describe', 'compare', 'predict', 'survival', 'relationships'
            df: DataFrame being analyzed

        Returns:
            Initialized AnalysisContext
        """
        context = AnalysisContext()

        # Map intent signal to flags
        if intent_signal == "describe":
            context.inferred_intent = AnalysisIntent.DESCRIBE
        elif intent_signal == "compare":
            context.compare_groups = True
            context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        elif intent_signal == "predict":
            context.find_predictors = True
            context.inferred_intent = AnalysisIntent.FIND_PREDICTORS
        elif intent_signal == "survival":
            context.time_to_event = True
            context.inferred_intent = AnalysisIntent.EXAMINE_SURVIVAL
        elif intent_signal == "relationships":
            context.inferred_intent = AnalysisIntent.EXPLORE_RELATIONSHIPS

        return context

    @staticmethod
    def render_progress_indicator(context: AnalysisContext):
        """Show user how much information we still need."""
        missing = context.get_missing_info()

        if missing:
            st.info(f"ℹ️ I still need to know: {', '.join(missing)}")
        else:
            st.success("✅ I have everything I need to run the analysis!")

    @staticmethod
    def ask_free_form_question(semantic_layer) -> AnalysisContext | None:
        """
        Ask user to type their question in natural language.

        Uses NLQueryEngine to parse the query and convert to AnalysisContext.
        Shows confidence and interpretation, asks for clarification if needed.

        Args:
            semantic_layer: SemanticLayer instance for NL parsing

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

        # Parse query with NLQueryEngine
        with st.spinner("Understanding your question..."):
            try:
                from clinical_analytics.core.nl_query_engine import NLQueryEngine

                # Initialize NL query engine
                nl_engine = NLQueryEngine(semantic_layer)

                # Parse query
                query_intent = nl_engine.parse_query(query)

                # Extract variables with collision suggestions
                matched_vars, collision_suggestions = nl_engine._extract_variables_from_query(query)

                # Show confidence
                if query_intent.confidence > 0.75:
                    st.success(f"✅ I understand! (Confidence: {query_intent.confidence:.0%})")
                elif query_intent.confidence > 0.5:
                    st.warning(f"⚠️ I think I understand, but please verify (Confidence: {query_intent.confidence:.0%})")
                else:
                    st.info("🤔 I'm not sure what you're asking. Let me ask some clarifying questions...")
                    return None  # Fall back to structured questions

                # Show interpretation
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
                            st.info("💡 Try rephrasing your question or use the structured questions below.")
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
                }
                context.inferred_intent = intent_map.get(query_intent.intent_type, AnalysisIntent.UNKNOWN)

                # Map variables
                context.research_question = query
                context.primary_variable = query_intent.primary_variable
                context.grouping_variable = query_intent.grouping_variable
                context.predictor_variables = query_intent.predictor_variables
                context.time_variable = query_intent.time_variable
                context.event_variable = query_intent.event_variable

                # Propagate collision suggestions to context
                context.match_suggestions = collision_suggestions

                # Set flags based on intent
                context.compare_groups = query_intent.intent_type == "COMPARE_GROUPS"
                context.find_predictors = query_intent.intent_type == "FIND_PREDICTORS"
                context.time_to_event = query_intent.intent_type == "SURVIVAL"

                return context

            except Exception as e:
                st.error(f"❌ Error parsing query: {str(e)}")
                st.info("💡 Please try using the structured questions below instead.")
                return None
