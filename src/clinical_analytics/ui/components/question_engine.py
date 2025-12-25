"""
Question Engine - Conversational analysis configuration through questions.

Guides users through analysis setup by asking natural questions,
infers the appropriate statistical test, and dynamically configures analysis.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


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
    research_question: Optional[str] = None

    # Variables
    primary_variable: Optional[str] = None
    grouping_variable: Optional[str] = None
    predictor_variables: List[str] = field(default_factory=list)
    time_variable: Optional[str] = None
    event_variable: Optional[str] = None

    # Analysis configuration
    compare_groups: Optional[bool] = None
    find_predictors: Optional[bool] = None
    time_to_event: Optional[bool] = None

    # Inferred intent (hidden from user)
    inferred_intent: AnalysisIntent = AnalysisIntent.UNKNOWN

    # Metadata
    variable_types: Dict[str, str] = field(default_factory=dict)

    def is_complete_for_intent(self) -> bool:
        """Check if we have enough information for the inferred analysis."""
        if self.inferred_intent == AnalysisIntent.DESCRIBE:
            return True  # Just needs data

        elif self.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
            return self.primary_variable is not None and self.grouping_variable is not None

        elif self.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
            return self.primary_variable is not None and len(self.predictor_variables) > 0

        elif self.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
            return (self.time_variable is not None and
                   self.event_variable is not None)

        elif self.inferred_intent == AnalysisIntent.EXPLORE_RELATIONSHIPS:
            return len(self.predictor_variables) >= 2

        return False

    def get_missing_info(self) -> List[str]:
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
    def ask_initial_question(df: pd.DataFrame) -> Optional[str]:
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
                "💭 I'm not sure - help me figure it out"
            ],
            label_visibility="collapsed"
        )

        # Map to intent signals
        intent_map = {
            "📊 See what's in my data": "describe",
            "📈 Compare something between groups": "compare",
            "🎯 Find what predicts or causes an outcome": "predict",
            "⏱️ Analyze time until an event happens": "survival",
            "🔗 See how variables relate to each other": "relationships",
            "💭 I'm not sure": "help"
        }

        for key, value in intent_map.items():
            if key in question_type:
                return value

        return None

    @staticmethod
    def ask_help_questions(df: pd.DataFrame) -> Dict[str, Any]:
        """
        If user selects 'help me figure it out', ask clarifying questions.

        Returns answers that help infer intent.
        """
        st.markdown("### Let me ask a few questions to understand what you need:")

        answers = {}

        # Question 1: Do they have an outcome?
        has_outcome = st.radio(
            "Do you have a specific outcome or result you're interested in?",
            ["Yes, I want to understand what affects a specific outcome",
             "No, I just want to explore and describe my data"]
        )
        answers['has_outcome'] = 'yes' in has_outcome.lower()

        if answers['has_outcome']:
            # Question 2: Groups or predictors?
            approach = st.radio(
                "What do you want to know about this outcome?",
                ["Compare it between groups (e.g., does treatment affect outcome?)",
                 "Find what variables predict or cause it"]
            )
            answers['approach'] = 'compare' if 'compare' in approach.lower() else 'predict'

        # Question 3: Time element?
        has_time = st.radio(
            "Does your question involve time?",
            ["Yes - I want to know how long until something happens",
             "No - time isn't important for my question"]
        )
        answers['has_time'] = 'yes' in has_time.lower()

        return answers

    @staticmethod
    def select_primary_variable(df: pd.DataFrame, context: AnalysisContext,
                               prompt: str = "What do you want to measure or analyze?") -> Optional[str]:
        """
        Ask user to select their primary variable of interest.

        Uses plain language prompt based on context.
        """
        available_cols = [c for c in df.columns if c not in ['patient_id', 'time_zero']]

        st.markdown(f"### {prompt}")

        # Provide context-specific help text
        if context.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
            help_text = "This is your outcome - what you want to predict or explain"
        elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
            help_text = "This is what you want to compare between groups"
        else:
            help_text = "Select the main variable you're interested in"

        variable = st.selectbox(
            "Variable:",
            ['(Choose one)'] + available_cols,
            help=help_text
        )

        return None if variable == '(Choose one)' else variable

    @staticmethod
    def select_grouping_variable(df: pd.DataFrame, exclude: List[str] = None) -> Optional[str]:
        """Ask user to select groups to compare."""
        available_cols = [c for c in df.columns if c not in ['patient_id', 'time_zero']]
        if exclude:
            available_cols = [c for c in available_cols if c not in exclude]

        st.markdown("### Which groups do you want to compare?")

        variable = st.selectbox(
            "Grouping variable:",
            ['(Choose one)'] + available_cols,
            help="This splits your data into groups (e.g., treatment arm, sex, age group)"
        )

        return None if variable == '(Choose one)' else variable

    @staticmethod
    def select_predictor_variables(df: pd.DataFrame, exclude: List[str] = None,
                                   min_vars: int = 1) -> List[str]:
        """Ask user to select predictor variables."""
        available_cols = [c for c in df.columns if c not in ['patient_id', 'time_zero']]
        if exclude:
            available_cols = [c for c in available_cols if c not in exclude]

        st.markdown("### Which variables might affect or predict this?")

        variables = st.multiselect(
            "Predictor variables:",
            available_cols,
            default=[],
            help=f"Select at least {min_vars} variable(s) that might influence the outcome"
        )

        return variables

    @staticmethod
    def select_time_variables(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Ask user to select time and event variables for survival analysis."""
        available_cols = [c for c in df.columns if c not in ['patient_id', 'time_zero']]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### How do you measure time?")
            time_var = st.selectbox(
                "Time variable:",
                ['(Choose one)'] + available_cols,
                help="Time from start until event or censoring (e.g., days, months)"
            )

        with col2:
            st.markdown("### What event are you tracking?")
            remaining_cols = [c for c in available_cols if c != time_var]
            event_var = st.selectbox(
                "Event variable:",
                ['(Choose one)'] + remaining_cols,
                help="Binary: did the event happen? (1=yes, 0=no/censored)"
            )

        time_var = None if time_var == '(Choose one)' else time_var
        event_var = None if event_var == '(Choose one)' else event_var

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
        if intent_signal == 'describe':
            context.inferred_intent = AnalysisIntent.DESCRIBE
        elif intent_signal == 'compare':
            context.compare_groups = True
            context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        elif intent_signal == 'predict':
            context.find_predictors = True
            context.inferred_intent = AnalysisIntent.FIND_PREDICTORS
        elif intent_signal == 'survival':
            context.time_to_event = True
            context.inferred_intent = AnalysisIntent.EXAMINE_SURVIVAL
        elif intent_signal == 'relationships':
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
