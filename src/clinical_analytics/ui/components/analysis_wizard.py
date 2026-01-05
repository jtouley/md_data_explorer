"""
Analysis Wizard Component

Guides clinicians to appropriate statistical tests based on their research question
and data characteristics.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st


@dataclass
class AnalysisType:
    """Definition of an analysis type."""

    id: str
    name: str
    icon: str
    description: str
    when_to_use: str
    requirements: dict[str, Any]
    page_path: str


class AnalysisRecommender:
    """
    Recommends appropriate statistical analyses based on data characteristics.

    Uses variable types, research questions, and data structure to suggest
    the most appropriate analysis method.
    """

    # Available analysis types
    ANALYSIS_TYPES = {
        "descriptive": AnalysisType(
            id="descriptive",
            name="Describe My Data",
            icon="📊",
            description="Create Table 1 with patient characteristics and summary statistics",
            when_to_use="When you want to describe your patient population or create demographic tables",
            requirements={"min_variables": 1, "outcome_required": False},
            page_path="2_📊_Descriptive_Stats",
        ),
        "compare_groups": AnalysisType(
            id="compare_groups",
            name="Compare Groups",
            icon="📈",
            description="Compare outcomes or characteristics between two or more groups",
            when_to_use="When you want to test if groups differ (t-test, chi-square, ANOVA)",
            requirements={"min_variables": 2, "grouping_variable": True, "outcome_required": True},
            page_path="3_📈_Compare_Groups",
        ),
        "risk_factors": AnalysisType(
            id="risk_factors",
            name="Identify Risk Factors",
            icon="🎯",
            description="Find which variables predict an outcome (logistic regression, linear regression)",
            when_to_use="When you want to identify predictors or risk factors for an outcome",
            requirements={
                "min_variables": 2,
                "outcome_required": True,
                "predictors_required": True,
            },
            page_path="4_🎯_Risk_Factors",
        ),
        "survival": AnalysisType(
            id="survival",
            name="Survival/Time-to-Event Analysis",
            icon="⏱️",
            description="Analyze time until an event occurs (Kaplan-Meier, Cox regression)",
            when_to_use="When you have time-to-event data (survival, time to discharge, etc.)",
            requirements={"time_variable": True, "event_variable": True},
            page_path="5_⏱️_Survival_Analysis",
        ),
        "correlation": AnalysisType(
            id="correlation",
            name="Explore Relationships",
            icon="🔗",
            description="Examine correlations and relationships between variables",
            when_to_use="When you want to see how variables relate to each other",
            requirements={"min_variables": 2, "numeric_variables": True},
            page_path="6_🔗_Correlations",
        ),
    }

    @classmethod
    def suggest_analyses(
        cls, df: pd.DataFrame, outcome_col: str | None = None, time_col: str | None = None
    ) -> list[tuple[AnalysisType, str]]:
        """
        Suggest appropriate analyses based on data characteristics.

        Args:
            df: DataFrame to analyze
            outcome_col: Name of outcome column (if any)
            time_col: Name of time variable (if any)

        Returns:
            List of (AnalysisType, reason) tuples
        """
        suggestions = []

        # Always suggest descriptive statistics
        suggestions.append(
            (
                cls.ANALYSIS_TYPES["descriptive"],
                "Always useful to describe your data and create Table 1",
            )
        )

        # Count variable types
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        n_numeric = len(numeric_cols)
        n_categorical = len(categorical_cols)
        n_total = len(df.columns)

        # Suggest group comparisons if we have grouping variables and outcome
        if outcome_col and (n_categorical > 0 or n_numeric > 1):
            suggestions.append(
                (
                    cls.ANALYSIS_TYPES["compare_groups"],
                    f"You have an outcome and {n_categorical} categorical variable(s) for grouping",
                )
            )

        # Suggest risk factor analysis if we have outcome and multiple predictors
        if outcome_col and n_total >= 3:
            suggestions.append(
                (
                    cls.ANALYSIS_TYPES["risk_factors"],
                    f"You have an outcome and {n_total - 1} potential predictor(s)",
                )
            )

        # Suggest survival analysis if we have time variable
        if time_col and outcome_col:
            suggestions.append(
                (
                    cls.ANALYSIS_TYPES["survival"],
                    "You have time-to-event data for survival analysis",
                )
            )

        # Suggest correlation if we have multiple numeric variables
        if n_numeric >= 2:
            suggestions.append(
                (
                    cls.ANALYSIS_TYPES["correlation"],
                    f"You have {n_numeric} numeric variables to explore relationships",
                )
            )

        return suggestions


class AnalysisWizard:
    """
    Interactive wizard component for selecting and configuring analyses.
    """

    @staticmethod
    def render_analysis_selector(
        df: pd.DataFrame, outcome_col: str | None = None, time_col: str | None = None
    ) -> str | None:
        """
        Render the "I want to..." analysis selector.

        Args:
            df: DataFrame being analyzed
            outcome_col: Outcome variable (if specified)
            time_col: Time variable (if specified)

        Returns:
            Selected analysis type ID or None
        """
        st.markdown("## 🧭 Choose Your Analysis")
        st.markdown(
            """
        Select what you want to do with your data. Don't worry about statistical terminology -
        we'll guide you to the right test.
        """
        )

        # Get suggestions
        suggestions = AnalysisRecommender.suggest_analyses(df, outcome_col, time_col)

        # Show suggested analyses
        st.markdown("### 💡 Suggested Analyses")
        st.caption("Based on your data characteristics")

        suggested_ids = {analysis.id for analysis, _ in suggestions}

        # Create analysis cards
        cols = st.columns(2)

        for idx, (analysis, reason) in enumerate(suggestions):
            with cols[idx % 2]:
                with st.container():
                    st.markdown(f"### {analysis.icon} {analysis.name}")
                    st.caption(reason)
                    st.markdown(f"**When to use:** {analysis.when_to_use}")

                    if st.button(
                        f"Start {analysis.name}",
                        key=f"select_{analysis.id}",
                        type="primary" if idx == 0 else "secondary",
                        width="stretch",
                    ):
                        return analysis.id

                    st.divider()

        # Show all other analyses
        other_analyses = [a for a in AnalysisRecommender.ANALYSIS_TYPES.values() if a.id not in suggested_ids]

        if other_analyses:
            with st.expander("📚 All Available Analyses"):
                for analysis in other_analyses:
                    st.markdown(f"**{analysis.icon} {analysis.name}**")
                    st.caption(analysis.description)
                    st.caption(f"*When to use: {analysis.when_to_use}*")

                    if st.button(
                        f"Start {analysis.name}",
                        key=f"select_other_{analysis.id}",
                        width="stretch",
                    ):
                        return analysis.id

                    st.markdown("---")

        return None

    @staticmethod
    def explain_test_choice(test_name: str, variable_types: dict[str, str], outcome_type: str | None = None) -> None:
        """
        Explain why a particular test was chosen.

        Args:
            test_name: Name of statistical test
            variable_types: Dictionary of variable names to types
            outcome_type: Type of outcome variable
        """
        st.info(f"**📖 Why {test_name}?**")

        explanations = {
            "Chi-square test": {
                "reason": "You're comparing categorical (yes/no) data between groups",
                "assumptions": [
                    "Independent observations",
                    "Expected frequency ≥5 in most cells",
                    "Categorical variables",
                ],
                "interpretation": "Tests if group proportions are significantly different",
            },
            "T-test": {
                "reason": "You're comparing a continuous variable between two groups",
                "assumptions": [
                    "Independent observations",
                    "Normally distributed data (or large sample)",
                    "Similar variance between groups",
                ],
                "interpretation": "Tests if group means are significantly different",
            },
            "ANOVA": {
                "reason": "You're comparing a continuous variable across 3+ groups",
                "assumptions": [
                    "Independent observations",
                    "Normally distributed data",
                    "Equal variance across groups",
                ],
                "interpretation": "Tests if at least one group mean differs",
            },
            "Logistic Regression": {
                "reason": f"You have a binary outcome ({outcome_type}) and want to find predictors",
                "assumptions": [
                    "Independent observations",
                    "Binary outcome",
                    "Linear relationship between predictors and log-odds",
                ],
                "interpretation": "Estimates odds ratios for each predictor",
            },
            "Linear Regression": {
                "reason": "You have a continuous outcome and want to find predictors",
                "assumptions": [
                    "Independent observations",
                    "Linear relationships",
                    "Normally distributed residuals",
                    "Homoscedasticity (constant variance)",
                ],
                "interpretation": "Estimates effect size for each predictor",
            },
            "Kaplan-Meier": {
                "reason": "You're analyzing time until an event occurs",
                "assumptions": ["Independent censoring", "Time starts at a common point"],
                "interpretation": "Shows survival probability over time",
            },
            "Cox Regression": {
                "reason": "You want to find predictors of time-to-event",
                "assumptions": [
                    "Proportional hazards",
                    "Independent observations",
                    "Censoring is non-informative",
                ],
                "interpretation": "Estimates hazard ratios for each predictor",
            },
        }

        if test_name in explanations:
            info = explanations[test_name]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Why this test:**")
                st.markdown(info["reason"])

            with col2:
                st.markdown("**What it tells you:**")
                st.markdown(info["interpretation"])

            with st.expander("📋 Assumptions to Check"):
                for assumption in info["assumptions"]:
                    st.markdown(f"- {assumption}")

    @staticmethod
    def render_variable_selector(
        df: pd.DataFrame,
        role: str,
        multiple: bool = False,
        filter_types: list[str] | None = None,
        exclude_cols: list[str] | None = None,
    ) -> str | None | list[str] | None:
        """
        Render a variable selector with helpful guidance.

        Args:
            df: DataFrame
            role: Role of variable (e.g., "outcome", "predictor", "grouping")
            multiple: Allow multiple selection
            filter_types: Only show variables of these types
            exclude_cols: Exclude these columns

        Returns:
            Selected column(s) or None
        """
        available_cols = list(df.columns)

        if exclude_cols:
            available_cols = [c for c in available_cols if c not in exclude_cols]

        # Apply type filter if specified
        if filter_types:
            if "numeric" in filter_types:
                numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
                available_cols = [c for c in available_cols if c in numeric_cols]
            elif "categorical" in filter_types:
                cat_cols = df.select_dtypes(include=["object", "category"]).columns
                available_cols = [c for c in available_cols if c in cat_cols]

        if not available_cols:
            st.warning(f"No suitable variables found for {role}")
            return None

        # Render selector
        if multiple:
            selected_multiple = st.multiselect(
                f"Select {role} variable(s)",
                available_cols,
                help=f"Choose one or more variables to use as {role}(s)",
            )
            return selected_multiple if selected_multiple else None
        else:
            selected_single: str | None = st.selectbox(
                f"Select {role} variable",
                ["(None)"] + available_cols,
                help=f"Choose the variable to use as {role}",
            )
            return None if selected_single == "(None)" else selected_single
