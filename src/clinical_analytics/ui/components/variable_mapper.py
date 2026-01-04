"""
Variable Mapping Wizard Component

Interactive UI component for mapping user columns to UnifiedCohort schema.
"""

from typing import Any

import pandas as pd
import streamlit as st


class VariableMappingWizard:
    """
    Interactive wizard for mapping uploaded columns to unified schema.

    Guides users through:
    1. Identifying patient ID column
    2. Selecting outcome variable
    3. Mapping optional time variables
    4. Categorizing predictors
    """

    @staticmethod
    def render_patient_id_selector(
        columns: list[str], suggested_column: str | None = None, key_prefix: str = "upload"
    ) -> str | None:
        """
        Render patient ID column selector.

        Args:
            columns: Available columns
            suggested_column: Auto-detected suggestion
            key_prefix: Unique key prefix for Streamlit widgets

        Returns:
            Selected column name or None
        """
        st.markdown("### 1️⃣ Patient ID Column")
        st.markdown("**Which column uniquely identifies each patient?**")

        if suggested_column:
            st.info(f"💡 Suggested: `{suggested_column}` (auto-detected)")

        # Set default index
        default_idx = 0
        if suggested_column and suggested_column in columns:
            default_idx = columns.index(suggested_column)

        selected = st.selectbox(
            "Select Patient ID Column",
            options=["(None)"] + columns,
            index=default_idx + 1 if suggested_column else 0,
            key=f"{key_prefix}_patient_id",
            help=("This column should contain unique values for each patient (e.g., patient_id, mrn, subject_id)"),
        )

        return None if selected == "(None)" else selected

    @staticmethod
    def render_outcome_selector(
        columns: list[str],
        variable_info: dict[str, dict],
        suggested_column: str | None = None,
        key_prefix: str = "upload",
    ) -> str | None:
        """
        Render outcome variable selector.

        Args:
            columns: Available columns
            variable_info: Variable type information from detector
            suggested_column: Auto-detected suggestion
            key_prefix: Unique key prefix for Streamlit widgets

        Returns:
            Selected column name or None
        """
        st.markdown("### 2️⃣ Outcome Variable")
        st.markdown("**What is the primary outcome you want to analyze?**")
        st.caption(
            "This is typically a yes/no, binary, or event indicator "
            "(e.g., death, hospitalization, response to treatment)"
        )

        if suggested_column:
            st.info(f"💡 Suggested: `{suggested_column}` (auto-detected as potential outcome)")

        # Filter to likely outcome columns (binary variables)
        binary_columns = [col for col in columns if variable_info.get(col, {}).get("type") == "binary"]

        if binary_columns:
            st.caption(f"📊 Binary variables found: {', '.join(binary_columns)}")

        # Set default index
        default_idx = 0
        if suggested_column and suggested_column in columns:
            default_idx = columns.index(suggested_column)

        selected = st.selectbox(
            "Select Outcome Column",
            options=["(None)"] + columns,
            index=default_idx + 1 if suggested_column else 0,
            key=f"{key_prefix}_outcome",
            help=("The outcome is your dependent variable - what you're trying to predict or explain"),
        )

        # Show outcome details if selected
        if selected and selected != "(None)":
            if selected in variable_info:
                info = variable_info[selected]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Type", info["type"].title())
                with col2:
                    if "values" in info["metadata"]:
                        values = info["metadata"]["values"]
                        st.metric("Unique Values", len(values))
                        st.caption(f"Values: {', '.join(str(v) for v in values[:5])}")

        return None if selected == "(None)" else selected

    @staticmethod
    def render_time_variable_selector(
        columns: list[str],
        variable_info: dict[str, dict],
        suggested_column: str | None = None,
        key_prefix: str = "upload",
    ) -> dict[str, Any] | None:
        """
        Render time variable selector (optional).

        Args:
            columns: Available columns
            variable_info: Variable type information
            suggested_column: Auto-detected suggestion
            key_prefix: Unique key prefix

        Returns:
            Dictionary with time configuration or None
        """
        st.markdown("### 3️⃣ Time Variables (Optional)")
        st.markdown("**Do you have time-to-event or date variables?**")
        st.caption("For survival analysis, longitudinal studies, or time-dependent analyses")

        with st.expander("Add Time Variables"):
            has_time = st.checkbox("My data includes time or date variables", key=f"{key_prefix}_has_time")

            if not has_time:
                return None

            # Show datetime columns if available
            datetime_columns = [col for col in columns if variable_info.get(col, {}).get("type") == "datetime"]

            if datetime_columns:
                st.info(f"📅 Date/time columns found: {', '.join(datetime_columns)}")

            # Select time zero (baseline)
            time_zero_col = st.selectbox(
                "Baseline Time (Time Zero)",
                options=["(None)"] + columns,
                key=f"{key_prefix}_time_zero",
                help="Date of study entry, diagnosis, or treatment start",
            )

            # Select event time (optional)
            event_time_col = st.selectbox(
                "Event Time (Optional)",
                options=["(None)"] + columns,
                key=f"{key_prefix}_event_time",
                help="Date of outcome event (for survival analysis)",
            )

            if time_zero_col != "(None)" or event_time_col != "(None)":
                return {
                    "time_zero": time_zero_col if time_zero_col != "(None)" else None,
                    "event_time": event_time_col if event_time_col != "(None)" else None,
                }

        return None

    @staticmethod
    def render_variable_roles(
        columns: list[str],
        variable_info: dict[str, dict],
        excluded_columns: list[str],
        key_prefix: str = "upload",
    ) -> dict[str, list[str]]:
        """
        Render variable role assignment.

        Args:
            columns: Available columns
            variable_info: Variable type information
            excluded_columns: Columns already assigned (ID, outcome, time)
            key_prefix: Unique key prefix

        Returns:
            Dictionary mapping roles to column lists
        """
        st.markdown("### 4️⃣ Variable Roles")
        st.markdown("**Categorize the remaining variables:**")

        # Filter available columns
        available = [col for col in columns if col not in excluded_columns]

        if not available:
            st.info("All columns have been assigned roles")
            return {"predictors": [], "exclude": []}

        st.caption(f"{len(available)} variables available for analysis")

        # Show variable type summary
        with st.expander("📊 Variable Type Summary"):
            type_counts: dict[str, int] = {}
            for col in available:
                var_type = variable_info.get(col, {}).get("type", "unknown")
                type_counts[var_type] = type_counts.get(var_type, 0) + 1

            cols = st.columns(len(type_counts))
            for idx, (var_type, count) in enumerate(type_counts.items()):
                cols[idx].metric(var_type.title(), count)

        # Default: include all as predictors
        st.markdown("**Select variables to include in analysis:**")

        selected_predictors = st.multiselect(
            "Predictor Variables (Risk Factors / Features)",
            options=available,
            default=available,  # Default to including all
            key=f"{key_prefix}_predictors",
            help="These will be used as independent variables in your analysis",
        )

        # Excluded variables
        excluded = [col for col in available if col not in selected_predictors]

        if excluded:
            st.caption(f"⚠️ Excluded variables ({len(excluded)}): {', '.join(excluded)}")

        return {"predictors": selected_predictors, "exclude": excluded}

    @classmethod
    def render_complete_wizard(
        cls,
        df: pd.DataFrame,
        variable_info: dict[str, dict],
        suggestions: dict[str, str | None],
        key_prefix: str = "upload",
    ) -> dict[str, Any] | None:
        """
        Render complete mapping wizard.

        Args:
            df: DataFrame being mapped
            variable_info: Variable type detection results
            suggestions: Auto-detected suggestions
            key_prefix: Unique key prefix

        Returns:
            Complete mapping configuration or None if incomplete
        """
        columns = list(df.columns)

        st.markdown("## 🗺️ Variable Mapping Wizard")
        st.markdown("Map your data columns to the standard analysis format.")

        # Step 1: Patient ID
        patient_id = cls.render_patient_id_selector(
            columns, suggested_column=suggestions.get("patient_id"), key_prefix=key_prefix
        )

        if not patient_id:
            st.warning("⚠️ Please select a patient ID column to continue")
            return None

        st.divider()

        # Step 2: Outcome
        outcome = cls.render_outcome_selector(
            columns,
            variable_info,
            suggested_column=suggestions.get("outcome"),
            key_prefix=key_prefix,
        )

        if not outcome:
            st.warning("⚠️ Please select an outcome variable to continue")
            return None

        st.divider()

        # Step 3: Time variables (optional)
        time_config = cls.render_time_variable_selector(
            columns,
            variable_info,
            suggested_column=suggestions.get("time_zero"),
            key_prefix=key_prefix,
        )

        st.divider()

        # Step 4: Remaining variables
        excluded_so_far = [patient_id, outcome]
        if time_config:
            if time_config.get("time_zero"):
                excluded_so_far.append(time_config["time_zero"])
            if time_config.get("event_time"):
                excluded_so_far.append(time_config["event_time"])

        variable_roles = cls.render_variable_roles(columns, variable_info, excluded_so_far, key_prefix=key_prefix)

        # Build complete mapping
        mapping = {
            "patient_id": patient_id,
            "outcome": outcome,
            "time_variables": time_config,
            "predictors": variable_roles["predictors"],
            "excluded": variable_roles["exclude"],
        }

        return mapping
