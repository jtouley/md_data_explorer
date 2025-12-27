"""
UI helper functions for common checks and validations.
"""

import pandas as pd
import streamlit as st

from clinical_analytics.core.schema import UnifiedCohort


def require_outcome(cohort: pd.DataFrame, analysis_name: str = "This analysis") -> None:
    """
    Require outcome column in cohort, show error and stop execution if missing.

    Args:
        cohort: Cohort DataFrame
        analysis_name: Name of analysis requiring outcome (for error message)

    Raises:
        SystemExit: If outcome is missing (stops page execution via st.stop())
    """
    if UnifiedCohort.OUTCOME not in cohort.columns:
        st.error(
            f"{analysis_name} requires an outcome variable, but none was found in the dataset. "
            "Please upload data with outcome mapping or select a dataset that includes outcomes."
        )
        st.stop()
