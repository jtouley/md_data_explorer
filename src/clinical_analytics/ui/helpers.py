"""
UI Helper Functions

Shared utilities for Streamlit pages.
"""

from typing import TYPE_CHECKING

import streamlit as st

from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.ui.config import ASK_QUESTIONS_PAGE, V1_MVP_MODE

if TYPE_CHECKING:
    # PANDAS EXCEPTION: ClinicalDataset.get_cohort() returns pd.DataFrame
    import pandas as pd


def gate_v1_mvp_legacy_page() -> bool:
    """
    Gate legacy analysis pages in V1 MVP mode.

    Shows redirect message and stops execution if V1_MVP_MODE is enabled.
    MUST be called at the start of main() before any expensive operations.

    Returns:
        True if gated (execution should stop), False if allowed to proceed
    """
    if not V1_MVP_MODE:
        return False

    st.info("🚧 This page is disabled in V1 MVP mode. Use the **Ask Questions** page for all analysis.")
    st.markdown(
        """
        **V1 MVP focuses on:**
        - Upload your data
        - Ask questions in natural language
        - Get answers with SQL preview

        All analysis is available through the Chat interface on the Ask Questions page.
        """
    )
    if st.button("Go to Ask Questions Page"):
        st.switch_page(ASK_QUESTIONS_PAGE)
    st.stop()

    return True  # Unreachable, but makes type checker happy


def require_outcome(cohort: "pd.DataFrame", analysis_name: str) -> None:
    """
    Check if outcome column exists in cohort, show error and stop if missing.

    Args:
        cohort: DataFrame to check (must have UnifiedCohort.OUTCOME column)
        analysis_name: Name of analysis requiring outcome (for error message)

    Raises:
        st.stop(): Stops Streamlit execution if outcome is missing
    """
    if UnifiedCohort.OUTCOME not in cohort.columns:
        st.error(f"❌ **{analysis_name} requires an outcome variable.**")
        st.markdown(
            """
            This dataset doesn't have an outcome column mapped.

            **To fix this:**
            1. Go to **📤 Add Your Data** page
            2. Re-upload your dataset and map a column to **outcome** in the mapping step
            3. Or use a dataset that already has an outcome defined

            **Note:** The semantic layer can map any column to outcome - you don't need
            a column literally named "outcome" in your raw data.
            """
        )
        st.stop()
