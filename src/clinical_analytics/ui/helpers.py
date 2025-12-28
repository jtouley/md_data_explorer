"""
UI Helper Functions

Shared utilities for Streamlit pages.
"""

import streamlit as st

from clinical_analytics.ui.config import ASK_QUESTIONS_PAGE, V1_MVP_MODE


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
