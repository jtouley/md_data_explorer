"""Clarifying Questions UI - Streamlit rendering for clarification requests.

Renders ClarificationRequest objects from core/clarifying_questions.py.
Streamlit-specific code isolated here.
"""

from typing import Any

import streamlit as st

from clinical_analytics.core.clarifying_questions import (
    ClarificationRequest,
    apply_clarification_response,
    generate_clarifications,
)
from clinical_analytics.core.nl_query_engine import QueryIntent


def render_clarifications(
    intent: QueryIntent,
    semantic_layer: Any,
    available_columns: list[str],
) -> QueryIntent:
    """Render clarifying questions and apply user responses.

    Streamlit-specific implementation of the clarification flow.

    Args:
        intent: Low-confidence QueryIntent to refine
        semantic_layer: SemanticLayer instance for metadata access
        available_columns: List of column names

    Returns:
        Refined QueryIntent with user selections applied
    """
    # Generate clarifications using pure Python logic
    clarifications = generate_clarifications(intent, semantic_layer, available_columns)

    if not clarifications:
        return intent

    # Render each clarification as Streamlit widgets
    for idx, clarification in enumerate(clarifications):
        if clarification.question_type == "quality_warning":
            # Informational warning, no selection needed
            st.warning(f"⚠️ Note: {clarification.prompt}")
            continue

        intent = _render_single_clarification(intent, clarification, available_columns, idx)

    return intent


def _render_single_clarification(
    intent: QueryIntent,
    clarification: ClarificationRequest,
    available_columns: list[str],
    idx: int,
) -> QueryIntent:
    """Render a single clarification question.

    Args:
        intent: Current QueryIntent
        clarification: ClarificationRequest to render
        available_columns: List of column names for reverse mapping
        idx: Index for unique widget keys

    Returns:
        Updated QueryIntent if user made selection
    """
    st.subheader(clarification.prompt)

    if clarification.help_text:
        st.caption(clarification.help_text)

    # Create unique key for widget
    widget_key = f"clarify_{clarification.question_type}_{idx}"

    if clarification.question_type == "collision":
        st.warning("⚠️ Some terms matched multiple columns. Please select:")

    # Render selectbox with options
    selected = st.selectbox(
        clarification.prompt,
        options=clarification.options,
        key=widget_key,
        label_visibility="collapsed",
    )

    if selected:
        intent = apply_clarification_response(intent, clarification, selected, available_columns)

    return intent


def render_clarifications_from_intent(
    intent: QueryIntent,
    available_columns: list[str],
) -> QueryIntent:
    """Render pending clarifications stored on intent.

    For backward compatibility with ClarifyingQuestionsEngine.ask_clarifying_questions().

    Args:
        intent: QueryIntent with _pending_clarifications attribute
        available_columns: List of column names

    Returns:
        Refined QueryIntent
    """
    pending = getattr(intent, "_pending_clarifications", None)
    if not pending:
        return intent

    for idx, clarification in enumerate(pending):
        if clarification.question_type == "quality_warning":
            st.warning(f"⚠️ Note: {clarification.prompt}")
            continue

        intent = _render_single_clarification(intent, clarification, available_columns, idx)

    # Clear pending clarifications after rendering
    intent._pending_clarifications = []  # type: ignore[attr-defined]

    return intent
