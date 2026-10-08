"""Clarifying Questions Engine for refining low-confidence NL queries.

Pure Python logic for generating clarification requests.
No UI dependencies - returns data structures for UI layer to render.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from clinical_analytics.core.nl_query_engine import QueryIntent

logger = structlog.get_logger()


@dataclass
class ClarificationRequest:
    """Data structure for UI layer to render clarifying questions.

    Pure Python dataclass - no UI dependencies.
    The UI layer (Streamlit, Electron, etc.) is responsible for rendering.
    """

    question_type: Literal["intent", "variable", "grouping", "collision", "quality_warning"]
    prompt: str
    options: list[str]
    current_value: str | None = None
    help_text: str | None = None
    variable_name: str | None = None  # For collision resolution


def generate_clarifications(
    intent: "QueryIntent",
    semantic_layer: Any,
    available_columns: list[str],
) -> list[ClarificationRequest]:
    """Generate list of clarifications needed for low-confidence queries.

    Pure logic - returns data structures, does NOT render UI.

    Args:
        intent: Low-confidence QueryIntent to refine
        semantic_layer: SemanticLayer instance for metadata access
        available_columns: List of column names (extracted from cohort before calling)

    Returns:
        List of ClarificationRequest objects for UI layer to render
    """
    from clinical_analytics.core.column_parser import parse_column_name
    from clinical_analytics.core.nl_query_config import ENABLE_CLARIFYING_QUESTIONS
    from clinical_analytics.core.nl_query_engine import VALID_INTENT_TYPES

    if not ENABLE_CLARIFYING_QUESTIONS:
        return []  # Feature flag disabled

    log = logger.bind(intent_type=intent.intent_type, confidence=intent.confidence)
    log.info("generate_clarifications_start")

    clarifications: list[ClarificationRequest] = []

    # 1. If intent_type is ambiguous or missing, ask user to select
    if not intent.intent_type or intent.confidence < 0.3:
        clarifications.append(
            ClarificationRequest(
                question_type="intent",
                prompt="What type of analysis do you want?",
                options=VALID_INTENT_TYPES,
                current_value=intent.intent_type,
                help_text="Select the type of analysis you're looking for",
            )
        )

    # 2. If primary_variable missing, show available columns
    if not intent.primary_variable:
        # Get display names using parse_column_name
        column_options = []
        for canonical_name in available_columns:
            meta = parse_column_name(canonical_name)
            display_name = meta.display_name or canonical_name
            column_options.append(display_name)

        if column_options:
            clarifications.append(
                ClarificationRequest(
                    question_type="variable",
                    prompt="Which variable are you interested in?",
                    options=column_options,
                    current_value=None,
                    help_text="Select the main outcome or variable you want to analyze",
                )
            )

    # 3. If grouping_variable missing for COMPARE_GROUPS
    if intent.intent_type == "COMPARE_GROUPS" and not intent.grouping_variable:
        available_dims = semantic_layer.get_available_dimensions()
        if available_dims:
            dim_options = list(available_dims.keys())
            clarifications.append(
                ClarificationRequest(
                    question_type="grouping",
                    prompt="How do you want to group the data?",
                    options=dim_options,
                    current_value=None,
                    help_text="This splits your data into groups (e.g., treatment arm, sex, age group)",
                )
            )

    # 4. Handle collisions
    if intent.primary_variable:
        suggestions = semantic_layer.get_collision_suggestions(intent.primary_variable)
        if suggestions:
            # Get display names for collision options
            display_options = []
            for opt in suggestions:
                meta = parse_column_name(opt)
                display_options.append(meta.display_name or opt)

            clarifications.append(
                ClarificationRequest(
                    question_type="collision",
                    prompt=f"Which '{intent.primary_variable}' did you mean?",
                    options=display_options,
                    current_value=None,
                    variable_name="primary_variable",
                )
            )

    # 5. Surface quality warnings (informational, not requiring selection)
    quality_warnings = semantic_layer.get_data_quality_warnings()
    if quality_warnings and intent.primary_variable:
        # Filter warnings relevant to selected variable
        relevant_warnings = [w for w in quality_warnings if w.get("column") == intent.primary_variable]
        for warning in relevant_warnings:
            clarifications.append(
                ClarificationRequest(
                    question_type="quality_warning",
                    prompt=warning.get("message", "Data quality issue detected"),
                    options=[],  # Informational only
                    variable_name=intent.primary_variable,
                )
            )

    log.info("generate_clarifications_complete", clarification_count=len(clarifications))
    return clarifications


def apply_clarification_response(
    intent: "QueryIntent",
    clarification: ClarificationRequest,
    selected_value: str,
    available_columns: list[str],
) -> "QueryIntent":
    """Apply user's response to a clarification request.

    Updates the intent based on user selection.

    Args:
        intent: Original QueryIntent to refine
        clarification: The clarification that was answered
        selected_value: User's selected option
        available_columns: List of column names for reverse mapping

    Returns:
        Updated QueryIntent with higher confidence
    """
    from clinical_analytics.core.column_parser import parse_column_name

    log = logger.bind(
        question_type=clarification.question_type,
        selected_value=selected_value,
    )

    if clarification.question_type == "intent":
        intent.intent_type = selected_value
        intent.confidence = max(intent.confidence, 0.6)
        log.info("intent_type_updated", new_intent=selected_value)

    elif clarification.question_type == "variable":
        # Reverse map display name to canonical name
        for canonical_name in available_columns:
            meta = parse_column_name(canonical_name)
            display_name = meta.display_name or canonical_name
            if display_name == selected_value:
                intent.primary_variable = canonical_name
                intent.confidence = max(intent.confidence, 0.7)
                log.info("primary_variable_updated", new_variable=canonical_name)
                break

    elif clarification.question_type == "grouping":
        intent.grouping_variable = selected_value
        intent.confidence = max(intent.confidence, 0.7)
        log.info("grouping_variable_updated", new_grouping=selected_value)

    elif clarification.question_type == "collision":
        # Reverse map display name to canonical name
        for canonical_name in available_columns:
            meta = parse_column_name(canonical_name)
            display_name = meta.display_name or canonical_name
            if display_name == selected_value:
                if clarification.variable_name == "primary_variable":
                    intent.primary_variable = canonical_name
                    intent.confidence = max(intent.confidence, 0.8)
                    log.info("collision_resolved", variable=canonical_name)
                break

    return intent


class ClarifyingQuestionsEngine:
    """Interactive clarifying questions to refine low-confidence queries.

    DEPRECATED: Use generate_clarifications() and apply_clarification_response() directly.
    This class is kept for backward compatibility during migration.

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
        semantic_layer: Any,
        available_columns: list[str],
    ) -> "QueryIntent":
        """Ask targeted questions to refine intent using semantic layer metadata.

        DEPRECATED: This method was designed for Streamlit. For API/Electron usage,
        use generate_clarifications() + apply_clarification_response() instead.

        This method now returns the intent unchanged - UI rendering must be done
        by the UI layer (see ui/components/clarifying_ui.py).

        Args:
            intent: Low-confidence QueryIntent to refine
            semantic_layer: SemanticLayer instance for metadata access
            available_columns: List of column names (extracted from cohort before calling)

        Returns:
            Original QueryIntent (unchanged - UI must handle clarification flow)
        """
        # Generate clarifications but don't render - return for UI layer to handle
        clarifications = generate_clarifications(intent, semantic_layer, available_columns)

        if clarifications:
            logger.info(
                "clarifications_generated",
                count=len(clarifications),
                types=[c.question_type for c in clarifications],
            )
            # Store clarifications on intent for UI layer to access
            # The UI layer must render these and call apply_clarification_response()
            intent._pending_clarifications = clarifications  # type: ignore[attr-defined]

        return intent
