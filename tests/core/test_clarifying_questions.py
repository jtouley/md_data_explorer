"""Tests for ClarifyingQuestionsEngine.

Updated for Phase 0 refactor: Core layer now returns data structures instead of
rendering Streamlit UI. Tests verify pure Python logic, not UI interactions.
"""

from unittest.mock import MagicMock, patch

import pytest
from clinical_analytics.core.clarifying_questions import (
    ClarificationRequest,
    ClarifyingQuestionsEngine,
    apply_clarification_response,
    generate_clarifications,
)
from clinical_analytics.core.nl_query_engine import QueryIntent


@pytest.fixture
def mock_semantic_layer():
    """Create a mock semantic layer for testing."""
    mock = MagicMock()
    mock.get_column_alias_index.return_value = {
        "mortality": "mortality",
        "treatment": "treatment_arm",
        "age": "age",
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_available_dimensions.return_value = {"treatment_arm": {"label": "Treatment Arm"}}
    mock.get_data_quality_warnings.return_value = []
    return mock


@pytest.fixture
def low_confidence_intent():
    """Create a low-confidence QueryIntent for testing."""
    return QueryIntent(intent_type="DESCRIBE", confidence=0.2)  # < 0.3 to trigger intent type question


class TestGenerateClarifications:
    """Test pure Python clarification generation logic."""

    def test_generate_clarifications_intent_ambiguous_returns_intent_question(
        self, mock_semantic_layer, low_confidence_intent
    ):
        """When intent is ambiguous, generate intent type clarification."""
        # Arrange
        with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
            from clinical_analytics.core.column_parser import ColumnMetadata

            mock_parse.return_value = ColumnMetadata(display_name="Mortality", canonical_name="mortality")

            # Act
            clarifications = generate_clarifications(
                low_confidence_intent, mock_semantic_layer, available_columns=["mortality"]
            )

            # Assert
            assert len(clarifications) >= 1
            intent_clarifications = [c for c in clarifications if c.question_type == "intent"]
            assert len(intent_clarifications) == 1
            assert intent_clarifications[0].prompt == "What type of analysis do you want?"

    def test_generate_clarifications_missing_variable_returns_variable_question(self, mock_semantic_layer):
        """When primary_variable missing, generate variable clarification."""
        # Arrange
        intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.5, primary_variable=None)

        with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
            from clinical_analytics.core.column_parser import ColumnMetadata

            mock_parse.return_value = ColumnMetadata(display_name="Mortality", canonical_name="mortality")

            # Act
            clarifications = generate_clarifications(intent, mock_semantic_layer, available_columns=["mortality"])

            # Assert
            variable_clarifications = [c for c in clarifications if c.question_type == "variable"]
            assert len(variable_clarifications) == 1
            assert variable_clarifications[0].prompt == "Which variable are you interested in?"

    def test_generate_clarifications_compare_groups_missing_grouping_returns_grouping_question(
        self, mock_semantic_layer
    ):
        """When grouping_variable missing for COMPARE_GROUPS, generate grouping clarification."""
        # Arrange
        intent = QueryIntent(
            intent_type="COMPARE_GROUPS",
            confidence=0.5,
            primary_variable="mortality",
            grouping_variable=None,
        )

        # Act
        clarifications = generate_clarifications(
            intent, mock_semantic_layer, available_columns=["mortality", "treatment"]
        )

        # Assert
        grouping_clarifications = [c for c in clarifications if c.question_type == "grouping"]
        assert len(grouping_clarifications) == 1
        assert grouping_clarifications[0].prompt == "How do you want to group the data?"

    def test_generate_clarifications_collisions_returns_collision_question(self, mock_semantic_layer):
        """When collisions exist, generate collision clarification."""
        # Arrange
        mock_semantic_layer.get_collision_suggestions.return_value = ["mortality_1", "mortality_2"]
        intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.5, primary_variable="mortality")

        with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
            from clinical_analytics.core.column_parser import ColumnMetadata

            def parse_side_effect(name):
                if name == "mortality_1":
                    return ColumnMetadata(display_name="Mortality 1", canonical_name="mortality_1")
                elif name == "mortality_2":
                    return ColumnMetadata(display_name="Mortality 2", canonical_name="mortality_2")
                return ColumnMetadata(display_name=name, canonical_name=name)

            mock_parse.side_effect = parse_side_effect

            # Act
            clarifications = generate_clarifications(
                intent, mock_semantic_layer, available_columns=["mortality_1", "mortality_2"]
            )

            # Assert
            collision_clarifications = [c for c in clarifications if c.question_type == "collision"]
            assert len(collision_clarifications) == 1
            assert "mortality" in collision_clarifications[0].prompt

    def test_generate_clarifications_quality_warnings_returns_warning(self, mock_semantic_layer):
        """Quality warnings are surfaced as informational clarifications."""
        # Arrange
        mock_semantic_layer.get_data_quality_warnings.return_value = [
            {"column": "mortality", "message": "15% missing values"}
        ]
        intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.5, primary_variable="mortality")

        # Act
        clarifications = generate_clarifications(intent, mock_semantic_layer, available_columns=["mortality"])

        # Assert
        warning_clarifications = [c for c in clarifications if c.question_type == "quality_warning"]
        assert len(warning_clarifications) == 1
        assert "15% missing values" in warning_clarifications[0].prompt


class TestApplyClarificationResponse:
    """Test applying user responses to clarifications."""

    def test_apply_clarification_response_intent_updates_intent_type(self):
        """User selecting intent type updates the intent."""
        # Arrange
        intent = QueryIntent(intent_type="DESCRIBE", confidence=0.2)
        clarification = ClarificationRequest(
            question_type="intent",
            prompt="What type of analysis?",
            options=["COMPARE_GROUPS", "DESCRIBE"],
        )

        # Act
        result = apply_clarification_response(intent, clarification, "COMPARE_GROUPS", available_columns=[])

        # Assert
        assert result.intent_type == "COMPARE_GROUPS"
        assert result.confidence >= 0.6

    def test_apply_clarification_response_variable_updates_primary_variable(self):
        """User selecting variable updates primary_variable."""
        # Arrange
        intent = QueryIntent(intent_type="DESCRIBE", confidence=0.5, primary_variable=None)
        clarification = ClarificationRequest(
            question_type="variable",
            prompt="Which variable?",
            options=["Mortality", "Age"],
        )

        with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
            from clinical_analytics.core.column_parser import ColumnMetadata

            def parse_side_effect(name):
                if name == "mortality":
                    return ColumnMetadata(display_name="Mortality", canonical_name="mortality")
                return ColumnMetadata(display_name=name, canonical_name=name)

            mock_parse.side_effect = parse_side_effect

            # Act
            result = apply_clarification_response(
                intent, clarification, "Mortality", available_columns=["mortality", "age"]
            )

            # Assert
            assert result.primary_variable == "mortality"
            assert result.confidence >= 0.7

    def test_apply_clarification_response_grouping_updates_grouping_variable(self):
        """User selecting grouping updates grouping_variable."""
        # Arrange
        intent = QueryIntent(
            intent_type="COMPARE_GROUPS",
            confidence=0.5,
            primary_variable="mortality",
            grouping_variable=None,
        )
        clarification = ClarificationRequest(
            question_type="grouping",
            prompt="How to group?",
            options=["treatment_arm"],
        )

        # Act
        result = apply_clarification_response(intent, clarification, "treatment_arm", available_columns=[])

        # Assert
        assert result.grouping_variable == "treatment_arm"
        assert result.confidence >= 0.7


class TestClarifyingQuestionsEngine:
    """Test backward compatibility and edge cases."""

    def test_ask_clarifying_questions_stores_pending_clarifications(self, mock_semantic_layer, low_confidence_intent):
        """ask_clarifying_questions stores clarifications for UI layer."""
        # Arrange
        with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
            from clinical_analytics.core.column_parser import ColumnMetadata

            mock_parse.return_value = ColumnMetadata(display_name="Mortality", canonical_name="mortality")

            # Act
            result = ClarifyingQuestionsEngine.ask_clarifying_questions(
                low_confidence_intent, mock_semantic_layer, available_columns=["mortality"]
            )

            # Assert: Clarifications are stored on intent for UI layer
            pending = getattr(result, "_pending_clarifications", None)
            assert pending is not None
            assert len(pending) >= 1

    def test_clarifying_questions_respects_feature_flag(self, mock_semantic_layer, low_confidence_intent):
        """Clarifying questions should respect ENABLE_CLARIFYING_QUESTIONS feature flag."""
        with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", False):
            result = ClarifyingQuestionsEngine.ask_clarifying_questions(
                low_confidence_intent, mock_semantic_layer, available_columns=["mortality"]
            )

            # Should return original intent when feature flag is disabled
            assert result.confidence == low_confidence_intent.confidence
            # No pending clarifications
            pending = getattr(result, "_pending_clarifications", None)
            assert pending is None or len(pending) == 0

    def test_generate_clarifications_returns_empty_when_feature_disabled(self, mock_semantic_layer):
        """generate_clarifications returns empty list when feature disabled."""
        with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", False):
            intent = QueryIntent(intent_type="DESCRIBE", confidence=0.2)
            result = generate_clarifications(intent, mock_semantic_layer, available_columns=["mortality"])

            assert result == []


# Keep these tests for backward compatibility during migration
def test_clarifying_questions_asks_about_intent_type(mock_semantic_layer, low_confidence_intent):
    """When intent is ambiguous, generate intent type clarification."""
    with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
        from clinical_analytics.core.column_parser import ColumnMetadata

        mock_parse.return_value = ColumnMetadata(display_name="Mortality", canonical_name="mortality")

        clarifications = generate_clarifications(
            low_confidence_intent, mock_semantic_layer, available_columns=["mortality", "treatment"]
        )

        # Verify intent clarification is generated
        intent_clarifications = [c for c in clarifications if c.question_type == "intent"]
        assert len(intent_clarifications) == 1


def test_clarifying_questions_uses_semantic_layer_metadata(mock_semantic_layer):
    """Clarifying questions should use semantic layer metadata for context."""
    with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
        from clinical_analytics.core.column_parser import ColumnMetadata

        mock_parse.return_value = ColumnMetadata(display_name="Mortality", canonical_name="mortality")

        intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.4, primary_variable=None)
        clarifications = generate_clarifications(intent, mock_semantic_layer, available_columns=["mortality"])

        # Verify variable clarification uses parse_column_name
        assert mock_parse.called
        variable_clarifications = [c for c in clarifications if c.question_type == "variable"]
        assert len(variable_clarifications) >= 1


def test_clarifying_questions_handles_collisions(mock_semantic_layer):
    """Clarifying questions should show collision suggestions when variables ambiguous."""
    mock_semantic_layer.get_collision_suggestions.return_value = ["mortality_1", "mortality_2"]

    intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.4, primary_variable="mortality")

    with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
        from clinical_analytics.core.column_parser import ColumnMetadata

        def parse_side_effect(name):
            if name == "mortality_1":
                return ColumnMetadata(display_name="Mortality 1", canonical_name="mortality_1")
            elif name == "mortality_2":
                return ColumnMetadata(display_name="Mortality 2", canonical_name="mortality_2")
            return ColumnMetadata(display_name=name, canonical_name=name)

        mock_parse.side_effect = parse_side_effect

        clarifications = generate_clarifications(
            intent, mock_semantic_layer, available_columns=["mortality_1", "mortality_2"]
        )

        # Verify collision clarification is generated
        mock_semantic_layer.get_collision_suggestions.assert_called()
        collision_clarifications = [c for c in clarifications if c.question_type == "collision"]
        assert len(collision_clarifications) == 1


def test_clarifying_questions_surfaces_quality_warnings(mock_semantic_layer):
    """Clarifying questions should show relevant quality warnings."""
    mock_semantic_layer.get_data_quality_warnings.return_value = [
        {"column": "mortality", "message": "15% missing values"}
    ]

    intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.4, primary_variable="mortality")

    clarifications = generate_clarifications(intent, mock_semantic_layer, available_columns=["mortality"])

    # Verify warning clarification is generated
    warning_clarifications = [c for c in clarifications if c.question_type == "quality_warning"]
    assert len(warning_clarifications) == 1


def test_clarifying_questions_asks_about_variables(mock_semantic_layer):
    """When variables are missing, generate variable clarifications."""
    intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.4, primary_variable=None)

    with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
        from clinical_analytics.core.column_parser import ColumnMetadata

        def parse_side_effect(name):
            if name == "mortality":
                return ColumnMetadata(display_name="Mortality", canonical_name="mortality")
            elif name == "treatment":
                return ColumnMetadata(display_name="Treatment", canonical_name="treatment")
            return ColumnMetadata(display_name=name, canonical_name=name)

        mock_parse.side_effect = parse_side_effect

        clarifications = generate_clarifications(
            intent, mock_semantic_layer, available_columns=["mortality", "treatment"]
        )

        # Verify variable clarifications are generated
        variable_clarifications = [c for c in clarifications if c.question_type == "variable"]
        assert len(variable_clarifications) >= 1
        # Verify display names are shown (using parse_column_name)
        assert mock_parse.called
