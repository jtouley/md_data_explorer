"""Tests for ClarifyingQuestionsEngine."""

from unittest.mock import MagicMock, patch

import pytest

from clinical_analytics.core.clarifying_questions import ClarifyingQuestionsEngine
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


def test_clarifying_questions_asks_about_intent_type(mock_semantic_layer, low_confidence_intent):
    """When intent is ambiguous, ask user to select analysis type."""

    # Mock st.selectbox to return different values based on first argument (label)
    def selectbox_side_effect(label, *args, **kwargs):
        # First argument is the label
        if "Analysis Type" in str(label):
            return "COMPARE_GROUPS"
        elif "Primary Variable" in str(label):
            return "Mortality"
        return "Mortality"

    with patch("streamlit.selectbox", side_effect=selectbox_side_effect):
        with patch("streamlit.subheader"):
            with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
                from clinical_analytics.core.column_parser import ColumnMetadata

                mock_parse.return_value = ColumnMetadata(display_name="Mortality", canonical_name="mortality")

                refined_intent = ClarifyingQuestionsEngine.ask_clarifying_questions(
                    low_confidence_intent, mock_semantic_layer, available_columns=["mortality", "treatment"]
                )

                # Verify refined intent has intent_type="COMPARE_GROUPS" and higher confidence
                assert refined_intent.intent_type == "COMPARE_GROUPS"
                assert refined_intent.confidence >= 0.6  # Should be boosted after user selection


def test_clarifying_questions_uses_semantic_layer_metadata(mock_semantic_layer, low_confidence_intent):
    """Clarifying questions should use semantic layer metadata for context."""
    # Mock st.selectbox to return a column
    with patch("streamlit.selectbox", return_value="Mortality"):
        with patch("streamlit.subheader"):
            # Mock parse_column_name to return display name
            with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
                from clinical_analytics.core.column_parser import ColumnMetadata

                mock_parse.return_value = ColumnMetadata(display_name="Mortality", canonical_name="mortality")

                refined_intent = ClarifyingQuestionsEngine.ask_clarifying_questions(
                    QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.4, primary_variable=None),
                    mock_semantic_layer,
                    available_columns=["mortality"],
                )

                # Verify get_column_alias_index was called
                mock_semantic_layer.get_column_alias_index.assert_called()
                # Verify primary_variable was set
                assert refined_intent.primary_variable == "mortality"


def test_clarifying_questions_handles_collisions(mock_semantic_layer):
    """Clarifying questions should show collision suggestions when variables ambiguous."""
    # Mock collision suggestions
    mock_semantic_layer.get_collision_suggestions.return_value = ["mortality_1", "mortality_2"]

    intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.4, primary_variable="mortality")

    with patch("streamlit.selectbox", return_value="Mortality 1"):
        with patch("streamlit.warning"):
            with patch("streamlit.subheader"):
                with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
                    from clinical_analytics.core.column_parser import ColumnMetadata

                    # Mock parse_column_name for collision options
                    def parse_side_effect(name):
                        if name == "mortality_1":
                            return ColumnMetadata(display_name="Mortality 1", canonical_name="mortality_1")
                        elif name == "mortality_2":
                            return ColumnMetadata(display_name="Mortality 2", canonical_name="mortality_2")
                        return ColumnMetadata(display_name=name, canonical_name=name)

                    mock_parse.side_effect = parse_side_effect

                    refined_intent = ClarifyingQuestionsEngine.ask_clarifying_questions(
                        intent, mock_semantic_layer, available_columns=["mortality_1", "mortality_2"]
                    )

                    # Verify collision suggestions were checked
                    mock_semantic_layer.get_collision_suggestions.assert_called()
                    # Verify selected option updates intent.primary_variable
                    assert refined_intent.primary_variable == "mortality_1"


def test_clarifying_questions_surfaces_quality_warnings(mock_semantic_layer):
    """Clarifying questions should show relevant quality warnings."""
    # Mock quality warnings
    mock_semantic_layer.get_data_quality_warnings.return_value = [
        {"column": "mortality", "message": "15% missing values"}
    ]

    intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.4, primary_variable="mortality")

    with patch("streamlit.warning") as mock_warning:
        with patch("streamlit.subheader"):
            ClarifyingQuestionsEngine.ask_clarifying_questions(
                intent, mock_semantic_layer, available_columns=["mortality"]
            )

            # Verify st.warning was called with warning message
            assert mock_warning.called


def test_clarifying_questions_asks_about_variables(mock_semantic_layer):
    """When variables are missing, ask user to select them."""
    intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.4, primary_variable=None)

    # Mock selectbox to return "Mortality" for variable selection, then "Treatment" for grouping
    selectbox_calls = ["Mortality", "Treatment Arm"]

    def selectbox_side_effect(*args, **kwargs):
        return selectbox_calls.pop(0) if selectbox_calls else "Treatment Arm"

    with patch("streamlit.selectbox", side_effect=selectbox_side_effect):
        with patch("streamlit.subheader"):
            with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
                from clinical_analytics.core.column_parser import ColumnMetadata

                def parse_side_effect(name):
                    if name == "mortality":
                        return ColumnMetadata(display_name="Mortality", canonical_name="mortality")
                    elif name == "treatment":
                        return ColumnMetadata(display_name="Treatment", canonical_name="treatment")
                    return ColumnMetadata(display_name=name, canonical_name=name)

                mock_parse.side_effect = parse_side_effect

                refined_intent = ClarifyingQuestionsEngine.ask_clarifying_questions(
                    intent, mock_semantic_layer, available_columns=["mortality", "treatment"]
                )

                # Verify engine asked for primary_variable
                assert refined_intent.primary_variable == "mortality"
                # Verify display names are shown (using parse_column_name)
                assert mock_parse.called


def test_clarifying_questions_refines_intent(mock_semantic_layer):
    """User answers should refine QueryIntent with higher confidence."""
    # Start with very low confidence (0.1) to ensure refinement
    original_intent = QueryIntent(intent_type="DESCRIBE", confidence=0.1)

    # Mock selectbox to return different values based on first argument (label)
    def selectbox_side_effect(label, *args, **kwargs):
        if "Analysis Type" in str(label):
            return "COMPARE_GROUPS"
        elif "Primary Variable" in str(label):
            return "Mortality"
        return "Mortality"

    with patch("streamlit.selectbox", side_effect=selectbox_side_effect):
        with patch("streamlit.subheader"):
            with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
                from clinical_analytics.core.column_parser import ColumnMetadata

                mock_parse.return_value = ColumnMetadata(display_name="Mortality", canonical_name="mortality")

                refined_intent = ClarifyingQuestionsEngine.ask_clarifying_questions(
                    original_intent, mock_semantic_layer, available_columns=["mortality"]
                )

                # Test that refined intent has confidence >= original (should be improved)
                assert refined_intent.confidence >= original_intent.confidence
                # If intent type was selected, confidence should be at least 0.6
                if refined_intent.intent_type != original_intent.intent_type:
                    assert refined_intent.confidence >= 0.6


def test_clarifying_questions_aborted_gracefully(mock_semantic_layer, low_confidence_intent):
    """Error path: User closes browser during clarifying questions returns gracefully."""
    # Mock st.selectbox to raise RuntimeError (simulating browser close)
    with patch("streamlit.selectbox", side_effect=RuntimeError("Widget session expired")):
        with patch("streamlit.subheader"):
            refined_intent = ClarifyingQuestionsEngine.ask_clarifying_questions(
                low_confidence_intent, mock_semantic_layer, available_columns=["mortality"]
            )

            # Verify function returns original intent (not crash)
            assert refined_intent is not None
            assert refined_intent.confidence == low_confidence_intent.confidence  # Unchanged if aborted


def test_clarifying_questions_respects_feature_flag(mock_semantic_layer, low_confidence_intent):
    """Clarifying questions should respect ENABLE_CLARIFYING_QUESTIONS feature flag."""
    with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", False):
        refined_intent = ClarifyingQuestionsEngine.ask_clarifying_questions(
            low_confidence_intent, mock_semantic_layer, available_columns=["mortality"]
        )

        # Should return original intent when feature flag is disabled
        assert refined_intent.confidence == low_confidence_intent.confidence
