"""Tests for confidence propagation from QueryIntent to AnalysisContext."""

from unittest.mock import MagicMock, patch

import pytest

from clinical_analytics.core.nl_query_engine import QueryIntent
from clinical_analytics.ui.components.question_engine import AnalysisContext, QuestionEngine


@pytest.fixture
def mock_semantic_layer(mock_semantic_layer):
    """File-specific columns on top of the shared conftest mock_semantic_layer factory."""
    return mock_semantic_layer(columns={"mortality": "mortality", "treatment": "treatment_arm"})


def test_ask_free_form_question_propagates_confidence_to_context(mock_semantic_layer):
    """ask_free_form_question() should propagate confidence from QueryIntent to AnalysisContext."""
    # Arrange: Create a high-confidence intent
    high_confidence_intent = QueryIntent(
        intent_type="COMPARE_GROUPS",
        confidence=0.85,
        primary_variable="mortality",
        grouping_variable="treatment_arm",
    )

    with patch("streamlit.text_input", return_value="compare mortality by treatment", spec=True):
        with patch("streamlit.markdown", spec=True):
            with patch("clinical_analytics.core.nl_query_engine.NLQueryEngine", spec=True) as mock_engine_class:
                mock_engine = MagicMock(spec=["parse_query", "_extract_variables_from_query"])
                mock_engine.parse_query.return_value = high_confidence_intent
                mock_engine._extract_variables_from_query.return_value = (["mortality", "treatment_arm"], {})
                mock_engine_class.return_value = mock_engine

                with patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", False, spec=True):
                    with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", False, spec=True):
                        with patch("streamlit.spinner", spec=True):
                            with patch("streamlit.expander", spec=True):
                                with patch("streamlit.radio", return_value="Yes, that's correct", spec=True):
                                    # Act: Call ask_free_form_question
                                    context = QuestionEngine.ask_free_form_question(mock_semantic_layer)

                                    # Assert: Context should have confidence propagated
                                    assert context is not None, "Should return AnalysisContext"
                                    assert hasattr(context, "confidence"), (
                                        "AnalysisContext should have confidence attribute"
                                    )
                                    assert context.confidence == 0.85, (
                                        "Confidence should be propagated from QueryIntent"
                                    )


def test_ask_free_form_question_propagates_low_confidence(mock_semantic_layer):
    """ask_free_form_question() should propagate low confidence values correctly."""
    # Arrange: Create a low-confidence intent
    low_confidence_intent = QueryIntent(
        intent_type="DESCRIBE",
        confidence=0.3,
    )

    with patch("streamlit.text_input", return_value="some unclear query", spec=True):
        with patch("streamlit.markdown", spec=True):
            with patch("clinical_analytics.core.nl_query_engine.NLQueryEngine", spec=True) as mock_engine_class:
                mock_engine = MagicMock(spec=["parse_query", "_extract_variables_from_query"])
                mock_engine.parse_query.return_value = low_confidence_intent
                mock_engine._extract_variables_from_query.return_value = ([], {})
                mock_engine_class.return_value = mock_engine

                with patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", False, spec=True):
                    with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", False, spec=True):
                        with patch("streamlit.spinner", spec=True):
                            with patch("streamlit.expander", spec=True):
                                with patch("streamlit.radio", return_value="Yes, that's correct", spec=True):
                                    # Act
                                    context = QuestionEngine.ask_free_form_question(mock_semantic_layer)

                                    # Assert: Low confidence should still be propagated
                                    if context is not None:
                                        assert context.confidence == 0.3, "Low confidence should be propagated"


def test_analysis_context_has_confidence_field():
    """AnalysisContext should have a confidence field for auto-execution logic."""
    # Arrange & Act: Create AnalysisContext
    context = AnalysisContext()

    # Assert: Should have confidence field (defaults to None or 0.0)
    assert hasattr(context, "confidence"), "AnalysisContext should have confidence attribute"
    # Should be able to set it
    context.confidence = 0.75
    assert context.confidence == 0.75, "Should be able to set confidence"
