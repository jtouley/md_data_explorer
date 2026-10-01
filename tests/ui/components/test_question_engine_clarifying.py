"""Tests for clarifying questions integration in question engine."""

from unittest.mock import MagicMock, patch

import pytest

from clinical_analytics.core.nl_query_engine import QueryIntent
from clinical_analytics.ui.components.question_engine import QuestionEngine


@pytest.fixture
def mock_semantic_layer(mock_semantic_layer):
    """File-specific columns on top of the shared conftest mock_semantic_layer factory."""
    return mock_semantic_layer(
        columns={"mortality": "mortality", "treatment": "treatment_arm"}, dimensions={}, quality_warnings=[]
    )


def test_ask_free_form_question_uses_clarifying_questions_for_low_confidence(mock_semantic_layer):
    """ask_free_form_question() should call clarifying questions when confidence is low."""

    # Create a low-confidence intent
    low_confidence_intent = QueryIntent(intent_type="DESCRIBE", confidence=0.3)

    with patch("streamlit.text_input", return_value="some unclear query"):
        with patch("streamlit.markdown"):
            with patch("clinical_analytics.core.nl_query_engine.NLQueryEngine") as mock_engine_class:
                mock_engine = MagicMock()
                mock_engine.parse_query.return_value = low_confidence_intent
                mock_engine._extract_variables_from_query.return_value = ([], {})
                mock_engine_class.return_value = mock_engine

                with patch(
                    "clinical_analytics.core.clarifying_questions.ClarifyingQuestionsEngine.ask_clarifying_questions"
                ) as mock_clarify:
                    # Mock clarifying questions to return refined intent
                    refined_intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.7)
                    mock_clarify.return_value = refined_intent

                    with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", True):
                        with patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", False):
                            with patch("streamlit.spinner"):
                                with patch("streamlit.expander"):
                                    with patch("streamlit.radio", return_value="Yes, that's correct"):
                                        QuestionEngine.ask_free_form_question(mock_semantic_layer)

                                        # Verify clarifying questions were called
                                        assert mock_clarify.called


def test_ask_free_form_question_skips_clarifying_questions_for_high_confidence(mock_semantic_layer):
    """ask_free_form_question() should skip clarifying questions when confidence is high."""
    high_confidence_intent = QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.9)

    with patch("streamlit.text_input", return_value="compare mortality by treatment"):
        with patch("streamlit.markdown"):
            with patch("clinical_analytics.core.nl_query_engine.NLQueryEngine") as mock_engine_class:
                mock_engine = MagicMock()
                mock_engine.parse_query.return_value = high_confidence_intent
                mock_engine._extract_variables_from_query.return_value = (["mortality", "treatment"], {})
                mock_engine_class.return_value = mock_engine

                with patch(
                    "clinical_analytics.core.clarifying_questions.ClarifyingQuestionsEngine.ask_clarifying_questions"
                ) as mock_clarify:
                    with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", True):
                        with patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", False):
                            with patch("streamlit.spinner"):
                                with patch("streamlit.expander"):
                                    with patch("streamlit.radio", return_value="Yes, that's correct"):
                                        QuestionEngine.ask_free_form_question(mock_semantic_layer)

                                        # Verify clarifying questions were NOT called (confidence too high)
                                        assert not mock_clarify.called
