"""Integration tests for end-to-end NL query flow."""

from unittest.mock import MagicMock, patch

import pytest

from clinical_analytics.core.nl_query_engine import QueryIntent
from clinical_analytics.ui.components.question_engine import QuestionEngine


@pytest.fixture
def mock_semantic_layer():
    """Create a mock semantic layer for testing."""
    mock = MagicMock()
    mock.get_column_alias_index.return_value = {
        "mortality": "mortality",
        "treatment": "treatment_arm",
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_available_dimensions.return_value = {"treatment_arm": {"label": "Treatment Arm"}}
    mock.get_data_quality_warnings.return_value = []
    return mock


def test_end_to_end_nl_query_flow(mock_semantic_layer):
    """Integration test: query → progressive feedback → clarifying questions → analysis."""
    # Create a query that will need clarifying questions
    query = "compare something by something else"

    with patch("streamlit.text_input", return_value=query):
        with patch("streamlit.markdown"):
            with patch("streamlit.status") as mock_status:
                mock_status.return_value.__enter__.return_value.update = MagicMock()

                with patch("clinical_analytics.core.nl_query_engine.NLQueryEngine") as mock_engine_class:
                    mock_engine = MagicMock()
                    # First parse returns low confidence
                    low_intent = QueryIntent(intent_type="DESCRIBE", confidence=0.3)
                    mock_engine._pattern_match.return_value = None
                    mock_engine._semantic_match.return_value = None
                    mock_engine._llm_parse.return_value = low_intent
                    mock_engine._extract_variables_from_query.return_value = ([], {})
                    mock_engine._generate_suggestions.return_value = ["Try mentioning specific variables"]
                    mock_engine_class.return_value = mock_engine

                    # Mock clarifying questions to refine intent
                    with patch(
                        "clinical_analytics.core.clarifying_questions.ClarifyingQuestionsEngine.ask_clarifying_questions"
                    ) as mock_clarify:
                        refined_intent = QueryIntent(
                            intent_type="COMPARE_GROUPS",
                            confidence=0.7,
                            primary_variable="mortality",
                            grouping_variable="treatment_arm",
                        )
                        mock_clarify.return_value = refined_intent

                        with patch("clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", True):
                            with patch("clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", True):
                                with patch("streamlit.expander"):
                                    with patch("streamlit.radio", return_value="Yes, that's correct"):
                                        result = QuestionEngine.ask_free_form_question(mock_semantic_layer)

                                        # Verify full pipeline executed
                                        assert result is not None
                                        assert result.inferred_intent.value == "compare_groups"
                                        assert result.primary_variable == "mortality"
                                        assert result.grouping_variable == "treatment_arm"
