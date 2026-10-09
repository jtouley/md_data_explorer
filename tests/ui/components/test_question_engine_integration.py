"""Integration tests for end-to-end NL query flow."""

from unittest.mock import MagicMock, patch

import pytest

from clinical_analytics.core.nl_query_engine import QueryIntent
from clinical_analytics.ui.components.question_engine import QuestionEngine


@pytest.fixture
def mock_semantic_layer(mock_semantic_layer):
    """File-specific columns on top of the shared conftest mock_semantic_layer factory."""
    return mock_semantic_layer(
        columns={"mortality": "mortality", "treatment": "treatment_arm"},
        dimensions={"treatment_arm": {"label": "Treatment Arm"}},
        quality_warnings=[],
    )


def test_end_to_end_nl_query_flow(mock_semantic_layer):
    """Integration test: query → progressive feedback → clarifying questions → analysis."""
    # Create a query that will need clarifying questions
    query = "compare something by something else"

    with patch("streamlit.text_input", return_value=query, spec=True):
        with patch("streamlit.markdown", spec=True):
            with patch("streamlit.status", spec=True) as mock_status:
                mock_status.return_value.__enter__.return_value.update = MagicMock()

                with patch("clinical_analytics.core.nl_query_engine.NLQueryEngine", spec=True) as mock_engine_class:
                    mock_engine = MagicMock(
                        spec=[
                            "_pattern_match",
                            "_semantic_match",
                            "_llm_parse",
                            "_extract_variables_from_query",
                            "_generate_suggestions",
                        ]
                    )
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
                        "clinical_analytics.core.clarifying_questions.ClarifyingQuestionsEngine.ask_clarifying_questions",
                        spec=True,
                    ) as mock_clarify:
                        refined_intent = QueryIntent(
                            intent_type="COMPARE_GROUPS",
                            confidence=0.7,
                            primary_variable="mortality",
                            grouping_variable="treatment_arm",
                        )
                        mock_clarify.return_value = refined_intent

                        with patch(
                            "clinical_analytics.core.nl_query_config.ENABLE_CLARIFYING_QUESTIONS", True, spec=True
                        ):
                            with patch(
                                "clinical_analytics.core.nl_query_config.ENABLE_PROGRESSIVE_FEEDBACK", True, spec=True
                            ):
                                with patch("streamlit.expander", spec=True):
                                    with patch("streamlit.radio", return_value="Yes, that's correct", spec=True):
                                        result = QuestionEngine.ask_free_form_question(mock_semantic_layer)

                                        # Verify full pipeline executed
                                        assert result is not None
                                        assert result.inferred_intent.value == "compare_groups"
                                        assert result.primary_variable == "mortality"
                                        assert result.grouping_variable == "treatment_arm"
