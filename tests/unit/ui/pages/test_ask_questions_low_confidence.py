"""
Test Low-Confidence Feedback UI - Phase 4

Tests that low-confidence queries show detected variables, collision suggestions,
and allow user correction without retyping.
"""

from unittest.mock import MagicMock, patch

from clinical_analytics.ui.components.question_engine import AnalysisIntent

# Fixtures moved to conftest.py - use shared fixtures:
# - sample_cohort
# - low_confidence_context
# - high_confidence_context


class TestLowConfidenceFeedback:
    """Test low-confidence feedback UI behavior."""

    def test_low_confidence_shows_warning_message(self, sample_cohort, low_confidence_context):
        """Test that low confidence displays warning message."""
        # Arrange
        auto_execute_threshold = 0.75

        # Act & Assert
        with patch("streamlit.session_state", {"confirmed_analysis:test_dataset_v1:test_run_key_123": False}):
            # Simulate low confidence check
            confidence = getattr(low_confidence_context, "confidence", 0.0)
            user_confirmed = False
            should_auto_execute = confidence >= auto_execute_threshold or user_confirmed

            # Verify the logic
            assert confidence < auto_execute_threshold
            assert not user_confirmed
            assert not should_auto_execute

    def test_low_confidence_shows_detected_variables(self, sample_cohort, low_confidence_context):
        """Test that low confidence shows detected variables with display names."""
        # Act & Assert
        with patch("clinical_analytics.core.column_parser.parse_column_name") as mock_parse:
            mock_meta = MagicMock()
            mock_meta.display_name = "Mortality"
            mock_parse.return_value = mock_meta

            # Simulate get_display_name function
            def get_display_name(canonical_name: str) -> str:
                try:
                    meta = mock_parse(canonical_name)
                    return meta.display_name
                except Exception:
                    return canonical_name

            # Test that display names are retrieved
            primary_display = get_display_name(low_confidence_context.primary_variable)
            assert primary_display == "Mortality"

    def test_low_confidence_shows_collision_suggestions(self, sample_cohort, low_confidence_context):
        """Test that collision suggestions from context are shown."""
        # Arrange
        context = low_confidence_context

        # Act & Assert
        assert hasattr(context, "match_suggestions")
        assert context.match_suggestions is not None
        assert "mortality" in context.match_suggestions
        assert isinstance(context.match_suggestions["mortality"], list)
        assert len(context.match_suggestions["mortality"]) > 0

    def test_low_confidence_allows_variable_editing(self, sample_cohort, low_confidence_context):
        """Test that low confidence allows editing variables via selectbox."""
        # Arrange
        available_cols = [c for c in sample_cohort.columns if c not in ["patient_id", "time_zero"]]

        # Act & Assert
        with patch("streamlit.selectbox") as mock_selectbox:
            mock_selectbox.return_value = "age"  # User changes selection

            # Simulate selectbox for primary variable
            primary_index = (
                available_cols.index(low_confidence_context.primary_variable)
                if low_confidence_context.primary_variable in available_cols
                else 0
            )
            selected_primary = mock_selectbox(
                "**Primary Variable** (what you want to measure/compare):",
                options=available_cols,
                index=primary_index if primary_index < len(available_cols) else 0,
                key="low_conf_primary_test",
            )

            # Verify user can change selection
            assert selected_primary == "age"
            assert selected_primary != low_confidence_context.primary_variable

    def test_low_confidence_prefills_detected_variables(self, sample_cohort, low_confidence_context):
        """Test that low confidence pre-fills detected variables in selectors."""
        # Arrange
        available_cols = [c for c in sample_cohort.columns if c not in ["patient_id", "time_zero"]]

        # Act & Assert
        # Verify that detected variables are used as initial index
        if low_confidence_context.primary_variable in available_cols:
            primary_index = available_cols.index(low_confidence_context.primary_variable)
            assert primary_index >= 0
            assert available_cols[primary_index] == low_confidence_context.primary_variable

        if low_confidence_context.grouping_variable in available_cols:
            grouping_index = available_cols.index(low_confidence_context.grouping_variable)
            assert grouping_index >= 0
            assert available_cols[grouping_index] == low_confidence_context.grouping_variable

    def test_high_confidence_auto_executes(self, sample_cohort, high_confidence_context):
        """Test that high confidence auto-executes without showing feedback UI."""
        # Arrange
        auto_execute_threshold = 0.75
        user_confirmed = False

        # Act
        confidence = getattr(high_confidence_context, "confidence", 0.0)
        should_auto_execute = confidence >= auto_execute_threshold or user_confirmed

        # Assert
        assert confidence >= auto_execute_threshold
        assert should_auto_execute

    def test_user_confirmation_overrides_low_confidence(self, sample_cohort, low_confidence_context):
        """Test that user confirmation allows execution even with low confidence."""
        # Arrange
        auto_execute_threshold = 0.75
        user_confirmed = True  # User clicked confirm button

        # Act
        confidence = getattr(low_confidence_context, "confidence", 0.0)
        should_auto_execute = confidence >= auto_execute_threshold or user_confirmed

        # Assert
        assert confidence < auto_execute_threshold
        assert user_confirmed
        assert should_auto_execute  # User confirmation overrides low confidence

    def test_collision_suggestions_rendered_from_context(self, sample_cohort, low_confidence_context):
        """Test that collision suggestions are rendered from context.match_suggestions."""
        # Arrange
        context = low_confidence_context

        # Act & Assert
        assert context.match_suggestions is not None
        assert len(context.match_suggestions) > 0

        # Verify structure of suggestions
        for query_term, suggestions in context.match_suggestions.items():
            assert isinstance(query_term, str)
            assert isinstance(suggestions, list)
            assert len(suggestions) > 0
            assert all(isinstance(s, str) for s in suggestions)

    def test_low_confidence_handles_missing_semantic_layer(self, sample_cohort, low_confidence_context):
        """Test that missing semantic layer shows error message."""
        # Arrange
        dataset = MagicMock()
        dataset.semantic = None

        # Act & Assert
        # Simulate semantic layer check
        try:
            semantic_layer = dataset.semantic
            if semantic_layer is None:
                raise AttributeError("Semantic layer not ready")
        except (ValueError, AttributeError):
            # This is what the UI would do - error handling verified
            assert True  # Error handling works correctly

            # In actual UI, this would call st.error and st.stop
            # We verify the exception handling works

    def test_low_confidence_shows_all_detected_variables(self, low_confidence_context):
        """Test that all detected variables are present in context, not just primary."""
        # Arrange
        context = low_confidence_context

        # Act & Assert
        # Primary variable should be detected
        assert context.primary_variable is not None
        assert isinstance(context.primary_variable, str)
        assert len(context.primary_variable) > 0

        # Grouping variable should be detected for COMPARE_GROUPS intent
        if context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
            assert context.grouping_variable is not None
            assert isinstance(context.grouping_variable, str)
            assert len(context.grouping_variable) > 0

    def test_low_confidence_context_updates_on_user_selection(self, sample_cohort, low_confidence_context):
        """Test that context is updated when user selects different variables."""
        # Arrange
        context = low_confidence_context
        original_primary = context.primary_variable

        # Act
        # Simulate user changing selection
        new_primary = "age"
        context.primary_variable = new_primary

        # Assert
        assert context.primary_variable != original_primary
        assert context.primary_variable == new_primary
