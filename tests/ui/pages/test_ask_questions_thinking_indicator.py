"""Tests for thinking indicator UI behavior in Ask Questions page.

Validates that:
- Completion status is NOT rendered (no success UI)
- Processing status IS rendered
- Error status IS rendered
- Results speak for themselves (no lifecycle noise)
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


def _load_ask_questions_module():
    """Load the Ask Questions page as a module."""
    page_path = (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "clinical_analytics"
        / "ui"
        / "pages"
        / "03_💬_Ask_Questions.py"
    )
    spec = importlib.util.spec_from_file_location("ask_questions", page_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ask_questions"] = module
    spec.loader.exec_module(module)
    return module


class TestThinkingIndicatorCompletionBehavior:
    """Test that completion status does NOT render UI."""

    def test_render_thinking_indicator_completed_status_no_ui_rendered(self):
        """Test that completed status does NOT render any status UI.

        Thinking is transient. Completion is invisible. Results speak for themselves.
        Status UI should never exist after completion.
        """
        # Arrange: Completed steps
        steps = [
            {"status": "processing", "text": "Interpreting query", "details": {}},
            {"status": "completed", "text": "Query complete", "details": {"result_rows": 42}},
        ]

        # Act: Import and call _render_thinking_indicator
        with patch("streamlit.status") as mock_status:
            # Load module dynamically to avoid Streamlit runtime errors
            ask_questions = _load_ask_questions_module()
            ask_questions._render_thinking_indicator(steps)

        # Assert: st.status should NOT be called for completed status
        # If status UI is rendered, this test will fail
        mock_status.assert_not_called()

    def test_render_thinking_indicator_processing_status_renders_ui(self):
        """Test that processing status DOES render status UI."""
        # Arrange: Processing steps
        steps = [
            {"status": "processing", "text": "Interpreting query", "details": {}},
        ]

        # Act: Import and call _render_thinking_indicator
        with patch("streamlit.status") as mock_status:
            ask_questions = _load_ask_questions_module()
            ask_questions._render_thinking_indicator(steps)

        # Assert: st.status should be called with "running" state
        mock_status.assert_called_once()
        call_args = mock_status.call_args
        assert "🤔" in call_args[0][0]  # Status label contains thinking emoji
        assert call_args[1]["state"] == "running"

    def test_render_thinking_indicator_error_status_renders_ui(self):
        """Test that error status DOES render status UI."""
        # Arrange: Error steps
        steps = [
            {"status": "processing", "text": "Interpreting query", "details": {}},
            {"status": "error", "text": "Query failed", "details": {"error": "Invalid column"}},
        ]

        # Act: Import and call _render_thinking_indicator
        with patch("streamlit.status") as mock_status:
            ask_questions = _load_ask_questions_module()
            ask_questions._render_thinking_indicator(steps)

        # Assert: st.status should be called with "error" state
        mock_status.assert_called_once()
        call_args = mock_status.call_args
        assert "❌" in call_args[0][0]  # Status label contains error emoji
        assert call_args[1]["state"] == "error"
        assert call_args[1]["expanded"] is True  # Errors should be expanded

    def test_render_thinking_indicator_empty_steps_no_ui_rendered(self):
        """Test that empty steps list does NOT render any UI."""
        # Arrange: Empty steps
        steps = []

        # Act: Import and call _render_thinking_indicator
        with patch("streamlit.status") as mock_status:
            ask_questions = _load_ask_questions_module()
            ask_questions._render_thinking_indicator(steps)

        # Assert: st.status should NOT be called
        mock_status.assert_not_called()
