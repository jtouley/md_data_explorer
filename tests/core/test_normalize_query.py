"""
Tests for query normalization and empty query rejection (Phase 1.3).

Tests that empty queries are properly normalized and can be detected.

Test name follows: test_unit_scenario_expectedBehavior
"""

import sys
from pathlib import Path

# Add src to path for importing from UI pages
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestQueryNormalization:
    """Test suite for normalize_query function."""

    def test_normalize_query_empty_string_returns_empty(self):
        """Empty string should normalize to empty string (Phase 1.3)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "03_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange
        empty_query = ""

        # Act
        result = ask_questions.normalize_query(empty_query)

        # Assert
        assert result == ""

    def test_normalize_query_whitespace_only_returns_empty(self):
        """Whitespace-only queries should normalize to empty string (Phase 1.3)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "03_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange
        whitespace_queries = ["   ", "\t", "\n", "\t\n  ", "  \t  \n  "]

        # Act & Assert
        for query in whitespace_queries:
            result = ask_questions.normalize_query(query)
            assert result == "", f"Whitespace query '{repr(query)}' should normalize to empty string"

    def test_normalize_query_none_returns_empty(self):
        """None query should normalize to empty string (Phase 1.3)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "03_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange
        none_query = None

        # Act
        result = ask_questions.normalize_query(none_query)

        # Assert
        assert result == ""

    def test_normalize_query_collapses_whitespace(self):
        """Multiple whitespace should collapse to single space (Phase 1.3)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "03_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange
        query_with_extra_spaces = "average   age    by   status"

        # Act
        result = ask_questions.normalize_query(query_with_extra_spaces)

        # Assert
        assert result == "average age by status"

    def test_normalize_query_lowercases_text(self):
        """Query should be lowercased (Phase 1.3)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "03_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange
        uppercase_query = "AVERAGE AGE BY STATUS"

        # Act
        result = ask_questions.normalize_query(uppercase_query)

        # Assert
        assert result == "average age by status"

    def test_normalize_query_strips_leading_trailing_whitespace(self):
        """Leading/trailing whitespace should be stripped (Phase 1.3)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "03_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange
        query_with_spaces = "  average age  "

        # Act
        result = ask_questions.normalize_query(query_with_spaces)

        # Assert
        assert result == "average age"
