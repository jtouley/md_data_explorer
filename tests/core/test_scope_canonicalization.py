"""
Tests for recursive scope canonicalization (Phase 1.4).

Tests that scope canonicalization handles nested structures recursively.

Test name follows: test_unit_scenario_expectedBehavior
"""

import sys
from pathlib import Path

# Add src to path for importing from UI pages
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestScopeCanonnicalization:
    """Test suite for canonicalize_scope function."""

    def test_scope_canonicalization_handles_nested_dicts(self):
        """Scope canonicalization should handle nested dicts recursively (Phase 1.4)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "3_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange: Nested scope with different key orders
        scope1 = {"filters": {"status": "active", "age": {"min": 18, "max": 65}}, "cohort": "all"}
        scope2 = {"cohort": "all", "filters": {"age": {"max": 65, "min": 18}, "status": "active"}}

        # Act
        canonical1 = ask_questions.canonicalize_scope(scope1)
        canonical2 = ask_questions.canonicalize_scope(scope2)

        # Assert: Should be identical after canonicalization
        assert canonical1 == canonical2

    def test_scope_canonicalization_handles_nested_lists(self):
        """Scope canonicalization should handle nested lists (Phase 1.4)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "3_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange: Scope with unsorted lists
        scope1 = {"filters": {"ids": [3, 1, 2]}}
        scope2 = {"filters": {"ids": [1, 2, 3]}}

        # Act
        canonical1 = ask_questions.canonicalize_scope(scope1)
        canonical2 = ask_questions.canonicalize_scope(scope2)

        # Assert: Lists should be sorted
        assert canonical1 == canonical2
        assert canonical1["filters"]["ids"] == [1, 2, 3]

    def test_scope_canonicalization_drops_none_recursively(self):
        """Scope canonicalization should drop None values recursively (Phase 1.4)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "3_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange: Nested scope with None values at different levels
        scope = {"filters": {"status": "active", "age": None, "nested": {"value": 10, "empty": None}}, "cohort": None}

        # Act
        canonical = ask_questions.canonicalize_scope(scope)

        # Assert: None values should be dropped at all levels
        assert "cohort" not in canonical
        assert "age" not in canonical["filters"]
        assert "empty" not in canonical["filters"]["nested"]
        assert canonical["filters"]["status"] == "active"
        assert canonical["filters"]["nested"]["value"] == 10

    def test_scope_canonicalization_sorts_keys_recursively(self):
        """Scope canonicalization should sort keys at all nesting levels (Phase 1.4)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "3_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange: Nested scope with unsorted keys
        scope = {"z_key": 1, "a_key": {"z_nested": 2, "a_nested": 3}}

        # Act
        canonical = ask_questions.canonicalize_scope(scope)

        # Assert: Keys should be sorted at all levels
        keys = list(canonical.keys())
        assert keys == ["a_key", "z_key"]
        nested_keys = list(canonical["a_key"].keys())
        assert nested_keys == ["a_nested", "z_nested"]

    def test_scope_canonicalization_handles_empty_dict(self):
        """Scope canonicalization should handle empty dict (Phase 1.4)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "3_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange
        scope = {}

        # Act
        canonical = ask_questions.canonicalize_scope(scope)

        # Assert
        assert canonical == {}

    def test_scope_canonicalization_handles_none(self):
        """Scope canonicalization should handle None input (Phase 1.4)."""
        # Import here to avoid import issues
        import importlib.util

        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "clinical_analytics"
            / "ui"
            / "pages"
            / "3_💬_Ask_Questions.py"
        )
        spec = importlib.util.spec_from_file_location("ask_questions", page_path)
        ask_questions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ask_questions)

        # Arrange
        scope = None

        # Act
        canonical = ask_questions.canonicalize_scope(scope)

        # Assert
        assert canonical == {}
