"""
normalize_query / canonicalize_scope (single implementation in core.conversation_manager).

Consolidates tests previously spread across test_normalize_query.py (re-imported the Streamlit page
per test) and tests/unit/ui/pages/test_ask_questions_run_key.py (tested the page's duplicate copy).

Test name follows: test_unit_scenario_expectedBehavior
"""

from enum import Enum

import pytest

from clinical_analytics.core.conversation_manager import ConversationManager, canonicalize_scope, normalize_query


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


class _NamedOnly:
    """Object exposing only .name (e.g. a lightweight descriptor)."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        ("", ""),
        ("   \t\n ", ""),
        ("  What   is   the AVERAGE age?  ", "what is the average age?"),
        ("compare\toutcome\nby  \t treatment", "compare outcome by treatment"),
    ],
    ids=["none", "empty", "whitespace_only", "collapse_lower_strip", "tabs_newlines"],
)
def test_normalize_query_variants_returnsCanonicalText(raw: str | None, expected: str) -> None:
    # Act / Assert
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        pytest.param(None, {}, id="none"),
        pytest.param({}, {}, id="empty"),
        pytest.param({"a": 1, "b": None}, {"a": 1}, id="drops_none"),
        pytest.param({"outer": {"x": None}, "k": 1}, {"k": 1}, id="drops_nested_dict_that_becomes_empty"),
        pytest.param({"outer": {"b": 2, "a": None}}, {"outer": {"b": 2}}, id="nested_dict_canonicalized"),
        pytest.param({"vals": [3, 1, 2]}, {"vals": [1, 2, 3]}, id="list_sorted"),
        pytest.param({"vals": [10, 9]}, {"vals": [10, 9]}, id="list_sorted_by_str_not_numeric"),
        pytest.param({"vals": [{"b": 1, "a": None}]}, {"vals": [{"b": 1}]}, id="list_of_dicts_canonicalized"),
        pytest.param({"vals": [_Color.BLUE, _Color.RED]}, {"vals": ["blue", "red"]}, id="list_enum_values"),
        pytest.param({"vals": [_NamedOnly("z"), _NamedOnly("a")]}, {"vals": ["a", "z"]}, id="list_name_only"),
        pytest.param({"c": _Color.RED}, {"c": "red"}, id="scalar_enum_value"),
        pytest.param({"c": _NamedOnly("n")}, {"c": "n"}, id="scalar_name_only"),
    ],
)
def test_canonicalize_scope_variants_returnsExpected(scope: dict | None, expected: dict) -> None:
    # Act
    result = canonicalize_scope(scope)

    # Assert
    assert result == expected


def test_canonicalize_scope_keysSorted_recursively() -> None:
    # Arrange
    scope = {"zebra": 1, "apple": {"y": 1, "b": 2}, "mango": 3}

    # Act
    result = canonicalize_scope(scope)

    # Assert: dict equality ignores order, so check order explicitly
    assert list(result) == ["apple", "mango", "zebra"]
    assert list(result["apple"]) == ["b", "y"]


def test_conversation_manager_methods_delegateToModuleFunctions() -> None:
    # Arrange
    manager = ConversationManager()
    scope = {"b": [2, 1], "a": None}

    # Act / Assert
    assert manager.normalize_query("  A  B ") == normalize_query("  A  B ")
    assert manager.canonicalize_scope(scope) == canonicalize_scope(scope)
