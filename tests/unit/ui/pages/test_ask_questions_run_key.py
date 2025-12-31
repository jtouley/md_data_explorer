"""
Test UI helper functions for query normalization and scope canonicalization.

Note: run_key generation tests are in tests/core/test_semantic_run_key_determinism.py
The semantic layer is the single source of truth for run_key generation (PR21 refactor).

Test name follows: test_unit_scenario_expectedBehavior
"""


def test_normalize_query_handles_tabs_and_newlines(ask_questions_page):
    """Test normalize_query handles tabs and newlines."""
    # Arrange
    query_with_tabs = "compare\toutcome\tby\ttreatment"
    query_with_newlines = "compare\noutcome\nby\ntreatment"
    query_mixed = "compare\t\noutcome  \n\tby   treatment"

    # Act
    normalized_tabs = ask_questions_page.normalize_query(query_with_tabs)
    normalized_newlines = ask_questions_page.normalize_query(query_with_newlines)
    normalized_mixed = ask_questions_page.normalize_query(query_mixed)

    # Assert: All normalized to same single-space separated string
    assert normalized_tabs == "compare outcome by treatment"
    assert normalized_newlines == "compare outcome by treatment"
    assert normalized_mixed == "compare outcome by treatment"


def test_normalize_query_lowercases_and_strips(ask_questions_page):
    """Test normalize_query lowercases and strips whitespace."""
    # Arrange
    query_mixed_case = "  Compare OUTCOME by Treatment  "

    # Act
    normalized = ask_questions_page.normalize_query(query_mixed_case)

    # Assert
    assert normalized == "compare outcome by treatment"


def test_normalize_query_handles_none_returns_empty_string(ask_questions_page):
    """Test normalize_query returns empty string for None."""
    # Arrange
    query_none = None

    # Act
    normalized = ask_questions_page.normalize_query(query_none)

    # Assert
    assert normalized == ""


def test_canonicalize_scope_drops_none_values(ask_questions_page):
    """Test canonicalize_scope drops None values."""
    # Arrange
    scope = {
        "filter_a": "value_a",
        "filter_b": None,
        "filter_c": "value_c",
    }

    # Act
    canonical = ask_questions_page.canonicalize_scope(scope)

    # Assert: None value dropped
    assert "filter_b" not in canonical
    assert canonical == {"filter_a": "value_a", "filter_c": "value_c"}


def test_canonicalize_scope_sorts_keys(ask_questions_page):
    """Test canonicalize_scope sorts dictionary keys."""
    # Arrange
    scope = {
        "z_filter": "value_z",
        "a_filter": "value_a",
        "m_filter": "value_m",
    }

    # Act
    canonical = ask_questions_page.canonicalize_scope(scope)

    # Assert: Keys are sorted
    assert list(canonical.keys()) == ["a_filter", "m_filter", "z_filter"]


def test_canonicalize_scope_sorts_list_values(ask_questions_page):
    """Test canonicalize_scope sorts list values."""
    # Arrange
    scope = {
        "predictors": ["age", "treatment", "bmi"],
    }

    # Act
    canonical = ask_questions_page.canonicalize_scope(scope)

    # Assert: List sorted
    assert canonical["predictors"] == ["age", "bmi", "treatment"]


def test_canonicalize_scope_handles_none_scope(ask_questions_page):
    """Test canonicalize_scope handles None scope."""
    # Arrange
    scope = None

    # Act
    canonical = ask_questions_page.canonicalize_scope(scope)

    # Assert: Returns empty dict
    assert canonical == {}
