"""Tests for type guards module."""

from clinical_analytics.core.type_guards import is_not_none, safe_get


def test_is_not_none_type_guard():
    """Test that is_not_none type guard works correctly."""
    value: dict[str, int] | None = {"key": 42}

    if is_not_none(value):
        # Type should be narrowed to dict[str, int]
        assert value["key"] == 42

    value = None
    assert not is_not_none(value)


def test_safe_get_with_none_dict():
    """Test safe_get with None dict."""
    result = safe_get(None, "key", "default")
    assert result == "default"


def test_safe_get_with_valid_dict():
    """Test safe_get with valid dict."""
    d = {"key": "value"}
    result = safe_get(d, "key", "default")
    assert result == "value"

    result = safe_get(d, "missing", "default")
    assert result == "default"


def test_safe_get_type_narrowing():
    """Test that safe_get works with type narrowing."""
    metadata: dict[str, str] | None = {"key": "value"}

    value = safe_get(metadata, "key")
    # Should not raise type errors
    assert value == "value"
