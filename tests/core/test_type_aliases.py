"""Tests for type aliases module."""

from clinical_analytics.core.type_aliases import (
    ColumnMapping,
    ConfigDict,
    FilterDict,
    MetadataDict,
    OptionalDict,
    OptionalFloat,
    OptionalInt,
    OptionalList,
    OptionalStr,
)


def test_type_aliases_are_importable():
    """Test that all type aliases can be imported."""
    assert ConfigDict is not None
    assert FilterDict is not None
    assert ColumnMapping is not None
    assert MetadataDict is not None
    assert OptionalStr is not None
    assert OptionalInt is not None
    assert OptionalFloat is not None
    assert OptionalDict is not None
    assert OptionalList is not None


def test_type_aliases_can_be_used_in_signatures():
    """Test that type aliases work in function signatures."""

    def example_func(config: ConfigDict, filter: FilterDict) -> OptionalStr:
        return config.get("key")

    # Should not raise type errors
    result = example_func({"key": "value"}, {"filter": "value"})
    assert result == "value"


def test_optional_types_accept_none():
    """Test that optional types accept None values."""

    def example_func(value: OptionalStr) -> OptionalStr:
        return value

    assert example_func(None) is None
    assert example_func("value") == "value"
