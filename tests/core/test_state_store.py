"""
Tests for StateStore - Protocol and implementations for session state abstraction.

Tests verify get/set/contains/delete operations and typed property accessors.
"""

from typing import Any

from clinical_analytics.core.state_store import InMemoryStateStore, StateStore


class TestStateStore:
    """Test suite for StateStore implementations."""

    def test_state_store_get_existing_key_returns_value(self):
        """Test that get returns value for existing key."""
        # Arrange
        store: StateStore = InMemoryStateStore()
        store.set("test_key", "test_value")

        # Act
        result = store.get("test_key")

        # Assert
        assert result == "test_value"

    def test_state_store_get_missing_key_returns_default(self):
        """Test that get returns default for missing key."""
        # Arrange
        store: StateStore = InMemoryStateStore()

        # Act
        result = store.get("missing_key", default="fallback")

        # Assert
        assert result == "fallback"

    def test_state_store_set_new_key_persists_value(self):
        """Test that set stores value retrievable via get."""
        # Arrange
        store: StateStore = InMemoryStateStore()
        data: dict[str, Any] = {"nested": {"value": 123}}

        # Act
        store.set("complex_key", data)
        retrieved = store.get("complex_key")

        # Assert
        assert retrieved == {"nested": {"value": 123}}

    def test_state_store_contains_existing_key_returns_true(self):
        """Test that contains returns True for existing key."""
        # Arrange
        store: StateStore = InMemoryStateStore()
        store.set("exists", True)

        # Act & Assert
        assert store.contains("exists") is True
        assert store.contains("does_not_exist") is False

    def test_state_store_delete_removes_key(self):
        """Test that delete removes key from store."""
        # Arrange
        store: StateStore = InMemoryStateStore()
        store.set("to_delete", "value")
        assert store.contains("to_delete") is True

        # Act
        store.delete("to_delete")

        # Assert
        assert store.contains("to_delete") is False
        assert store.get("to_delete") is None
