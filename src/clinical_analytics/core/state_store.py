"""
StateStore - Protocol and implementations for session state abstraction.

This module provides a testable interface for session state management,
decoupling business logic from Streamlit's st.session_state.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StateStore(Protocol):
    """
    Protocol for session state storage.

    Abstracts session state access to enable unit testing without Streamlit.
    Production uses StreamlitStateStore, tests use InMemoryStateStore.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key, returning default if not found."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set value for key."""
        ...

    def contains(self, key: str) -> bool:
        """Check if key exists in store."""
        ...

    def delete(self, key: str) -> None:
        """Delete key from store."""
        ...


class InMemoryStateStore:
    """
    In-memory implementation of StateStore for testing.

    Provides a simple dict-backed state store that matches
    the behavior of Streamlit's session_state.
    """

    def __init__(self) -> None:
        """Initialize empty state store."""
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key, returning default if not found."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value for key."""
        self._data[key] = value

    def contains(self, key: str) -> bool:
        """Check if key exists in store."""
        return key in self._data

    def delete(self, key: str) -> None:
        """Delete key from store if it exists."""
        self._data.pop(key, None)
