"""Type guard functions for safe None handling."""

from typing import Any, TypeGuard


def is_not_none(value: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    """Type guard for non-None dict."""
    return value is not None


def safe_get(d: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """Safely get value from potentially None dict."""
    if d is None:
        return default
    return d.get(key, default)
