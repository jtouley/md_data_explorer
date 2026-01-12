"""
ConversationManager - Pure Python conversation state management.

Extracted from Streamlit UI to enable UI-agnostic execution.
Manages conversation history, dataset context, and follow-up logic.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

__all__ = ["Message", "ConversationManager"]


@dataclass
class Message:
    """Represents a single message in the conversation transcript."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    run_key: str | None = None  # For result association
    status: Literal["pending", "completed", "error"] = "completed"


class ConversationManager:
    """
    Manages conversation state: messages, dataset context, active query, follow-ups.

    Pure Python class with zero Streamlit dependencies.
    Extracted from Ask_Questions.py to enable UI-agnostic execution.
    """

    def __init__(self) -> None:
        """Initialize empty conversation manager."""
        self._messages: list[Message] = []
        self._current_dataset: str | None = None
        self._active_query: str | None = None
        self._follow_ups: list[str] = []

    def add_message(
        self,
        role: Literal["user", "assistant"],
        content: str,
        run_key: str | None = None,
        status: Literal["pending", "completed", "error"] = "completed",
    ) -> str:
        """
        Add a message to the conversation transcript.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content
            run_key: Optional run key for result association
            status: Message status ("pending", "completed", or "error")

        Returns:
            Message ID (unique identifier)
        """
        message_id = str(uuid4())
        message = Message(
            id=message_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            run_key=run_key,
            status=status,
        )
        self._messages.append(message)
        return message_id

    def get_transcript(self) -> list[Message]:
        """
        Get full conversation transcript.

        Returns:
            List of messages in chronological order
        """
        return self._messages.copy()

    def update_message(self, message_id: str, **updates: Any) -> bool:
        """
        Update an existing message by ID.

        Args:
            message_id: The unique ID of the message to update
            **updates: Fields to update (status, content, run_key)

        Returns:
            True if message was found and updated, False if not found
        """
        for msg in self._messages:
            if msg.id == message_id:
                for key, value in updates.items():
                    if hasattr(msg, key):
                        setattr(msg, key, value)
                return True
        return False

    def get_current_dataset(self) -> str | None:
        """
        Get current dataset ID.

        Returns:
            Dataset ID or None if not set
        """
        return self._current_dataset

    def set_dataset(self, dataset_id: str) -> None:
        """
        Set current dataset ID.

        Args:
            dataset_id: Dataset identifier
        """
        self._current_dataset = dataset_id

    def get_active_query(self) -> str | None:
        """
        Get active query.

        Returns:
            Active query text or None if not set
        """
        return self._active_query

    def set_active_query(self, query: str) -> None:
        """
        Set active query.

        Args:
            query: Query text
        """
        self._active_query = query

    def get_follow_ups(self) -> list[str]:
        """
        Get follow-up questions.

        Returns:
            List of follow-up question strings
        """
        return self._follow_ups.copy()

    def set_follow_ups(self, follow_ups: list[str]) -> None:
        """
        Set follow-up questions.

        Args:
            follow_ups: List of follow-up question strings
        """
        self._follow_ups = follow_ups.copy()

    def clear(self) -> None:
        """Clear all conversation state (reset to initial state)."""
        self._messages = []
        self._current_dataset = None
        self._active_query = None
        self._follow_ups = []

    def normalize_query(self, q: str | None) -> str:
        """
        Normalize query text: collapse whitespace, lowercase, strip.

        Extracted from Ask_Questions.py (lines 141-158).
        This is the single source of truth for query normalization.

        Args:
            q: Raw query text (may be None)

        Returns:
            Normalized query string (lowercase, single spaces, stripped)
        """
        if q is None:
            return ""
        # Collapse whitespace, lowercase, strip
        return " ".join(q.strip().split()).lower()

    def canonicalize_scope(self, scope: dict[str, Any] | None) -> dict[str, Any]:
        """
        Canonicalize semantic scope dict for stable hashing.

        Extracted from Ask_Questions.py (lines 160-228).
        - Drops None values recursively
        - Sorts dictionary keys recursively
        - Sorts list values recursively
        - Ensures stable JSON serialization

        Args:
            scope: Semantic scope dict (may be None)

        Returns:
            Canonicalized scope dict (stable, sorted, no Nones)

        Raises:
            TypeError: If scope contains non-serializable objects (enums/dataclasses
                       should be converted to primitives before calling)
        """
        if scope is None:
            return {}

        canonical: dict[str, Any] = {}
        for key in sorted(scope.keys()):
            value = scope[key]
            if value is None:
                continue  # Drop None values
            elif isinstance(value, dict):
                # Recursively canonicalize nested dicts
                nested_canonical = self.canonicalize_scope(value)
                if nested_canonical:  # Only add non-empty dicts
                    canonical[key] = nested_canonical
            elif isinstance(value, list):
                # Sort lists, recursively canonicalize list items if they are dicts
                sorted_list: list[Any] = []
                for item in value:
                    if isinstance(item, dict):
                        sorted_list.append(self.canonicalize_scope(item))
                    else:
                        # Handle enums and other objects with .value or .name attributes
                        if hasattr(item, "value"):
                            sorted_list.append(item.value)
                        elif hasattr(item, "name"):
                            sorted_list.append(item.name)
                        else:
                            sorted_list.append(item)
                # Sort the list (works for primitives, dicts as JSON strings for comparison)
                try:
                    canonical[key] = sorted(sorted_list, key=lambda x: str(x))
                except TypeError as e:
                    # If sorting fails, raise with helpful error message
                    raise TypeError(
                        f"Scope contains non-serializable value for key '{key}': {type(value).__name__}. "
                        "Convert enums/dataclasses to primitives (use .value or .name) before canonicalizing."
                    ) from e
            else:
                # Handle enums and other objects with .value or .name attributes
                if hasattr(value, "value"):
                    canonical[key] = value.value
                elif hasattr(value, "name"):
                    canonical[key] = value.name
                else:
                    canonical[key] = value

        return canonical

    def serialize(self) -> dict[str, Any]:
        """
        Serialize conversation state to dict for persistence.

        Returns:
            Serializable dict representation
        """
        return {
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "run_key": msg.run_key,
                    "status": msg.status,
                }
                for msg in self._messages
            ],
            "dataset_id": self._current_dataset,
            "active_query": self._active_query,
            "follow_ups": self._follow_ups,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "ConversationManager":
        """
        Deserialize conversation state from dict.

        Args:
            data: Serializable dict representation

        Returns:
            ConversationManager instance with restored state
        """
        manager = cls()
        manager._current_dataset = data.get("dataset_id")
        manager._active_query = data.get("active_query")
        manager._follow_ups = data.get("follow_ups", [])

        # Restore messages
        for msg_data in data.get("messages", []):
            message = Message(
                id=msg_data["id"],
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                run_key=msg_data.get("run_key"),
                status=msg_data.get("status", "completed"),  # Backward compat default
            )
            manager._messages.append(message)

        return manager
