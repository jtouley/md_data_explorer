"""
StateStore - Persistence interface for conversation state.

Extracted from Streamlit UI to enable UI-agnostic persistence.
Provides pluggable persistence backends (file-based, database, etc.).
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from clinical_analytics.core.conversation_manager import ConversationManager
from clinical_analytics.core.result_cache import ResultCache


@dataclass
class ConversationState:
    """Serializable conversation state for persistence."""

    conversation_manager: ConversationManager
    result_cache: ResultCache
    dataset_id: str
    upload_id: str | None
    dataset_version: str
    last_updated: datetime


class StateStore(ABC):
    """
    Abstract base class for conversation state persistence.

    Provides pluggable persistence backends (file, database, etc.).
    """

    @abstractmethod
    def save(self, state: ConversationState) -> None:
        """
        Save conversation state to persistent storage.

        Args:
            state: ConversationState to save
        """
        pass

    @abstractmethod
    def load(self, upload_id: str, dataset_version: str) -> ConversationState | None:
        """
        Load conversation state from persistent storage.

        Args:
            upload_id: Upload identifier
            dataset_version: Dataset version identifier

        Returns:
            ConversationState if found, None otherwise
        """
        pass

    @abstractmethod
    def list_sessions(self) -> list[tuple[str, str, datetime]]:
        """
        List all saved sessions.

        Returns:
            List of tuples: (upload_id, dataset_version, last_updated)
        """
        pass


class FileStateStore(StateStore):
    """
    File-based state persistence backend.

    Stores state as JSON files in: {base_path}/{upload_id}_{dataset_version}.json
    """

    def __init__(self, base_path: Path | str = Path("data/sessions")) -> None:
        """
        Initialize file-based state store.

        Args:
            base_path: Base directory for storing session files (default: data/sessions)
        """
        self.base_path = Path(base_path)

    def _get_file_path(self, upload_id: str, dataset_version: str) -> Path:
        """
        Get file path for session.

        Args:
            upload_id: Upload identifier
            dataset_version: Dataset version identifier

        Returns:
            Path to session file
        """
        filename = f"{upload_id}_{dataset_version}.json"
        return self.base_path / filename

    def save(self, state: ConversationState) -> None:
        """
        Save conversation state to JSON file.

        Args:
            state: ConversationState to save
        """
        # Create base_path if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Serialize state
        serialized = {
            "conversation_manager": state.conversation_manager.serialize(),
            "result_cache": state.result_cache.serialize(),
            "dataset_id": state.dataset_id,
            "upload_id": state.upload_id,
            "dataset_version": state.dataset_version,
            "last_updated": state.last_updated.isoformat(),
        }

        # Write to file
        file_path = self._get_file_path(state.upload_id or "unknown", state.dataset_version)
        with open(file_path, "w") as f:
            json.dump(serialized, f, indent=2)

    def load(self, upload_id: str, dataset_version: str) -> ConversationState | None:
        """
        Load conversation state from JSON file.

        Args:
            upload_id: Upload identifier
            dataset_version: Dataset version identifier

        Returns:
            ConversationState if file exists and is valid, None otherwise
        """
        file_path = self._get_file_path(upload_id, dataset_version)

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Deserialize components
            manager = ConversationManager.deserialize(data["conversation_manager"])
            cache = ResultCache.deserialize(data["result_cache"])

            return ConversationState(
                conversation_manager=manager,
                result_cache=cache,
                dataset_id=data["dataset_id"],
                upload_id=data.get("upload_id"),
                dataset_version=data["dataset_version"],
                last_updated=datetime.fromisoformat(data["last_updated"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Handle corrupt JSON or missing fields gracefully
            return None

    def list_sessions(self) -> list[tuple[str, str, datetime]]:
        """
        List all saved sessions.

        Returns:
            List of tuples: (upload_id, dataset_version, last_updated)
        """
        if not self.base_path.exists():
            return []

        sessions = []
        for file_path in self.base_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                upload_id = data.get("upload_id", "unknown")
                dataset_version = data.get("dataset_version", "unknown")
                last_updated = datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))

                sessions.append((upload_id, dataset_version, last_updated))
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip corrupt files
                continue

        return sessions
