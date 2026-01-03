"""
Tests for StateStore - Persistence interface for conversation state.

Tests verify UI-agnostic state persistence with pluggable backends.
"""

from datetime import datetime
from pathlib import Path

import pytest

from clinical_analytics.core.conversation_manager import ConversationManager
from clinical_analytics.core.result_cache import CachedResult, ResultCache
from clinical_analytics.core.state_store import ConversationState, FileStateStore, StateStore


class TestStateStoreInterface:
    """Test suite for StateStore ABC interface."""

    def test_state_store_is_abstract(self):
        """Test that StateStore cannot be instantiated directly."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError):
            StateStore()  # type: ignore[abstract]


class TestConversationState:
    """Test suite for ConversationState dataclass."""

    def test_conversation_state_initializes(self):
        """Test that ConversationState can be created with required fields."""
        # Arrange
        manager = ConversationManager()
        cache = ResultCache(max_size=50)

        # Act
        state = ConversationState(
            conversation_manager=manager,
            result_cache=cache,
            dataset_id="test_dataset",
            upload_id="test_upload",
            dataset_version="test_version",
            last_updated=datetime.now(),
        )

        # Assert
        assert state.conversation_manager is manager
        assert state.result_cache is cache
        assert state.dataset_id == "test_dataset"
        assert state.upload_id == "test_upload"
        assert state.dataset_version == "test_version"
        assert isinstance(state.last_updated, datetime)

    def test_conversation_state_accepts_none_upload_id(self):
        """Test that ConversationState accepts None for upload_id."""
        # Arrange
        manager = ConversationManager()
        cache = ResultCache(max_size=50)

        # Act
        state = ConversationState(
            conversation_manager=manager,
            result_cache=cache,
            dataset_id="test_dataset",
            upload_id=None,
            dataset_version="test_version",
            last_updated=datetime.now(),
        )

        # Assert
        assert state.upload_id is None


class TestFileStateStore:
    """Test suite for FileStateStore implementation."""

    def test_file_state_store_initializes_with_default_path(self):
        """Test that FileStateStore initializes with default base path."""
        # Arrange & Act
        store = FileStateStore()

        # Assert
        assert store.base_path == Path("data/sessions")

    def test_file_state_store_initializes_with_custom_path(self, tmp_path):
        """Test that FileStateStore initializes with custom base path."""
        # Arrange
        custom_path = tmp_path / "custom_sessions"

        # Act
        store = FileStateStore(base_path=custom_path)

        # Assert
        assert store.base_path == custom_path

    def test_file_state_store_save_creates_file(self, tmp_path):
        """Test that save() creates a JSON file with state."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)
        manager = ConversationManager()
        manager.add_message("user", "test query")
        cache = ResultCache(max_size=50)
        state = ConversationState(
            conversation_manager=manager,
            result_cache=cache,
            dataset_id="test_dataset",
            upload_id="test_upload",
            dataset_version="test_version",
            last_updated=datetime.now(),
        )

        # Act
        store.save(state)

        # Assert
        expected_file = tmp_path / "test_upload_test_version.json"
        assert expected_file.exists()
        assert expected_file.is_file()

    def test_file_state_store_load_restores_state(self, tmp_path):
        """Test that load() restores ConversationState from file."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)
        manager = ConversationManager()
        manager.add_message("user", "test query")
        manager.set_dataset("test_dataset")
        cache = ResultCache(max_size=50)
        cache.put(
            CachedResult(
                run_key="test_key",
                query="test query",
                result={"data": 1},
                timestamp=datetime.now(),
                dataset_version="test_version",
            )
        )
        original_state = ConversationState(
            conversation_manager=manager,
            result_cache=cache,
            dataset_id="test_dataset",
            upload_id="test_upload",
            dataset_version="test_version",
            last_updated=datetime.now(),
        )
        store.save(original_state)

        # Act
        loaded_state = store.load("test_upload", "test_version")

        # Assert
        assert loaded_state is not None
        assert loaded_state.dataset_id == "test_dataset"
        assert loaded_state.upload_id == "test_upload"
        assert loaded_state.dataset_version == "test_version"
        assert len(loaded_state.conversation_manager.get_transcript()) == 1
        assert loaded_state.conversation_manager.get_transcript()[0].content == "test query"
        assert loaded_state.conversation_manager.get_current_dataset() == "test_dataset"
        cached_result = loaded_state.result_cache.get("test_key", "test_version")
        assert cached_result is not None
        assert cached_result.result == {"data": 1}

    def test_file_state_store_load_returns_none_for_missing_file(self, tmp_path):
        """Test that load() returns None for non-existent file."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)

        # Act
        loaded_state = store.load("nonexistent_upload", "nonexistent_version")

        # Assert
        assert loaded_state is None

    def test_file_state_store_list_sessions_returns_saved_sessions(self, tmp_path):
        """Test that list_sessions() returns all saved sessions."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)
        manager1 = ConversationManager()
        manager2 = ConversationManager()
        cache1 = ResultCache(max_size=50)
        cache2 = ResultCache(max_size=50)

        state1 = ConversationState(
            conversation_manager=manager1,
            result_cache=cache1,
            dataset_id="dataset1",
            upload_id="upload1",
            dataset_version="version1",
            last_updated=datetime(2024, 1, 1, 12, 0, 0),
        )
        state2 = ConversationState(
            conversation_manager=manager2,
            result_cache=cache2,
            dataset_id="dataset2",
            upload_id="upload2",
            dataset_version="version2",
            last_updated=datetime(2024, 1, 2, 12, 0, 0),
        )

        store.save(state1)
        store.save(state2)

        # Act
        sessions = store.list_sessions()

        # Assert
        assert len(sessions) == 2
        # Check that both sessions are present (order may vary)
        upload_ids = {session[0] for session in sessions}
        assert "upload1" in upload_ids
        assert "upload2" in upload_ids

    def test_file_state_store_list_sessions_returns_empty_for_no_sessions(self, tmp_path):
        """Test that list_sessions() returns empty list when no sessions exist."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)

        # Act
        sessions = store.list_sessions()

        # Assert
        assert sessions == []

    def test_file_state_store_save_overwrites_existing_file(self, tmp_path):
        """Test that save() overwrites existing file with new state."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)
        manager1 = ConversationManager()
        manager1.add_message("user", "first query")
        cache1 = ResultCache(max_size=50)
        state1 = ConversationState(
            conversation_manager=manager1,
            result_cache=cache1,
            dataset_id="test_dataset",
            upload_id="test_upload",
            dataset_version="test_version",
            last_updated=datetime(2024, 1, 1, 12, 0, 0),
        )
        store.save(state1)

        # Act: Save updated state
        manager2 = ConversationManager()
        manager2.add_message("user", "second query")
        cache2 = ResultCache(max_size=50)
        state2 = ConversationState(
            conversation_manager=manager2,
            result_cache=cache2,
            dataset_id="test_dataset",
            upload_id="test_upload",
            dataset_version="test_version",
            last_updated=datetime(2024, 1, 2, 12, 0, 0),
        )
        store.save(state2)

        # Assert: Loaded state should be the second one
        loaded_state = store.load("test_upload", "test_version")
        assert loaded_state is not None
        assert len(loaded_state.conversation_manager.get_transcript()) == 1
        assert loaded_state.conversation_manager.get_transcript()[0].content == "second query"

    def test_file_state_store_handles_corrupt_json_gracefully(self, tmp_path):
        """Test that load() handles corrupt JSON files gracefully."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)
        corrupt_file = tmp_path / "test_upload_test_version.json"
        corrupt_file.write_text("{ invalid json }")

        # Act & Assert
        # Should return None or raise a specific exception, not crash
        loaded_state = store.load("test_upload", "test_version")
        # Either None or we catch the exception - both are acceptable
        assert loaded_state is None or isinstance(loaded_state, ConversationState)

    def test_file_state_store_creates_base_path_if_missing(self, tmp_path):
        """Test that save() creates base_path directory if it doesn't exist."""
        # Arrange
        base_path = tmp_path / "new_sessions"
        store = FileStateStore(base_path=base_path)
        manager = ConversationManager()
        cache = ResultCache(max_size=50)
        state = ConversationState(
            conversation_manager=manager,
            result_cache=cache,
            dataset_id="test_dataset",
            upload_id="test_upload",
            dataset_version="test_version",
            last_updated=datetime.now(),
        )

        # Act
        store.save(state)

        # Assert
        assert base_path.exists()
        assert base_path.is_dir()
