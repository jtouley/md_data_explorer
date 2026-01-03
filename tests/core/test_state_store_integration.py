"""
Integration tests for StateStore persistence flow.

Tests verify end-to-end persistence: save -> restart -> restore -> continue.
"""

from datetime import datetime

from clinical_analytics.core.conversation_manager import ConversationManager
from clinical_analytics.core.result_cache import CachedResult, ResultCache
from clinical_analytics.core.state_store import ConversationState, FileStateStore


class TestStateStoreIntegration:
    """Integration tests for StateStore persistence flow."""

    def test_persistence_flow_save_restore_continue(self, tmp_path):
        """Test complete persistence flow: save -> restore -> continue."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)
        upload_id = "test_upload_123"
        dataset_version = "test_version_v1"
        dataset_id = "test_dataset"

        # Step 1: Initial state - ask question, get result
        manager1 = ConversationManager()
        manager1.set_dataset(dataset_id)
        manager1.add_message("user", "what is the average age?")
        cache1 = ResultCache(max_size=50)
        cache1.put(
            CachedResult(
                run_key="run_key_1",
                query="what is the average age?",
                result={"mean": 45.5, "type": "descriptive"},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )

        state1 = ConversationState(
            conversation_manager=manager1,
            result_cache=cache1,
            dataset_id=dataset_id,
            upload_id=upload_id,
            dataset_version=dataset_version,
            last_updated=datetime.now(),
        )

        # Act: Save state
        store.save(state1)

        # Assert: Verify file was created
        expected_file = tmp_path / f"{upload_id}_{dataset_version}.json"
        assert expected_file.exists()

        # Step 2: Simulate restart - load state
        loaded_state = store.load(upload_id, dataset_version)

        # Assert: Verify state was restored
        assert loaded_state is not None
        assert loaded_state.dataset_id == dataset_id
        assert loaded_state.upload_id == upload_id
        assert len(loaded_state.conversation_manager.get_transcript()) == 1
        assert loaded_state.conversation_manager.get_transcript()[0].content == "what is the average age?"
        assert loaded_state.conversation_manager.get_current_dataset() == dataset_id

        # Verify cached result was restored
        cached_result = loaded_state.result_cache.get("run_key_1", dataset_version)
        assert cached_result is not None
        assert cached_result.result == {"mean": 45.5, "type": "descriptive"}

        # Step 3: Continue conversation - add new query and result
        manager2 = loaded_state.conversation_manager
        manager2.add_message("user", "what is the median age?")
        cache2 = loaded_state.result_cache
        cache2.put(
            CachedResult(
                run_key="run_key_2",
                query="what is the median age?",
                result={"median": 44.0, "type": "descriptive"},
                timestamp=datetime.now(),
                dataset_version=dataset_version,
            )
        )

        # Act: Save updated state
        state2 = ConversationState(
            conversation_manager=manager2,
            result_cache=cache2,
            dataset_id=dataset_id,
            upload_id=upload_id,
            dataset_version=dataset_version,
            last_updated=datetime.now(),
        )
        store.save(state2)

        # Step 4: Simulate another restart - load updated state
        loaded_state2 = store.load(upload_id, dataset_version)

        # Assert: Verify both messages and both results are present
        assert loaded_state2 is not None
        transcript = loaded_state2.conversation_manager.get_transcript()
        assert len(transcript) == 2
        assert transcript[0].content == "what is the average age?"
        assert transcript[1].content == "what is the median age?"

        # Verify both cached results are present
        result1 = loaded_state2.result_cache.get("run_key_1", dataset_version)
        result2 = loaded_state2.result_cache.get("run_key_2", dataset_version)
        assert result1 is not None
        assert result2 is not None
        assert result1.result == {"mean": 45.5, "type": "descriptive"}
        assert result2.result == {"median": 44.0, "type": "descriptive"}

    def test_persistence_isolates_datasets(self, tmp_path):
        """Test that persistence isolates state per dataset version."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)

        # Create state for dataset 1
        manager1 = ConversationManager()
        manager1.set_dataset("dataset1")
        manager1.add_message("user", "query for dataset 1")
        cache1 = ResultCache(max_size=50)
        state1 = ConversationState(
            conversation_manager=manager1,
            result_cache=cache1,
            dataset_id="dataset1",
            upload_id="upload1",
            dataset_version="version1",
            last_updated=datetime.now(),
        )
        store.save(state1)

        # Create state for dataset 2
        manager2 = ConversationManager()
        manager2.set_dataset("dataset2")
        manager2.add_message("user", "query for dataset 2")
        cache2 = ResultCache(max_size=50)
        state2 = ConversationState(
            conversation_manager=manager2,
            result_cache=cache2,
            dataset_id="dataset2",
            upload_id="upload2",
            dataset_version="version2",
            last_updated=datetime.now(),
        )
        store.save(state2)

        # Act: Load each state
        loaded1 = store.load("upload1", "version1")
        loaded2 = store.load("upload2", "version2")

        # Assert: States are isolated
        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.conversation_manager.get_current_dataset() == "dataset1"
        assert loaded2.conversation_manager.get_current_dataset() == "dataset2"
        assert loaded1.conversation_manager.get_transcript()[0].content == "query for dataset 1"
        assert loaded2.conversation_manager.get_transcript()[0].content == "query for dataset 2"

    def test_persistence_handles_missing_state_gracefully(self, tmp_path):
        """Test that loading non-existent state returns None gracefully."""
        # Arrange
        store = FileStateStore(base_path=tmp_path)

        # Act
        loaded = store.load("nonexistent_upload", "nonexistent_version")

        # Assert
        assert loaded is None
