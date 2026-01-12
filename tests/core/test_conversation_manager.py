"""
Tests for ConversationManager - Pure Python conversation state management.

Tests verify UI-agnostic conversation history, dataset context, and follow-up logic.
"""

from datetime import datetime

from clinical_analytics.core.conversation_manager import ConversationManager


class TestConversationManager:
    """Test suite for ConversationManager."""

    def test_conversation_manager_initializes_empty(self):
        """Test that ConversationManager initializes with empty state."""
        # Arrange
        manager = ConversationManager()

        # Act
        transcript = manager.get_transcript()
        current_dataset = manager.get_current_dataset()
        active_query = manager.get_active_query()
        follow_ups = manager.get_follow_ups()

        # Assert
        assert transcript == []
        assert current_dataset is None
        assert active_query is None
        assert follow_ups == []

    def test_conversation_manager_add_message_returns_message_id(self):
        """Test that add_message returns a message ID."""
        # Arrange
        manager = ConversationManager()

        # Act
        message_id = manager.add_message("user", "What is the average age?")

        # Assert
        assert isinstance(message_id, str)
        assert len(message_id) > 0

    def test_conversation_manager_add_message_stores_message(self):
        """Test that add_message stores message in transcript."""
        # Arrange
        manager = ConversationManager()

        # Act
        message_id = manager.add_message("user", "What is the average age?")
        transcript = manager.get_transcript()

        # Assert
        assert len(transcript) == 1
        message = transcript[0]
        assert message.id == message_id
        assert message.role == "user"
        assert message.content == "What is the average age?"
        assert isinstance(message.timestamp, datetime)
        assert message.run_key is None

    def test_conversation_manager_add_message_with_run_key(self):
        """Test that add_message can associate message with run_key."""
        # Arrange
        manager = ConversationManager()
        run_key = "test_run_123"

        # Act
        manager.add_message("assistant", "The average age is 45", run_key=run_key)
        transcript = manager.get_transcript()

        # Assert
        message = transcript[0]
        assert message.run_key == run_key

    def test_conversation_manager_add_message_accepts_optional_run_key(self):
        """Test that add_message accepts optional run_key parameter."""
        # Arrange
        manager = ConversationManager()

        # Act
        manager.add_message("assistant", "Answer", run_key=None)
        transcript = manager.get_transcript()

        # Assert
        message = transcript[0]
        assert message.run_key is None

    def test_conversation_manager_get_transcript_returns_all_messages(self):
        """Test that get_transcript returns all messages in order."""
        # Arrange
        manager = ConversationManager()

        # Act
        manager.add_message("user", "Question 1")
        manager.add_message("assistant", "Answer 1")
        manager.add_message("user", "Question 2")
        transcript = manager.get_transcript()

        # Assert
        assert len(transcript) == 3
        assert transcript[0].content == "Question 1"
        assert transcript[1].content == "Answer 1"
        assert transcript[2].content == "Question 2"

    def test_conversation_manager_set_dataset_stores_dataset(self):
        """Test that set_dataset stores current dataset."""
        # Arrange
        manager = ConversationManager()
        dataset_id = "test_dataset_123"

        # Act
        manager.set_dataset(dataset_id)
        current_dataset = manager.get_current_dataset()

        # Assert
        assert current_dataset == dataset_id

    def test_conversation_manager_get_current_dataset_returns_none_initially(self):
        """Test that get_current_dataset returns None initially."""
        # Arrange
        manager = ConversationManager()

        # Act
        current_dataset = manager.get_current_dataset()

        # Assert
        assert current_dataset is None

    def test_conversation_manager_set_active_query_stores_query(self):
        """Test that set_active_query stores active query."""
        # Arrange
        manager = ConversationManager()
        query = "What is the average age?"

        # Act
        manager.set_active_query(query)
        active_query = manager.get_active_query()

        # Assert
        assert active_query == query

    def test_conversation_manager_get_active_query_returns_none_initially(self):
        """Test that get_active_query returns None initially."""
        # Arrange
        manager = ConversationManager()

        # Act
        active_query = manager.get_active_query()

        # Assert
        assert active_query is None

    def test_conversation_manager_set_follow_ups_stores_follow_ups(self):
        """Test that set_follow_ups stores follow-up questions."""
        # Arrange
        manager = ConversationManager()
        follow_ups = ["What about by gender?", "What about by age group?"]

        # Act
        manager.set_follow_ups(follow_ups)
        stored_follow_ups = manager.get_follow_ups()

        # Assert
        assert stored_follow_ups == follow_ups

    def test_conversation_manager_get_follow_ups_returns_empty_list_initially(self):
        """Test that get_follow_ups returns empty list initially."""
        # Arrange
        manager = ConversationManager()

        # Act
        follow_ups = manager.get_follow_ups()

        # Assert
        assert follow_ups == []

    def test_conversation_manager_clear_resets_all_state(self):
        """Test that clear resets all state to initial values."""
        # Arrange
        manager = ConversationManager()
        manager.set_dataset("test_dataset")
        manager.set_active_query("test query")
        manager.set_follow_ups(["follow up 1"])
        manager.add_message("user", "test message")

        # Act
        manager.clear()

        # Assert
        assert manager.get_transcript() == []
        assert manager.get_current_dataset() is None
        assert manager.get_active_query() is None
        assert manager.get_follow_ups() == []

    def test_conversation_manager_normalize_query_collapses_whitespace(self):
        """Test that normalize_query collapses whitespace and lowercases."""
        # Arrange
        manager = ConversationManager()
        raw_query = "  What   is   the   average   age?  "

        # Act
        normalized = manager.normalize_query(raw_query)

        # Assert
        assert normalized == "what is the average age?"

    def test_conversation_manager_normalize_query_handles_none(self):
        """Test that normalize_query handles None input."""
        # Arrange
        manager = ConversationManager()

        # Act
        normalized = manager.normalize_query(None)

        # Assert
        assert normalized == ""

    def test_conversation_manager_canonicalize_scope_drops_none_values(self):
        """Test that canonicalize_scope drops None values."""
        # Arrange
        manager = ConversationManager()
        scope = {"key1": "value1", "key2": None, "key3": "value3"}

        # Act
        canonical = manager.canonicalize_scope(scope)

        # Assert
        assert "key1" in canonical
        assert "key2" not in canonical
        assert "key3" in canonical

    def test_conversation_manager_canonicalize_scope_sorts_keys(self):
        """Test that canonicalize_scope sorts dictionary keys."""
        # Arrange
        manager = ConversationManager()
        scope = {"zebra": 1, "apple": 2, "banana": 3}

        # Act
        canonical = manager.canonicalize_scope(scope)

        # Assert
        keys = list(canonical.keys())
        assert keys == ["apple", "banana", "zebra"]

    def test_conversation_manager_canonicalize_scope_handles_nested_dicts(self):
        """Test that canonicalize_scope recursively handles nested dicts."""
        # Arrange
        manager = ConversationManager()
        scope = {"outer": {"inner": "value", "none_key": None}, "other": "value"}

        # Act
        canonical = manager.canonicalize_scope(scope)

        # Assert
        assert "outer" in canonical
        assert "inner" in canonical["outer"]
        assert "none_key" not in canonical["outer"]
        assert "other" in canonical

    def test_conversation_manager_canonicalize_scope_handles_none_input(self):
        """Test that canonicalize_scope handles None input."""
        # Arrange
        manager = ConversationManager()

        # Act
        canonical = manager.canonicalize_scope(None)

        # Assert
        assert canonical == {}

    def test_conversation_manager_serialize_returns_dict(self):
        """Test that serialize returns serializable dict."""
        # Arrange
        manager = ConversationManager()
        manager.set_dataset("test_dataset")
        manager.add_message("user", "test message")

        # Act
        serialized = manager.serialize()

        # Assert
        assert isinstance(serialized, dict)
        assert "dataset_id" in serialized
        assert "messages" in serialized

    def test_conversation_manager_deserialize_restores_state(self):
        """Test that deserialize restores ConversationManager state."""
        # Arrange
        manager = ConversationManager()
        manager.set_dataset("test_dataset")
        manager.add_message("user", "test message")
        serialized = manager.serialize()

        # Act
        restored = ConversationManager.deserialize(serialized)

        # Assert
        assert restored.get_current_dataset() == "test_dataset"
        assert len(restored.get_transcript()) == 1
        assert restored.get_transcript()[0].content == "test message"


class TestMessageStatus:
    """Test suite for Message.status field and related functionality."""

    def test_message_dataclass_status_defaults_to_completed(self):
        """Test that Message.status defaults to 'completed'."""
        # Arrange & Act
        from clinical_analytics.core.conversation_manager import Message

        message = Message(
            id="test-id",
            role="assistant",
            content="test content",
            timestamp=datetime.now(),
        )

        # Assert
        assert message.status == "completed"

    def test_message_dataclass_status_accepts_pending(self):
        """Test that Message.status accepts 'pending' value."""
        # Arrange & Act
        from clinical_analytics.core.conversation_manager import Message

        message = Message(
            id="test-id",
            role="assistant",
            content="",
            timestamp=datetime.now(),
            status="pending",
        )

        # Assert
        assert message.status == "pending"

    def test_message_dataclass_status_accepts_error(self):
        """Test that Message.status accepts 'error' value."""
        # Arrange & Act
        from clinical_analytics.core.conversation_manager import Message

        message = Message(
            id="test-id",
            role="assistant",
            content="Error occurred",
            timestamp=datetime.now(),
            status="error",
        )

        # Assert
        assert message.status == "error"

    def test_add_message_status_parameter_creates_pending_message(self):
        """Test that add_message accepts status parameter for pending messages."""
        # Arrange
        manager = ConversationManager()

        # Act
        message_id = manager.add_message("assistant", "", status="pending")
        transcript = manager.get_transcript()

        # Assert
        assert len(transcript) == 1
        message = transcript[0]
        assert message.id == message_id
        assert message.status == "pending"
        assert message.content == ""

    def test_add_message_status_defaults_to_completed(self):
        """Test that add_message defaults status to 'completed'."""
        # Arrange
        manager = ConversationManager()

        # Act
        manager.add_message("user", "What is the average age?")
        transcript = manager.get_transcript()

        # Assert
        message = transcript[0]
        assert message.status == "completed"


class TestUpdateMessage:
    """Test suite for ConversationManager.update_message() method."""

    def test_update_message_status_changes_status(self):
        """Test that update_message changes status from pending to completed."""
        # Arrange
        manager = ConversationManager()
        message_id = manager.add_message("assistant", "", status="pending")

        # Act
        result = manager.update_message(message_id, status="completed")
        transcript = manager.get_transcript()

        # Assert
        assert result is True
        assert transcript[0].status == "completed"

    def test_update_message_content_changes_content(self):
        """Test that update_message changes message content."""
        # Arrange
        manager = ConversationManager()
        message_id = manager.add_message("assistant", "", status="pending")

        # Act
        result = manager.update_message(message_id, content="The answer is 42")
        transcript = manager.get_transcript()

        # Assert
        assert result is True
        assert transcript[0].content == "The answer is 42"

    def test_update_message_run_key_changes_run_key(self):
        """Test that update_message changes run_key."""
        # Arrange
        manager = ConversationManager()
        message_id = manager.add_message("assistant", "", status="pending")

        # Act
        result = manager.update_message(message_id, run_key="run_123")
        transcript = manager.get_transcript()

        # Assert
        assert result is True
        assert transcript[0].run_key == "run_123"

    def test_update_message_multiple_fields_changes_all(self):
        """Test that update_message can change multiple fields at once."""
        # Arrange
        manager = ConversationManager()
        message_id = manager.add_message("assistant", "", status="pending")

        # Act
        result = manager.update_message(
            message_id,
            status="completed",
            content="Analysis complete",
            run_key="run_456",
        )
        transcript = manager.get_transcript()

        # Assert
        assert result is True
        message = transcript[0]
        assert message.status == "completed"
        assert message.content == "Analysis complete"
        assert message.run_key == "run_456"

    def test_update_message_invalid_id_returns_false(self):
        """Test that update_message returns False for non-existent message ID."""
        # Arrange
        manager = ConversationManager()
        manager.add_message("assistant", "test")

        # Act
        result = manager.update_message("invalid-id", status="completed")

        # Assert
        assert result is False

    def test_update_message_ignores_invalid_fields(self):
        """Test that update_message ignores fields that don't exist on Message."""
        # Arrange
        manager = ConversationManager()
        message_id = manager.add_message("assistant", "test")

        # Act - should not raise, just ignore invalid field
        result = manager.update_message(message_id, invalid_field="value")

        # Assert
        assert result is True  # Message was found, field just ignored


class TestSerializeDeserializeStatus:
    """Test suite for serialization/deserialization of Message.status."""

    def test_serialize_includes_status_field(self):
        """Test that serialize includes status field in messages."""
        # Arrange
        manager = ConversationManager()
        manager.add_message("assistant", "test", status="pending")

        # Act
        serialized = manager.serialize()

        # Assert
        assert "messages" in serialized
        assert len(serialized["messages"]) == 1
        assert "status" in serialized["messages"][0]
        assert serialized["messages"][0]["status"] == "pending"

    def test_deserialize_restores_status_field(self):
        """Test that deserialize restores status field."""
        # Arrange
        manager = ConversationManager()
        manager.add_message("assistant", "test", status="pending")
        serialized = manager.serialize()

        # Act
        restored = ConversationManager.deserialize(serialized)
        transcript = restored.get_transcript()

        # Assert
        assert len(transcript) == 1
        assert transcript[0].status == "pending"

    def test_deserialize_backward_compat_defaults_status_to_completed(self):
        """Test that deserialize defaults status to 'completed' for old data."""
        # Arrange - simulate old serialized data without status field
        old_data = {
            "messages": [
                {
                    "id": "old-msg-id",
                    "role": "assistant",
                    "content": "old message",
                    "timestamp": datetime.now().isoformat(),
                    "run_key": None,
                    # No status field - old format
                }
            ],
            "dataset_id": None,
            "active_query": None,
            "follow_ups": [],
        }

        # Act
        restored = ConversationManager.deserialize(old_data)
        transcript = restored.get_transcript()

        # Assert
        assert len(transcript) == 1
        assert transcript[0].status == "completed"  # Should default to completed

    def test_serialize_deserialize_roundtrip_preserves_status(self):
        """Test that serialize/deserialize roundtrip preserves all status values."""
        # Arrange
        manager = ConversationManager()
        manager.add_message("user", "question", status="completed")
        manager.add_message("assistant", "", status="pending")
        manager.add_message("assistant", "error msg", status="error")

        # Act
        serialized = manager.serialize()
        restored = ConversationManager.deserialize(serialized)
        transcript = restored.get_transcript()

        # Assert
        assert len(transcript) == 3
        assert transcript[0].status == "completed"
        assert transcript[1].status == "pending"
        assert transcript[2].status == "error"
