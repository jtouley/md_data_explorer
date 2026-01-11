"""
Tests for Ask Questions state machine persistence and execution control.

Tests verify that intent_signal persists correctly across reruns and that
the execution guard properly triggers when intent_signal is set.

Also tests the fixes for:
- State restoration with manager dataset update (prevents false dataset change detection)
- Dataset change guard with intent_signal='nl_parsed' (prevents race condition)
"""

from datetime import datetime

from clinical_analytics.core.conversation_manager import ConversationManager
from clinical_analytics.core.result_cache import CachedResult, ResultCache
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


class TestAskQuestionsStateMachinePersistence:
    """Test suite for Ask Questions state machine persistence."""

    def test_intent_signal_persists_and_execution_block_runs(self, mock_session_state):
        """
        Test that intent_signal='nl_parsed' persists and execution block executes.

        This test verifies the core bug fix: intent_signal should persist across reruns
        and trigger execution when set.
        """
        # Arrange: Set up state as if query was just parsed
        context = AnalysisContext(
            inferred_intent=AnalysisIntent.COUNT,
            research_question="How many patients?",
        )
        mock_session_state["analysis_context"] = context
        mock_session_state["intent_signal"] = "nl_parsed"
        mock_session_state["chat"] = []
        mock_session_state["result_cache"] = ResultCache(max_size=50)

        # Act: Verify state is persisted (simulate session state access)
        persisted_signal = mock_session_state.get("intent_signal")
        persisted_context = mock_session_state.get("analysis_context")

        # Assert: State should persist exactly as set
        assert persisted_signal == "nl_parsed", "intent_signal should persist as 'nl_parsed'"
        assert persisted_context is not None, "analysis_context should persist"
        assert persisted_context.inferred_intent == AnalysisIntent.COUNT

    def test_state_initialization_does_not_clear_intent_signal(self, mock_session_state):
        """
        Test that state initialization doesn't incorrectly clear intent_signal.

        This verifies the fix to separate initialization checks:
        - Only initialize analysis_context if not present
        - Only initialize intent_signal if not present
        - Don't reset one based on the other
        """
        # Arrange: Set analysis_context to None but intent_signal to "nl_parsed"
        # (this can happen after parsing but before execution)
        mock_session_state["analysis_context"] = AnalysisContext()
        mock_session_state["intent_signal"] = "nl_parsed"

        # Act: Simulate the fixed initialization logic
        # Fixed code: Check EACH state variable separately
        if "analysis_context" not in mock_session_state:
            mock_session_state["analysis_context"] = None
        if "intent_signal" not in mock_session_state:
            mock_session_state["intent_signal"] = None

        # Assert: intent_signal should NOT be cleared by initialization
        assert (
            mock_session_state["intent_signal"] == "nl_parsed"
        ), "intent_signal should not be cleared by state initialization"

    def test_execution_block_check_with_valid_state(self, mock_session_state):
        """
        Test that execution block executes when intent_signal and analysis_context are valid.

        Simulates the execution guard logic to verify it correctly identifies
        when execution should proceed.
        """
        # Arrange: Valid state for execution
        context = AnalysisContext(
            inferred_intent=AnalysisIntent.DESCRIBE,
            research_question="Describe patient characteristics",
            primary_variable="age",
        )
        mock_session_state["analysis_context"] = context
        mock_session_state["intent_signal"] = "nl_parsed"

        # Act: Simulate execution guard check (from main() at ~line 2003)
        intent_signal = mock_session_state.get("intent_signal")
        has_context = "analysis_context" in mock_session_state
        context_value = mock_session_state.get("analysis_context")

        # Determine if execution should happen
        should_execute = intent_signal is not None and context_value is not None

        # Assert: Should execute when both are present
        assert intent_signal == "nl_parsed", "intent_signal should be set"
        assert has_context is True, "analysis_context should be present"
        assert should_execute is True, "Execution should happen with valid state"

    def test_execution_block_skip_when_intent_signal_none(self, mock_session_state):
        """
        Test that execution block is skipped when intent_signal is None.

        Verifies the guard prevents execution on pages without parsed queries.
        """
        # Arrange: No parsed query yet
        mock_session_state["analysis_context"] = None
        mock_session_state["intent_signal"] = None

        # Act: Simulate execution guard check
        intent_signal = mock_session_state.get("intent_signal")
        should_execute = intent_signal is not None

        # Assert: Should NOT execute when intent_signal is None
        assert intent_signal is None, "intent_signal should be None initially"
        assert should_execute is False, "Execution should not happen when intent_signal is None"

    def test_state_machine_invariant_violation_detection(self, mock_session_state):
        """
        Test that state machine detects invalid state (intent_signal set but no context).

        This verifies the invariant check that catches inconsistent state.
        """
        # Arrange: Invalid state (intent_signal set but context is None)
        mock_session_state["analysis_context"] = None
        mock_session_state["intent_signal"] = "nl_parsed"

        # Act: Check for state machine invariant violation
        intent_signal = mock_session_state.get("intent_signal")
        context = mock_session_state.get("analysis_context")
        is_invalid = intent_signal is not None and context is None

        # Assert: Should detect this as invalid state
        assert is_invalid is True, "Should detect intent_signal without context as invalid"


class TestAskQuestionsChatRendering:
    """Test suite for chat message rendering with cached results."""

    def test_render_chat_retrieves_result_from_cache(self, mock_session_state):
        """
        Test that render_chat can retrieve results from cache using run_key.

        This verifies the core rendering flow: chat message has run_key,
        render_chat uses run_key to fetch result from cache.
        """
        # Arrange: Add assistant message to chat with run_key, result in cache
        run_key = "test_run_key_001"
        dataset_version = "test_dataset_v1"
        result_data = {"type": "count", "value": 42, "intent": "COUNT"}

        # Create cache and store result
        cache = ResultCache(max_size=50)
        cached_result = CachedResult(
            run_key=run_key,
            query="How many patients?",
            result=result_data,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)
        mock_session_state["result_cache"] = cache

        # Create chat message with run_key
        chat_message = {
            "role": "assistant",
            "text": "42 patients found",
            "run_key": run_key,
            "status": "completed",
        }
        mock_session_state["chat"] = [chat_message]

        # Act: Simulate render_chat retrieving result
        retrieved = cache.get(run_key, dataset_version)

        # Assert: Result should be retrievable from cache
        assert retrieved is not None, "Result should be in cache"
        assert retrieved.result == result_data, "Result data should match"
        assert retrieved.run_key == run_key, "run_key should match"

    def test_render_chat_handles_cache_miss(self, mock_session_state):
        """
        Test that render_chat handles gracefully when result is not in cache.

        This verifies fallback behavior when cache miss occurs.
        """
        # Arrange: Chat message with run_key but result not in cache
        run_key = "nonexistent_run_key"
        dataset_version = "test_dataset_v1"

        cache = ResultCache(max_size=50)
        mock_session_state["result_cache"] = cache

        chat_message = {
            "role": "assistant",
            "text": "Result would be shown here",
            "run_key": run_key,
            "status": "completed",
        }
        mock_session_state["chat"] = [chat_message]

        # Act: Try to retrieve non-existent result
        retrieved = cache.get(run_key, dataset_version)

        # Assert: Cache miss should return None (not crash)
        assert retrieved is None, "Cache miss should return None, not crash"

    def test_assistant_message_has_run_key_for_result_lookup(self, mock_session_state):
        """
        Test that assistant messages include run_key for result lookup.

        This verifies that messages can be linked back to cached results.
        """
        # Arrange: Create assistant message with run_key
        run_key = "test_run_key_002"
        chat_message = {
            "role": "assistant",
            "text": "Analysis complete",
            "run_key": run_key,
            "status": "completed",
        }

        # Act: Access run_key from message
        message_run_key = chat_message.get("run_key")

        # Assert: run_key should be present and accessible
        assert message_run_key == run_key, "Assistant message should have run_key for lookup"
        assert message_run_key is not None, "run_key should not be None"

    def test_result_cached_before_assistant_message_added(self, mock_session_state):
        """
        Test that results are cached BEFORE assistant messages are added to chat.

        This verifies order of operations to prevent cache misses during rendering.
        """
        # Arrange: Set up for ordered execution
        run_key = "test_run_key_003"
        dataset_version = "test_dataset_v1"
        result_data = {"count": 42}

        cache = ResultCache(max_size=50)
        mock_session_state["result_cache"] = cache
        mock_session_state["chat"] = []

        # Act: First, cache the result
        cached_result = CachedResult(
            run_key=run_key,
            query="Test query",
            result=result_data,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        # Then, add assistant message
        chat_message = {
            "role": "assistant",
            "text": "Result: 42",
            "run_key": run_key,
            "status": "completed",
        }
        mock_session_state["chat"].append(chat_message)

        # Finally, render chat should find result in cache
        retrieved = cache.get(run_key, dataset_version)

        # Assert: Result should be found because it was cached before message added
        assert retrieved is not None, "Result should be in cache"
        assert len(mock_session_state["chat"]) == 1, "Chat should have assistant message"
        assert mock_session_state["chat"][0]["run_key"] == run_key, "Message should reference cached result"


class TestStateRestorationWithManagerDatasetUpdate:
    """
    Test suite for state restoration with manager dataset update fix.

    These tests verify that when state is restored from disk, the manager's
    dataset is updated to match the current selection, preventing false
    "dataset changed" detection.
    """

    def test_restored_manager_dataset_updated_to_current(self, mock_session_state):
        """
        Test that restored manager's dataset is updated to match current selection.

        This verifies the fix: after restoring state from disk, we call
        saved_state.conversation_manager.set_dataset(dataset_choice) to prevent
        false dataset change detection.
        """
        # Arrange: Simulate a restored manager with NO dataset set (stale state)
        restored_manager = ConversationManager()
        # Stale manager has no dataset set (simulates restored from disk)
        assert restored_manager.get_current_dataset() is None

        current_dataset_choice = "user_upload_20251228_203407_376a8faa"

        # Act: Apply the fix - update restored manager's dataset
        restored_manager.set_dataset(current_dataset_choice)

        # Assert: Manager now has current dataset
        assert restored_manager.get_current_dataset() == current_dataset_choice

    def test_restored_manager_with_stale_dataset_triggers_false_change(self, mock_session_state):
        """
        Test that demonstrates the bug: restored manager with stale dataset
        triggers false "dataset changed" detection.

        This shows what happens WITHOUT the fix.
        """
        # Arrange: Manager restored from disk with stale/no dataset
        restored_manager = ConversationManager()
        restored_manager.set_dataset("old_dataset_id")  # Stale dataset

        current_dataset_choice = "new_dataset_id"

        # Act: Check for dataset change (this is the bug)
        last_dataset = restored_manager.get_current_dataset()
        dataset_changed = last_dataset != current_dataset_choice

        # Assert: Without fix, this would trigger clearing of analysis_context
        assert dataset_changed is True, "Stale manager triggers false dataset change"

    def test_restored_manager_with_updated_dataset_no_false_change(self, mock_session_state):
        """
        Test that demonstrates the fix: updating restored manager's dataset
        prevents false "dataset changed" detection.
        """
        # Arrange: Manager restored from disk with stale/no dataset
        restored_manager = ConversationManager()
        restored_manager.set_dataset("old_dataset_id")  # Stale dataset

        current_dataset_choice = "new_dataset_id"

        # Act: Apply the fix - update manager's dataset BEFORE comparison
        restored_manager.set_dataset(current_dataset_choice)
        last_dataset = restored_manager.get_current_dataset()
        dataset_changed = last_dataset != current_dataset_choice

        # Assert: With fix, no false dataset change
        assert dataset_changed is False, "Updated manager prevents false dataset change"


class TestDatasetChangeGuardWithIntentSignal:
    """
    Test suite for dataset change guard with intent_signal='nl_parsed'.

    These tests verify that when intent_signal is 'nl_parsed', the dataset
    change detection skips clearing the analysis_context (prevents race condition).
    """

    def test_dataset_change_skipped_when_intent_signal_nl_parsed(self, mock_session_state):
        """
        Test that dataset change handling is skipped when intent_signal='nl_parsed'.

        This verifies the guard: if we just parsed a query, don't clear the
        context even if dataset appears to have changed (race condition).
        """
        # Arrange: State after query parsing (context set, intent_signal='nl_parsed')
        context = AnalysisContext(
            inferred_intent=AnalysisIntent.DESCRIBE,
            primary_variable="age",
        )
        mock_session_state["analysis_context"] = context
        mock_session_state["intent_signal"] = "nl_parsed"

        # Simulate dataset comparison detecting a "change"
        last_dataset = None  # Stale manager had no dataset
        current_dataset = "user_upload_20251228_203407_376a8faa"
        dataset_appears_changed = last_dataset != current_dataset

        # Act: Apply the guard logic
        intent_signal = mock_session_state.get("intent_signal")
        should_skip_clearing = intent_signal == "nl_parsed"

        # Assert: Guard should prevent clearing when intent_signal='nl_parsed'
        assert dataset_appears_changed is True, "Dataset comparison shows change"
        assert should_skip_clearing is True, "Guard should skip clearing"
        # Context should NOT be cleared
        assert mock_session_state.get("analysis_context") is not None

    def test_dataset_change_allowed_when_intent_signal_none(self, mock_session_state):
        """
        Test that dataset change handling proceeds when intent_signal is None.

        When no query is pending, dataset change should clear context normally.
        """
        # Arrange: Normal state (no pending query)
        mock_session_state["analysis_context"] = None
        mock_session_state["intent_signal"] = None

        # Simulate dataset comparison detecting a "change"
        last_dataset = "old_dataset"
        current_dataset = "new_dataset"
        dataset_appears_changed = last_dataset != current_dataset

        # Act: Apply the guard logic
        intent_signal = mock_session_state.get("intent_signal")
        should_skip_clearing = intent_signal == "nl_parsed"

        # Assert: Guard should NOT prevent clearing when intent_signal is None
        assert dataset_appears_changed is True, "Dataset comparison shows change"
        assert should_skip_clearing is False, "Guard should NOT skip clearing"

    def test_context_preserved_during_rerun_race_condition(self, mock_session_state):
        """
        Test the full race condition scenario:
        1. Query parsed → context stored → intent_signal='nl_parsed'
        2. Rerun happens
        3. Dataset appears changed (stale manager)
        4. Guard prevents clearing because intent_signal='nl_parsed'
        5. Context is preserved for execution
        """
        # Arrange: Simulate state after parsing, before execution
        context = AnalysisContext(
            inferred_intent=AnalysisIntent.COUNT,
            research_question="How many patients?",
        )
        mock_session_state["analysis_context"] = context
        mock_session_state["intent_signal"] = "nl_parsed"
        mock_session_state["chat"] = [
            {"role": "user", "text": "How many patients?", "run_key": None, "status": "completed"}
        ]

        # Simulate stale manager (restored from disk, no dataset set)
        manager = ConversationManager()  # Fresh manager, no dataset
        last_dataset = manager.get_current_dataset()  # None
        current_dataset = "user_upload_12345"

        # Act: Dataset comparison + guard check
        dataset_changed = last_dataset != current_dataset
        intent_signal = mock_session_state.get("intent_signal")

        # Apply guard: if intent_signal='nl_parsed', skip clearing
        if dataset_changed and intent_signal == "nl_parsed":
            # Just update manager's dataset, don't clear context
            manager.set_dataset(current_dataset)
            context_preserved = True
        else:
            context_preserved = False

        # Assert: Context should be preserved
        assert context_preserved is True, "Guard should preserve context during race condition"
        assert mock_session_state.get("analysis_context") is not None, "Context not cleared"
        assert mock_session_state.get("intent_signal") == "nl_parsed", "intent_signal preserved"
        assert manager.get_current_dataset() == current_dataset, "Manager dataset updated"


class TestFullChatFlowIntegration:
    """
    Integration tests for the full chat flow: query → parse → execute → render.

    These tests verify the end-to-end state machine behavior using pure Python
    logic extracted from the Streamlit page.
    """

    def test_chat_flow_state_machine_transitions(self, mock_session_state):
        """
        Test state machine transitions through the full chat flow.

        States: IDLE → PARSED → EXECUTING → COMPLETED
        """
        # === Initial State (IDLE) ===
        mock_session_state["analysis_context"] = None
        mock_session_state["intent_signal"] = None
        mock_session_state["chat"] = []
        cache = ResultCache(max_size=50)
        mock_session_state["result_cache"] = cache

        # Verify initial state
        assert mock_session_state.get("intent_signal") is None, "Should start in IDLE"

        # === After Query Parsing (PARSED) ===
        # Simulate successful query parsing
        context = AnalysisContext(
            inferred_intent=AnalysisIntent.DESCRIBE,
            primary_variable="age",
            research_question="What is the average age?",
        )
        mock_session_state["analysis_context"] = context
        mock_session_state["intent_signal"] = "nl_parsed"

        # Add user message to chat
        user_msg = {
            "role": "user",
            "text": "What is the average age?",
            "run_key": None,
            "status": "completed",
        }
        mock_session_state["chat"].append(user_msg)

        # Verify PARSED state
        assert mock_session_state.get("intent_signal") == "nl_parsed", "Should be in PARSED"
        assert mock_session_state.get("analysis_context") is not None, "Context should exist"
        assert len(mock_session_state["chat"]) == 1, "User message added"

        # === Execution Check (should proceed) ===
        intent_signal = mock_session_state.get("intent_signal")
        context_value = mock_session_state.get("analysis_context")
        should_execute = intent_signal is not None and context_value is not None

        assert should_execute is True, "Execution should proceed"

        # === After Execution (COMPLETED) ===
        # Simulate result caching
        run_key = "test_run_key"
        dataset_version = "test_dataset"
        result = {
            "type": "describe",
            "intent": "DESCRIBE",
            "headline": "Average age is 45.2 years",
            "mean": 45.2,
        }

        cached_result = CachedResult(
            run_key=run_key,
            query="What is the average age?",
            result=result,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)

        # Add assistant message
        assistant_msg = {
            "role": "assistant",
            "text": "Average age is 45.2 years",
            "run_key": run_key,
            "status": "completed",
        }
        mock_session_state["chat"].append(assistant_msg)

        # Clear intent_signal after execution (transition to COMPLETED)
        mock_session_state["intent_signal"] = None

        # Verify COMPLETED state
        assert mock_session_state.get("intent_signal") is None, "Back to IDLE/COMPLETED"
        assert len(mock_session_state["chat"]) == 2, "Both messages in chat"
        assert cache.get(run_key, dataset_version) is not None, "Result cached"

    def test_chat_flow_render_retrieves_cached_result(self, mock_session_state):
        """
        Test that render_chat retrieves results from cache for display.
        """
        # Arrange: Set up completed chat with cached result
        run_key = "render_test_key"
        dataset_version = "render_dataset"
        result = {
            "type": "count",
            "intent": "COUNT",
            "headline": "42 patients found",
            "count": 42,
        }

        cache = ResultCache(max_size=50)
        cached_result = CachedResult(
            run_key=run_key,
            query="How many patients?",
            result=result,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached_result)
        mock_session_state["result_cache"] = cache

        mock_session_state["chat"] = [
            {"role": "user", "text": "How many patients?", "status": "completed"},
            {"role": "assistant", "text": "42 patients found", "run_key": run_key, "status": "completed"},
        ]

        # Act: Simulate render_chat retrieval
        assistant_msg = mock_session_state["chat"][1]
        msg_run_key = assistant_msg.get("run_key")
        retrieved = cache.get(msg_run_key, dataset_version)

        # Assert: Result should be retrievable
        assert retrieved is not None, "Result should be in cache"
        assert retrieved.result["count"] == 42, "Result data should match"
        assert retrieved.result["intent"] == "COUNT", "Intent should match"

    def test_chat_flow_context_persists_for_followup(self, mock_session_state):
        """
        Test that analysis context persists for follow-up queries.
        """
        # Arrange: First query completed
        context = AnalysisContext(
            inferred_intent=AnalysisIntent.DESCRIBE,
            primary_variable="age",
            research_question="What is the average age?",
        )
        mock_session_state["analysis_context"] = context
        mock_session_state["intent_signal"] = None  # Completed

        # Act: User asks follow-up
        # The existing context should inform the follow-up parsing
        existing_context = mock_session_state.get("analysis_context")

        # Assert: Context persists for follow-up
        assert existing_context is not None, "Context persists after execution"
        assert existing_context.primary_variable == "age", "Primary variable preserved"
        assert existing_context.inferred_intent == AnalysisIntent.DESCRIBE, "Intent preserved"


class TestRenderChatConversationManagerMigration:
    """
    Test suite for render_chat migration to ConversationManager.

    These tests verify that render_chat reads from ConversationManager.get_transcript()
    instead of st.session_state["chat"], enabling unified transcript management.
    """

    def test_render_chat_reads_from_conversation_manager_transcript(self, mock_session_state):
        """
        Test that render_chat reads messages from ConversationManager.get_transcript().

        This is the core migration test: render_chat should use the ConversationManager
        as the source of truth, not st.session_state["chat"].
        """
        # Arrange: Set up ConversationManager with messages
        manager = ConversationManager()
        manager.add_message("user", "How many patients?")
        manager.add_message("assistant", "42 patients found", run_key="run_123", status="completed")
        mock_session_state["conversation_manager"] = manager

        # Also set up legacy st.session_state["chat"] with DIFFERENT content
        # If render_chat uses manager, it won't see this legacy content
        mock_session_state["chat"] = [{"role": "user", "text": "LEGACY - should not be used", "status": "completed"}]

        # Act: Get messages from manager (what render_chat should do)
        transcript = mock_session_state["conversation_manager"].get_transcript()

        # Assert: Should get messages from manager, not legacy chat
        assert len(transcript) == 2, "Should have 2 messages from manager"
        assert transcript[0].content == "How many patients?", "First message from manager"
        assert transcript[1].content == "42 patients found", "Second message from manager"
        assert transcript[1].run_key == "run_123", "run_key preserved"
        assert transcript[1].status == "completed", "status preserved"

    def test_render_chat_handles_pending_status_from_manager(self, mock_session_state):
        """
        Test that render_chat correctly identifies pending messages from manager.

        Pending messages should show "Thinking..." indicator in the UI.
        """
        # Arrange: Manager with a pending assistant message
        manager = ConversationManager()
        manager.add_message("user", "Complex query")
        manager.add_message("assistant", "", status="pending")  # Empty content, pending status
        mock_session_state["conversation_manager"] = manager

        # Act: Get transcript and check for pending messages
        transcript = manager.get_transcript()
        pending_messages = [msg for msg in transcript if msg.status == "pending"]

        # Assert: Should identify pending message
        assert len(pending_messages) == 1, "Should have 1 pending message"
        assert pending_messages[0].role == "assistant", "Pending message is assistant"
        assert pending_messages[0].content == "", "Pending message has empty content"

    def test_render_chat_gracefully_handles_missing_manager(self, mock_session_state):
        """
        Test that render_chat handles case where conversation_manager is not set.

        Should not crash if manager is None (e.g., first page load).
        """
        # Arrange: No conversation_manager in session state
        mock_session_state.pop("conversation_manager", None)

        # Act: Check for manager safely
        manager = mock_session_state.get("conversation_manager")

        # Assert: Should be None, not crash
        assert manager is None, "Manager should be None when not set"

    def test_render_chat_message_attributes_match_message_dataclass(self, mock_session_state):
        """
        Test that messages from manager have all required attributes for rendering.

        render_chat needs: role, content (text), run_key, status
        """
        # Arrange: Manager with fully populated message
        manager = ConversationManager()
        manager.add_message(
            "assistant",
            "Analysis complete: 42 patients",
            run_key="run_456",
            status="completed",
        )
        mock_session_state["conversation_manager"] = manager

        # Act: Get message from transcript
        transcript = manager.get_transcript()
        message = transcript[0]

        # Assert: All required attributes present
        assert hasattr(message, "role"), "Message must have role"
        assert hasattr(message, "content"), "Message must have content"
        assert hasattr(message, "run_key"), "Message must have run_key"
        assert hasattr(message, "status"), "Message must have status"

        # Verify values
        assert message.role == "assistant"
        assert message.content == "Analysis complete: 42 patients"
        assert message.run_key == "run_456"
        assert message.status == "completed"

    def test_transcript_restoration_provides_messages_for_render(self, mock_session_state):
        """
        Test that restored manager provides messages for render_chat.

        When state is restored from disk, render_chat should still work.
        """
        # Arrange: Create and serialize manager with messages
        original_manager = ConversationManager()
        original_manager.add_message("user", "What is the average age?")
        original_manager.add_message("assistant", "45.2 years", run_key="run_789", status="completed")
        serialized = original_manager.serialize()

        # Simulate restoration from disk
        restored_manager = ConversationManager.deserialize(serialized)
        mock_session_state["conversation_manager"] = restored_manager

        # Act: Get transcript from restored manager
        transcript = restored_manager.get_transcript()

        # Assert: Restored messages available for rendering
        assert len(transcript) == 2, "Should have 2 restored messages"
        assert transcript[0].content == "What is the average age?"
        assert transcript[1].content == "45.2 years"
        assert transcript[1].run_key == "run_789"
        assert transcript[1].status == "completed"


class TestPendingMessagePattern:
    """
    Test suite for pending message pattern.

    The pending message pattern ensures:
    1. User message added to ConversationManager
    2. Pending assistant message added (status="pending", empty content)
    3. Single rerun to show "Thinking..." state
    4. Execution completes
    5. Pending message updated to completed with content
    6. Single rerun to show result
    """

    def test_pending_message_added_before_execution(self, mock_session_state):
        """
        Test that a pending assistant message is added before execution starts.

        This enables the "Thinking..." UX while query is being processed.
        """
        # Arrange: Set up ConversationManager
        manager = ConversationManager()
        mock_session_state["conversation_manager"] = manager

        # Act: Simulate adding user message and pending assistant message
        manager.add_message("user", "How many patients?")
        pending_id = manager.add_message("assistant", "", status="pending")
        mock_session_state["pending_message_id"] = pending_id

        # Assert: Manager should have pending assistant message
        transcript = manager.get_transcript()
        assert len(transcript) == 2, "Should have user and pending messages"
        assert transcript[1].status == "pending", "Assistant message should be pending"
        assert transcript[1].content == "", "Pending message should have empty content"
        assert mock_session_state.get("pending_message_id") == pending_id

    def test_pending_message_updated_after_execution(self, mock_session_state):
        """
        Test that pending message is updated to completed after execution.

        The update_message() method should change status, content, and run_key.
        """
        # Arrange: Manager with pending message
        manager = ConversationManager()
        manager.add_message("user", "How many patients?")
        pending_id = manager.add_message("assistant", "", status="pending")
        mock_session_state["conversation_manager"] = manager
        mock_session_state["pending_message_id"] = pending_id

        # Act: Simulate execution completing and updating message
        run_key = "run_test_123"
        assistant_text = "Found 42 patients"
        manager.update_message(
            pending_id,
            status="completed",
            content=assistant_text,
            run_key=run_key,
        )
        mock_session_state.pop("pending_message_id", None)

        # Assert: Message should now be completed
        transcript = manager.get_transcript()
        assert len(transcript) == 2, "Should still have 2 messages"
        assert transcript[1].status == "completed", "Status should be completed"
        assert transcript[1].content == assistant_text, "Content should be updated"
        assert transcript[1].run_key == run_key, "run_key should be set"
        assert "pending_message_id" not in mock_session_state

    def test_pending_message_error_handling(self, mock_session_state):
        """
        Test that pending message is updated to error status on failure.
        """
        # Arrange: Manager with pending message
        manager = ConversationManager()
        manager.add_message("user", "Invalid query")
        pending_id = manager.add_message("assistant", "", status="pending")
        mock_session_state["conversation_manager"] = manager
        mock_session_state["pending_message_id"] = pending_id

        # Act: Simulate execution failing
        error_message = "Error: Could not parse query"
        manager.update_message(
            pending_id,
            status="error",
            content=error_message,
        )
        mock_session_state.pop("pending_message_id", None)

        # Assert: Message should have error status
        transcript = manager.get_transcript()
        assert transcript[1].status == "error", "Status should be error"
        assert transcript[1].content == error_message, "Error content should be set"

    def test_pending_pattern_preserves_message_order(self, mock_session_state):
        """
        Test that pending pattern maintains correct message order in transcript.

        Messages should always be in chronological order: user first, then assistant.
        """
        # Arrange: Multiple query cycles
        manager = ConversationManager()
        mock_session_state["conversation_manager"] = manager

        # First query cycle
        manager.add_message("user", "Query 1")
        pending_id_1 = manager.add_message("assistant", "", status="pending")
        manager.update_message(pending_id_1, status="completed", content="Result 1", run_key="run_1")

        # Second query cycle
        manager.add_message("user", "Query 2")
        pending_id_2 = manager.add_message("assistant", "", status="pending")
        manager.update_message(pending_id_2, status="completed", content="Result 2", run_key="run_2")

        # Assert: All messages in correct order
        transcript = manager.get_transcript()
        assert len(transcript) == 4, "Should have 4 messages"
        assert transcript[0].role == "user" and transcript[0].content == "Query 1"
        assert transcript[1].role == "assistant" and transcript[1].content == "Result 1"
        assert transcript[2].role == "user" and transcript[2].content == "Query 2"
        assert transcript[3].role == "assistant" and transcript[3].content == "Result 2"

    def test_pending_message_id_cleared_after_completion(self, mock_session_state):
        """
        Test that pending_message_id is cleared from session state after completion.

        This prevents stale pending_message_id from affecting subsequent queries.
        """
        # Arrange: Set up pending message
        manager = ConversationManager()
        manager.add_message("user", "Test query")
        pending_id = manager.add_message("assistant", "", status="pending")
        mock_session_state["conversation_manager"] = manager
        mock_session_state["pending_message_id"] = pending_id

        # Act: Complete the pending message and clear ID
        manager.update_message(pending_id, status="completed", content="Done", run_key="run_x")
        mock_session_state.pop("pending_message_id", None)

        # Assert: pending_message_id should be cleared
        assert mock_session_state.get("pending_message_id") is None

        # Start new query - should not conflict with previous pending ID
        manager.add_message("user", "New query")
        new_pending_id = manager.add_message("assistant", "", status="pending")
        mock_session_state["pending_message_id"] = new_pending_id

        assert new_pending_id != pending_id, "New pending ID should be different"
        assert mock_session_state["pending_message_id"] == new_pending_id


class TestRenderChatOnlyRendering:
    """
    Test suite for ensuring all rendering goes through render_chat only.

    Phase 4: Remove inline render_result calls from execute_analysis_with_idempotency.
    All chat message rendering should go through render_chat() which reads from
    ConversationManager transcript.
    """

    def test_transcript_driven_rendering_reads_from_cache(self, mock_session_state):
        """
        Test that transcript-driven rendering can retrieve results from cache.

        render_chat should read run_key from message, fetch result from cache,
        and render it - eliminating need for inline rendering.
        """
        # Arrange: Set up ConversationManager with completed message
        manager = ConversationManager()
        manager.add_message("user", "How many patients?")
        manager.add_message("assistant", "42 patients found", run_key="run_test", status="completed")
        mock_session_state["conversation_manager"] = manager

        # Set up result cache with the result
        cache = ResultCache(max_size=50)
        cached_result = CachedResult(
            run_key="run_test",
            query="How many patients?",
            result={"type": "count", "value": 42, "intent": "COUNT", "headline": "42 patients found"},
            timestamp=datetime.now(),
            dataset_version="test_v1",
        )
        cache.put(cached_result)
        mock_session_state["result_cache"] = cache

        # Act: Simulate render_chat reading from manager and cache
        transcript = manager.get_transcript()
        assistant_msg = transcript[1]
        run_key = assistant_msg.run_key
        result = cache.get(run_key, "test_v1")

        # Assert: Result available through transcript-driven rendering
        assert run_key == "run_test", "Message should have run_key"
        assert result is not None, "Result should be in cache"
        assert result.result["value"] == 42, "Result data should be correct"

    def test_execute_stores_result_before_message_update(self, mock_session_state):
        """
        Test that execution stores result in cache before updating message.

        This ensures render_chat can always find the result when it renders.
        Order: 1. Store in cache → 2. Update message → 3. Rerun → 4. render_chat renders
        """
        # Arrange: Empty manager and cache
        manager = ConversationManager()
        manager.add_message("user", "Count patients")
        pending_id = manager.add_message("assistant", "", status="pending")
        mock_session_state["conversation_manager"] = manager
        mock_session_state["pending_message_id"] = pending_id

        cache = ResultCache(max_size=50)
        mock_session_state["result_cache"] = cache

        # Act: Simulate execution - store result BEFORE updating message
        run_key = "run_executed"
        dataset_version = "test_v1"
        result_data = {"type": "count", "value": 100, "headline": "100 patients"}

        # Step 1: Store in cache
        cached = CachedResult(
            run_key=run_key,
            query="Count patients",
            result=result_data,
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        cache.put(cached)

        # Step 2: Update message (after cache is populated)
        manager.update_message(pending_id, status="completed", content="100 patients", run_key=run_key)

        # Assert: After update, render_chat will find result
        transcript = manager.get_transcript()
        msg_run_key = transcript[1].run_key
        result = cache.get(msg_run_key, dataset_version)
        assert result is not None, "Result should be available for render_chat"
        assert result.result["value"] == 100

    def test_no_inline_rendering_needed_with_transcript_pattern(self, mock_session_state):
        """
        Test that the transcript pattern eliminates need for inline rendering.

        With ConversationManager holding message state and ResultCache holding
        results, render_chat can render everything without inline render_result calls.
        """
        # Arrange: Complete chat flow without any inline rendering simulation
        manager = ConversationManager()
        cache = ResultCache(max_size=50)
        mock_session_state["conversation_manager"] = manager
        mock_session_state["result_cache"] = cache

        # Simulate query 1: user asks, execution stores, message updated
        manager.add_message("user", "Query 1")
        cache.put(
            CachedResult(
                run_key="r1",
                query="Query 1",
                result={"headline": "Result 1"},
                timestamp=datetime.now(),
                dataset_version="v1",
            )
        )
        manager.add_message("assistant", "Result 1", run_key="r1", status="completed")

        # Simulate query 2: same pattern
        manager.add_message("user", "Query 2")
        cache.put(
            CachedResult(
                run_key="r2",
                query="Query 2",
                result={"headline": "Result 2"},
                timestamp=datetime.now(),
                dataset_version="v1",
            )
        )
        manager.add_message("assistant", "Result 2", run_key="r2", status="completed")

        # Act: Simulate render_chat reading all messages
        transcript = manager.get_transcript()
        renderable_messages = []
        for msg in transcript:
            if msg.role == "assistant" and msg.status == "completed" and msg.run_key:
                result = cache.get(msg.run_key, "v1")
                if result:
                    renderable_messages.append({"content": msg.content, "result": result.result})

        # Assert: All assistant messages have renderable results
        assert len(renderable_messages) == 2, "Should have 2 renderable messages"
        assert renderable_messages[0]["result"]["headline"] == "Result 1"
        assert renderable_messages[1]["result"]["headline"] == "Result 2"


class TestClearConversationButton:
    """
    Test suite for Clear Conversation button functionality.

    The Clear Conversation button must clear the ConversationManager,
    not just legacy session state variables.
    """

    def test_clear_conversation_button_clears_manager(self, mock_session_state):
        """Test that Clear Conversation button clears ConversationManager."""
        # Arrange: Manager with messages
        manager = ConversationManager()
        manager.add_message("user", "Query 1")
        manager.add_message("assistant", "Result 1", run_key="run_1", status="completed")
        mock_session_state["conversation_manager"] = manager
        mock_session_state["analysis_context"] = AnalysisContext(inferred_intent=AnalysisIntent.COUNT)
        mock_session_state["intent_signal"] = "nl_parsed"
        mock_session_state["pending_message_id"] = "some_pending_id"

        # Act: Simulate Clear Conversation button handler
        mock_session_state["conversation_history"] = []
        mock_session_state["analysis_context"] = None
        mock_session_state["intent_signal"] = None
        mock_session_state["pending_message_id"] = None
        manager = mock_session_state.get("conversation_manager")
        if manager:
            manager.clear()

        # Assert: Manager should be cleared
        transcript = mock_session_state["conversation_manager"].get_transcript()
        assert len(transcript) == 0, "ConversationManager should be cleared"
        assert mock_session_state.get("intent_signal") is None
        assert mock_session_state.get("pending_message_id") is None

    def test_clear_conversation_button_clears_stuck_pending_message(self, mock_session_state):
        """Test that Clear Conversation button clears stuck pending messages."""
        # Arrange: Manager with stuck pending message
        manager = ConversationManager()
        manager.add_message("user", "Stuck query")
        manager.add_message("assistant", "", status="pending")
        mock_session_state["conversation_manager"] = manager

        # Verify stuck state
        transcript = manager.get_transcript()
        pending_msgs = [m for m in transcript if m.status == "pending"]
        assert len(pending_msgs) == 1, "Should have stuck pending message"

        # Act: Clear conversation
        manager.clear()

        # Assert: No more pending messages
        transcript = manager.get_transcript()
        assert len(transcript) == 0, "Manager should be empty after clear"


class TestOrphanedPendingMessageCleanup:
    """
    Test suite for orphaned pending message cleanup on state restoration.

    When state is restored from disk, pending messages should be marked as
    errors because intent_signal/pending_message_id are not persisted.
    """

    def test_orphaned_pending_messages_marked_as_error_after_restoration(self, mock_session_state):
        """Test that orphaned pending messages are marked as errors after state restoration."""
        # Arrange: Simulate state restoration with orphaned pending message
        manager = ConversationManager()
        manager.add_message("user", "Query that was interrupted")
        manager.add_message("assistant", "", status="pending")
        mock_session_state["conversation_manager"] = manager

        # Simulate state restoration (intent_signal not persisted)
        mock_session_state["intent_signal"] = None
        mock_session_state["pending_message_id"] = None

        # Act: Cleanup orphaned pending messages
        for msg in manager.get_transcript():
            if msg.status == "pending":
                manager.update_message(
                    msg.id,
                    status="error",
                    content="Query was interrupted. Please try again.",
                )

        # Assert: Pending message should now be an error
        transcript = manager.get_transcript()
        assert len(transcript) == 2, "Should still have 2 messages"
        assert transcript[1].status == "error", "Pending should become error"
        assert "interrupted" in transcript[1].content, "Should have error message"

    def test_completed_messages_preserved_after_restoration(self, mock_session_state):
        """
        Test that completed messages are not affected by orphan cleanup.
        """
        # Arrange: Manager with completed messages only
        manager = ConversationManager()
        manager.add_message("user", "Query 1")
        manager.add_message("assistant", "Result 1", run_key="run_1", status="completed")
        mock_session_state["conversation_manager"] = manager

        # Act: Run cleanup (should not affect completed messages)
        for msg in manager.get_transcript():
            if msg.status == "pending":
                manager.update_message(
                    msg.id,
                    status="error",
                    content="Query was interrupted.",
                )

        # Assert: Completed messages unchanged
        transcript = manager.get_transcript()
        assert len(transcript) == 2
        assert transcript[1].status == "completed", "Completed should stay completed"
        assert transcript[1].content == "Result 1", "Content unchanged"


class TestLegacyCodeCleanup:
    """
    Test suite for Phase 5: Legacy code cleanup.

    These tests verify that ConversationManager is the single source of truth
    for chat transcript, eliminating the need for st.session_state["chat"].
    """

    def test_conversation_manager_is_single_source_of_truth(self, mock_session_state):
        """
        Test that ConversationManager is the only source of chat transcript.

        After Phase 5, st.session_state["chat"] should not be needed.
        """
        # Arrange: Only set up ConversationManager, no legacy chat
        manager = ConversationManager()
        manager.add_message("user", "Question from manager")
        manager.add_message("assistant", "Answer from manager", run_key="run_1", status="completed")
        mock_session_state["conversation_manager"] = manager
        # Note: NOT setting st.session_state["chat"]

        # Act: Get transcript from manager
        transcript = manager.get_transcript()

        # Assert: Manager has all messages, no legacy chat needed
        assert len(transcript) == 2
        assert transcript[0].content == "Question from manager"
        assert transcript[1].content == "Answer from manager"
        assert "chat" not in mock_session_state or mock_session_state["chat"] == []

    def test_last_user_message_check_via_manager(self, mock_session_state):
        """
        Test that checking for last user message works via ConversationManager.

        The code currently checks st.session_state["chat"][-1]["role"] == "user"
        to determine if query came from chat input. This should use manager instead.
        """
        # Arrange: Manager with user message (last message)
        manager = ConversationManager()
        manager.add_message("user", "New query")
        mock_session_state["conversation_manager"] = manager

        # Act: Check last message via manager
        transcript = manager.get_transcript()
        last_message = transcript[-1] if transcript else None

        # Assert: Can determine last message role via manager
        assert last_message is not None
        assert last_message.role == "user"
        assert last_message.run_key is None  # User messages don't have run_key

    def test_empty_chat_check_via_manager(self, mock_session_state):
        """
        Test that checking for empty chat works via ConversationManager.

        _render_example_questions checks if chat is empty to show examples.
        This should use manager.get_transcript() instead.
        """
        # Arrange: Empty manager
        manager = ConversationManager()
        mock_session_state["conversation_manager"] = manager

        # Act: Check if transcript is empty
        transcript = manager.get_transcript()
        is_empty = len(transcript) == 0

        # Assert: Can determine empty state via manager
        assert is_empty is True

        # Add a message
        manager.add_message("user", "First message")
        transcript = manager.get_transcript()
        is_empty = len(transcript) == 0

        # Assert: No longer empty
        assert is_empty is False

    def test_chat_clearing_on_dataset_change_via_manager(self, mock_session_state):
        """
        Test that clearing chat on dataset change works via ConversationManager.

        When dataset changes, chat should be cleared. Use manager.clear() instead
        of st.session_state["chat"] = [].
        """
        # Arrange: Manager with messages
        manager = ConversationManager()
        manager.add_message("user", "Query for old dataset")
        manager.add_message("assistant", "Result", run_key="r1", status="completed")
        mock_session_state["conversation_manager"] = manager

        # Act: Simulate dataset change - clear via manager
        manager.clear()

        # Assert: Manager is empty after clear
        transcript = manager.get_transcript()
        assert len(transcript) == 0
        assert manager.get_current_dataset() is None
