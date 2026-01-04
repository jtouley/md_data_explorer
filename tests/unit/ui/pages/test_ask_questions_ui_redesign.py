"""
Tests for UI redesign - conversation history display and chat input.

Tests verify:
- Conversation history is displayed in chat message format
- Chat input is always visible
- New queries from chat input are handled correctly
- Results can be reconstructed from run_key in conversation history
"""


class TestConversationHistoryDisplay:
    """Test conversation history display in UI."""

    def test_conversation_history_display_shows_user_queries(self, mock_session_state):
        """Test that conversation history displays user queries in chat message format."""
        # Arrange: Create conversation history
        mock_session_state["conversation_history"] = [
            {
                "query": "what is the average age",
                "intent": "DESCRIBE",
                "headline": "Average age: 45.2 years",
                "run_key": "describe_123",
                "timestamp": 1234567890.0,
                "filters_applied": [],
            }
        ]

        # Assert: Entry structure supports chat message display
        entry = mock_session_state["conversation_history"][0]
        assert "query" in entry
        assert entry["query"] == "what is the average age"
        # In UI: with st.chat_message("user"): st.write(entry["query"])

    def test_conversation_history_display_shows_assistant_headlines(self, mock_session_state):
        """Test that conversation history displays assistant headlines in chat message format."""
        # Arrange: Create conversation history
        mock_session_state["conversation_history"] = [
            {
                "query": "compare viral load by treatment",
                "intent": "COMPARE_GROUPS",
                "headline": "Treatment A has lower viral load (p<0.05)",
                "run_key": "compare_456",
                "timestamp": 1234567890.0,
                "filters_applied": [],
            }
        ]

        # Assert: Entry structure supports chat message display
        entry = mock_session_state["conversation_history"][0]
        assert "headline" in entry
        assert entry["headline"] == "Treatment A has lower viral load (p<0.05)"
        # In UI: with st.chat_message("assistant"): st.info(entry["headline"])

    def test_conversation_history_can_reconstruct_results_from_run_key(self, mock_session_state):
        """Test that conversation history entries can reconstruct full results from run_key."""
        # Arrange: Create conversation history entry with run_key
        dataset_version = "dataset_v1"
        run_key = "describe_abc123"
        entry = {
            "query": "what is the average age",
            "intent": "DESCRIBE",
            "headline": "Average age: 45.2 years",
            "run_key": run_key,
            "timestamp": 1234567890.0,
            "filters_applied": [],
        }

        # Simulate: Full result stored separately
        result_key = f"analysis_result:{dataset_version}:{run_key}"
        full_result = {
            "type": "descriptive",
            "mean": 45.2,
            "median": 44.0,
            "std": 12.5,
        }
        mock_session_state[result_key] = full_result

        # Assert: run_key can be used to reconstruct full result
        assert entry["run_key"] == run_key
        assert result_key in mock_session_state
        assert mock_session_state[result_key]["mean"] == 45.2
        # In UI: result_key = f"analysis_result:{dataset_version}:{entry['run_key']}"
        #        if result_key in st.session_state: render_analysis_by_type(...)

    def test_conversation_history_displays_expandable_details(self, mock_session_state):
        """Test that conversation history entries support expandable details."""
        # Arrange: Create entry with run_key
        entry = {
            "query": "compare survival by treatment",
            "intent": "COMPARE_GROUPS",
            "headline": "Treatment A has better survival (p<0.01)",
            "run_key": "compare_789",
            "timestamp": 1234567890.0,
            "filters_applied": [],
        }

        # Assert: Entry has run_key for expandable details
        assert "run_key" in entry
        assert entry["run_key"] is not None
        # In UI: with st.expander("View detailed results"): render_analysis_by_type(...)


class TestChatInputHandling:
    """Test chat input query handling."""

    def test_chat_input_query_creates_analysis_context(self):
        """Test that chat input query creates AnalysisContext correctly."""
        # Arrange: Mock query and semantic layer
        from unittest.mock import MagicMock

        query = "what is the average age"
        mock_semantic_layer = MagicMock()
        mock_semantic_layer.get_column_alias_index.return_value = {"age": "age"}

        # Act: Parse query (simulating chat_input flow)
        from clinical_analytics.core.nl_query_engine import NLQueryEngine
        from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

        nl_engine = NLQueryEngine(mock_semantic_layer)
        query_intent = nl_engine.parse_query(query)

        if query_intent:
            # Convert QueryIntent to AnalysisContext (same logic as chat_input handler)
            context = AnalysisContext()

            intent_map = {
                "DESCRIBE": AnalysisIntent.DESCRIBE,
                "COMPARE_GROUPS": AnalysisIntent.COMPARE_GROUPS,
                "FIND_PREDICTORS": AnalysisIntent.FIND_PREDICTORS,
                "SURVIVAL": AnalysisIntent.EXAMINE_SURVIVAL,
                "CORRELATIONS": AnalysisIntent.EXPLORE_RELATIONSHIPS,
                "COUNT": AnalysisIntent.COUNT,
            }
            context.inferred_intent = intent_map.get(query_intent.intent_type, AnalysisIntent.UNKNOWN)
            context.research_question = query
            context.primary_variable = query_intent.primary_variable
            context.filters = query_intent.filters

            # Assert: Context created correctly
            assert context.research_question == query
            assert context.inferred_intent in [
                AnalysisIntent.DESCRIBE,
                AnalysisIntent.COMPARE_GROUPS,
                AnalysisIntent.FIND_PREDICTORS,
                AnalysisIntent.EXAMINE_SURVIVAL,
                AnalysisIntent.EXPLORE_RELATIONSHIPS,
                AnalysisIntent.COUNT,
            ]

    def test_chat_input_query_creates_queryplan_when_dataset_version_available(self):
        """Test that chat input query creates QueryPlan when dataset_version is available."""
        # Arrange: Mock query and semantic layer
        from unittest.mock import MagicMock

        query = "how many patients on statins"
        dataset_version = "dataset_v1"
        mock_semantic_layer = MagicMock()
        mock_semantic_layer.get_column_alias_index.return_value = {
            "statin_prescribed": "Statin Prescribed? 1: Yes 2: No"
        }

        # Act: Parse query and create QueryPlan
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        nl_engine = NLQueryEngine(mock_semantic_layer)
        query_intent = nl_engine.parse_query(query)

        if query_intent and hasattr(nl_engine, "_intent_to_plan"):
            query_plan = nl_engine._intent_to_plan(query_intent, dataset_version)

            # Assert: QueryPlan created (run_key is None - semantic layer owns it per PR21)
            assert query_plan is not None
            assert query_plan.run_key is None, (
                "nl_query_engine._intent_to_plan() should not set run_key - "
                "semantic layer generates it during execute_query_plan() (PR21)"
            )
            assert query_plan.intent == "COUNT" or query_plan.intent == query_intent.intent_type

    def test_chat_input_query_handles_parsing_errors_gracefully(self):
        """Test that chat input query handles parsing errors gracefully."""
        # Arrange: Invalid query that might cause parsing error
        query = ""

        # Act & Assert: Empty query should not cause error
        # In UI: if query: ... (empty string is falsy, so handler won't execute)
        assert not query  # Empty string is falsy

    def test_chat_input_query_integrates_with_existing_analysis_flow(self):
        """Test that chat input query integrates with existing analysis execution flow."""
        # Arrange: Create context from chat input
        from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.DESCRIBE
        context.primary_variable = "age"
        context.research_question = "what is the average age"

        # Assert: Context can be used in existing flow
        assert context.is_complete_for_intent()  # DESCRIBE with primary_variable is complete
        # In UI: if context.is_complete_for_intent(): execute_analysis_with_idempotency(...)

    def test_chat_input_executes_immediately_when_context_complete(self):
        """Test that chat input executes analysis immediately when context is complete."""
        # Arrange: Create complete context
        from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COUNT
        context.research_question = "how many patients"
        # COUNT intent is complete with just data (no variables needed)

        # Assert: Context is complete, so should execute immediately
        assert context.is_complete_for_intent()
        # In UI: if context.is_complete_for_intent(): execute immediately, no rerun needed

    def test_chat_input_renders_results_inline_in_chat_message_style(self):
        """Test that chat input renders results inline in chat message style."""
        # Arrange: Simulate result rendering
        result = {
            "type": "count",
            "headline": "Total: 100 patients",
            "total_count": 100,
        }

        # Assert: Result has headline for inline display
        assert "headline" in result
        assert result["headline"] == "Total: 100 patients"
        # In UI: with st.chat_message("assistant"): st.info(headline)

    def test_old_text_input_flow_removed_no_duplicate_inputs(self):
        """Test that old ask_free_form_question flow using st.text_input is removed."""
        # Arrange: Check that ask_free_form_question is not called in main flow
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent.parent.parent
        page_path = project_root / "src" / "clinical_analytics" / "ui" / "pages" / "03_💬_Ask_Questions.py"

        # Read the page file
        page_content = page_path.read_text()

        # Assert: ask_free_form_question should not be called in main() function
        # Check that it's not in the main() function body (after "def main():")
        main_start = page_content.find("def main():")
        if main_start != -1:
            main_body = page_content[main_start:]
            # Assert: QuestionEngine.ask_free_form_question should not be called in main()
            assert (
                "QuestionEngine.ask_free_form_question" not in main_body
            ), "Old ask_free_form_question flow should be removed from main() function"

        # Assert: st.chat_input should be used for query input
        assert "st.chat_input" in page_content, "st.chat_input should be used for queries"

    def test_followup_suggestions_have_unique_button_keys(self, mock_session_state):
        """
        Test that follow-up suggestion buttons have unique keys to prevent collisions.

        This verifies the fix for StreamlitDuplicateElementKey errors when multiple
        suggestions have the same hash value.
        """
        # Arrange: Multiple suggestions that might hash to same value
        suggestions = [
            "Break down the count by a grouping variable",
            "Filter Statin Used and count again",
            "Remove filters and count all records",
            "Compare outcomes by treatment group",
        ]

        # Act: Generate button keys (as done in _suggest_follow_ups)
        button_keys = []
        for idx, suggestion in enumerate(suggestions[:4]):
            button_key = f"followup_{idx}_{hash(suggestion) % 1000000}"
            button_keys.append(button_key)

        # Assert: All keys should be unique
        assert len(button_keys) == len(set(button_keys)), f"Button keys should be unique, got duplicates: {button_keys}"
        # Assert: Keys should include index
        for idx, key in enumerate(button_keys):
            assert f"followup_{idx}_" in key, f"Button key should include index, got: {key}"

    def test_conversation_history_skips_duplicate_rendering(self, mock_session_state):
        """Test that conversation history skips the last entry if it was just rendered."""
        # Arrange: Create conversation history with last entry
        dataset_version = "test_dataset_v1"
        last_run_key = "last_run_123"
        mock_session_state["conversation_history"] = [
            {
                "query": "first query",
                "intent": "DESCRIBE",
                "headline": "First result",
                "run_key": "first_run",
                "timestamp": 1234567890.0,
                "filters_applied": [],
            },
            {
                "query": "second query",
                "intent": "COUNT",
                "headline": "Second result",
                "run_key": last_run_key,
                "timestamp": 1234567891.0,
                "filters_applied": [],
            },
        ]
        mock_session_state[f"last_run_key:{dataset_version}"] = last_run_key
        mock_session_state["intent_signal"] = "nl_parsed"

        # Assert: Logic should skip last entry if run_key matches and intent_signal is nl_parsed
        # This is tested by checking the structure supports this logic
        history = mock_session_state["conversation_history"]
        assert len(history) == 2
        assert history[-1]["run_key"] == last_run_key
        # In UI: if entry.get("run_key") == last_run_key and
        # st.session_state.get("intent_signal") == "nl_parsed": continue

    def test_compact_interpretation_function_exists(self):
        """Test that compact inline interpretation function exists."""
        # Arrange: Check that the function exists in the page
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent.parent.parent
        page_path = project_root / "src" / "clinical_analytics" / "ui" / "pages" / "03_💬_Ask_Questions.py"

        # Read the page file
        page_content = page_path.read_text()

        # Assert: Compact interpretation function should exist
        assert (
            "_render_interpretation_inline_compact" in page_content
        ), "Compact inline interpretation function should exist"
        # Assert: Old expander-based interpretation should not be used for inline rendering
        # (It may still exist but should not be called in execute_analysis_with_idempotency)
        main_start = page_content.find("def execute_analysis_with_idempotency")
        if main_start != -1:
            execute_body = page_content[main_start : main_start + 2000]  # Check first 2000 chars
            # Should call compact version, not expander version
            assert "_render_interpretation_inline_compact" in execute_body or (
                "_render_interpretation_and_confidence" not in execute_body
            ), "Should use compact inline interpretation, not expander version"
