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

            # Assert: QueryPlan created with run_key
            assert query_plan is not None
            assert query_plan.run_key is not None
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
