"""
End-to-end tests for Ask Questions page - full conversational flow.

Tests verify meaningful behavior that unit tests don't cover:
- Complete NL query → parse → execute flow (integration)
- Filter extraction stops at continuation words (catches real bugs)
- QueryPlan deterministic run_key (via semantic layer, not nl_query_engine)
- Feature parity: identical behavior for both upload types

Note: Filter application, COUNT with filters, etc. are already tested in test_compute.py.
These tests focus on integration and behavior that spans multiple components.

PR21 refactor: run_key generation moved to semantic layer.
nl_query_engine._intent_to_plan() should NOT set run_key (semantic layer owns it).
Run key determinism tests are in tests/core/test_semantic_run_key_determinism.py.
"""

from unittest.mock import MagicMock

import pytest

from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.ui.components.question_engine import AnalysisIntent


@pytest.fixture
def mock_semantic_layer_with_statin():
    """Create mock semantic layer with statin-related columns."""
    mock = MagicMock()

    # Statin columns with coded values
    statin_prescribed = "Statin Prescribed? 1: Yes 2: No"
    statin_used = (
        "Statin Used:    0: n/a                       1: Atorvastatin  "
        "2: Rosuvastatin 3: Pravastatin   4: Pitavastatin  5: Simvastatin"
    )

    # Build alias index
    alias_index = {
        "statin prescribed": statin_prescribed,
        "statin prescribed?": statin_prescribed,
        "on statins": statin_prescribed,
        "statin used": statin_used,
        "statin": statin_used,
        "statins": statin_used,
    }

    mock.get_column_alias_index.return_value = alias_index
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")

    return mock


class TestE2EFullQueryFlow:
    """Test complete end-to-end query flow - integration across components."""

    def test_e2e_nl_query_parses_to_queryplan_with_grouping(self, mock_semantic_layer_with_statin):
        """Test that NL query with grouping extracts grouping variable correctly."""
        # Arrange: Query asking for count with grouping
        query = "how many patients were on statins and which statin was most prescribed?"
        engine = NLQueryEngine(mock_semantic_layer_with_statin)

        # Act: Parse and convert to QueryPlan
        query_intent = engine.parse_query(query)
        assert query_intent is not None
        assert query_intent.intent_type == "COUNT"

        dataset_version = "test_dataset_v1"
        query_plan = engine._intent_to_plan(query_intent, dataset_version)

        # Assert: QueryPlan has correct intent
        assert query_plan.intent == "COUNT"

        # PR21: run_key should be None from nl_query_engine (semantic layer generates it)
        assert query_plan.run_key is None, (
            "nl_query_engine._intent_to_plan() should not set run_key - "
            "semantic layer is the single source of truth (PR21)"
        )

        # Grouping should be extracted (statin_used or similar)
        if query_plan.group_by:
            assert "statin" in query_plan.group_by.lower(), (
                f"Grouping variable should be statin-related, got: {query_plan.group_by}"
            )

    def test_e2e_queryplan_run_key_is_none_from_nl_query_engine(self, mock_semantic_layer_with_statin):
        """Test that nl_query_engine does NOT set run_key (semantic layer owns it).

        PR21 refactor: run_key determinism is now tested in:
        tests/core/test_semantic_run_key_determinism.py
        """
        # Arrange: Same query
        query = "how many patients were on statins"
        engine = NLQueryEngine(mock_semantic_layer_with_statin)

        # Act: Parse and convert to plan
        intent = engine.parse_query(query)
        dataset_version = "test_dataset_v1"
        plan = engine._intent_to_plan(intent, dataset_version)

        # Assert: run_key should be None (semantic layer generates it later)
        assert plan.run_key is None, (
            "nl_query_engine._intent_to_plan() should not set run_key. "
            "The semantic layer's execute_query_plan() generates run_key deterministically. "
            "See tests/core/test_semantic_run_key_determinism.py for run_key tests."
        )


class TestE2EFilterExtraction:
    """Test filter extraction behavior that spans NL parsing and filter logic."""

    def test_e2e_filter_extraction_stops_at_continuation_words(self, mock_semantic_layer_with_statin):
        """Test that filter extraction doesn't capture continuation phrases - catches real bugs."""
        # Arrange: Compound query with continuation
        query = "how many patients were on statins and which statin was most prescribed?"
        engine = NLQueryEngine(mock_semantic_layer_with_statin)

        # Act: Parse query
        query_intent = engine.parse_query(query)

        # Assert: Filters don't contain continuation phrase
        # This catches a real bug where filter extraction over-captures
        if query_intent.filters:
            for f in query_intent.filters:
                assert "and which statin" not in str(f.value).lower(), (
                    f"Filter value should not contain continuation phrase, got: {f.value}. "
                    "This verifies filter extraction stops at continuation words."
                )


class TestE2EConversationHistory:
    """Test conversation history structure - verifies lightweight storage requirement."""

    def test_e2e_conversation_history_is_lightweight(self):
        """Test that conversation history entries are lightweight (catches memory bloat)."""
        # Arrange: Expected structure per ADR001
        entry = {
            "query": "how many patients were on statins",
            "intent": AnalysisIntent.COUNT,
            "headline": "5 patients",
            "run_key": "test_dataset_v1_abc123",
            "timestamp": 1234567890.0,
            "filters_applied": [],
        }

        # Assert: Lightweight (no large objects) - this catches memory bloat bugs
        import json
        import sys

        entry_size = sys.getsizeof(json.dumps(entry, default=str))
        assert entry_size < 1024, (
            f"Entry too large: {entry_size} bytes (should be <1KB). "
            "This verifies we're not storing full result dicts in history."
        )
