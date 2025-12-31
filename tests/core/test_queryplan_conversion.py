"""
Tests for QueryIntent to QueryPlan conversion and execution gating.

Tests verify:
- QueryIntent converts to QueryPlan with deterministic run_key
- QueryPlan is stored in AnalysisContext
- Execution gating uses QueryPlan confidence and run_key
"""

import pytest

from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan


@pytest.fixture
def mock_semantic_layer():
    """Create mock semantic layer for testing."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.get_column_alias_index.return_value = {
        "mortality": "mortality",
        "treatment": "treatment_arm",
        "age": "age",
        "statin_prescribed": "Statin Prescribed? 1: Yes 2: No",
    }
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")
    return mock


class TestQueryPlanConversion:
    """Test QueryIntent to QueryPlan conversion."""

    def test_intent_to_plan_creates_queryplan_without_run_key(self, mock_semantic_layer):
        """Test that _intent_to_plan creates QueryPlan with run_key=None (Phase 1.1.5)."""
        # Arrange: Create QueryIntent
        engine = NLQueryEngine(mock_semantic_layer)
        intent = QueryIntent(
            intent_type="COUNT",
            confidence=0.9,
            filters=[FilterSpec(column="statin_prescribed", operator="==", value=1, exclude_nulls=True)],
        )

        # Act: Convert to QueryPlan
        plan1 = engine._intent_to_plan(intent, "dataset_v1")
        plan2 = engine._intent_to_plan(intent, "dataset_v1")

        # Assert: Phase 1.1.5 - nl_query_engine should NOT set run_key (semantic layer will generate it)
        assert isinstance(plan1, QueryPlan)
        assert plan1.run_key is None, (
            "Phase 1.1.5: nl_query_engine should not set run_key - semantic layer will generate it"
        )
        assert plan2.run_key is None, (
            "Phase 1.1.5: nl_query_engine should not set run_key - semantic layer will generate it"
        )
        assert plan1.intent == "COUNT"
        assert plan1.confidence == 0.9
        assert len(plan1.filters) == 1

    def test_intent_to_plan_does_not_set_run_key(self, mock_semantic_layer):
        """Test that _intent_to_plan does not set run_key (Phase 1.1.5)."""
        # Arrange: Create QueryIntent
        engine = NLQueryEngine(mock_semantic_layer)
        intent = QueryIntent(intent_type="DESCRIBE", primary_variable="age", confidence=0.9)

        # Act: Convert to QueryPlan with different dataset versions
        plan1 = engine._intent_to_plan(intent, "dataset_v1")
        plan2 = engine._intent_to_plan(intent, "dataset_v2")

        # Assert: Phase 1.1.5 - run_key should be None (semantic layer will generate it deterministically)
        assert plan1.run_key is None, "Phase 1.1.5: nl_query_engine should not set run_key"
        assert plan2.run_key is None, "Phase 1.1.5: nl_query_engine should not set run_key"
        assert plan1.intent == plan2.intent
        assert plan1.metric == plan2.metric

    def test_intent_to_plan_preserves_filters_without_run_key(self, mock_semantic_layer):
        """Test that filters are preserved but run_key is not set (Phase 1.1.5)."""
        # Arrange: Create QueryIntent with filters
        engine = NLQueryEngine(mock_semantic_layer)
        intent1 = QueryIntent(
            intent_type="COUNT",
            filters=[FilterSpec(column="age", operator=">", value=50, exclude_nulls=True)],
        )
        intent2 = QueryIntent(
            intent_type="COUNT",
            filters=[FilterSpec(column="age", operator=">", value=60, exclude_nulls=True)],
        )

        # Act: Convert to QueryPlan
        plan1 = engine._intent_to_plan(intent1, "dataset_v1")
        plan2 = engine._intent_to_plan(intent2, "dataset_v1")

        # Assert: Phase 1.1.5 - run_key should be None (semantic layer will generate it)
        assert plan1.run_key is None, "Phase 1.1.5: nl_query_engine should not set run_key"
        assert plan2.run_key is None, "Phase 1.1.5: nl_query_engine should not set run_key"
        # Filters should be preserved
        assert len(plan1.filters) == 1
        assert len(plan2.filters) == 1
        assert plan1.filters[0].value == 50
        assert plan2.filters[0].value == 60

    def test_intent_to_plan_preserves_all_intent_fields(self, mock_semantic_layer):
        """Test that all QueryIntent fields are preserved in QueryPlan."""
        # Arrange: Create QueryIntent with all fields
        engine = NLQueryEngine(mock_semantic_layer)
        intent = QueryIntent(
            intent_type="COMPARE_GROUPS",
            primary_variable="mortality",
            grouping_variable="treatment_arm",
            confidence=0.85,
            filters=[FilterSpec(column="age", operator=">=", value=18, exclude_nulls=True)],
        )

        # Act: Convert to QueryPlan
        plan = engine._intent_to_plan(intent, "dataset_v1")

        # Assert: All fields preserved
        assert plan.intent == "COMPARE_GROUPS"
        assert plan.metric == "mortality"
        assert plan.group_by == "treatment_arm"
        assert plan.confidence == 0.85
        assert len(plan.filters) == 1
        assert plan.filters[0].column == "age"
        assert plan.filters[0].operator == ">="
        assert plan.filters[0].value == 18


class TestAnalysisContextQueryPlan:
    """Test QueryPlan integration with AnalysisContext."""

    def test_analysiscontext_has_query_plan_field(self):
        """Test that AnalysisContext has query_plan field."""
        # Arrange
        from clinical_analytics.ui.components.question_engine import AnalysisContext

        # Act
        context = AnalysisContext()

        # Assert: query_plan field exists (defaults to None)
        assert hasattr(context, "query_plan")
        assert context.query_plan is None

    def test_queryplan_can_be_assigned_to_analysiscontext(self, mock_semantic_layer):
        """Test that QueryPlan can be assigned to AnalysisContext."""
        # Arrange
        from clinical_analytics.ui.components.question_engine import AnalysisContext

        engine = NLQueryEngine(mock_semantic_layer)
        intent = QueryIntent(intent_type="COUNT", confidence=0.9)
        query_plan = engine._intent_to_plan(intent, "dataset_v1")

        context = AnalysisContext()

        # Act
        context.query_plan = query_plan

        # Assert
        assert context.query_plan is not None
        assert context.query_plan.intent == "COUNT"
        # Phase 1.1.5: run_key should be None (semantic layer will generate it)
        assert context.query_plan.run_key is None, "Phase 1.1.5: nl_query_engine should not set run_key"
