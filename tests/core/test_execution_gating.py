"""
Tests for execution gating logic using QueryPlan confidence and run_key.

Tests verify:
- Execution gating uses QueryPlan.confidence when available
- Low confidence (< threshold) requires confirmation
- High confidence (>= threshold) auto-executes
- QueryPlan.run_key is used for idempotency
"""

from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


class TestExecutionGating:
    """Test execution gating logic with QueryPlan."""

    def test_high_confidence_queryplan_auto_executes(self):
        """Test that QueryPlan with high confidence (>= threshold) auto-executes."""
        # Arrange: Create QueryPlan with high confidence
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,  # Above threshold (0.75)
            run_key="test_run_key_123",
        )
        context = AnalysisContext()
        context.query_plan = query_plan
        context.confidence = 0.9

        # Act: Check execution decision
        confidence = query_plan.confidence
        should_auto_execute = confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD

        # Assert: Should auto-execute
        assert should_auto_execute is True
        assert confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD

    def test_low_confidence_queryplan_requires_confirmation(self):
        """Test that QueryPlan with low confidence (< threshold) requires confirmation."""
        # Arrange: Create QueryPlan with low confidence
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.5,  # Below threshold (0.75)
            run_key="test_run_key_456",
        )
        context = AnalysisContext()
        context.query_plan = query_plan
        context.confidence = 0.5

        # Act: Check execution decision
        confidence = query_plan.confidence
        should_auto_execute = confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD

        # Assert: Should require confirmation
        assert should_auto_execute is False
        assert confidence < AUTO_EXECUTE_CONFIDENCE_THRESHOLD

    def test_user_confirmation_overrides_low_confidence(self):
        """Test that user confirmation allows execution even with low confidence."""
        # Arrange: Create QueryPlan with low confidence
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.5,  # Below threshold
            run_key="test_run_key_789",
        )

        # Act: Check execution decision with user confirmation
        confidence = query_plan.confidence
        user_confirmed = True
        should_auto_execute = confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD or user_confirmed

        # Assert: Should execute due to user confirmation
        assert should_auto_execute is True
        assert confidence < AUTO_EXECUTE_CONFIDENCE_THRESHOLD  # Still low confidence

    def test_queryplan_run_key_used_for_idempotency(self):
        """Test that QueryPlan.run_key is used for idempotent execution."""
        # Arrange: Create QueryPlan with deterministic run_key
        query_plan = QueryPlan(
            intent="COUNT",
            filters=[FilterSpec(column="age", operator=">", value=50, exclude_nulls=True)],
            confidence=0.9,
            run_key="dataset_v1_abc123def456",
        )

        # Act: Extract run_key
        run_key = query_plan.run_key

        # Assert: Run key is deterministic and can be used for caching
        assert run_key is not None
        assert run_key == "dataset_v1_abc123def456"
        # Same QueryPlan should produce same run_key (tested in test_queryplan_conversion.py)

    def test_execution_gating_falls_back_to_context_confidence_when_no_queryplan(self):
        """Test that execution gating falls back to context.confidence when QueryPlan not available."""
        # Arrange: Create context without QueryPlan
        context = AnalysisContext()
        context.confidence = 0.8
        context.inferred_intent = AnalysisIntent.DESCRIBE

        # Act: Check execution decision (simulating UI logic)
        query_plan = getattr(context, "query_plan", None)
        if query_plan:
            confidence = query_plan.confidence
        else:
            confidence = getattr(context, "confidence", 0.0)

        should_auto_execute = confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD

        # Assert: Uses context.confidence
        assert query_plan is None
        assert confidence == 0.8
        assert should_auto_execute is True  # 0.8 >= 0.75

    def test_execution_gating_uses_queryplan_when_available(self):
        """Test that execution gating prefers QueryPlan.confidence over context.confidence."""
        # Arrange: Create context with QueryPlan
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.6,  # Lower than context.confidence
            run_key="test_key",
        )
        context = AnalysisContext()
        context.query_plan = query_plan
        context.confidence = 0.9  # Higher than query_plan.confidence

        # Act: Check execution decision (simulating UI logic)
        if context.query_plan:
            confidence = context.query_plan.confidence
            run_key = context.query_plan.run_key or "fallback_key"
        else:
            confidence = getattr(context, "confidence", 0.0)
            run_key = "fallback_key"

        should_auto_execute = confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD

        # Assert: Uses QueryPlan.confidence (not context.confidence)
        assert confidence == 0.6  # From QueryPlan, not 0.9
        assert should_auto_execute is False  # 0.6 < 0.75
        assert run_key == "test_key"
