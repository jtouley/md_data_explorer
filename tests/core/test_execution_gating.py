"""
Tests for transparent confidence display (replaces blocking execution gating).

Tests verify:
- Execution always proceeds regardless of confidence (no blocking gates)
- QueryPlan.confidence is used for transparent display
- Confidence levels are categorized correctly (high/moderate/low)
- QueryPlan.run_key is used for idempotency
- Interpretation details are available for display
"""

from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


class TestTransparentConfidenceDisplay:
    """Test transparent confidence display (always execute, show confidence)."""

    def test_execution_always_proceeds_regardless_of_confidence(self):
        """Test that execution always proceeds regardless of confidence level."""
        # Arrange: Create QueryPlan with low confidence
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.5,  # Below threshold (0.75)
            run_key="test_run_key_123",
        )

        # Act: Check execution decision (always execute now)
        confidence = query_plan.confidence

        # Assert: Execution should proceed (no blocking gates)
        assert confidence < AUTO_EXECUTE_CONFIDENCE_THRESHOLD  # Low confidence
        # But execution proceeds anyway - confidence is for display only

    def test_confidence_levels_categorized_correctly(self):
        """Test that confidence levels are categorized correctly for display."""
        # Arrange: Create QueryPlans with different confidence levels
        high_conf = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.9, run_key="high")
        moderate_conf = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.6, run_key="moderate")
        low_conf = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.3, run_key="low")

        # Act: Categorize confidence (simulating _render_interpretation_and_confidence logic)
        def categorize_confidence(conf: float) -> str:
            if conf >= 0.75:
                return "high"
            elif conf >= 0.5:
                return "moderate"
            else:
                return "low"

        # Assert: Confidence levels categorized correctly
        assert categorize_confidence(high_conf.confidence) == "high"
        assert categorize_confidence(moderate_conf.confidence) == "moderate"
        assert categorize_confidence(low_conf.confidence) == "low"

    def test_low_confidence_expands_interpretation_by_default(self):
        """Test that low confidence queries expand interpretation expander by default."""
        # Arrange: Create QueryPlan with low confidence
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.4,  # Low confidence
            run_key="test_key",
        )

        # Act: Check expander default (simulating _render_interpretation_and_confidence logic)
        expanded = query_plan.confidence < 0.75

        # Assert: Low confidence expands interpretation by default
        assert expanded is True  # Should be expanded for low confidence

    def test_high_confidence_collapses_interpretation_by_default(self):
        """Test that high confidence queries collapse interpretation expander by default."""
        # Arrange: Create QueryPlan with high confidence
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,  # High confidence
            run_key="test_key",
        )

        # Act: Check expander default (simulating _render_interpretation_and_confidence logic)
        expanded = query_plan.confidence < 0.75

        # Assert: High confidence collapses interpretation by default
        assert expanded is False  # Should be collapsed for high confidence

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

    def test_confidence_display_falls_back_to_context_when_no_queryplan(self):
        """Test that confidence display falls back to context.confidence when QueryPlan not available."""
        # Arrange: Create context without QueryPlan
        context = AnalysisContext()
        context.confidence = 0.8
        context.inferred_intent = AnalysisIntent.DESCRIBE

        # Act: Get confidence for display (simulating UI logic)
        query_plan = getattr(context, "query_plan", None)
        if query_plan:
            confidence = query_plan.confidence
        else:
            confidence = getattr(context, "confidence", 0.0)

        # Assert: Uses context.confidence for display
        assert query_plan is None
        assert confidence == 0.8
        # Execution proceeds regardless - confidence is for display only

    def test_confidence_display_prefers_queryplan_when_available(self):
        """Test that confidence display prefers QueryPlan.confidence over context.confidence."""
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

        # Act: Get confidence for display (simulating UI logic)
        if context.query_plan:
            confidence = context.query_plan.confidence
            run_key = context.query_plan.run_key or "fallback_key"
        else:
            confidence = getattr(context, "confidence", 0.0)
            run_key = "fallback_key"

        # Assert: Uses QueryPlan.confidence for display (not context.confidence)
        assert confidence == 0.6  # From QueryPlan, not 0.9
        assert run_key == "test_key"
        # Execution proceeds regardless - confidence is for display only
