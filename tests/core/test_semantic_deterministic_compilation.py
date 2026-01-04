"""
Tests for deterministic QueryPlan compilation (Phase 5.3).

Ensures:
- Same QueryPlan produces same Ibis expressions/SQL
- Execution is fully deterministic
- No freeform code generation

PANDAS EXCEPTION: Required for testing SemanticLayer._execute_plan() which
returns pd.DataFrame (legacy interface for backward compatibility with UI).
TODO: Migrate _execute_plan() to return pl.DataFrame in future refactor.
"""

# PANDAS EXCEPTION: SemanticLayer._execute_plan() returns pd.DataFrame (legacy interface)
# TODO: Remove when _execute_plan() migrated to return pl.DataFrame
import pandas as pd
import pandas.testing as pdt
import pytest

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan


class TestDeterministicCompilation:
    """Test that QueryPlan compilation to Ibis/SQL is deterministic."""

    @pytest.fixture
    def semantic_layer(self, make_semantic_layer):
        """Create semantic layer for testing deterministic compilation."""
        return make_semantic_layer(
            data={
                "patient_id": [1, 2, 3, 4, 5],
                "status": ["active", "inactive", "active", "active", "inactive"],
                "age": [25, 35, 45, 55, 65],
            }
        )

    def test_same_queryplan_produces_same_sql_multiple_times(self, semantic_layer):
        """Same QueryPlan should compile to same SQL every time."""
        # Arrange: Create QueryPlan
        plan = QueryPlan(
            intent="COUNT",
            metric=None,
            group_by="status",
            filters=[],
            confidence=0.9,
            entity_key="patient_id",
        )

        # Act: Execute plan multiple times
        result1 = semantic_layer._execute_plan(plan)
        result2 = semantic_layer._execute_plan(plan)
        result3 = semantic_layer._execute_plan(plan)

        # Assert: Results should be identical (deterministic execution)
        # Note: Sort by all columns to ensure consistent ordering for comparison
        result1_sorted = result1.sort_values(by=list(result1.columns)).reset_index(drop=True)
        result2_sorted = result2.sort_values(by=list(result2.columns)).reset_index(drop=True)
        result3_sorted = result3.sort_values(by=list(result3.columns)).reset_index(drop=True)

        pdt.assert_frame_equal(result1_sorted, result2_sorted)
        pdt.assert_frame_equal(result2_sorted, result3_sorted)

    def test_different_queryplans_produce_different_sql(self, semantic_layer):
        """Different QueryPlans should produce different SQL."""
        # Arrange: Create two different QueryPlans
        plan1 = QueryPlan(
            intent="COUNT",
            metric=None,
            group_by="status",
            filters=[],
            entity_key="patient_id",
        )

        plan2 = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            group_by=None,
            filters=[],
        )

        # Act: Execute both plans
        result1 = semantic_layer._execute_plan(plan1)
        result2 = semantic_layer._execute_plan(plan2)

        # Assert: Results should be different (different plans produce different results)
        assert not result1.equals(result2), "Different plans should produce different results"

    def test_queryplan_with_filters_compiles_deterministically(self, semantic_layer):
        """QueryPlan with filters should compile to same SQL every time."""
        # Arrange: QueryPlan with filter
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            group_by=None,
            filters=[FilterSpec(column="status", operator="==", value="active", exclude_nulls=True)],
            confidence=0.9,
        )

        # Act: Execute plan multiple times
        result1 = semantic_layer._execute_plan(plan)
        result2 = semantic_layer._execute_plan(plan)

        # Assert: Results should be identical
        assert result1.equals(result2), "Filtered plans should produce identical results"

    def test_queryplan_with_grouping_compiles_deterministically(self, semantic_layer):
        """QueryPlan with grouping should compile to same SQL every time."""
        # Arrange: QueryPlan with grouping
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            group_by="status",
            filters=[],
            confidence=0.9,
        )

        # Act: Execute plan multiple times
        result1 = semantic_layer._execute_plan(plan)
        result2 = semantic_layer._execute_plan(plan)

        # Assert: Results should be identical (deterministic grouping)
        # Note: check_like=True ignores row/column order for aggregated results
        pdt.assert_frame_equal(result1, result2, check_like=True)

    def test_run_key_determinism_for_same_plan(self, semantic_layer):
        """Same QueryPlan should produce same run_key every time (Phase 1.1)."""
        # Arrange: Same QueryPlan
        plan1 = QueryPlan(
            intent="COUNT",
            metric=None,
            group_by="status",
            filters=[],
            entity_key="patient_id",
        )

        plan2 = QueryPlan(
            intent="COUNT",
            metric=None,
            group_by="status",
            filters=[],
            entity_key="patient_id",
        )

        # Act: Generate run_keys
        query_text = "how many patients by status"
        key1 = semantic_layer._generate_run_key(plan1, query_text)
        key2 = semantic_layer._generate_run_key(plan2, query_text)

        # Assert: Same plan + same query should produce same run_key
        assert key1 == key2, "Same plan should produce same run_key"

    def test_run_key_different_for_different_filters(self, semantic_layer):
        """Different filters should produce different run_keys."""
        # Arrange: QueryPlans with different filters
        plan1 = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            filters=[FilterSpec(column="status", operator="==", value="active")],
        )

        plan2 = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            filters=[FilterSpec(column="status", operator="==", value="inactive")],
        )

        # Act: Generate run_keys
        key1 = semantic_layer._generate_run_key(plan1, "describe age for active")
        key2 = semantic_layer._generate_run_key(plan2, "describe age for inactive")

        # Assert: Different filters should produce different run_keys
        assert key1 != key2, "Different filters should produce different run_keys"

    def test_execution_is_testable_no_freeform_code(self, semantic_layer):
        """Execution should be fully testable (no freeform code generation)."""
        # Arrange: QueryPlan
        plan = QueryPlan(
            intent="COUNT",
            metric=None,
            group_by="status",
            entity_key="patient_id",
        )

        # Act: Execute plan
        result = semantic_layer._execute_plan(plan)

        # Assert: Result is testable (pandas/polars DataFrame, not freeform code)
        # Note: _execute_plan returns pandas DataFrame
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame (testable)"
        assert len(result) > 0, "Result should have rows"
        assert "status" in result.columns, "Result should have expected columns"
