"""
Tests for run_key determinism (Phase 1.1).

Tests that run_key is deterministic based on canonical plan, normalized query,
dataset version, and canonical scope.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pandas as pd
import pytest

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.semantic import SemanticLayer


class TestRunKeyDeterminism:
    """Test suite for run_key determinism."""

    @pytest.fixture
    def semantic_layer(self, tmp_path):
        """Create minimal semantic layer for testing run_key generation."""
        # Create minimal config
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create minimal data directory
        data_dir = workspace / "data" / "raw" / "test_dataset"
        data_dir.mkdir(parents=True)

        # Create test CSV
        test_csv = data_dir / "test.csv"
        df = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [45, 62, 38],
                "status": ["active", "inactive", "active"],
            }
        )
        df.to_csv(test_csv, index=False)

        # Minimal config
        config = {
            "init_params": {"source_path": "data/raw/test_dataset/test.csv"},
            "column_mapping": {"patient_id": "patient_id"},
            "time_zero": {"value": "2024-01-01"},
            "outcomes": {},
            "analysis": {"default_outcome": "outcome"},
        }

        # Create SemanticLayer
        semantic = SemanticLayer("test_dataset", config=config, workspace_root=workspace)
        semantic.dataset_version = "test_v1"

        return semantic

    def test_run_key_deterministic_for_same_plan_and_query(self, semantic_layer):
        """Same QueryPlan + same query should produce same run_key (Phase 1.1)."""
        # Arrange: Same plan, same query
        plan1 = QueryPlan(intent="COUNT", metric="age", group_by="status", confidence=0.9)
        plan2 = QueryPlan(intent="COUNT", metric="age", group_by="status", confidence=0.9)
        query = "average age by status"

        # Act
        key1 = semantic_layer._generate_run_key(plan1, query)
        key2 = semantic_layer._generate_run_key(plan2, query)

        # Assert: Should be identical
        assert key1 == key2

    def test_run_key_different_for_different_queries(self, semantic_layer):
        """Different query text should produce different run_keys (Phase 1.1)."""
        # Arrange: Same plan, different queries
        plan = QueryPlan(intent="COUNT", metric="age", group_by="status", confidence=0.9)
        query1 = "average age by status"
        query2 = "mean age grouped by status"

        # Act
        key1 = semantic_layer._generate_run_key(plan, query1)
        key2 = semantic_layer._generate_run_key(plan, query2)

        # Assert: Different queries should produce different keys
        # (even though they might produce same plan)
        assert key1 != key2

    def test_run_key_same_for_whitespace_variations(self, semantic_layer):
        """Whitespace variations should produce same run_key (Phase 1.1)."""
        # Arrange: Same plan, query with different whitespace
        plan = QueryPlan(intent="COUNT", metric="age", group_by="status", confidence=0.9)
        query1 = "average age by status"
        query2 = "average  age   by    status"  # Extra whitespace
        query3 = "AVERAGE AGE BY STATUS"  # Different case

        # Act
        key1 = semantic_layer._generate_run_key(plan, query1)
        key2 = semantic_layer._generate_run_key(plan, query2)
        key3 = semantic_layer._generate_run_key(plan, query3)

        # Assert: All should be identical (normalized)
        assert key1 == key2 == key3

    def test_run_key_includes_all_plan_fields(self, semantic_layer):
        """Run key should include all QueryPlan fields for determinism (Phase 1.1)."""
        # Arrange: Plans with different fields
        plan1 = QueryPlan(intent="COUNT", metric="age", group_by="status", entity_key="patient_id")
        plan2 = QueryPlan(intent="COUNT", metric="age", group_by="status", entity_key=None)
        query = "query"

        # Act
        key1 = semantic_layer._generate_run_key(plan1, query)
        key2 = semantic_layer._generate_run_key(plan2, query)

        # Assert: Different entity_key should produce different key
        assert key1 != key2

    def test_run_key_includes_filters_sorted(self, semantic_layer):
        """Run key should include filters in canonical sorted order (Phase 1.1)."""
        # Arrange: Plans with same filters in different order
        plan1 = QueryPlan(
            intent="COUNT",
            metric="age",
            filters=[
                FilterSpec(column="status", operator="==", value="active"),
                FilterSpec(column="gender", operator="==", value="M"),
            ],
        )
        plan2 = QueryPlan(
            intent="COUNT",
            metric="age",
            filters=[
                FilterSpec(column="gender", operator="==", value="M"),
                FilterSpec(column="status", operator="==", value="active"),
            ],
        )
        query = "query"

        # Act
        key1 = semantic_layer._generate_run_key(plan1, query)
        key2 = semantic_layer._generate_run_key(plan2, query)

        # Assert: Same filters (different order) should produce same key
        assert key1 == key2

    def test_run_key_includes_dataset_version(self, semantic_layer):
        """Run key should include dataset_version for cache invalidation (Phase 1.1)."""
        # Arrange: Same plan, different dataset versions
        plan = QueryPlan(intent="COUNT", metric="age", confidence=0.9)
        query = "average age"

        # Modify dataset version
        original_version = semantic_layer.dataset_version
        semantic_layer.dataset_version = "v1"
        key1 = semantic_layer._generate_run_key(plan, query)

        semantic_layer.dataset_version = "v2"
        key2 = semantic_layer._generate_run_key(plan, query)

        # Restore
        semantic_layer.dataset_version = original_version

        # Assert: Different versions should produce different keys
        assert key1 != key2

    def test_run_key_handles_none_query_text(self, semantic_layer):
        """Run key should handle None query_text gracefully (Phase 1.1)."""
        # Arrange: Plan with None query text
        plan = QueryPlan(intent="COUNT", metric="age", confidence=0.9)

        # Act
        key1 = semantic_layer._generate_run_key(plan, None)
        key2 = semantic_layer._generate_run_key(plan, None)

        # Assert: Should be deterministic even with None
        assert key1 == key2

    def test_run_key_different_for_different_intents(self, semantic_layer):
        """Different intents should produce different run_keys (Phase 1.1)."""
        # Arrange: Plans with different intents
        plan1 = QueryPlan(intent="COUNT", metric="age", confidence=0.9)
        plan2 = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.9)
        query = "query"

        # Act
        key1 = semantic_layer._generate_run_key(plan1, query)
        key2 = semantic_layer._generate_run_key(plan2, query)

        # Assert: Different intents should produce different keys
        assert key1 != key2

    def test_run_key_different_for_different_scope(self, semantic_layer):
        """Different scope values should produce different run_keys (Phase 1.1)."""
        # Arrange: Plans with different scope
        plan1 = QueryPlan(intent="COUNT", metric="age", scope="all")
        plan2 = QueryPlan(intent="COUNT", metric="age", scope="filtered")
        query = "query"

        # Act
        key1 = semantic_layer._generate_run_key(plan1, query)
        key2 = semantic_layer._generate_run_key(plan2, query)

        # Assert: Different scope should produce different keys
        assert key1 != key2


class TestRunKeyDeterminismAllExecutionPaths:
    """Test suite for Phase 1.1.5: Verify run_key determinism across all execution paths."""

    @pytest.fixture
    def semantic_layer(self, tmp_path):
        """Create minimal semantic layer for testing run_key generation."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

        data_dir = workspace / "data" / "raw" / "test_dataset"
        data_dir.mkdir(parents=True)

        test_csv = data_dir / "test.csv"
        import pandas as pd

        df = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [45, 62, 38],
                "status": ["active", "inactive", "active"],
            }
        )
        df.to_csv(test_csv, index=False)

        config = {
            "init_params": {"source_path": "data/raw/test_dataset/test.csv"},
            "column_mapping": {"patient_id": "patient_id"},
            "time_zero": {"value": "2024-01-01"},
            "outcomes": {},
            "analysis": {"default_outcome": "outcome"},
        }

        from clinical_analytics.core.semantic import SemanticLayer

        semantic = SemanticLayer("test_dataset", config=config, workspace_root=workspace)
        semantic.dataset_version = "test_v1"

        return semantic

    def test_all_execution_paths_produce_same_run_key(self, semantic_layer):
        """All execution paths should produce same run_key for same query (Phase 1.1.5)."""
        # Arrange: Same query, same plan
        query = "average age by status"
        plan = QueryPlan(intent="COUNT", metric="age", group_by="status", entity_key="patient_id", confidence=0.9)

        # Act: Generate run_key from different paths
        # Path 1: semantic._generate_run_key() directly
        key1 = semantic_layer._generate_run_key(plan, query)

        # Path 2: execute_query_plan() (which calls _generate_run_key())
        result = semantic_layer.execute_query_plan(plan, query_text=query)
        key2 = result["run_key"]

        # Assert: All paths produce same key
        assert key1 == key2, f"Run keys should match: {key1} != {key2}"
        assert result["success"] is True, "Execution should succeed"

    def test_query_plan_run_key_always_from_semantic_layer(self, semantic_layer):
        """QueryPlan.run_key should always come from semantic layer, never from nl_query_engine (Phase 1.1.5)."""
        # Arrange: QueryIntent from nl_query_engine
        from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent

        engine = NLQueryEngine(semantic_layer)
        intent = QueryIntent(
            intent_type="COUNT",
            primary_variable="age",
            grouping_variable="status",
            confidence=0.9,
        )

        # Act: Convert to plan
        plan = engine._intent_to_plan(intent, "test_v1")

        # Assert: run_key should be None (semantic layer will generate it)
        assert plan.run_key is None, "nl_query_engine should not set run_key - semantic layer will generate it"

    def test_execute_query_plan_always_generates_run_key(self, semantic_layer):
        """execute_query_plan() should always generate run_key even if plan.run_key is None (Phase 1.1.5)."""
        # Arrange: Plan with run_key=None (as nl_query_engine should produce)
        plan = QueryPlan(intent="COUNT", metric="age", group_by="status", entity_key="patient_id", run_key=None)
        query = "average age by status"

        # Act
        result = semantic_layer.execute_query_plan(plan, query_text=query)

        # Assert: Execution result should always include run_key
        assert "run_key" in result, "Execution result must include run_key"
        assert result["run_key"] is not None, "Run key should not be None"
        assert len(result["run_key"]) > 0, "Run key should not be empty"
        assert result["success"] is True, "Execution should succeed"

    def test_execute_query_plan_run_key_matches_direct_generation(self, semantic_layer):
        """execute_query_plan() run_key should match direct _generate_run_key() call (Phase 1.1.5)."""
        # Arrange: Same plan and query
        plan = QueryPlan(intent="COUNT", metric="age", group_by="status", entity_key="patient_id", confidence=0.9)
        query = "average age by status"

        # Act: Generate run_key directly
        direct_key = semantic_layer._generate_run_key(plan, query)

        # Act: Generate run_key via execute_query_plan()
        result = semantic_layer.execute_query_plan(plan, query_text=query)
        execution_key = result["run_key"]

        # Assert: Keys should match
        assert direct_key == execution_key, (
            f"Direct generation and execution should produce same key: {direct_key} != {execution_key}"
        )

    def test_run_key_deterministic_across_multiple_executions(self, semantic_layer):
        """Same query executed multiple times should produce same run_key (Phase 1.1.5)."""
        # Arrange: Same plan and query
        plan = QueryPlan(intent="COUNT", metric="age", group_by="status", entity_key="patient_id", confidence=0.9)
        query = "average age by status"

        # Act: Execute multiple times
        result1 = semantic_layer.execute_query_plan(plan, query_text=query)
        result2 = semantic_layer.execute_query_plan(plan, query_text=query)
        result3 = semantic_layer.execute_query_plan(plan, query_text=query)

        # Assert: All executions should produce same run_key
        assert result1["run_key"] == result2["run_key"] == result3["run_key"], (
            "Multiple executions should produce same run_key: "
            f"{result1['run_key']} != {result2['run_key']} != {result3['run_key']}"
        )
