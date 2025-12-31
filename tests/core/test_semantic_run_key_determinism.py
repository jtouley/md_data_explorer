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
