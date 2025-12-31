"""
Tests for semantic layer observability (Phase 2.1).

Tests that warnings are collected and explanations provided before removing gating.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pandas as pd
import pytest

from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.semantic import SemanticLayer


class TestSemanticLayerObservability:
    """Test suite for semantic layer observability features."""

    @pytest.fixture
    def semantic_layer(self, tmp_path):
        """Create minimal semantic layer for testing."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

        data_dir = workspace / "data" / "raw" / "test_dataset"
        data_dir.mkdir(parents=True)

        test_csv = data_dir / "test.csv"
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

        semantic = SemanticLayer("test_dataset", config=config, workspace_root=workspace)
        semantic.dataset_version = "test_v1"

        return semantic

    def test_execute_query_plan_includes_warnings_field(self, semantic_layer):
        """Execution result should include warnings field (Phase 2.1)."""
        # Arrange: Valid QueryPlan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)

        # Act
        result = semantic_layer.execute_query_plan(plan)

        # Assert: Warnings field should exist (even if empty)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_low_confidence_adds_warning_with_explanation(self, semantic_layer):
        """Low confidence should add warning with explanation (Phase 2.2)."""
        # Arrange: Low confidence plan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.3)

        # Act
        result = semantic_layer.execute_query_plan(plan, confidence_threshold=0.75)

        # Assert: Warning should explain low confidence (Phase 2.2: no blocking)
        assert "warnings" in result
        assert len(result["warnings"]) > 0
        assert "Low confidence" in result["warnings"][0]
        assert "0.30" in result["warnings"][0]
        # Phase 2.2: Should still execute successfully despite low confidence
        assert result["success"] is True

    def test_incomplete_plan_adds_warning_with_explanation(self, semantic_layer):
        """Incomplete plan should add warning with explanation (Phase 2.2)."""
        # Arrange: Incomplete COUNT plan (no entity_key, no group_by)
        plan = QueryPlan(intent="COUNT", confidence=0.9)

        # Act
        result = semantic_layer.execute_query_plan(plan)

        # Assert: Warning should explain incompleteness (Phase 2.2: warning only, no blocking)
        assert "warnings" in result
        assert len(result["warnings"]) > 0
        assert "Incomplete plan" in result["warnings"][0]
        # Phase 2.2: Still attempts execution (may succeed or fail depending on implementation)
        # The key is that incompleteness doesn't block - it only warns

    def test_validation_failure_adds_warning_with_explanation(self, semantic_layer):
        """Validation failure should add warning with explanation (Phase 2.2)."""
        # Arrange: Plan with nonexistent column
        plan = QueryPlan(intent="COUNT", entity_key="nonexistent_column", confidence=0.9)

        # Act
        result = semantic_layer.execute_query_plan(plan)

        # Assert: Warning should explain validation failure (Phase 2.2: warnings collected, then execution fails)
        assert "warnings" in result
        assert len(result["warnings"]) >= 1  # At least one warning
        # Should have validation warning (may also have execution error)
        warnings_text = " ".join(result["warnings"])
        assert "Validation failed" in warnings_text or "nonexistent_column" in warnings_text
        # Phase 2.2: Execution will fail (not gated, but actual execution error)
        assert result["success"] is False

    def test_successful_execution_has_empty_warnings(self, semantic_layer):
        """Successful execution with no issues should have empty warnings (Phase 2.1)."""
        # Arrange: Valid plan with high confidence
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)

        # Act
        result = semantic_layer.execute_query_plan(plan)

        # Assert: No warnings for successful execution
        assert "warnings" in result
        if result.get("success", False):
            assert len(result["warnings"]) == 0
