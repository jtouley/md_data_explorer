"""
Tests for QueryPlan-driven chart specification (Phase 3.3).

Ensures chart_spec is deterministic from QueryPlan and included in result artifact.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pytest

from clinical_analytics.core.query_plan import QueryPlan


class TestChartSpecGeneration:
    """Test suite for chart_spec generation from QueryPlan."""

    def test_result_artifact_includes_chart_spec(self, mock_semantic_layer):
        """Execution result should include chart_spec for visualization."""
        # Arrange: QueryPlan with grouping (typically visualizable)
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", group_by="status", confidence=0.9)

        # Act
        result = mock_semantic_layer.execute_query_plan(plan)

        # Assert: Chart spec present
        assert "chart_spec" in result, "Result must include chart_spec for visualization"
        assert result["chart_spec"] is not None, "chart_spec should not be None"

    def test_chart_spec_has_required_fields(self, mock_semantic_layer):
        """chart_spec should have required fields (type, title)."""
        # Arrange: QueryPlan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", group_by="status", confidence=0.9)

        # Act
        result = mock_semantic_layer.execute_query_plan(plan)

        # Assert: Required fields present
        chart_spec = result["chart_spec"]
        assert "type" in chart_spec, "chart_spec must include type field"
        assert "title" in chart_spec, "chart_spec must include title field"

    def test_chart_spec_type_is_valid(self, mock_semantic_layer):
        """chart_spec type should be one of allowed values."""
        # Arrange: QueryPlan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", group_by="status", confidence=0.9)

        # Act
        result = mock_semantic_layer.execute_query_plan(plan)

        # Assert: Type is valid
        chart_spec = result["chart_spec"]
        valid_types = ["bar", "line", "hist"]
        assert chart_spec["type"] in valid_types, f"chart_spec type must be one of {valid_types}"

    def test_chart_spec_deterministic_from_plan(self, mock_semantic_layer):
        """Chart spec should be deterministic from QueryPlan."""
        # Arrange: Same QueryPlan executed twice
        plan1 = QueryPlan(intent="COUNT", entity_key="patient_id", group_by="status", confidence=0.9)
        plan2 = QueryPlan(intent="COUNT", entity_key="patient_id", group_by="status", confidence=0.9)

        # Act: Generate chart spec multiple times
        result1 = mock_semantic_layer.execute_query_plan(plan1)
        result2 = mock_semantic_layer.execute_query_plan(plan2)

        # Assert: Same spec (deterministic)
        assert result1["chart_spec"] == result2["chart_spec"], "chart_spec should be deterministic"

    def test_chart_spec_differs_for_different_plans(self, mock_semantic_layer):
        """Chart spec should differ for different QueryPlans."""
        # Arrange: Different QueryPlans
        plan1 = QueryPlan(intent="COUNT", entity_key="patient_id", group_by="status", confidence=0.9)
        plan2 = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.9)

        # Act
        result1 = mock_semantic_layer.execute_query_plan(plan1)
        result2 = mock_semantic_layer.execute_query_plan(plan2)

        # Assert: Different specs for different plans
        # Note: Specs might be the same if logic produces same chart for different intents
        # but typically COUNT vs DESCRIBE should produce different visualizations
        assert (
            result1["chart_spec"]["type"] != result2["chart_spec"]["type"]
            or result1["chart_spec"]["title"] != result2["chart_spec"]["title"]
        ), "Different plans should typically produce different chart specs"

    def test_chart_spec_none_for_non_visualizable_plans(self, mock_semantic_layer):
        """chart_spec may be None for non-visualizable plans."""
        # Arrange: QueryPlan without obvious visualization (e.g., COUNT without grouping)
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)

        # Act
        result = mock_semantic_layer.execute_query_plan(plan)

        # Assert: chart_spec present (may be None or minimal)
        # This is permissive - allows None for non-visualizable queries
        assert "chart_spec" in result, "Result must include chart_spec field"


@pytest.fixture
def mock_semantic_layer(tmp_path):
    """Create minimal semantic layer for testing."""
    import pandas as pd

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

    from clinical_analytics.core.semantic import SemanticLayer

    semantic = SemanticLayer("test_dataset", config=config, workspace_root=workspace)
    semantic.dataset_version = "test_v1"

    return semantic
