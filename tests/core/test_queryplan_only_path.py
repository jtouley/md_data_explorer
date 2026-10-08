"""
Tests for QueryPlan-only execution path enforcement (Phase 3.1).

Ensures all queries go through QueryPlan → execute_query_plan() with no legacy bypasses.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pytest

from clinical_analytics.core.query_plan import QueryPlan


class TestQueryPlanOnlyPath:
    """Test suite enforcing QueryPlan-only execution."""

    def test_semantic_layer_execute_requires_queryplan(self, mock_semantic_layer):
        """semantic_layer.execute_query_plan() should require QueryPlan instance."""
        # Arrange: Invalid input (not a QueryPlan)
        invalid_plan = {"intent": "COUNT"}  # dict, not QueryPlan

        # Act & Assert: Should raise TypeError
        with pytest.raises((TypeError, AttributeError)):
            mock_semantic_layer.execute_query_plan(invalid_plan)

    def test_execute_query_plan_accepts_only_queryplan_type(self, mock_semantic_layer):
        """execute_query_plan() should accept only QueryPlan instances."""
        # Arrange: Valid QueryPlan
        valid_plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)

        # Act: Should not raise
        result = mock_semantic_layer.execute_query_plan(valid_plan)

        # Assert: Returns valid result
        assert result is not None
        assert "success" in result
        assert "run_key" in result

    def test_format_execution_result_should_not_reanalyze_result_dataframe(self):
        """format_execution_result() should format result DataFrame, not call compute_analysis_by_type on it."""
        # This is a static code analysis test
        import ast
        from pathlib import Path

        semantic_file = Path("src/clinical_analytics/core/semantic.py")

        with open(semantic_file) as f:
            source = f.read()

        # Parse AST to find format_execution_result method
        tree = ast.parse(source)

        # Find format_execution_result method
        format_method_found = False
        calls_compute_on_result = False

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "format_execution_result":
                format_method_found = True
                # Check if it calls compute_analysis_by_type on result_df
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        # Check if calling compute_analysis_by_type
                        if isinstance(child.func, ast.Name) and child.func.id == "compute_analysis_by_type":
                            # Check arguments - if first arg is result_df, that's wrong
                            if child.args:
                                first_arg = child.args[0]
                                # Check if first arg is result_df (the result DataFrame)
                                if isinstance(first_arg, ast.Name) and first_arg.id == "result_df":
                                    calls_compute_on_result = True
                                elif isinstance(first_arg, ast.Name) and first_arg.id == "result_df_pl":
                                    calls_compute_on_result = True
                break

        assert format_method_found, "format_execution_result() method not found"

        # Assert: Should NOT call compute_analysis_by_type on result DataFrame
        # (result_df is already aggregated - compute_analysis_by_type expects raw cohort)
        assert not calls_compute_on_result, (
            "format_execution_result() should not call compute_analysis_by_type() on result DataFrame. "
            "result_df is already aggregated from execute_query_plan(). "
            "compute_analysis_by_type() expects raw cohort data, not aggregated results."
        )

    def test_format_execution_result_formats_count_result_correctly(self, mock_semantic_layer):
        """format_execution_result() should format COUNT result DataFrame correctly."""
        # Arrange: COUNT query result (already aggregated)
        import pandas as pd

        from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
        from clinical_analytics.core.query_plan import QueryPlan

        # Simulate result from execute_query_plan() for COUNT with group_by
        # Result DataFrame has: [group_by_column, "count"]
        result_df = pd.DataFrame(
            {
                "Statin Used": ["Atorvastatin", "Pravastatin", "Simvastatin"],
                "count": [1, 1, 1],
            }
        )

        execution_result = {
            "success": True,
            "result": result_df,
            "run_key": "test_key",
            "warnings": [],
        }

        # Create context for COUNT intent
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COUNT
        context.grouping_variable = "Statin Used"

        # Create QueryPlan
        query_plan = QueryPlan(intent="COUNT", group_by="Statin Used", confidence=0.9)
        context.query_plan = query_plan

        # Act
        formatted = mock_semantic_layer.format_execution_result(execution_result, context)

        # Assert: Should return count result format
        assert formatted["type"] == "count"
        assert "total_count" in formatted
        assert "grouped_by" in formatted
        assert formatted["grouped_by"] == "Statin Used"
        assert "group_counts" in formatted
        assert isinstance(formatted["group_counts"], list)
        assert len(formatted["group_counts"]) == 3
        # Each group_count should have the group value and count
        assert all("Statin Used" in gc and "count" in gc for gc in formatted["group_counts"])
        assert "headline" in formatted


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
