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

    def test_no_direct_compute_analysis_by_type_calls_in_ui(self):
        """UI pages should not call compute_analysis_by_type() directly."""
        # This is a static code analysis test
        import ast
        from pathlib import Path

        ui_pages_dir = Path("src/clinical_analytics/ui/pages")
        ask_questions_file = ui_pages_dir / "3_💬_Ask_Questions.py"

        # Read and parse the file
        with open(ask_questions_file) as f:
            tree = ast.parse(f.read())

        # Find all function calls
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        # Assert: compute_analysis_by_type should not be called
        assert "compute_analysis_by_type" not in calls, (
            "Found legacy compute_analysis_by_type() call in Ask_Questions.py"
        )

    def test_no_direct_get_or_compute_result_calls_in_ui(self):
        """UI pages should not call get_or_compute_result() legacy path."""
        import ast
        from pathlib import Path

        ui_pages_dir = Path("src/clinical_analytics/ui/pages")
        ask_questions_file = ui_pages_dir / "3_💬_Ask_Questions.py"

        with open(ask_questions_file) as f:
            tree = ast.parse(f.read())

        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        # Assert: get_or_compute_result should not be called
        assert "get_or_compute_result" not in calls, "Found legacy get_or_compute_result() call in Ask_Questions.py"

    def test_all_execution_paths_use_semantic_layer_execute_query_plan(self):
        """All query execution should go through semantic_layer.execute_query_plan()."""
        import ast
        from pathlib import Path

        ui_pages_dir = Path("src/clinical_analytics/ui/pages")
        ask_questions_file = ui_pages_dir / "3_💬_Ask_Questions.py"

        with open(ask_questions_file) as f:
            source = f.read()

        # Check that execute_query_plan is called
        assert "execute_query_plan" in source, "execute_query_plan() should be present in Ask_Questions.py"

        # Parse AST to verify calls
        tree = ast.parse(source)

        # Find execute_query_plan calls
        execute_query_plan_calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "execute_query_plan":
                    execute_query_plan_calls += 1

        # Assert: Should have at least one execute_query_plan call
        assert execute_query_plan_calls > 0, "No execute_query_plan() calls found in Ask_Questions.py"


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
