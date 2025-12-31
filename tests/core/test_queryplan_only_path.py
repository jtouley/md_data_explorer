"""
Tests for QueryPlan-only execution path enforcement (Phase 3.1).

Ensures all queries go through QueryPlan → execute_query_plan() with no legacy bypasses.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pytest

from clinical_analytics.core.query_plan import QueryPlan


class TestQueryPlanOnlyPath:
    """Test suite enforcing QueryPlan-only execution."""

    def test_semantic_layer_execute_requires_queryplan(self, make_semantic_layer):
        """semantic_layer.execute_query_plan() should require QueryPlan instance."""
        # Arrange: Invalid input (not a QueryPlan)
        invalid_plan = {"intent": "COUNT"}  # dict, not QueryPlan
        semantic_layer = make_semantic_layer()

        # Act & Assert: Should raise TypeError
        with pytest.raises((TypeError, AttributeError)):
            semantic_layer.execute_query_plan(invalid_plan)

    def test_execute_query_plan_accepts_only_queryplan_type(self, make_semantic_layer):
        """execute_query_plan() should accept only QueryPlan instances."""
        # Arrange: Valid QueryPlan
        valid_plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        semantic_layer = make_semantic_layer()

        # Act: Should not raise
        result = semantic_layer.execute_query_plan(valid_plan)

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

    def test_chat_handler_should_not_execute_query(self):
        """Chat input handler should not execute query - only parse and rerun."""
        # This is a static code analysis test
        from pathlib import Path

        ui_pages_dir = Path("src/clinical_analytics/ui/pages")
        ask_questions_file = ui_pages_dir / "3_💬_Ask_Questions.py"

        with open(ask_questions_file) as f:
            source = f.read()

        # Parse AST to find chat handler execution
        # Find the chat input handler section (look for chat_input_analysis_execution_triggered log)
        # If chat handler executes query, it will call execute_query_plan() before st.rerun()
        # We want to verify that chat handler does NOT execute - it should only parse and rerun

        # Find all st.rerun() calls in chat handler context
        # Chat handler should rerun immediately after parsing, not after execution
        # Look for pattern: execute_query_plan() followed by st.rerun() in chat handler context

        # Check: chat_input_analysis_execution_triggered should NOT be followed by execute_query_plan
        # The execution should happen in main flow, not chat handler
        lines = source.split("\n")
        chat_handler_executes = False
        for i, line in enumerate(lines):
            if "chat_input_analysis_execution_triggered" in line:
                # Check if execute_query_plan is called in chat handler (within next 50 lines)
                for j in range(i + 1, min(i + 50, len(lines))):
                    if "execute_query_plan" in lines[j] and "semantic_layer" in lines[j]:
                        chat_handler_executes = True
                        break
                break

        # Assert: Chat handler should NOT execute query
        assert not chat_handler_executes, (
            "Chat handler should not execute query - it should only parse and rerun. "
            "Execution should happen in main flow after rerun."
        )

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

    def test_format_execution_result_formats_count_result_correctly(self, make_semantic_layer):
        """format_execution_result() should format COUNT result DataFrame correctly."""
        # Arrange: COUNT query result (already aggregated)
        import pandas as pd

        from clinical_analytics.core.query_plan import QueryPlan
        from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

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
        semantic_layer = make_semantic_layer()

        # Act
        formatted = semantic_layer.format_execution_result(execution_result, context)

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
