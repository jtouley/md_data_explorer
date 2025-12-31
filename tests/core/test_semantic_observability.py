"""
Tests for semantic layer observability (Phase 2.1) and retry logic (Phase 2.5.2).

Tests that warnings are collected and explanations provided before removing gating.
Tests retry logic with exponential backoff for transient errors.

Test name follows: test_unit_scenario_expectedBehavior
"""

from unittest.mock import patch

import pandas as pd
import pytest

from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.semantic import SemanticLayer


class TestSemanticLayerObservability:
    """Test suite for semantic layer observability features."""

    @pytest.fixture
    def semantic_layer(self, make_semantic_layer):
        """Create minimal semantic layer for testing."""
        return make_semantic_layer(
            dataset_name="test_dataset",
            data={
                "patient_id": [1, 2, 3],
                "age": [45, 62, 38],
                "status": ["active", "inactive", "active"],
            },
            config_overrides={
                "column_mapping": {"patient_id": "patient_id"},
                "time_zero": {"value": "2024-01-01"},
            },
            workspace_name="workspace",
        )

    def test_execute_query_plan_includes_warnings_field(self, semantic_layer):
        """Execution result should include warnings field (Phase 2.1)."""
        # Arrange: Valid QueryPlan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)

        # Act
        result = semantic_layer.execute_query_plan(plan)

        # Assert: Warnings field should exist (even if empty)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_execute_query_plan_includes_steps_field(self, semantic_layer):
        """Execution result should include steps field for progressive thinking indicator (Phase 2.5.1)."""
        # Arrange: Valid QueryPlan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)

        # Act
        result = semantic_layer.execute_query_plan(plan)

        # Assert: Steps field should exist (core layer provides step data)
        assert "steps" in result
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) > 0

        # Assert: Steps should have required structure
        for step in result["steps"]:
            assert "status" in step
            assert "text" in step
            assert step["status"] in ["processing", "completed", "error"]
            assert isinstance(step["text"], str)

        # Assert: Last step should indicate completion or error
        last_step = result["steps"][-1]
        if result["success"]:
            assert last_step["status"] == "completed"
            assert "Query complete" in last_step["text"]
        else:
            assert last_step["status"] == "error"
            assert "Query failed" in last_step["text"]

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


class TestSemanticLayerRetryLogic:
    """Test suite for retry logic with exponential backoff (Phase 2.5.2)."""

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

    def test_retry_succeeds_after_backend_error(self, semantic_layer):
        """Retry should succeed after backend initialization error (Phase 2.5.2)."""
        # Arrange: Mock _execute_plan to fail once with AttributeError, then succeed
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        expected_df = pd.DataFrame({"count": [3]})

        call_count = 0

        def mock_execute_plan(plan):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: backend error
                raise AttributeError("'NoneType' object has no attribute '_record_batch_readers_consumed'")
            # Second call: success
            return expected_df

        # Act
        with patch.object(semantic_layer, "_execute_plan", side_effect=mock_execute_plan):
            result_df = semantic_layer._execute_plan_with_retry(plan, max_retries=3, initial_delay=0.01)

        # Assert
        assert call_count == 2  # Failed once, succeeded on retry
        assert result_df.equals(expected_df)

    def test_retry_succeeds_after_connection_error(self, semantic_layer):
        """Retry should succeed after connection error (Phase 2.5.2)."""
        # Arrange: Mock _execute_plan to fail with ConnectionError, then succeed
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        expected_df = pd.DataFrame({"count": [3]})

        call_count = 0

        def mock_execute_plan(plan):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection refused")
            return expected_df

        # Act
        with patch.object(semantic_layer, "_execute_plan", side_effect=mock_execute_plan):
            result_df = semantic_layer._execute_plan_with_retry(plan, max_retries=3, initial_delay=0.01)

        # Assert
        assert call_count == 2
        assert result_df.equals(expected_df)

    def test_retry_succeeds_after_transient_error(self, semantic_layer):
        """Retry should succeed after transient error with pattern match (Phase 2.5.2)."""
        # Arrange: Mock _execute_plan to fail with transient error, then succeed
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        expected_df = pd.DataFrame({"count": [3]})

        call_count = 0

        def mock_execute_plan(plan):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Database temporarily unavailable")
            return expected_df

        # Act
        with patch.object(semantic_layer, "_execute_plan", side_effect=mock_execute_plan):
            result_df = semantic_layer._execute_plan_with_retry(plan, max_retries=3, initial_delay=0.01)

        # Assert
        assert call_count == 2
        assert result_df.equals(expected_df)

    def test_retry_raises_after_max_attempts(self, semantic_layer):
        """Retry should raise exception after exhausting max retries (Phase 2.5.2)."""
        # Arrange: Mock _execute_plan to always fail
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)

        call_count = 0

        def mock_execute_plan(plan):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection refused")

        # Act & Assert
        with patch.object(semantic_layer, "_execute_plan", side_effect=mock_execute_plan):
            with pytest.raises(ConnectionError, match="Connection refused"):
                semantic_layer._execute_plan_with_retry(plan, max_retries=2, initial_delay=0.01)

        # Should try: initial + 2 retries = 3 attempts total
        assert call_count == 3

    def test_retry_non_transient_error_no_retry(self, semantic_layer):
        """Non-transient errors should not trigger retry (Phase 2.5.2)."""
        # Arrange: Mock _execute_plan to fail with non-transient error
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)

        call_count = 0

        def mock_execute_plan(plan):
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid column name")  # Not transient

        # Act & Assert
        with patch.object(semantic_layer, "_execute_plan", side_effect=mock_execute_plan):
            with pytest.raises(ValueError, match="Invalid column name"):
                semantic_layer._execute_plan_with_retry(plan, max_retries=3, initial_delay=0.01)

        # Should only try once (no retry for non-transient errors)
        assert call_count == 1

    def test_retry_exponential_backoff_timing(self, semantic_layer):
        """Retry should use exponential backoff (Phase 2.5.2)."""
        # Arrange: Mock _execute_plan to fail multiple times
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        expected_df = pd.DataFrame({"count": [3]})

        call_count = 0
        sleep_delays = []

        def mock_execute_plan(plan):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Connection refused")
            return expected_df

        def mock_sleep(delay):
            sleep_delays.append(delay)

        # Act
        with patch.object(semantic_layer, "_execute_plan", side_effect=mock_execute_plan):
            with patch("time.sleep", side_effect=mock_sleep):
                result_df = semantic_layer._execute_plan_with_retry(plan, max_retries=3, initial_delay=0.5)

        # Assert
        assert call_count == 3  # Failed twice, succeeded on third attempt
        assert len(sleep_delays) == 2  # Slept twice (after 1st and 2nd failure)
        assert sleep_delays[0] == 0.5  # First retry: 0.5s
        assert sleep_delays[1] == 1.0  # Second retry: 0.5 * 2 = 1.0s (exponential backoff)
        assert result_df.equals(expected_df)
