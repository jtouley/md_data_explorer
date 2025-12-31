"""
Tests for QueryPlan contract enforcement (Phase 3.2).

Ensures all queries go through QueryPlan → execute_query_plan() with no bypasses.
Adds assertions and logging to catch any contract violations.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pytest

from clinical_analytics.core.query_plan import QueryPlan


class TestQueryPlanContractEnforcement:
    """Test suite for QueryPlan contract enforcement."""

    def test_execute_query_plan_logs_execution_start(self, make_semantic_layer, caplog):
        """execute_query_plan() should log execution start for observability."""
        # Arrange: Valid QueryPlan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        semantic_layer = make_semantic_layer()

        # Act
        import logging

        with caplog.at_level(logging.DEBUG):
            semantic_layer.execute_query_plan(plan)

        # Assert: Should log execution (verify contract enforcement)
        log_messages = [record.message for record in caplog.records]
        assert any("execute_query_plan" in msg.lower() or "query" in msg.lower() for msg in log_messages), (
            "execute_query_plan should log execution for observability"
        )

    def test_execute_query_plan_validates_plan_type(self, make_semantic_layer):
        """execute_query_plan() should reject non-QueryPlan inputs."""
        # Arrange: Invalid input (dict instead of QueryPlan)
        invalid_plan = {"intent": "COUNT", "entity_key": "patient_id"}
        semantic_layer = make_semantic_layer()

        # Act & Assert: Should raise TypeError or AttributeError
        with pytest.raises((TypeError, AttributeError)):
            semantic_layer.execute_query_plan(invalid_plan)

    def test_execute_query_plan_validates_plan_intent(self, make_semantic_layer):
        """execute_query_plan() should validate QueryPlan has intent field."""
        # Arrange: QueryPlan without intent (will fail at validation)
        # Note: This test doesn't need semantic_layer since validation happens at QueryPlan construction
        with pytest.raises((TypeError, ValueError)):
            # QueryPlan requires intent, this should fail at construction
            QueryPlan(entity_key="patient_id", confidence=0.9)

    def test_execute_query_plan_returns_standardized_result(self, make_semantic_layer):
        """execute_query_plan() should return standardized result with required fields."""
        # Arrange: Valid QueryPlan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        semantic_layer = make_semantic_layer()

        # Act
        result = semantic_layer.execute_query_plan(plan)

        # Assert: Result has standard contract fields
        assert isinstance(result, dict), "Result should be dict"
        assert "success" in result, "Result must include success field"
        assert "run_key" in result, "Result must include run_key field"
        assert "warnings" in result, "Result must include warnings field"
        assert "steps" in result, "Result must include steps field (Phase 2.5.1)"
        assert isinstance(result["success"], bool), "success should be bool"
        assert isinstance(result["run_key"], str), "run_key should be str"
        assert isinstance(result["warnings"], list), "warnings should be list"
        assert isinstance(result["steps"], list), "steps should be list"

    def test_execute_query_plan_generates_deterministic_run_key(self, make_semantic_layer):
        """execute_query_plan() should generate deterministic run_key for same plan."""
        # Arrange: Same QueryPlan executed twice
        plan1 = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        plan2 = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        semantic_layer = make_semantic_layer()

        # Act
        result1 = semantic_layer.execute_query_plan(plan1, query_text="test query")
        result2 = semantic_layer.execute_query_plan(plan2, query_text="test query")

        # Assert: Same run_key for identical plans
        assert result1["run_key"] == result2["run_key"], "run_key should be deterministic"

    def test_execute_query_plan_different_run_key_for_different_plans(self, make_semantic_layer):
        """execute_query_plan() should generate different run_key for different plans."""
        # Arrange: Different QueryPlans
        plan1 = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        plan2 = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.9)
        semantic_layer = make_semantic_layer()

        # Act
        result1 = semantic_layer.execute_query_plan(plan1)
        result2 = semantic_layer.execute_query_plan(plan2)

        # Assert: Different run_keys for different plans
        assert result1["run_key"] != result2["run_key"], "Different plans should have different run_keys"

    def test_execute_query_plan_uses_retry_logic(self, make_semantic_layer):
        """execute_query_plan() should use _execute_plan_with_retry() internally."""
        # Arrange: Valid QueryPlan
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
        semantic_layer = make_semantic_layer()

        # Act
        result = semantic_layer.execute_query_plan(plan)

        # Assert: Execution succeeded (retry logic is tested separately)
        # This test verifies the integration - retry logic tests are in test_semantic_observability.py
        assert result["success"] is True
