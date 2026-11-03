"""
Tests for QueryService - Pure Python query execution service.

Tests verify UI-agnostic query parsing, validation, and execution.
"""

from unittest.mock import MagicMock

import pytest

from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.query_service import QueryResult, QueryService


class TestQueryService:
    """Test suite for QueryService."""

    def test_query_service_initializes_with_semantic_layer(self, mock_semantic_layer):
        """Test that QueryService initializes with semantic layer."""
        # Arrange & Act
        service = QueryService(mock_semantic_layer)

        # Assert
        assert service.semantic_layer is mock_semantic_layer
        assert service.nl_engine is not None

    def test_query_service_ask_parses_query(self, mock_semantic_layer):
        """Test that ask() parses query using NLQueryEngine."""
        # Arrange
        mock_semantic_layer = mock_semantic_layer()
        mock_semantic_layer.format_execution_result.return_value = {"type": "descriptive", "mean": 45.5}
        service = QueryService(mock_semantic_layer)
        query = "what is the average age?"

        # Mock NLQueryEngine.parse_query to return QueryIntent
        mock_intent = MagicMock()
        mock_intent.intent_type = "DESCRIBE"
        mock_intent.primary_variable = "age"
        mock_intent.confidence = 0.9
        mock_intent.filters = []
        mock_intent.interpretation = "Calculate descriptive statistics for age"
        mock_intent.confidence_explanation = "High confidence pattern match"

        # Mock NLQueryEngine to return QueryPlan
        mock_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,
            explanation="Calculate descriptive statistics for age",
        )
        service.nl_engine.parse_query = MagicMock(return_value=mock_intent)
        service.nl_engine._intent_to_plan = MagicMock(return_value=mock_plan)

        # Mock semantic layer methods
        mock_semantic_layer._generate_run_key = MagicMock(return_value="test_run_key_123")
        mock_semantic_layer._validate_query_plan = MagicMock(return_value={"valid": True, "issues": []})
        mock_semantic_layer.execute_query_plan = MagicMock(return_value={"headline": "Average age: 45.5"})

        # Act
        result = service.ask(query, dataset_id="test_dataset", dataset_version="v1")

        # Assert
        assert isinstance(result, QueryResult)
        assert result.plan.intent == "DESCRIBE"
        assert result.plan.metric == "age"
        assert result.confidence == 0.9
        assert result.run_key == "test_run_key_123"
        assert result.issues == []
        assert result.result is not None

    def test_query_service_ask_generates_run_key(self, mock_semantic_layer):
        """Test that ask() generates deterministic run_key."""
        # Arrange
        mock_semantic_layer = mock_semantic_layer()  # factory fixture: build the layer
        service = QueryService(mock_semantic_layer)
        query = "what is the average age?"

        # Mock NLQueryEngine
        mock_intent = MagicMock()
        mock_intent.intent_type = "DESCRIBE"
        mock_intent.primary_variable = "age"
        mock_intent.confidence = 0.9
        mock_intent.filters = []
        mock_plan = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.9)
        service.nl_engine.parse_query = MagicMock(return_value=mock_intent)
        service.nl_engine._intent_to_plan = MagicMock(return_value=mock_plan)

        # Mock semantic layer
        mock_semantic_layer._generate_run_key = MagicMock(return_value="deterministic_key")
        mock_semantic_layer._validate_query_plan = MagicMock(return_value={"valid": True, "issues": []})
        mock_semantic_layer.execute_query_plan = MagicMock(return_value={"headline": "Result"})

        # Act
        result = service.ask(query, dataset_id="test_dataset", dataset_version="v1")

        # Assert
        assert result.run_key == "deterministic_key"
        assert result is not None
        # Verify _generate_run_key was called with plan and normalized query
        mock_semantic_layer._generate_run_key.assert_called_once()
        call_args = mock_semantic_layer._generate_run_key.call_args
        assert call_args[0][0] == mock_plan  # First arg is plan
        assert call_args[0][1] is not None  # Second arg is normalized query

    def test_query_service_ask_returns_validation_issues(self, mock_semantic_layer):
        """Test that ask() returns validation issues when plan is invalid."""
        # Arrange
        service = QueryService(mock_semantic_layer)
        query = "compare groups"

        # Mock NLQueryEngine
        mock_intent = MagicMock()
        mock_intent.intent_type = "COMPARE_GROUPS"
        mock_intent.confidence = 0.5
        mock_intent.filters = []
        mock_plan = QueryPlan(intent="COMPARE_GROUPS", confidence=0.5)
        service.nl_engine.parse_query = MagicMock(return_value=mock_intent)
        service.nl_engine._intent_to_plan = MagicMock(return_value=mock_plan)

        # Mock validation to return issues
        mock_semantic_layer._generate_run_key = MagicMock(return_value="test_key")
        mock_semantic_layer._validate_query_plan = MagicMock(
            return_value={
                "valid": False,
                "issues": [
                    {"message": "Missing grouping variable", "severity": "error"},
                    {"message": "Low confidence", "severity": "warning"},
                ],
            }
        )

        # Act
        result = service.ask(query, dataset_id="test_dataset", dataset_version="v1")

        # Assert
        assert len(result.issues) == 2
        assert result.issues[0]["message"] == "Missing grouping variable"
        assert result.issues[1]["severity"] == "warning"
        assert result.result is None  # No execution when validation fails

    def test_query_service_ask_handles_parse_failure(self, mock_semantic_layer):
        """Test that ask() handles query parse failure gracefully."""
        # Arrange
        service = QueryService(mock_semantic_layer)
        query = "invalid query xyz"

        # Mock NLQueryEngine to return None (parse failure)
        service.nl_engine.parse_query = MagicMock(return_value=None)

        # Act
        result = service.ask(query, dataset_id="test_dataset", dataset_version="v1")

        # Assert
        assert result.plan is None
        assert len(result.issues) > 0
        assert any(
            "parse" in issue["message"].lower() or "failed" in issue["message"].lower() for issue in result.issues
        )
        assert result.result is None

    def test_query_service_ask_normalizes_query(self, mock_semantic_layer):
        """Test that ask() normalizes query before parsing."""
        # Arrange
        mock_semantic_layer = mock_semantic_layer()  # factory fixture: build the layer
        service = QueryService(mock_semantic_layer)
        query = "  What  IS   the  Average  AGE?  "  # Has extra spaces and mixed case

        # Mock NLQueryEngine
        mock_intent = MagicMock()
        mock_intent.intent_type = "DESCRIBE"
        mock_intent.primary_variable = "age"
        mock_intent.confidence = 0.9
        mock_intent.filters = []
        mock_plan = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.9)
        service.nl_engine.parse_query = MagicMock(return_value=mock_intent)
        service.nl_engine._intent_to_plan = MagicMock(return_value=mock_plan)

        # Mock semantic layer
        mock_semantic_layer._generate_run_key = MagicMock(return_value="test_key")
        mock_semantic_layer._validate_query_plan = MagicMock(return_value={"valid": True, "issues": []})
        mock_semantic_layer.execute_query_plan = MagicMock(return_value={"headline": "Result"})

        # Act
        result = service.ask(query, dataset_id="test_dataset", dataset_version="v1")

        # Assert
        assert result is not None
        assert result.plan is not None
        # Verify parse_query was called with normalized query (lowercase, collapsed whitespace)
        call_args = service.nl_engine.parse_query.call_args
        normalized_query = call_args[0][0]  # First positional arg
        assert normalized_query == "what is the average age?"
        assert normalized_query == normalized_query.lower()
        assert "  " not in normalized_query  # No double spaces

    def test_query_service_generate_run_key_delegates_to_semantic_layer(self, mock_semantic_layer):
        """Test that _generate_run_key delegates to semantic_layer._generate_run_key."""
        # Arrange
        service = QueryService(mock_semantic_layer)
        query = "test query"
        plan = QueryPlan(intent="DESCRIBE", metric="age")

        # Mock semantic layer method
        mock_semantic_layer._generate_run_key = MagicMock(return_value="delegated_key")

        # Act
        run_key = service._generate_run_key(plan, query)

        # Assert
        assert run_key == "delegated_key"
        mock_semantic_layer._generate_run_key.assert_called_once_with(plan, query)


class TestQueryServiceAnalysis:
    """ask() runs the statistics for the plan via SemanticLayer.format_execution_result."""

    def _service(self, mock_semantic_layer, execution_result, formatted=None):
        layer = mock_semantic_layer()
        layer._generate_run_key.return_value = "run_key"
        layer._validate_query_plan.return_value = {"valid": True, "issues": []}
        layer.execute_query_plan.return_value = execution_result
        layer.format_execution_result.return_value = formatted
        service = QueryService(layer)
        intent = MagicMock(
            intent_type="COMPARE_GROUPS",
            primary_variable="ldl",
            confidence=0.95,
            filters=[],
            predictor_variables=[],
            time_variable=None,
            event_variable=None,
        )
        service.nl_engine.parse_query = MagicMock(return_value=intent)
        service.nl_engine._intent_to_plan = MagicMock(
            return_value=QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="treatment", confidence=0.95)
        )
        return service, layer

    @pytest.mark.parametrize(
        ("formatted", "expected_errors"),
        [
            ({"type": "comparison", "p_value": 0.01}, []),
            (
                {"type": "predictor", "error": "No predictors specified for outcome."},
                ["No predictors specified for outcome."],
            ),
        ],
        ids=["analysis_ok", "analysis_error_becomes_issue"],
    )
    def test_ask_executedPlan_attachesFormatterAnalysis(self, mock_semantic_layer, formatted, expected_errors):
        # Arrange
        execution_result = {"success": True, "result": "rows"}
        service, layer = self._service(mock_semantic_layer, execution_result, formatted)

        # Act
        result = service.ask("compare ldl by treatment", dataset_id="ds")

        # Assert
        assert result.analysis == formatted
        assert layer.format_execution_result.call_args.args[0] == execution_result
        assert layer.format_execution_result.call_args.args[1] is result.context
        assert [i["message"] for i in result.issues if i["severity"] == "error"] == expected_errors

    def test_ask_failedExecution_skipsFormatter(self, mock_semantic_layer):
        # Arrange
        service, layer = self._service(mock_semantic_layer, {"success": False, "warnings": ["boom"]})

        # Act
        result = service.ask("compare ldl by treatment", dataset_id="ds")

        # Assert
        layer.format_execution_result.assert_not_called()
        assert result.analysis is None
        assert {"message": "boom", "severity": "error"} in result.issues

    @pytest.mark.parametrize("threshold", [0.5, 0.95], ids=["lenient", "strict"])
    def test_ask_execution_usesConfiguredConfidenceThreshold(self, mock_semantic_layer, monkeypatch, threshold):
        """The hold threshold comes from config/nl_query.yaml, not execute_query_plan's default."""
        # Arrange
        from clinical_analytics.core import query_service as query_service_module

        monkeypatch.setattr(query_service_module, "AUTO_EXECUTE_CONFIDENCE_THRESHOLD", threshold)
        service, layer = self._service(mock_semantic_layer, {"success": True, "run_key": "rk"}, {"type": "comparison"})

        # Act
        service.ask("compare ldl by treatment", dataset_id="ds")

        # Assert
        assert layer.execute_query_plan.call_args.kwargs["confidence_threshold"] == threshold

    @pytest.mark.parametrize(
        ("confirm_plan_line", "expected_confirmed", "expected_reason"),
        [
            ("SHOWN", True, None),
            ("COMPARE_GROUPS · outcome hdl · grouping sex", False, "changed"),
            (None, False, "held"),
            ("", False, "held"),
        ],
        ids=["same_plan_released", "different_plan_held_again", "no_plan_line_not_released", "empty_plan_line"],
    )
    def test_ask_confirmPlanLine_releasesHoldOnlyForTheConfirmedPlan(
        self, mock_semantic_layer, confirm_plan_line, expected_confirmed, expected_reason
    ):
        """Confirm covers exactly the plan the user saw; anything else stays held."""
        # Arrange
        from clinical_analytics.core.execution_gate import HOLD_FAILURE_REASON, PLAN_CHANGED_REASON, format_plan_line

        held = {"success": False, "requires_confirmation": True, "failure_reason": HOLD_FAILURE_REASON}
        service, layer = self._service(mock_semantic_layer, held, None)
        plan = service.nl_engine._intent_to_plan.return_value
        shown = format_plan_line(plan, service._intent_to_context(service.nl_engine.parse_query.return_value, plan))
        sent = shown if confirm_plan_line == "SHOWN" else confirm_plan_line

        # Act
        result = service.ask("compare ldl by treatment", dataset_id="ds", confirmed=True, confirm_plan_line=sent)

        # Assert
        if expected_reason == "changed":
            # A different plan is held before execution (see the high-confidence test below).
            layer.execute_query_plan.assert_not_called()
            assert result.requires_confirmation is True
        else:
            assert layer.execute_query_plan.call_args.kwargs["confirmed"] is expected_confirmed
        if expected_reason == "changed":
            assert result.failure_reason == PLAN_CHANGED_REASON
            assert result.plan_line == shown
        elif expected_reason == "held":
            assert result.failure_reason == HOLD_FAILURE_REASON

    @pytest.mark.parametrize("confidence", [0.95, 0.75, 1.0], ids=["high", "at_threshold", "certain"])
    def test_ask_changedPlanAtHighConfidence_staysHeldWithoutExecuting(self, mock_semantic_layer, confidence):
        """A confirmation for one plan never runs a different reparse, however confident."""
        # Arrange
        from clinical_analytics.core.execution_gate import PLAN_CHANGED_REASON, format_plan_line

        service, layer = self._service(mock_semantic_layer, {"success": True, "run_key": "rk"}, {"type": "comparison"})
        plan = QueryPlan(intent="FIND_PREDICTORS", metric="ldl", group_by="treatment", confidence=confidence)
        service.nl_engine._intent_to_plan.return_value = plan
        shown = format_plan_line(plan, service._intent_to_context(service.nl_engine.parse_query.return_value, plan))
        confirmed_line = "COMPARE_GROUPS · outcome ldl · grouping treatment"

        # Act
        result = service.ask(
            "compare ldl by treatment", dataset_id="ds", confirmed=True, confirm_plan_line=confirmed_line
        )

        # Assert
        assert result.requires_confirmation is True
        assert result.failure_reason == PLAN_CHANGED_REASON
        assert result.plan_line == shown
        assert result.claim is None
        assert result.analysis is None
        assert result.run_key is None
        layer.execute_query_plan.assert_not_called()
        layer.format_execution_result.assert_not_called()
