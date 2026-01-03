"""
Tests for QueryService - Pure Python query execution service.

Tests verify UI-agnostic query parsing, validation, and execution.
"""

from unittest.mock import MagicMock

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
