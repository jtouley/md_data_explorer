"""
Tests for LLM-constrained QueryPlan generation (Phase 5.1).

Ensures:
- LLM returns JSON matching QueryPlan schema
- Malformed plans are rejected with clear errors
- All required fields are present
- Field values match allowed types/literals
"""

import json
from unittest.mock import MagicMock, patch

from clinical_analytics.core.nl_query_engine import NLQueryEngine


class TestLLMQueryPlanSchema:
    """Test that LLM returns JSON matching QueryPlan schema."""

    def test_llm_prompt_specifies_queryplan_schema(self, mock_semantic_layer):
        """LLM prompt should specify QueryPlan JSON schema (Phase 5.1)."""
        # Arrange: Mock semantic layer
        mock = mock_semantic_layer(columns={"age": "Patient Age"})
        engine = NLQueryEngine(mock)

        # Build context for LLM
        context = {
            "columns": ["age"],
            "aliases": {},
            "examples": [],
        }

        # Act: Build LLM prompt
        system_prompt, user_prompt = engine._build_llm_prompt("test query", context)

        # Assert: Prompt should use QueryPlan schema fields (NOT QueryIntent fields)
        # QueryPlan fields: intent, metric, group_by
        # QueryIntent fields (legacy): intent_type, primary_variable, grouping_variable
        assert "- intent:" in system_prompt or "intent:" in system_prompt, (
            "Prompt should specify 'intent' field (QueryPlan schema)"
        )
        assert "- metric:" in system_prompt or "metric:" in system_prompt, (
            "Prompt should specify 'metric' field (QueryPlan schema)"
        )
        assert "- group_by:" in system_prompt or "group_by:" in system_prompt, (
            "Prompt should specify 'group_by' field (QueryPlan schema)"
        )

        # Verify QueryPlan schema is being used by checking field list
        assert "- intent: One of" in system_prompt, "Should specify intent field with allowed values"
        assert "- metric: Main variable" in system_prompt or "- metric:" in system_prompt, "Should specify metric field"
        assert "- group_by: Variable to group by" in system_prompt or "- group_by:" in system_prompt, (
            "Should specify group_by field"
        )

        # Should specify valid intent types
        assert "COUNT" in system_prompt or "DESCRIBE" in system_prompt, "Prompt should list valid intents"

    def test_llm_response_parses_to_queryplan_compatible_dict(self, mock_semantic_layer):
        """LLM JSON response should parse QueryPlan schema (Phase 5.1)."""
        # Arrange
        mock = mock_semantic_layer(columns={"age": "Patient Age"})
        engine = NLQueryEngine(mock)

        # Phase 5.1: Valid LLM JSON response using QueryPlan schema fields
        llm_response = json.dumps(
            {
                "intent": "DESCRIBE",  # QueryPlan field (not intent_type)
                "metric": "age",  # QueryPlan field (not primary_variable)
                "group_by": None,  # QueryPlan field (not grouping_variable)
                "filters": [],  # QueryPlan field
                "confidence": 0.9,
                "explanation": "Describe age statistics",
            }
        )

        # Act: Extract QueryIntent from LLM response
        result = engine._extract_query_intent_from_llm_response(llm_response)

        # Assert: Should parse successfully and convert to QueryIntent
        assert result is not None, "Valid QueryPlan JSON should parse successfully"
        assert result.intent_type == "DESCRIBE", "Should convert QueryPlan.intent to QueryIntent.intent_type"
        assert result.primary_variable == "age", "Should convert QueryPlan.metric to QueryIntent.primary_variable"
        assert result.grouping_variable is None, "Should convert QueryPlan.group_by to QueryIntent.grouping_variable"
        assert result.confidence == 0.9

    def test_llm_rejects_invalid_intent_type(self, mock_semantic_layer):
        """LLM should reject responses with invalid intent types (Phase 5.1)."""
        # Arrange
        mock = mock_semantic_layer(columns={"age": "Patient Age"})
        engine = NLQueryEngine(mock)

        # Phase 5.1: Invalid intent (using QueryPlan schema)
        llm_response = json.dumps(
            {
                "intent": "INVALID_INTENT",  # Invalid value
                "metric": None,
                "group_by": None,
                "filters": [],
                "confidence": 0.9,
                "explanation": "",
            }
        )

        # Act: Extract QueryIntent
        result = engine._extract_query_intent_from_llm_response(llm_response)

        # Assert: Should return None (QueryPlan.from_dict() validation failed)
        # Falls back to legacy format, which also fails
        assert result is None, "Invalid intent should be rejected by QueryPlan.from_dict() validation"

    def test_llm_rejects_malformed_json(self, mock_semantic_layer):
        """LLM should reject malformed JSON responses."""
        # Arrange
        mock = mock_semantic_layer(columns={"age": "Patient Age"})
        engine = NLQueryEngine(mock)

        # Malformed JSON
        llm_response = "{ invalid json }"

        # Act: Extract QueryIntent
        result = engine._extract_query_intent_from_llm_response(llm_response)

        # Assert: Should return None for malformed JSON
        assert result is None, "Malformed JSON should be rejected"

    def test_llm_rejects_missing_required_fields(self, mock_semantic_layer):
        """LLM should reject responses missing required fields (Phase 5.1)."""
        # Arrange
        mock = mock_semantic_layer(columns={"age": "Patient Age"})
        engine = NLQueryEngine(mock)

        # Phase 5.1: Missing required 'intent' field
        llm_response = json.dumps(
            {
                "metric": "age",
                "group_by": None,
                "confidence": 0.9,
                # Missing 'intent' - required field
            }
        )

        # Act: Extract QueryIntent
        result = engine._extract_query_intent_from_llm_response(llm_response)

        # Assert: Should return None (QueryPlan.from_dict() requires 'intent')
        assert result is None, "Missing required 'intent' field should be rejected"

    def test_llm_rejects_nested_query_object_structure(self, mock_semantic_layer):
        """LLM should reject nested 'query' object structure (actual failure from logs)."""
        # Arrange
        mock = mock_semantic_layer(columns={"statin_used": "Statin Used"})
        engine = NLQueryEngine(mock)

        # Actual invalid response from logs: { "query": { "remove": [...], "recalc": true } }
        # This is the EXACT anti-pattern that caused the regression
        llm_response = json.dumps(
            {
                "query": {  # WRONG: Nested structure
                    "remove": ["{n/a}"],
                    "recalc": True,
                }
            }
        )

        # Act: Extract QueryIntent
        result = engine._extract_query_intent_from_llm_response(llm_response)

        # Assert: Should return None (no 'intent' field at top level)
        assert result is None, (
            "Nested 'query' object structure should be rejected - missing required 'intent' field at top level"
        )


class TestLLMConstrainedOutput:
    """Test that LLM output is constrained to QueryPlan schema."""

    @patch("clinical_analytics.core.ollama_manager.get_ollama_manager")
    def test_llm_fallback_returns_valid_queryplan_compatible_dict(self, mock_get_manager, mock_semantic_layer):
        """LLM fallback should return QueryPlan JSON schema (Phase 5.1)."""
        # Arrange: Mock Ollama manager and client
        mock_manager = MagicMock()
        mock_client = MagicMock()

        # Configure mock client to return QueryPlan schema JSON
        mock_client.is_available.return_value = True
        # Phase 5.1: Return QueryPlan schema fields
        mock_client.generate.return_value = json.dumps(
            {
                "intent": "DESCRIBE",  # QueryPlan field
                "metric": "age",  # QueryPlan field
                "group_by": None,  # QueryPlan field
                "filters": [],
                "confidence": 0.85,
                "explanation": "Describe age statistics",
            }
        )

        mock_manager.get_client.return_value = mock_client
        mock_get_manager.return_value = mock_manager

        mock = mock_semantic_layer(columns={"age": "Patient Age"})
        engine = NLQueryEngine(mock)

        # Act: Parse query with no pattern match (trigger LLM fallback)
        # Use a query that references columns not in semantic layer to force LLM fallback
        query = "analyze xyz123"  # No pattern match, will use LLM
        intent = engine.parse_query(query)

        # Assert: Should return valid QueryIntent (converted from validated QueryPlan)
        # LLM is mocked to return "age" as metric, which engine uses
        assert intent is not None
        assert intent.intent_type == "DESCRIBE"  # From mocked LLM response
        assert intent.confidence == 0.85  # From mocked LLM response
        assert 0.0 <= intent.confidence <= 1.0

    def test_confidence_clamped_to_valid_range(self, mock_semantic_layer):
        """Confidence values should be clamped to [0.0, 1.0] (Phase 5.1)."""
        # Arrange
        mock = mock_semantic_layer(columns={"age": "Patient Age"})
        engine = NLQueryEngine(mock)

        # Phase 5.1: Confidence > 1.0 (using QueryPlan schema)
        llm_response = json.dumps(
            {
                "intent": "DESCRIBE",
                "metric": None,
                "group_by": None,
                "filters": [],
                "confidence": 1.5,  # Invalid: > 1.0
                "explanation": "",
            }
        )

        # Act
        result = engine._extract_query_intent_from_llm_response(llm_response)

        # Assert: Confidence should be clamped to 1.0 by QueryPlan.from_dict()
        assert result is not None
        assert result.confidence == 1.0, "Confidence should be clamped to 1.0"

        # Phase 5.1: Confidence < 0.0 (using QueryPlan schema)
        llm_response = json.dumps(
            {
                "intent": "DESCRIBE",
                "metric": None,
                "group_by": None,
                "filters": [],
                "confidence": -0.5,  # Invalid: < 0.0
                "explanation": "",
            }
        )

        result = engine._extract_query_intent_from_llm_response(llm_response)

        assert result is not None
        assert result.confidence == 0.0, "Confidence should be clamped to 0.0"
