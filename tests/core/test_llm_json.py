"""
Tests for centralized LLM JSON parsing and validation module.

Tests cover:
- JSON parsing from raw LLM responses (malformed, valid, nested)
- Schema validation against expected shapes
- Error handling and graceful degradation
- Edge cases: empty responses, non-JSON text, partial JSON
"""

from clinical_analytics.core.llm_json import (
    ValidationResult,
    parse_json_response,
    validate_shape,
)


class TestParseJsonResponse:
    """Test JSON parsing from raw LLM responses."""

    def test_parse_valid_json_returns_dict(self):
        # Arrange
        raw = '{"intent": "DESCRIBE", "metric": "age", "confidence": 0.9}'

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is not None
        assert isinstance(result, dict)
        assert result["intent"] == "DESCRIBE"
        assert result["metric"] == "age"
        assert result["confidence"] == 0.9

    def test_parse_malformed_json_returns_none(self):
        # Arrange
        raw = '{"intent": "DESCRIBE", "metric": "age"'  # Missing closing brace

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is None

    def test_parse_empty_string_returns_none(self):
        # Arrange
        raw = ""

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is None

    def test_parse_none_input_returns_none(self):
        # Arrange
        raw = None

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is None

    def test_parse_non_json_text_returns_none(self):
        # Arrange
        raw = "This is just plain text, not JSON"

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is None

    def test_parse_json_array_returns_list(self):
        # Arrange
        raw = '[{"filter": "age"}, {"filter": "status"}]'

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_nested_json_preserves_structure(self):
        # Arrange
        raw = '{"plan": {"intent": "DESCRIBE", "filters": [{"column": "age", "op": ">=", "value": 18}]}}'

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is not None
        assert "plan" in result
        assert result["plan"]["intent"] == "DESCRIBE"
        assert len(result["plan"]["filters"]) == 1

    def test_parse_json_with_whitespace_succeeds(self):
        # Arrange
        raw = """
        {
            "intent": "DESCRIBE",
            "metric": "age"
        }
        """

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is not None
        assert result["intent"] == "DESCRIBE"

    def test_parse_json_with_unicode_succeeds(self):
        # Arrange
        raw = '{"message": "Patient improved: ✓"}'

        # Act
        result = parse_json_response(raw)

        # Assert
        assert result is not None
        assert "✓" in result["message"]


class TestValidateShape:
    """Test schema validation against expected shapes."""

    def test_validate_valid_queryplan_schema_passes(self):
        # Arrange
        payload = {
            "intent": "DESCRIBE",
            "metric": "age",
            "group_by": None,
            "filters": [],
            "confidence": 0.9,
        }
        schema_name = "queryplan"

        # Act
        result = validate_shape(payload, schema_name)

        # Assert
        assert result.valid is True
        assert result.errors == []

    def test_validate_missing_required_field_fails(self):
        # Arrange
        payload = {
            "metric": "age",
            "confidence": 0.9,
            # Missing 'intent'
        }
        schema_name = "queryplan"

        # Act
        result = validate_shape(payload, schema_name)

        # Assert
        assert result.valid is False
        assert len(result.errors) > 0
        assert any("intent" in err.lower() for err in result.errors)

    def test_validate_wrong_field_type_fails(self):
        # Arrange
        payload = {
            "intent": "DESCRIBE",
            "metric": "age",
            "confidence": "high",  # Should be float, not str
        }
        schema_name = "queryplan"

        # Act
        result = validate_shape(payload, schema_name)

        # Assert
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_followups_schema_passes(self):
        # Arrange
        payload = {
            "follow_ups": ["What predicts mortality?", "Compare by age group"],
            "follow_up_explanation": "These questions explore the data further",
        }
        schema_name = "followups"

        # Act
        result = validate_shape(payload, schema_name)

        # Assert
        assert result.valid is True

    def test_validate_unknown_schema_fails(self):
        # Arrange
        payload = {"key": "value"}
        schema_name = "unknown_schema"

        # Act
        result = validate_shape(payload, schema_name)

        # Assert
        assert result.valid is False
        assert len(result.errors) > 0
        assert "unknown" in result.errors[0].lower() or "not found" in result.errors[0].lower()

    def test_validate_none_payload_fails(self):
        # Arrange
        payload = None
        schema_name = "queryplan"

        # Act
        result = validate_shape(payload, schema_name)

        # Assert
        assert result.valid is False

    def test_validate_interpretation_schema_passes(self):
        # Arrange
        payload = {
            "interpretation": "This query analyzes age distribution",
            "confidence_explanation": "High confidence due to clear metric and intent",
        }
        schema_name = "interpretation"

        # Act
        result = validate_shape(payload, schema_name)

        # Assert
        assert result.valid is True

    def test_validate_filter_array_schema_passes(self):
        # Arrange
        payload = [
            {"column": "age", "operator": ">=", "value": 18},
            {"column": "status", "operator": "==", "value": "active"},
        ]
        schema_name = "filters"

        # Act
        result = validate_shape(payload, schema_name)

        # Assert
        assert result.valid is True


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_valid_has_no_errors(self):
        # Arrange & Act
        result = ValidationResult(valid=True, errors=[])

        # Assert
        assert result.valid is True
        assert result.errors == []

    def test_validation_result_invalid_has_errors(self):
        # Arrange & Act
        result = ValidationResult(valid=False, errors=["Missing required field 'intent'"])

        # Assert
        assert result.valid is False
        assert len(result.errors) == 1
        assert "intent" in result.errors[0]
