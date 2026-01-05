"""
Centralized LLM JSON parsing and validation module.

This module provides a single choke point for all LLM JSON parsing and validation,
preventing duplicated brittle parsing logic across features.

Key functions:
- parse_json_response: Parse raw LLM text into Python dict/list
- validate_shape: Validate parsed payload against known schemas

Design principles:
- Single source of truth for JSON parsing
- Graceful degradation (return None on failures, never crash)
- Standardized error logging
- Schema validation with clear error messages
"""

import json
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of schema validation."""

    valid: bool
    errors: list[str]


def parse_json_response(raw: str | None) -> dict[str, Any] | list[Any] | None:
    """
    Parse raw LLM response into Python dict or list.

    This is the single choke point for all LLM JSON parsing. All LLM features
    must use this function instead of calling json.loads() directly.

    Args:
        raw: Raw text from LLM response (may be None, empty, or malformed)

    Returns:
        Parsed dict/list if valid JSON, None otherwise

    Examples:
        >>> parse_json_response('{"intent": "DESCRIBE"}')
        {'intent': 'DESCRIBE'}
        >>> parse_json_response('not json')
        None
        >>> parse_json_response(None)
        None
    """
    if raw is None or raw == "":
        logger.debug("llm_json_parse_empty", raw=raw)
        return None

    try:
        parsed = json.loads(raw)
        logger.debug("llm_json_parse_success", length=len(str(parsed)))
        from typing import cast

        return cast(dict[str, Any] | list[Any], parsed)
    except json.JSONDecodeError as e:
        logger.warning(
            "llm_json_parse_failed",
            error=str(e),
            raw_length=len(raw),
            raw_preview=raw[:100] if len(raw) > 100 else raw,
        )
        return None
    except Exception as e:
        logger.error(
            "llm_json_parse_unexpected_error",
            error_type=type(e).__name__,
            error=str(e),
        )
        return None


# Schema definitions
# Each schema defines required fields and their expected types
_SCHEMAS: dict[str, dict[str, Any]] = {
    "queryplan": {
        "required_fields": ["intent"],
        "optional_fields": ["metric", "group_by", "filters", "confidence", "explanation"],
        "field_types": {
            "intent": str,
            "metric": (str, type(None)),
            "group_by": (str, type(None)),
            "filters": list,
            "confidence": (int, float),
            "explanation": str,
        },
    },
    "followups": {
        "required_fields": ["follow_ups"],
        "optional_fields": ["follow_up_explanation"],
        "field_types": {
            "follow_ups": list,
            "follow_up_explanation": str,
        },
    },
    "interpretation": {
        "required_fields": [],
        "optional_fields": ["interpretation", "confidence_explanation"],
        "field_types": {
            "interpretation": str,
            "confidence_explanation": str,
        },
    },
    "filters": {
        "is_array": True,
        "array_item_required_fields": ["column", "operator", "value"],
        "array_item_optional_fields": ["exclude_nulls"],
        "array_item_field_types": {
            "column": str,
            "operator": str,
            "value": (str, int, float, list),
            "exclude_nulls": bool,
        },
    },
}


def validate_shape(payload: dict[str, Any] | list[Any] | None, schema_name: str) -> ValidationResult:
    """
    Validate parsed JSON payload against expected schema.

    Args:
        payload: Parsed JSON (dict or list)
        schema_name: Name of schema to validate against (e.g., "queryplan", "followups")

    Returns:
        ValidationResult with valid flag and error list

    Examples:
        >>> result = validate_shape({"intent": "DESCRIBE"}, "queryplan")
        >>> result.valid
        True
        >>> result = validate_shape({"metric": "age"}, "queryplan")
        >>> result.valid
        False
        >>> "intent" in result.errors[0].lower()
        True
    """
    if payload is None:
        return ValidationResult(valid=False, errors=["Payload is None"])

    if schema_name not in _SCHEMAS:
        return ValidationResult(
            valid=False,
            errors=[f"Unknown schema: {schema_name}. Available schemas: {list(_SCHEMAS.keys())}"],
        )

    schema = _SCHEMAS[schema_name]
    errors: list[str] = []

    # Handle array schemas (like filters)
    if schema.get("is_array"):
        if not isinstance(payload, list):
            errors.append(f"Expected array for schema '{schema_name}', got {type(payload).__name__}")
            return ValidationResult(valid=False, errors=errors)

        # Validate each array item
        required_fields = schema.get("array_item_required_fields", [])
        field_types = schema.get("array_item_field_types", {})

        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                errors.append(f"Array item {idx} is not a dict: {type(item).__name__}")
                continue

            # Check required fields in item
            for field in required_fields:
                if field not in item:
                    errors.append(f"Array item {idx} missing required field: {field}")

            # Check types for all fields in item
            for field, expected_type in field_types.items():
                if field in item and not isinstance(item[field], expected_type):
                    errors.append(
                        f"Array item {idx} field '{field}' has wrong type: "
                        f"expected {expected_type}, got {type(item[field]).__name__}"
                    )

    else:
        # Handle dict schemas (like queryplan, followups, interpretation)
        if not isinstance(payload, dict):
            errors.append(f"Expected dict for schema '{schema_name}', got {type(payload).__name__}")
            return ValidationResult(valid=False, errors=errors)

        required_fields = schema.get("required_fields", [])
        field_types = schema.get("field_types", {})

        # Check required fields
        for field in required_fields:
            if field not in payload:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, expected_type in field_types.items():
            if field in payload:
                if not isinstance(payload[field], expected_type):
                    errors.append(
                        f"Field '{field}' has wrong type: expected {expected_type}, got {type(payload[field]).__name__}"
                    )

    if errors:
        logger.warning(
            "llm_json_validation_failed",
            schema=schema_name,
            errors=errors,
            payload_keys=list(payload.keys()) if isinstance(payload, dict) else f"array[{len(payload)}]",
        )
        return ValidationResult(valid=False, errors=errors)

    logger.debug("llm_json_validation_success", schema=schema_name)
    return ValidationResult(valid=True, errors=[])
