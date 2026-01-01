"""
LLM-powered filter extraction (ADR009 Phase 5).

This module provides LLM-powered extraction of complex filter patterns that
regex struggles with, such as "get rid of the n/a" or "exclude missing values".

Key functions:
- _extract_filters_with_llm: Extract filters using LLM with independent validation
- _validate_filter: Validate individual filter against semantic layer
"""

from typing import Any

import structlog

from clinical_analytics.core.llm_feature import LLMFeature, call_llm
from clinical_analytics.core.nl_query_config import LLM_TIMEOUT_FILTER_EXTRACTION_S, LLM_TIMEOUT_MAX_S
from clinical_analytics.core.query_plan import FilterSpec

logger = structlog.get_logger()

# Valid operators (from FilterSpec Literal type)
VALID_OPERATORS = {"==", "!=", ">", ">=", "<", "<=", "IN", "NOT_IN"}


def _validate_filter(
    filter_dict: dict[str, Any],
    semantic_layer: Any,
) -> tuple[bool, str | None]:
    """
    Validate a single filter against semantic layer.

    Validates:
    - Column exists in semantic layer
    - Operator is valid for FilterSpec
    - Value type matches column type (if metadata available)
    - FilterSpec construction succeeds

    Args:
        filter_dict: Dictionary with column, operator, value keys
        semantic_layer: SemanticLayer instance for validation

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if filter is valid
        - error_message: None if valid, error description if invalid
    """
    # Check required fields
    if "column" not in filter_dict:
        return False, "Missing 'column' field"
    if "operator" not in filter_dict:
        return False, "Missing 'operator' field"
    if "value" not in filter_dict:
        return False, "Missing 'value' field"

    column = filter_dict["column"]
    operator = filter_dict["operator"]
    value = filter_dict["value"]

    # Validate operator
    if operator not in VALID_OPERATORS:
        return False, f"Invalid operator '{operator}'. Must be one of {VALID_OPERATORS}"

    # Check column exists in semantic layer
    try:
        view = semantic_layer.get_base_view()
        available_columns = set(view.columns)
    except Exception as e:
        logger.warning("filter_validation_column_check_failed", column=column, error=str(e))
        return False, f"Could not check column existence: {str(e)}"

    if column not in available_columns:
        return False, f"Column '{column}' not found in dataset"

    # Try to construct FilterSpec (validates value type compatibility)
    try:
        FilterSpec(
            column=column,
            operator=operator,
            value=value,
            exclude_nulls=filter_dict.get("exclude_nulls", True),
        )
    except (TypeError, ValueError) as e:
        return False, f"FilterSpec construction failed: {str(e)}"

    # All validations passed
    return True, None


def _extract_filters_with_llm(
    query: str,
    semantic_layer: Any,
    current_confidence: float = 1.0,
) -> tuple[list[FilterSpec], float, list[str]]:
    """
    Extract filter conditions from query using LLM (ADR009 Phase 5).

    Uses LLM to extract complex filter patterns that regex struggles with,
    such as "get rid of the n/a" or "exclude missing values".

    **Critical validation requirement**: Each filter is validated independently.
    - Apply only valid filters (not all-or-nothing)
    - Log invalid filters with reasons
    - Reduce confidence if any invalid filters detected

    Args:
        query: User's natural language query
        semantic_layer: SemanticLayer instance for column validation
        current_confidence: Current query plan confidence (reduced if invalid filters)

    Returns:
        Tuple of (valid_filters, confidence_delta, validation_failures)
        - valid_filters: List of validated FilterSpec objects
        - confidence_delta: Confidence adjustment (negative if invalid filters)
        - validation_failures: List of error messages for invalid filters

    Example:
        >>> filters, delta, failures = _extract_filters_with_llm(
        ...     "get rid of the n/a",
        ...     semantic_layer,
        ...     current_confidence=0.8
        ... )
        >>> # Returns filters excluding n/a (code 0), delta=0 if all valid, failures=[]
    """
    # Enforce timeout cap
    timeout_s = min(LLM_TIMEOUT_FILTER_EXTRACTION_S, LLM_TIMEOUT_MAX_S)

    # Build prompt with filter extraction instructions
    system_prompt = """You are an expert at extracting filter conditions from clinical queries.

Your task: Extract filter conditions and return them as JSON.

Output format:
{
  "filters": [
    {
      "column": "exact_column_name_from_dataset",
      "operator": "==" | "!=" | ">" | ">=" | "<" | "<=" | "IN" | "NOT_IN",
      "value": value_matching_column_type,
      "exclude_nulls": true | false
    }
  ]
}

CRITICAL RULES:
1. Column names MUST be exact matches from the available columns list
2. For coded/categorical columns (like "Statin Used: 0=n/a, 1=Atorvastatin..."):
   - Use the NUMERIC CODE as the value (e.g., 0 for n/a)
   - "remove the n/a" → value: 0 (not "n/a" string)
   - "exclude missing" → value: 0 (not "missing" string)
3. For numeric columns, use numeric values (not strings)
4. When user says "exclude" or "remove", use operator "!="
5. When user says "only" or "just", use operator "=="
6. Infer column from context if value reference like "n/a" lacks explicit column name
7. Return empty array if no clear filters can be extracted

Examples:
- Query: "remove the n/a"
  Context: "Statin Used: 0=n/a, 1=Atorvastatin, 2=Rosuvastatin"
  Result: {"column": "Statin Used", "operator": "!=", "value": 0}

- Query: "exclude missing values"
  Context: Column "Status" has codes 0=unknown
  Result: {"column": "Status", "operator": "!=", "value": 0}

- Query: "only on statins"
  Context: Column "Statin Prescribed?" with codes 0=no, 1=yes
  Result: {"column": "Statin Prescribed?", "operator": "==", "value": 1}"""

    # Get available columns with their metadata for context
    try:
        view = semantic_layer.get_base_view()
        all_columns = list(view.columns)

        # Build column descriptions including any metadata about coded values
        column_descriptions = []
        for col in all_columns[:30]:  # Limit for prompt size
            try:
                col_metadata = semantic_layer.get_column_metadata(col)
                col_type = col_metadata.get("type", "unknown")

                # If this is a categorical column with codes, include the code descriptions
                if col_type == "categorical" and ":" in col:
                    column_descriptions.append(f"  - {col}")
                else:
                    column_descriptions.append(f"  - {col} ({col_type})")
            except Exception:
                column_descriptions.append(f"  - {col}")

        columns_context = "\n".join(column_descriptions)
    except Exception:
        columns_context = "columns not available"

    user_prompt = f"""Extract filter conditions from this clinical query:

Query: "{query}"

Available columns in the dataset:
{columns_context}

IMPORTANT: Look for keywords like "remove", "exclude", "only", "get rid of", "without" which indicate filtering intent.
For vague references like "the n/a" or "missing values", infer the most relevant column from the context.

Return JSON with "filters" array. Use EXACT column names from the list above."""

    # Call LLM
    result = call_llm(
        feature=LLMFeature.FILTER_EXTRACTION,
        system=system_prompt,
        user=user_prompt,
        timeout_s=timeout_s,
    )

    # Handle LLM failures (graceful degradation)
    if result.error or result.payload is None:
        logger.debug(
            "filter_extraction_llm_failed",
            query=query[:100],  # Truncate for logging
            error=result.error,
            timed_out=result.timed_out,
        )
        return [], 0.0, []

    # Extract filters from payload
    filters_data = result.payload.get("filters", [])
    if not isinstance(filters_data, list):
        logger.warning(
            "filter_extraction_invalid_payload",
            query=query[:100],
            payload_type=type(filters_data).__name__,
        )
        return [], 0.0, []

    # Validate each filter independently
    valid_filters = []
    validation_failures = []
    confidence_delta = 0.0

    for filter_dict in filters_data:
        is_valid, error_msg = _validate_filter(filter_dict, semantic_layer)

        if is_valid:
            # Construct FilterSpec
            try:
                filter_spec = FilterSpec(
                    column=filter_dict["column"],
                    operator=filter_dict["operator"],
                    value=filter_dict["value"],
                    exclude_nulls=filter_dict.get("exclude_nulls", True),
                )
                valid_filters.append(filter_spec)
            except (TypeError, ValueError) as e:
                # Construction failed even though validation passed (edge case)
                validation_failures.append(f"FilterSpec construction failed: {str(e)}")
                confidence_delta -= 0.1
        else:
            # Invalid filter - log and reduce confidence
            validation_failures.append(f"Invalid filter: {error_msg}")
            confidence_delta -= 0.1

    # If any invalid filters, reduce confidence (min 0.6 per plan requirement)
    if validation_failures:
        logger.warning(
            "filter_extraction_validation_failures",
            query=query[:100],
            failure_count=len(validation_failures),
            failures=validation_failures,
        )
        # Reduce confidence: -0.1 per invalid filter
        # confidence_delta is already negative (accumulated from -0.1 per failure)
        # Per plan requirement: "Set plan.confidence = min(plan.confidence, 0.6)"
        # This means if current_confidence > 0.6, reduce to 0.6
        # Calculate the delta needed to reach 0.6 (if current > 0.6)
        min_allowed_confidence = 0.6
        if current_confidence > min_allowed_confidence:
            # Reduce to 0.6: delta = 0.6 - current_confidence (negative)
            target_delta = min_allowed_confidence - current_confidence
            # Use the more negative of: accumulated delta or target delta
            confidence_delta = min(confidence_delta, target_delta)

    logger.debug(
        "filter_extraction_completed",
        query=query[:100],
        valid_count=len(valid_filters),
        invalid_count=len(validation_failures),
        confidence_delta=confidence_delta,
    )

    return valid_filters, confidence_delta, validation_failures
