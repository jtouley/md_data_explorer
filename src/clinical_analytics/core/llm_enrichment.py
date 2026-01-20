"""
LLM Enrichment Suggestion Generation for ADR011.

This module generates metadata enrichment suggestions using LLM,
with privacy-safe prompt building and strict validation.

Key Design Decisions:
- Schema + profiling only (no raw data in prompts)
- All suggestions are PENDING until explicitly accepted
- Validation rejects unknown columns and operations
- Graceful degradation when LLM unavailable
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog

from clinical_analytics.core.llm_client import OllamaClient
from clinical_analytics.core.llm_json import parse_json_response, validate_shape
from clinical_analytics.core.metadata_patch import (
    ExclusionPatternPatch,
    MetadataPatch,
    PatchOperation,
    PatchStatus,
    RelationshipPatch,
)
from clinical_analytics.core.schema_inference import InferredSchema

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating LLM suggestions."""

    valid: bool
    suggestions: list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def build_enrichment_prompt(
    schema: InferredSchema,
    doc_context: str | None = None,
) -> str:
    """
    Build privacy-safe prompt for LLM enrichment suggestions.

    Args:
        schema: InferredSchema with column information
        doc_context: Optional documentation context (from PDF/text)

    Returns:
        Prompt string for LLM

    Note:
        This prompt contains ONLY schema metadata, never raw data values.
    """
    # Collect all columns
    columns = []
    if schema.patient_id_column:
        columns.append(f"- {schema.patient_id_column} (identifier)")
    for col in schema.outcome_columns:
        columns.append(f"- {col} (outcome, binary)")
    for col in schema.time_columns:
        columns.append(f"- {col} (temporal)")
    for col in schema.event_columns:
        columns.append(f"- {col} (event indicator)")
    for col in schema.categorical_columns:
        columns.append(f"- {col} (categorical)")
    for col in schema.continuous_columns:
        columns.append(f"- {col} (continuous)")

    columns_text = "\n".join(columns)

    # Add codebook information if available
    codebook_text = ""
    if schema.dictionary_metadata and schema.dictionary_metadata.codebooks:
        codebook_lines = []
        for col, codes in schema.dictionary_metadata.codebooks.items():
            code_str = ", ".join(f"{k}: {v}" for k, v in codes.items())
            codebook_lines.append(f"- {col}: {code_str}")
        if codebook_lines:
            codebook_text = "\n\nExisting Codebooks:\n" + "\n".join(codebook_lines)

    # Add doc context if provided
    doc_text = ""
    if doc_context:
        doc_text = f"\n\nDocumentation Context:\n{doc_context[:2000]}"  # Limit length

    prompt = f"""You are a clinical data analyst. Analyze this dataset schema and suggest metadata enrichments.

Dataset Columns:
{columns_text}
{codebook_text}
{doc_text}

For each column, suggest appropriate metadata enrichments. Return a JSON object with this structure:

{{
    "suggestions": [
        {{
            "operation": "set_description",
            "column": "column_name",
            "value": "Human-readable description of what this column represents"
        }},
        {{
            "operation": "set_semantic_type",
            "column": "column_name",
            "value": "identifier|demographic|clinical|temporal|outcome|measurement|coded"
        }},
        {{
            "operation": "set_codebook_entry",
            "column": "column_name",
            "value": {{"code": "0", "label": "No/n/a"}}
        }},
        {{
            "operation": "set_exclusion_pattern",
            "column": "column_name",
            "pattern": "n/a",
            "coded_value": 0,
            "context": "Use != 0 to exclude missing values"
        }}
    ]
}}

Valid operations:
- set_label, add_alias, set_description, set_semantic_type
- mark_phi, set_unit, set_codebook_entry, set_exclusion_pattern

Valid semantic_type values: identifier, demographic, clinical, temporal, outcome, measurement, coded

Focus on:
1. Descriptions that explain clinical meaning
2. Semantic types for proper categorization
3. Codebook entries for coded columns (e.g., 0=No, 1=Yes)
4. Exclusion patterns for n/a or missing value codes

Return ONLY valid JSON, no explanatory text."""

    return prompt


def validate_llm_suggestions(
    raw_json: str,
    schema: InferredSchema,
    model_id: str | None = None,
) -> ValidationResult:
    """
    Validate LLM-generated suggestions against schema.

    Args:
        raw_json: Raw JSON string from LLM
        schema: InferredSchema for column validation

    Returns:
        ValidationResult with valid suggestions and rejected items
    """
    # Parse JSON
    parsed = parse_json_response(raw_json)
    if parsed is None:
        return ValidationResult(
            valid=False,
            errors=["Failed to parse JSON response"],
        )

    # Validate top-level structure
    shape_result = validate_shape(parsed, "metadata_patch")
    if not shape_result.valid:
        return ValidationResult(
            valid=False,
            errors=shape_result.errors,
        )

    # Get all valid column names
    valid_columns = set()
    if schema.patient_id_column:
        valid_columns.add(schema.patient_id_column)
    valid_columns.update(schema.outcome_columns)
    valid_columns.update(schema.time_columns)
    valid_columns.update(schema.event_columns)
    valid_columns.update(schema.categorical_columns)
    valid_columns.update(schema.continuous_columns)

    # Valid operations
    valid_operations = {op.value for op in PatchOperation}

    # Process suggestions
    suggestions: list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch] = []
    rejected: list[dict[str, Any]] = []

    raw_suggestions = parsed.get("suggestions", []) if isinstance(parsed, dict) else []

    for idx, item in enumerate(raw_suggestions):
        if not isinstance(item, dict):
            rejected.append({"index": idx, "reason": "Not a dict", "item": item})
            continue

        operation = item.get("operation")
        column = item.get("column")

        # Validate operation
        if operation not in valid_operations:
            rejected.append(
                {
                    "index": idx,
                    "reason": f"Invalid operation: {operation}",
                    "item": item,
                }
            )
            continue

        # Validate column (for column-specific operations)
        if operation != PatchOperation.SET_RELATIONSHIP.value:
            if column not in valid_columns:
                rejected.append(
                    {
                        "index": idx,
                        "reason": f"Unknown column: {column}",
                        "item": item,
                    }
                )
                continue

        # Create patch based on operation type
        try:
            patch = _create_patch_from_suggestion(item, model_id=model_id)
            if patch:
                suggestions.append(patch)
            else:
                rejected.append(
                    {
                        "index": idx,
                        "reason": "Failed to create patch",
                        "item": item,
                    }
                )
        except Exception as e:
            rejected.append(
                {
                    "index": idx,
                    "reason": f"Error creating patch: {e}",
                    "item": item,
                }
            )

    logger.info(
        "enrichment_validation_result",
        valid=len(suggestions),
        rejected=len(rejected),
    )

    return ValidationResult(
        valid=len(suggestions) > 0 or len(raw_suggestions) == 0,
        suggestions=suggestions,
        rejected=rejected,
    )


def _create_patch_from_suggestion(
    item: dict[str, Any],
    model_id: str | None = None,
) -> MetadataPatch | ExclusionPatternPatch | RelationshipPatch | None:
    """Create patch object from suggestion dict."""
    operation = item.get("operation")
    now = datetime.now(UTC)
    patch_id = str(uuid4())

    if operation == PatchOperation.SET_EXCLUSION_PATTERN.value:
        return ExclusionPatternPatch(
            patch_id=patch_id,
            column=item["column"],
            pattern=item.get("pattern", ""),
            coded_value=item.get("coded_value", 0),
            context=item.get("context", ""),
            auto_apply=item.get("auto_apply", False),
            status=PatchStatus.PENDING,
            created_at=now,
            provenance="llm",
            model_id=model_id,
            confidence=item.get("confidence"),
        )
    elif operation == PatchOperation.SET_RELATIONSHIP.value:
        return RelationshipPatch(
            patch_id=patch_id,
            columns=item.get("columns", []),
            relationship_type=item.get("relationship_type", "correlation"),
            rule=item.get("rule", ""),
            inference=item.get("inference", ""),
            confidence=item.get("confidence", 0.8),
            status=PatchStatus.PENDING,
            created_at=now,
            provenance="llm",
            model_id=model_id,
        )
    else:
        return MetadataPatch(
            patch_id=patch_id,
            operation=PatchOperation(operation),
            column=item["column"],
            value=item.get("value"),
            status=PatchStatus.PENDING,
            created_at=now,
            provenance="llm",
            model_id=model_id,
            confidence=item.get("confidence"),
        )


def generate_enrichment_suggestions(
    schema: InferredSchema,
    doc_context: str | None = None,
    model_id: str | None = None,
) -> list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch]:
    """
    Generate enrichment suggestions using LLM.

    Args:
        schema: InferredSchema with column information
        doc_context: Optional documentation context
        model_id: LLM model to use (default: from config)

    Returns:
        List of pending patches (empty list if LLM unavailable)
    """
    try:
        # Build prompt
        prompt = build_enrichment_prompt(schema, doc_context=doc_context)

        # Get LLM client
        if model_id:
            client = OllamaClient(model=model_id)
        else:
            client = OllamaClient()

        # Generate response
        raw_response = client.generate(
            prompt=prompt,
            system_prompt="You are a clinical data analyst. Return only valid JSON.",
            json_mode=True,
        )

        if not raw_response:
            logger.warning("enrichment_llm_empty_response")
            return []

        # Validate and parse suggestions
        result = validate_llm_suggestions(raw_response, schema, model_id=model_id)

        if not result.valid and result.errors:
            logger.warning(
                "enrichment_validation_failed",
                errors=result.errors,
            )

        logger.info(
            "enrichment_suggestions_generated",
            count=len(result.suggestions),
            model_id=model_id,
        )

        return result.suggestions

    except Exception as e:
        logger.error(
            "enrichment_generation_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return []
