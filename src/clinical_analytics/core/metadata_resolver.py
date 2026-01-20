"""
Metadata Resolver for ADR011 - LLM-Assisted Metadata Enrichment.

This module implements deterministic merge resolution for metadata patches.
Resolves metadata from multiple sources with defined precedence rules.

Merge Precedence (deterministic):
1. Base (column names from InferredSchema)
2. Inferred (from DictionaryMetadata)
3. Accepted patches (chronological order)

Later patches override earlier patches for the same operation on the same column.
Rejected and pending patches are excluded from resolution.

Usage:
    from clinical_analytics.core.metadata_resolver import resolve_metadata

    resolved = resolve_metadata(schema, patches)
    print(resolved.columns["age"].description)
"""

from typing import Any

import structlog

from clinical_analytics.core.metadata_patch import (
    ExclusionPatternPatch,
    MetadataPatch,
    PatchOperation,
    PatchStatus,
    RelationshipPatch,
    ResolvedColumnMetadata,
    ResolvedDatasetMetadata,
    ResolvedExclusionPattern,
    ResolvedRelationship,
    SemanticType,
)
from clinical_analytics.core.schema_inference import InferredSchema

logger = structlog.get_logger(__name__)


def resolve_metadata(
    schema: InferredSchema,
    patches: list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch],
) -> ResolvedDatasetMetadata:
    """
    Resolve metadata from schema and patches with deterministic precedence.

    Args:
        schema: InferredSchema with column information and optional DictionaryMetadata
        patches: List of metadata patches to apply

    Returns:
        ResolvedDatasetMetadata with merged column metadata

    Merge Precedence:
        1. Base: Column names from schema
        2. Inferred: DictionaryMetadata descriptions/codebooks
        3. Accepted patches: Applied in chronological order

    Notes:
        - Rejected patches are excluded
        - Pending patches are excluded
        - Later patches override earlier patches for same operation+column
        - Resolution is deterministic (same inputs = same output)
    """
    # Collect all column names from schema
    all_columns: set[str] = set()
    if schema.patient_id_column:
        all_columns.add(schema.patient_id_column)
    all_columns.update(schema.outcome_columns)
    all_columns.update(schema.time_columns)
    all_columns.update(schema.event_columns)
    all_columns.update(schema.categorical_columns)
    all_columns.update(schema.continuous_columns)

    # Initialize resolved columns with base (column names only)
    resolved_columns: dict[str, ResolvedColumnMetadata] = {}
    for col in all_columns:
        resolved_columns[col] = ResolvedColumnMetadata(name=col)

    # Apply inferred metadata from DictionaryMetadata (precedence level 2)
    if schema.dictionary_metadata:
        dict_meta = schema.dictionary_metadata
        for col in resolved_columns:
            # Apply descriptions
            desc = dict_meta.get_description(col)
            if desc:
                resolved_columns[col] = _update_column_metadata(
                    resolved_columns[col],
                    description=desc,
                )

            # Apply codebooks from dictionary
            if col in dict_meta.codebooks:
                resolved_columns[col] = _update_column_metadata(
                    resolved_columns[col],
                    codebook=dict_meta.codebooks[col],
                )

    # Filter to only accepted patches
    accepted_patches = [p for p in patches if hasattr(p, "status") and p.status == PatchStatus.ACCEPTED]

    # Sort patches chronologically (deterministic)
    accepted_patches.sort(key=lambda p: p.created_at)

    # Collect relationships separately
    relationships: list[ResolvedRelationship] = []

    # Apply patches in chronological order (precedence level 3)
    for patch in accepted_patches:
        if isinstance(patch, RelationshipPatch):
            relationships.append(
                ResolvedRelationship(
                    columns=patch.columns,
                    relationship_type=patch.relationship_type,
                    rule=patch.rule,
                    inference=patch.inference,
                    confidence=patch.confidence,
                )
            )
        elif isinstance(patch, ExclusionPatternPatch):
            col = patch.column
            if col in resolved_columns:
                resolved_columns[col] = _apply_exclusion_pattern(
                    resolved_columns[col],
                    patch,
                )
        elif isinstance(patch, MetadataPatch):
            col = patch.column
            if col not in resolved_columns:
                # Column referenced by patch but not in schema - create it
                resolved_columns[col] = ResolvedColumnMetadata(name=col)

            resolved_columns[col] = _apply_metadata_patch(
                resolved_columns[col],
                patch,
            )

    logger.info(
        "metadata_resolved",
        column_count=len(resolved_columns),
        patch_count=len(accepted_patches),
        relationship_count=len(relationships),
    )

    return ResolvedDatasetMetadata(
        columns=resolved_columns,
        relationships=relationships if relationships else None,
    )


def _update_column_metadata(
    col_meta: ResolvedColumnMetadata,
    **kwargs: Any,
) -> ResolvedColumnMetadata:
    """Create updated ResolvedColumnMetadata with new values."""
    return ResolvedColumnMetadata(
        name=col_meta.name,
        label=kwargs.get("label", col_meta.label),
        description=kwargs.get("description", col_meta.description),
        semantic_type=kwargs.get("semantic_type", col_meta.semantic_type),
        unit=kwargs.get("unit", col_meta.unit),
        is_phi=kwargs.get("is_phi", col_meta.is_phi),
        aliases=kwargs.get("aliases", col_meta.aliases.copy()),
        codebook=kwargs.get("codebook", col_meta.codebook),
        exclusion_patterns=kwargs.get("exclusion_patterns", col_meta.exclusion_patterns),
    )


def _apply_metadata_patch(
    col_meta: ResolvedColumnMetadata,
    patch: MetadataPatch,
) -> ResolvedColumnMetadata:
    """Apply a single metadata patch to column metadata."""
    if patch.operation == PatchOperation.SET_LABEL:
        return _update_column_metadata(col_meta, label=patch.value)

    elif patch.operation == PatchOperation.SET_DESCRIPTION:
        return _update_column_metadata(col_meta, description=patch.value)

    elif patch.operation == PatchOperation.SET_SEMANTIC_TYPE:
        # Convert string to SemanticType enum
        semantic_type = SemanticType(patch.value) if isinstance(patch.value, str) else patch.value
        return _update_column_metadata(col_meta, semantic_type=semantic_type)

    elif patch.operation == PatchOperation.SET_UNIT:
        return _update_column_metadata(col_meta, unit=patch.value)

    elif patch.operation == PatchOperation.MARK_PHI:
        return _update_column_metadata(col_meta, is_phi=bool(patch.value))

    elif patch.operation == PatchOperation.ADD_ALIAS:
        new_aliases = col_meta.aliases.copy()
        if patch.value not in new_aliases:
            new_aliases.append(patch.value)
        return _update_column_metadata(col_meta, aliases=new_aliases)

    elif patch.operation == PatchOperation.SET_CODEBOOK_ENTRY:
        # Codebook entry value is {"code": "0", "label": "n/a"}
        codebook = col_meta.codebook.copy() if col_meta.codebook else {}
        if isinstance(patch.value, dict):
            code = str(patch.value.get("code", ""))
            label = patch.value.get("label", "")
            codebook[code] = label
        return _update_column_metadata(col_meta, codebook=codebook)

    # Unknown operation - return unchanged
    return col_meta


def _apply_exclusion_pattern(
    col_meta: ResolvedColumnMetadata,
    patch: ExclusionPatternPatch,
) -> ResolvedColumnMetadata:
    """Apply an exclusion pattern patch to column metadata."""
    patterns = list(col_meta.exclusion_patterns) if col_meta.exclusion_patterns else []
    patterns.append(
        ResolvedExclusionPattern(
            pattern=patch.pattern,
            coded_value=patch.coded_value,
            context=patch.context,
            auto_apply=patch.auto_apply,
        )
    )
    return _update_column_metadata(col_meta, exclusion_patterns=patterns)
